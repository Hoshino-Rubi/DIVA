using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Any2Xsampa_Desktop.Services;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using Windows.ApplicationModel.DataTransfer;
using Windows.Foundation;
using Windows.Media.Core;
using Windows.Media.Playback;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.System;
using Path = System.IO.Path;

namespace Any2Xsampa_Desktop;

public sealed class PhoneRow : INotifyPropertyChanged
{
    public int Index { get; init; }
    public double Start { get; init; }
    public double End { get; init; }
    public string TimeLabel => $"{Start:F2}–{End:F2}";
    public string MachineIpa { get; init; } = "";
    private string xsampa = "";
    public string XSampa { get => string.IsNullOrEmpty(xsampa) ? UiLanguage.Text("缺少转换规则", "No conversion rule") : xsampa; init => xsampa=value; }
    private string editedIpa = "";
    public string EditedIpa
    {
        get => editedIpa;
        set { if (value == editedIpa) return; editedIpa=value; PropertyChanged?.Invoke(this,new(nameof(EditedIpa))); Changed?.Invoke(); }
    }
    public Action? Changed { get; set; }
    public void RefreshLanguage() => PropertyChanged?.Invoke(this,new(nameof(XSampa)));
    public event PropertyChangedEventHandler? PropertyChanged;
}
public sealed record ModelChoice(string Id, string Label, bool Available, string Detail);
public sealed record ScoreRow(double Start, double Duration, int Pitch, string Phones, bool Placeholder) : INotifyPropertyChanged
{
    public string Label => UiLanguage.Text(
        $"{Start:F2} 秒 / {Duration:F2} 秒 / MIDI {Pitch} / {(Placeholder ? "待定占位：" : "")}{Phones}",
        $"{Start:F2} s / {Duration:F2} s / MIDI {Pitch} / {(Placeholder ? "Placeholder: " : "")}{Phones}");
    public event PropertyChangedEventHandler? PropertyChanged;
    public void RefreshLanguage() => PropertyChanged?.Invoke(this,new(nameof(Label)));
}

public sealed partial class MainPage : Page
{
    private readonly LocalTaskService service;
    private readonly ObservableCollection<PhoneRow> rows = new();
    private readonly ObservableCollection<ScoreRow> scoreRows = new();
    private readonly MediaPlayer media = new();
    private readonly DispatcherTimer playbackTimer = new() { Interval=TimeSpan.FromMilliseconds(60) };
    private CancellationTokenSource? active, draftDebounce;
    private readonly SemaphoreSlim draftGate = new(1);
    private string audio="", resultPath="", resultHash="", outputDir="";
    private string scorePath="", scoreHash="", scoreSource="";
    private double audioDuration, playStart, playEnd;
    private double[] peaks = Array.Empty<double>();
    private bool initialized, closed, dirty, closingAuthorized;
    private bool setupPreferencesReady;
    private bool languageReady, modelSetupEnabled, refreshingModelChoices;
    private readonly Dictionary<DependencyObject, Action> presentations = new();
    private Action? statusTitlePresentation, statusMessagePresentation;
    private readonly List<(string Id, bool Available, string Detail)> modelChoices = new();
    private long editVersion;
    private readonly Queue<string> log = new();
    public MainPage(LocalTaskService service)
    {
        this.service=service;
        InitializeComponent();
        InitializeLanguageSelection();
        PhoneList.ItemsSource=rows;
        ScoreList.ItemsSource=scoreRows;
        Player.SetMediaPlayer(media);
        service.Log += message => DispatcherQueue.TryEnqueue(() => Log(message));
        service.Progress += e => DispatcherQueue.TryEnqueue(() =>
        {
            var stage=e.TryGetProperty("stage",out var s)?s.GetString():"";
            if(stage is "model_download" or "model_import" or "model_ready")
            {
                ModelDownloadProgress.Visibility=Visibility.Visible;
                if(e.TryGetProperty("total_bytes",out var size) && size.GetInt64()>0 && e.TryGetProperty("downloaded_bytes",out var done))
                {
                    ModelDownloadProgress.Value=done.GetInt64()*100d/size.GetInt64();
                    Show(ModelSetupStatus, () => UiLanguage.Text($"{e.GetProperty("file").GetString()} · {done.GetInt64()/1048576d:F1}/{size.GetInt64()/1048576d:F1} MiB", $"{e.GetProperty("file").GetString()} · {done.GetInt64()/1048576d:F1}/{size.GetInt64()/1048576d:F1} MiB"));
                }
                Message(() => stage=="model_ready"?UiLanguage.Text("模型已安装并校验", "Models installed and verified"):UiLanguage.Text("正在安装模型，可取消；已下载部分会保留", "Installing models; you can cancel and keep downloaded data"));
            }
            else if(e.TryGetProperty("chunk_total",out var total) && total.GetInt32()>0)
            {
                var index=e.TryGetProperty("chunk_index",out var current)?current.GetInt32():0;
                if(e.TryGetProperty("processed_seconds",out var processed) && e.TryGetProperty("total_seconds",out var seconds) && seconds.GetDouble()>0)
                    WholeProgress.Value=Math.Clamp(processed.GetDouble()/seconds.GetDouble()*100,0,100);
                Message(() => UiLanguage.Text($"{(stage is "pitch_analysis" or "note_transcription"?"音符识别":"整首识别")}：第 {index}/{total.GetInt32()} 段 · {ProgressStage(stage)}", $"{(stage is "pitch_analysis" or "note_transcription"?"Note transcription":"Whole-song recognition")}: chunk {index}/{total.GetInt32()} · {ProgressStage(stage)}"));
            }
            else Message(() => ProgressStage(stage));
            Log(e.ToString());
        });
        playbackTimer.Tick += (_,_) =>
        {
            if(media.PlaybackSession.Position.TotalSeconds>=playEnd)
            {
                if(LoopBox.IsChecked==true)media.PlaybackSession.Position=TimeSpan.FromSeconds(playStart);
                else StopPlayback();
            }
        };
        media.MediaFailed += (_,e)=>DispatcherQueue.TryEnqueue(()=>
        {
            Log(e.ErrorMessage);
            Error(new PresentationException(() => UiLanguage.Text("音频播放失败。请选择有效的 PCM WAV 文件，并检查音频输出设备。", "Audio playback failed. Select a valid PCM WAV file and check the audio output device.")));
        });
    }

    // Store presentation callbacks only: changing language never reloads task state or starts I/O.
    private void Show(TextBlock target, Func<string> text)
    { Action render=()=>target.Text=text(); presentations[target]=render; render(); }
    private void Title(Func<string> text)
    { statusTitlePresentation=()=>StatusBar.Title=text(); statusTitlePresentation(); }
    private void Message(Func<string> text)
    { statusMessagePresentation=()=>StatusBar.Message=text(); statusMessagePresentation(); }

    private void InitializeLanguageSelection()
    {
        languageReady=false;
        string preference=UiLanguage.Preference;
        LanguageBox.SelectedItem=LanguageBox.Items.OfType<ComboBoxItem>().FirstOrDefault(x=>x.Tag?.ToString()==preference);
        languageReady=true;
        AutoDownloadBox.IsChecked=false;
        AutoDownloadBox.IsEnabled=false;
        AutoDownloadBox.Content=new TextBlock { Text=UiLanguage.Text("仅手动开始下载；启动时不会联网下载模型", "Downloads start manually; startup never downloads models"), TextWrapping=TextWrapping.Wrap };
    }

    private void Language_Changed(object sender, SelectionChangedEventArgs e)
    {
        if(!languageReady || LanguageBox.SelectedItem is not ComboBoxItem item)return;
        string preference=item.Tag?.ToString()??"system";
        try
        {
            UiLanguage.SetPreference(preference);
        }
        catch(Exception ex){Error(ex);}
        RenderModelChoices();
        foreach(var render in presentations.Values.ToArray())render();
        statusTitlePresentation?.Invoke();statusMessagePresentation?.Invoke();
        foreach(var row in rows)row.RefreshLanguage();
        foreach(var row in scoreRows)row.RefreshLanguage();
        AutoDownloadBox.Content=new TextBlock { Text=UiLanguage.Text("仅手动开始下载；启动时不会联网下载模型", "Downloads start manually; startup never downloads models"), TextWrapping=TextWrapping.Wrap };
    }

    private static string ProgressStage(string? stage) => stage switch
    {
        "loading"=>UiLanguage.Text("加载模型……", "Loading model…"),
        "recognizing"=>UiLanguage.Text("识别中……", "Recognizing…"),
        "saving"=>UiLanguage.Text("保存与验证……", "Saving and verifying…"),
        "validating"=>UiLanguage.Text("核验音频与运行包……", "Checking audio and local runtime…"),
        "pitch_analysis"=>UiLanguage.Text("分析音高……", "Analyzing pitch…"),
        "note_transcription"=>UiLanguage.Text("识别音符……", "Transcribing notes…"),
        _=>UiLanguage.Text("正在处理……", "Processing…")
    };

    private sealed class PresentationException(Func<string> text) : InvalidOperationException(text())
    { public string LocalizedMessage => text(); }

    private static string MissingNoteSetup() => UiLanguage.Text(
        "GAME 音符模型尚未就绪。请导入离线模型文件夹或选择已有 GAME 模型目录；也可关闭自动生成 VSQX，仅识别音素。启动不会下载模型。",
        "The GAME note model is not ready. Import an offline model folder or select an existing GAME model directory. You can also turn off automatic VSQX generation to recognize phones only. Startup never downloads models.");

    private static string FriendlyError(Exception error)
    {
        if(error is PresentationException localized)return localized.LocalizedMessage;
        string detail=error.ToString();
        if(error is TimeoutException || detail.Contains("timed out",StringComparison.OrdinalIgnoreCase) || detail.Contains("timeout",StringComparison.OrdinalIgnoreCase))
            return UiLanguage.Text("连接超时。请检查网络或代理后手动继续下载；已下载部分会保留，也可导入离线模型文件夹。不会自动重试。", "The connection timed out. Check your network or proxy, then resume manually; downloaded data is kept. You can also import an offline model folder. No automatic retry.");
        if(detail.Contains("urlopen",StringComparison.OrdinalIgnoreCase) || detail.Contains("HTTP",StringComparison.OrdinalIgnoreCase) || detail.Contains("SSL",StringComparison.OrdinalIgnoreCase) || detail.Contains("network",StringComparison.OrdinalIgnoreCase) || detail.Contains("connection",StringComparison.OrdinalIgnoreCase) || detail.Contains("socket",StringComparison.OrdinalIgnoreCase))
            return UiLanguage.Text("无法连接下载服务。请检查网络和代理设置后手动继续，或导入离线模型文件夹；已下载部分会保留。", "Cannot reach the download service. Check your network and proxy, then resume manually, or import an offline model folder. Downloaded data is kept.");
        if(error is FileNotFoundException or DirectoryNotFoundException || detail.Contains("WinError",StringComparison.OrdinalIgnoreCase) || detail.Contains("snapshot",StringComparison.OrdinalIgnoreCase) || detail.Contains("model",StringComparison.OrdinalIgnoreCase) || detail.Contains("模型",StringComparison.Ordinal))
            return UiLanguage.Text("本地模型或运行文件尚未就绪。请在模型安装区手动下载、导入离线模型，或选择已有模型目录后重试；启动不会自动下载。", "Local model or runtime files are not ready. Use Model setup to download manually, import offline models, or select an existing model folder, then retry. Startup never downloads files.");
        return UiLanguage.Text("操作未完成。请检查输入和本地配置后重试；可导出诊断包查看错误详情。", "The operation could not be completed. Check your input and local configuration, then retry. Export diagnostics for error details.");
    }

    private async void Page_Loaded(object sender,RoutedEventArgs e)
    {
        if(initialized)return; initialized=true;
        App.Window.Closed += (_,_) => {
            App.StartupTrace("page-close-start");
            closed=true; active?.Cancel(); draftDebounce?.Cancel(); playbackTimer.Stop();
            App.StartupTrace("page-close-dispose-media");
            Player.SetMediaPlayer(null);
            media.Source = null;
            media.Dispose();
            App.StartupTrace("page-close-complete");
        };
        App.Window.AppWindow.Closing += async (_, args) =>
        {
            if(closingAuthorized || !dirty)return;
            args.Cancel=true;
            try { await SaveDraft(); if(!dirty){closingAuthorized=true;App.Window.Close();} }
            catch(Exception error){Error(error);}
        };
        await RefreshModels();
        await RefreshNoteRuntime();
        await InitializeModelSetup();
        if (App.StartupReportPath is string startupReport)
        {
            try
            {
                var doctor = await service.CallAsync("doctor", new { });
                await Task.Delay(250);
                UpdateLayout();
                bool ready = doctor.GetProperty("transport").GetString() == "stdio"
                    && RunButton.ActualWidth > 0 && RunButton.ActualHeight > 0
                    && RunButton.Visibility == Visibility.Visible;
                var uiChecks = await CaptureStartupUiAsync(startupReport);
                ready &= uiChecks.Passed;
                using var report = new FileStream(startupReport, FileMode.CreateNew, FileAccess.Write);
                await JsonSerializer.SerializeAsync(report, new {
                    status = ready ? "passed" : "failed", version = ReleaseUpdates.CurrentTag,
                    real_winui_window_loaded = true, local_worker_ready = true,
                    inference_button_visible = RunButton.ActualWidth > 0 && RunButton.ActualHeight > 0,
                    run_button_width = RunButton.ActualWidth, run_button_height = RunButton.ActualHeight,
                    ui_checks = uiChecks,
                    model_forwards = 0, model_downloads = 0,
                    state_separate_from_program = !service.StateDirectory.StartsWith(service.Root + Path.DirectorySeparatorChar,
                        StringComparison.OrdinalIgnoreCase)
                });
            }
            catch (Exception ex)
            {
                using var report = new FileStream(startupReport, FileMode.CreateNew, FileAccess.Write);
                await JsonSerializer.SerializeAsync(report, new { status = "failed",
                    error_code = ex.GetType().Name, model_forwards = 0, model_downloads = 0 });
            }
            finally { DispatcherQueue.TryEnqueue(() => App.Window.Close()); }
            return;
        }
        try
        {
            string session=Path.Combine(service.StateDirectory,"local-ui-session.json");
            if(File.Exists(session))
            {
                using var saved=JsonDocument.Parse(await File.ReadAllTextAsync(session));
                var last=saved.RootElement.GetProperty("result_path").GetString();
                if(last is not null && File.Exists(last))
                {
                    await LoadResult(last);
                    if(saved.RootElement.TryGetProperty("score_delivery",out var delivery))await LoadScore(delivery);
                }
            }
        }catch(Exception error){Log(UiLanguage.Text("上次会话未恢复：", "Previous session could not be restored: ")+error.Message);}
    }
    private void Log(string line) { log.Enqueue(line);while(log.Count>70)log.Dequeue();ActivityLog.Text=string.Join(Environment.NewLine,log); }
    private void Error(Exception e) { service.Diagnostics.Record("desktop", "failed", e);StatusBar.Severity=InfoBarSeverity.Error;Title(() => UiLanguage.Text("操作未完成", "Operation incomplete"));Message(() => FriendlyError(e));Log(e.Message); }
    private async Task RefreshModels()
    {
        try
        {
            var data=await service.CallAsync("models",new{});
            modelChoices.Clear();
            foreach(var m in data.GetProperty("models").EnumerateArray())
            {
                bool ok=m.GetProperty("available").GetBoolean();string id=m.GetProperty("id").GetString()!;
                string detail=ok?m.GetProperty("quality").GetString()!:m.GetProperty("reason").GetString()!;
                modelChoices.Add((id,ok,detail));
            }
            RenderModelChoices();
            string gpu=await LocalTaskService.GpuStatusAsync(service.Diagnostics);
            Show(GpuStatus, () => LocalizeGpuStatus(gpu));
            Show(ModelStatus, () => modelChoices.Count==0 || !modelChoices.Any(m=>m.Available)
                ?UiLanguage.Text("尚无可用真实模型。请手动下载或导入离线模型；启动不会下载，也不会自动使用 fixture。", "No real model is ready. Download manually or import offline models; startup never downloads models or falls back to fixtures.")
                :UiLanguage.Text("运行包已核验；请选择模型。质量状态与可加载状态分开。", "Runtime verified; please select a model. Loadability and recognition quality are separate."));
        }catch(Exception e){Error(e);Show(ModelStatus, () => UiLanguage.Text("环境检查失败；未启动推理。", "Environment check failed; inference has not started."));}
    }

    private void RenderModelChoices()
    {
        string? selected=(ModelBox.SelectedItem as ModelChoice)?.Id;
        var choices=modelChoices.Select(m=>
        {
            string quality=LocalizeModelQuality(m.Detail);
            string detail=m.Available
                ?UiLanguage.Text($"本地模型已就绪；质量标记：{quality}（可加载不代表准确率已验证）", $"Local model ready; quality tag: {quality} (loadability does not establish accuracy)")
                :UiLanguage.Text("模型文件尚未就绪。请手动下载或导入离线模型；不会自动下载。", "Model files are not ready. Download manually or import offline models; downloads never start automatically.");
            return new ModelChoice(m.Id,m.Id+" · "+(m.Available?UiLanguage.Text("可用", "Available"):UiLanguage.Text("未就绪", "Not ready")),m.Available,detail);
        }).ToList();
        refreshingModelChoices=true;
        try
        {
            ModelBox.ItemsSource=choices;
            if(selected is not null)ModelBox.SelectedItem=choices.FirstOrDefault(m=>m.Id==selected);
        }
        finally { refreshingModelChoices=false; }
    }

    private static string LocalizeModelQuality(string quality) => quality
        .Replace("研究候选", UiLanguage.Text("研究候选", "Research candidate"), StringComparison.Ordinal)
        .Replace("非质量验收通过", UiLanguage.Text("非质量验收通过", "Not quality-approved"), StringComparison.Ordinal);

    private static string LocalizeGpuStatus(string text)
    {
        // The transport supplies localized snapshots. Reuse the captured probe without running it again.
        if(text.StartsWith("GPU 状态不可用：",StringComparison.Ordinal) || text.StartsWith("GPU status unavailable:",StringComparison.Ordinal))
            return UiLanguage.Text("GPU 状态不可用；请检查显卡驱动。未运行模型。", "GPU status unavailable; check the GPU driver. No model was run.");
        string chinese=text.Replace("GPU (name / used MiB / total MiB): ", "GPU 快照（名称 / 已用 MiB / 总 MiB）：")
            .Replace("GPU status timed out. No model was run.", "GPU 状态查询超时；未运行模型")
            .Replace("GPU status unavailable: ", "GPU 状态不可用：")
            .Replace("; budget: 7 GiB per process / 14 GiB total GPU", "；进程预算 7 GiB / 整卡 14 GiB");
        string english=text.Replace("GPU 快照（名称 / 已用 MiB / 总 MiB）：", "GPU (name / used MiB / total MiB): ")
            .Replace("GPU 状态查询超时；未运行模型", "GPU status timed out. No model was run.")
            .Replace("GPU 状态不可用：", "GPU status unavailable: ")
            .Replace("；进程预算 7 GiB / 整卡 14 GiB", "; budget: 7 GiB per process / 14 GiB total GPU");
        return UiLanguage.Text(chinese,english);
    }

    private string SetupPreferencesPath => Path.Combine(service.StateDirectory, "model-setup-preferences.json");

    private async Task InitializeModelSetup()
    {
        try
        {
            var info=await service.CallAsync("model_setup_status",new{});
            bool enabled=modelSetupEnabled=info.GetProperty("enabled").GetBoolean();
            InstallModelsButton.IsEnabled=ImportModelsButton.IsEnabled=enabled;
            AutoDownloadBox.IsChecked=false;
            AutoDownloadBox.IsEnabled=false;
            if(File.Exists(SetupPreferencesPath))
            {
                using var doc=JsonDocument.Parse(await File.ReadAllTextAsync(SetupPreferencesPath));
                if(doc.RootElement.TryGetProperty("proxy",out var proxy))ModelProxyBox.Text=proxy.GetString() ?? "";
            }
            setupPreferencesReady=true;
            bool ready=info.GetProperty("ready").GetBoolean();
            Show(ModelSetupStatus, () => !enabled?UiLanguage.Text("此版本未配置模型安装清单；可选择已有本地模型。启动不会下载。", "No model installation catalog is configured; select existing local models. Startup never downloads files.")
                :ready?UiLanguage.Text("模型文件已完整核验；识别阶段不联网。", "Model files verified; recognition runs offline.")
                :UiLanguage.Text("模型尚未安装。请手动点击“下载 / 继续下载模型”，或导入离线模型文件夹（约 2.4 GiB）。启动不会下载。", "Models are not installed. Click Download / Resume models manually, or import an offline model folder (about 2.4 GiB). Startup never downloads files."));
            if(!ready)
            {
                StatusBar.Severity=InfoBarSeverity.Warning;
                Title(() => UiLanguage.Text("需要安装本地模型", "Local models need setup"));
                Message(() => UiLanguage.Text("请手动下载或导入离线模型后开始识别；本次启动没有下载模型。", "Download manually or import offline models before recognition; this startup has not downloaded any models."));
            }
        }
        catch(Exception ex){Error(ex);Show(ModelSetupStatus, () => UiLanguage.Text("无法检查本地模型。请检查模型目录或导入离线模型；启动不会下载。", "Local models could not be checked. Check the model directory or import offline models; startup never downloads files."));}
    }

    private async Task SaveSetupPreferences()
    {
        if(!setupPreferencesReady)return;
        Directory.CreateDirectory(Path.GetDirectoryName(SetupPreferencesPath)!);
        string tmp=SetupPreferencesPath+".tmp";
        await File.WriteAllTextAsync(tmp,JsonSerializer.Serialize(new{auto_download=false,proxy=ModelProxyBox.Text.Trim()}));
        File.Move(tmp,SetupPreferencesPath,true);
    }

    private async void AutoDownload_Changed(object sender,RoutedEventArgs e)
    {
        if(!setupPreferencesReady)return;
        try{await SaveSetupPreferences();}catch(Exception ex){Error(ex);}
    }

    private async Task InstallModels(string? offline=null)
    {
        if(active is not null)return;
        active=new CancellationTokenSource();
        SetBusy(true);ModelDownloadProgress.Visibility=Visibility.Visible;ModelDownloadProgress.Value=0;
        StatusBar.Severity=InfoBarSeverity.Informational;Title(() => offline is null?UiLanguage.Text("安装模型", "Installing models"):UiLanguage.Text("导入离线模型", "Importing offline models"));
        try
        {
            await SaveSetupPreferences();
            var p=new Dictionary<string,object>{{"proxy",ModelProxyBox.Text.Trim()}};
            if(offline is not null)p["offline_dir"]=offline;
            await service.CallAsync("install_models",p,active.Token);
            await RefreshModels();NoteSnapshotBox.Text="";await RefreshNoteRuntime();
            ModelDownloadProgress.Value=100;Show(ModelSetupStatus, () => UiLanguage.Text("模型安装完成，文件哈希均已核验。请选择真实模型后识别。", "Models installed and file hashes verified. Select a real model to recognize audio."));
            StatusBar.Severity=InfoBarSeverity.Success;Message(() => UiLanguage.Text("模型就绪。", "Models ready."));
        }
        catch(OperationCanceledException)
        {Show(ModelSetupStatus, () => UiLanguage.Text("已取消安装，下载部分保留；点击“继续下载”可恢复。", "Installation cancelled; downloaded data is kept. Click Resume to continue."));StatusBar.Severity=InfoBarSeverity.Warning;Message(() => ModelSetupStatus.Text);}
        catch(Exception ex)
        {
            service.Diagnostics.Record("desktop", "failed", ex);Log(ex.Message);
            StatusBar.Severity=InfoBarSeverity.Warning;
            Title(() => UiLanguage.Text("安装未完成", "Installation incomplete"));
            Show(ModelSetupStatus, () => FriendlyError(ex));
            Message(() => ModelSetupStatus.Text);
        }
        finally{active.Dispose();active=null;SetBusy(false);}
    }

    private async void InstallModels_Click(object sender,RoutedEventArgs e)=>await InstallModels();
    private async void ImportModels_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        try
        {
            var picker=new FolderPicker { CommitButtonText=UiLanguage.Text("导入离线模型", "Import offline models") };picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();
            if(folder is not null)await InstallModels(folder.Path);
        }
        catch(Exception ex){Error(ex);}
    }

    private void Model_Changed(object sender,SelectionChangedEventArgs e)
    { if(!refreshingModelChoices && ModelBox.SelectedItem is ModelChoice)Show(ModelStatus, () => (ModelBox.SelectedItem as ModelChoice)?.Detail??""); }
    private FileOpenPicker OpenPicker(string extension)
    {
        var p=new FileOpenPicker { CommitButtonText=UiLanguage.Text("选择", "Select") };p.FileTypeFilter.Add(extension);
        WinRT.Interop.InitializeWithWindow.Initialize(p,App.WindowHandle);return p;
    }
    private async void Browse_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        try{var file=await OpenPicker(".wav").PickSingleFileAsync();if(file is not null)await SetAudio(file.Path);}catch(Exception ex){Error(ex);}
    }
    private async Task SetAudio(string file)
    {
        StopPlayback();
        if(!string.IsNullOrEmpty(resultPath))
        {
            await SaveDraft();draftDebounce?.Cancel();rows.Clear();resultPath="";resultHash="";outputDir="";dirty=false;
            scoreRows.Clear();scorePath="";scoreHash="";scoreSource="";Show(ScoreStatus, () => UiLanguage.Text("尚无音符数据。", "No note data yet."));
            Show(ResultStatus, () => UiLanguage.Text("已切换音频；旧结果保留在原目录，不与新音频混用。", "Audio changed; previous results remain in their original folder and are not used with this audio."));Show(DraftStatus, () => "");
        }
        var info=await service.CallAsync("inspect_audio",new{audio=file});
        var storage=await StorageFile.GetFileFromPathAsync(file);
        media.Source=MediaSource.CreateFromStorageFile(storage);
        audio=file;AudioPathBox.Text=file;audioDuration=info.GetProperty("duration").GetDouble();
        peaks=info.GetProperty("peaks").EnumerateArray().Select(x=>x.GetDouble()).ToArray();
        StartBox.Maximum=Math.Max(0,audioDuration-.1);StartBox.Value=0;DurationBox.Value=Math.Min(8,audioDuration);
        bool waveformWarning=info.TryGetProperty("waveform_warning",out _);
        Show(AudioStatus, () => UiLanguage.Text($"{audioDuration:F2} 秒 · {info.GetProperty("sample_rate").GetInt32()} Hz · 原文件只读", $"{audioDuration:F2} seconds · {info.GetProperty("sample_rate").GetInt32()} Hz · Original file is read-only")
            +(waveformWarning?UiLanguage.Text(" · 波形预览仅支持 16-bit PCM；音频未修改", " · Waveform preview requires 16-bit PCM; audio was not modified"):""));
        DrawWaveform();
    }
    private void Audio_DragOver(object sender,DragEventArgs e)
    { e.AcceptedOperation=active is null && e.DataView.Contains(StandardDataFormats.StorageItems)?DataPackageOperation.Copy:DataPackageOperation.None; }
    private async void Audio_Drop(object sender,DragEventArgs e)
    {
        try
        {
            if(active is not null || !e.DataView.Contains(StandardDataFormats.StorageItems))return;
            var files=await e.DataView.GetStorageItemsAsync();
            if(files.Count!=1 || files[0] is not StorageFile file || !file.FileType.Equals(".wav",StringComparison.OrdinalIgnoreCase))throw new PresentationException(() => UiLanguage.Text("请拖入一个 PCM WAV 文件。", "Drop a single PCM WAV file."));
            await SetAudio(file.Path);
        }catch(Exception ex){Error(ex);}
    }
    private void Waveform_SizeChanged(object sender,SizeChangedEventArgs e)=>DrawWaveform();
    private void Range_Changed(NumberBox sender,NumberBoxValueChangedEventArgs e)=>DrawWaveform();
    private void Waveform_PointerPressed(object sender,PointerRoutedEventArgs e)
    {
        if(audioDuration<=0 || Waveform.ActualWidth<=0)return;
        StartBox.Value=Math.Clamp(e.GetCurrentPoint(Waveform).Position.X/Waveform.ActualWidth*audioDuration,0,Math.Max(0,audioDuration-DurationBox.Value));
    }
    private void DrawWaveform()
    {
        if(Waveform is null || StartBox is null || DurationBox is null)return;
        Waveform.Children.Clear();double w=Waveform.ActualWidth,h=Waveform.ActualHeight;
        if(w<=0 || h<=0 || audioDuration<=0)return;
        var stroke=new SolidColorBrush(Colors.Teal);
        for(int i=0;i<peaks.Length;i++)
        {
            double x=(i+.5)*w/peaks.Length;
            Waveform.Children.Add(new Line{X1=x,X2=x,Y1=h*(.5-peaks[i]*.44),Y2=h*(.5+peaks[i]*.44),Stroke=stroke,StrokeThickness=1});
        }
        if(!double.IsFinite(StartBox.Value)||!double.IsFinite(DurationBox.Value))return;
        double left=StartBox.Value/audioDuration*w,right=Math.Min(audioDuration,StartBox.Value+DurationBox.Value)/audioDuration*w;
        var range=new Rectangle{Width=Math.Max(0,right-left),Height=h,Fill=new SolidColorBrush(Colors.DeepSkyBlue),Opacity=.2};
        Canvas.SetLeft(range,left);Waveform.Children.Add(range);
    }
    private void Play(double start,double end)
    {
        if(string.IsNullOrEmpty(audio))return;
        playStart=Math.Clamp(start,0,audioDuration);playEnd=Math.Clamp(end,playStart,audioDuration);
        media.PlaybackSession.Position=TimeSpan.FromSeconds(playStart);media.Play();playbackTimer.Start();
    }
    private void StopPlayback(){playbackTimer.Stop();media.Pause();}
    private void Play_Click(object sender,RoutedEventArgs e)=>Play(StartBox.Value,StartBox.Value+DurationBox.Value);
    private void Stop_Click(object sender,RoutedEventArgs e)=>StopPlayback();
    private void Phone_Selected(object sender,SelectionChangedEventArgs e)
    { if(PhoneList.SelectedItem is PhoneRow row)Play(Math.Max(0,row.Start-.18),Math.Min(audioDuration,row.End+.18)); }
    private void SetBusy(bool busy)
    {
        ScorePitchMode.IsEnabled=!busy;
        ScoreTranspose.IsEnabled=!busy && SelectedScorePitchMode=="manual";
        SetTaskControlsBusy(busy);
    }

    private void SetTaskControlsBusy(bool busy)
    {RunButton.IsEnabled=!busy;CancelButton.IsEnabled=busy;BusyRing.IsActive=busy;ModelBox.IsEnabled=!busy;DeviceBox.IsEnabled=!busy;PhoneList.IsEnabled=!busy;WholeMode.IsEnabled=!busy;ClipMode.IsEnabled=!busy;AnalyzeScoreButton.IsEnabled=!busy;ImportScoreButton.IsEnabled=!busy;ExportScoreButton.IsEnabled=!busy;ScoreTempo.IsEnabled=!busy;ImportNotesButton.IsEnabled=!busy;NoteSnapshotBox.IsEnabled=!busy;AutoVsqxBox.IsEnabled=!busy;InstallModelsButton.IsEnabled=!busy && modelSetupEnabled;ImportModelsButton.IsEnabled=!busy && modelSetupEnabled;AutoDownloadBox.IsEnabled=false;ModelProxyBox.IsEnabled=!busy;WholeProgress.Visibility=busy && WholeMode.IsChecked==true?Visibility.Visible:Visibility.Collapsed;if(!busy)WholeProgress.Value=0;}
    private async Task<bool> RunTask(string action,object parameters)
    {
        if(active is not null)return false;
        active=new CancellationTokenSource();SetBusy(true);StatusBar.Severity=InfoBarSeverity.Informational;Title(() => action=="recognize-whole"?UiLanguage.Text("整首真实推理", "Whole-song inference"):action=="recognize"?UiLanguage.Text("片段真实推理", "Clip inference"):action is "build_score" or "import_score"?UiLanguage.Text("生成 VSQX", "Generating VSQX"):UiLanguage.Text("显式回放（零推理）", "Explicit replay (no inference)"));
        try
        {
            await SaveDraft();
            var result=await service.CallAsync(action,parameters,active.Token);
            if(action is "build_score" or "import_score")await LoadScore(result);
            else await LoadResult(result.GetProperty("result_path").GetString()!);
            StatusBar.Severity=InfoBarSeverity.Success;Message(() => UiLanguage.Text("完成；结果已保存。", "Done; results saved."));
            return true;
        }
        catch(OperationCanceledException){StatusBar.Severity=InfoBarSeverity.Warning;Message(() => UiLanguage.Text("任务已取消；没有自动重试。", "Task cancelled; no automatic retry."));}
        catch(Exception e){Error(e);}
        finally{active.Dispose();active=null;SetBusy(false);}
        return false;
    }
    private async void Recognize_Click(object sender,RoutedEventArgs e)
    {
        if(ModelBox.SelectedItem is not ModelChoice {Available:true} model){Error(new PresentationException(() => UiLanguage.Text("请先明确选择可用的真实模型。缺少模型时请手动下载或导入离线模型。", "Select an available real model first. Download manually or import offline models if needed.")));return;}
        if(string.IsNullOrEmpty(audio)){Error(new PresentationException(() => UiLanguage.Text("请先选择音频。", "Select an audio file first.")));return;}
        string device=(DeviceBox.SelectedItem as ComboBoxItem)?.Tag?.ToString()??"cuda";
        try
        {
            if(AutoVsqxBox.IsChecked==true)
            {
                var runtime=await service.CallAsync("note_runtime",NoteRuntimeParameters());
                if(!runtime.GetProperty("available").GetBoolean())throw new PresentationException(MissingNoteSetup);
            }
            bool success=WholeMode.IsChecked==true
                ? await RunTask("recognize-whole",new{audio,model=model.Id,device})
                : await RunTask("recognize",new{audio,model=model.Id,device,start=StartBox.Value,duration=DurationBox.Value});
            if(success && AutoVsqxBox.IsChecked==true)await GenerateGameScore();
        }
        catch(Exception ex){Error(ex);}
    }
    private async void Replay_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        try
        {
            var file=await OpenPicker(".json").PickSingleFileAsync();if(file is null)return;
            var p=new Dictionary<string,object>{{"fixture",file.Path},{"start",StartBox.Value},{"duration",DurationBox.Value}};
            if(!string.IsNullOrEmpty(audio))p["audio"]=audio;
            await RunTask("replay",p);
        }catch(Exception ex){Error(ex);}
    }
    private void Cancel_Click(object sender,RoutedEventArgs e)=>active?.Cancel();

    private async void AnalyzeScore_Click(object sender,RoutedEventArgs e)
    {
        if(string.IsNullOrEmpty(resultPath)){Error(new PresentationException(() => UiLanguage.Text("请先完成识别或回放，再分析音高和时值。", "Complete recognition or replay before analyzing pitch and duration.")));return;}
        if(!double.IsFinite(ScoreTempo.Value)){Error(new PresentationException(() => UiLanguage.Text("请输入有效 BPM。", "Enter a valid BPM.")));return;}
        await GenerateGameScore();
    }

    private Dictionary<string,object> NoteRuntimeParameters()
    {
        var p=new Dictionary<string,object>();
        if(!string.IsNullOrWhiteSpace(NoteSnapshotBox.Text))p["snapshot"]=NoteSnapshotBox.Text.Trim();
        return p;
    }

    private async Task RefreshNoteRuntime()
    {
        try
        {
            var info=await service.CallAsync("note_runtime",NoteRuntimeParameters());
            bool ready=info.GetProperty("available").GetBoolean();
            if(ready)NoteSnapshotBox.Text=info.GetProperty("snapshot").GetString()!;
            Show(NoteRuntimeStatus, () => ready?UiLanguage.Text("GAME 已配置 · CPU 四线程 · 运行时无需联网", "GAME configured · four CPU threads · runs offline"):MissingNoteSetup());
        }
        catch(Exception ex){Log(ex.Message);Show(NoteRuntimeStatus, MissingNoteSetup);}
    }

    private async void BrowseNoteModel_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        try
        {
            var picker=new FolderPicker { CommitButtonText=UiLanguage.Text("选择 GAME 模型", "Select GAME model") };picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();if(folder is null)return;
            NoteSnapshotBox.Text=folder.Path;await RefreshNoteRuntime();
        }
        catch(Exception ex){Error(ex);}
    }

    private Task<bool> GenerateGameScore(string? notes=null)
    {
        var p=NoteRuntimeParameters();
        p["result_path"]=resultPath;p["result_sha256"]=resultHash;p["tempo"]=ScoreTempo.Value;p["backend"]="game";
        if(notes is not null){p.Remove("snapshot");p["notes_path"]=notes;}
        return RunTask("build_score",p);
    }

    private async void ImportNotes_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        if(string.IsNullOrEmpty(resultPath)){Error(new PresentationException(() => UiLanguage.Text("先加载音素结果，再选择同一音频的 GAME 音符 JSON。", "Load phone results first, then select GAME note JSON for the same audio.")));return;}
        try
        {
            var file=await OpenPicker(".json").PickSingleFileAsync();
            if(file is not null)await GenerateGameScore(file.Path);
        }
        catch(Exception ex){Error(ex);}
    }

    private async void ImportScore_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        if(string.IsNullOrEmpty(resultPath)){Error(new PresentationException(() => UiLanguage.Text("请先加载识别结果，再导入已有 VSQX。", "Load recognition results before importing an existing VSQX.")));return;}
        try
        {
            var file=await OpenPicker(".vsqx").PickSingleFileAsync();if(file is null)return;
            await RunTask("import_score",new{result_path=resultPath,result_sha256=resultHash,template_path=file.Path});
        }
        catch(Exception ex){Error(ex);}
    }

    private string SelectedScorePitchMode => (ScorePitchMode.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "miku_v4c_octave";

    private void ScorePitchMode_Changed(object sender,SelectionChangedEventArgs e)
    {
        if(ScoreTranspose is not null)
            ScoreTranspose.IsEnabled=ScorePitchMode.IsEnabled && SelectedScorePitchMode=="manual";
    }

    private void ScoreTranspose_Changed(NumberBox sender,NumberBoxValueChangedEventArgs e)
    {
        double value=double.IsFinite(e.NewValue)?Math.Clamp(Math.Round(e.NewValue),-36,36):0;
        if(sender.Value!=value)sender.Value=value;
    }

    private async void ExportScore_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        if(string.IsNullOrEmpty(scorePath)){Error(new PresentationException(() => UiLanguage.Text("请先分析音高/时值，或导入已有 VSQX。", "Analyze pitch and duration, or import an existing VSQX first.")));return;}
        active=new CancellationTokenSource();SetBusy(true);
        try
        {
            var picker=new FolderPicker { CommitButtonText=UiLanguage.Text("导出到此文件夹", "Export to this folder") };picker.FileTypeFilter.Add("*");WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();if(folder is null)return;
            string output=Path.Combine(folder.Path,$"DIVA-{DateTime.Now:yyyyMMdd-HHmmssfff}.vsqx");
            var parameters=new Dictionary<string,object>
            {
                ["score_path"]=scorePath,["score_sha256"]=scoreHash,["output"]=output,
                ["pitch_mode"]=SelectedScorePitchMode
            };
            if(SelectedScorePitchMode=="manual")
                parameters["transpose_semitones"]=(int)Math.Clamp(Math.Round(double.IsFinite(ScoreTranspose.Value)?ScoreTranspose.Value:0),-36,36);
            var response=await service.CallAsync("export_score",parameters,active.Token);
            var adaptation=response.GetProperty("pitch_adaptation");
            int transpose=adaptation.GetProperty("transpose_semitones").GetInt32();
            var after=adaptation.GetProperty("after");
            int below=after.GetProperty("below_range_note_count").GetInt32(),above=after.GetProperty("above_range_note_count").GetInt32();
            string exported=response.GetProperty("output").GetString()!;
            StatusBar.Severity=InfoBarSeverity.Success;
            Message(() => UiLanguage.Text(
                $"已导出 VSQX：{exported}；实际移调 {transpose:+0;-0;0} 半音；剩余音域外：低于 {below}，高于 {above}。",
                $"VSQX exported: {exported}; actual transposition: {transpose:+0;-0;0} semitones; remaining out of range: {below} below, {above} above."));
        }
        catch(Exception ex){Error(ex);}
        finally{active.Dispose();active=null;SetBusy(false);}
    }

    private void Score_Selected(object sender,SelectionChangedEventArgs e)
    {
        if(ScoreList.SelectedItem is ScoreRow row && scoreSource is "audio_pyin_draft" or "audio_game")Play(row.Start,row.Start+row.Duration);
    }

    private async Task LoadScore(JsonElement response)
    {
        string file=response.GetProperty("score_path").GetString()!;
        byte[] bytes=await File.ReadAllBytesAsync(file);using var doc=JsonDocument.Parse(bytes);var data=doc.RootElement;
        string digest=Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if(digest!=response.GetProperty("score_sha256").GetString() || data.GetProperty("result_sha256").GetString()!=resultHash)
            throw new PresentationException(() => UiLanguage.Text("音符数据与识别结果版本不一致。", "Note data and recognition results belong to different versions."));
        scoreRows.Clear();scorePath=file;scoreHash=digest;scoreSource=data.GetProperty("source").GetString()!;
        ScorePitchMode.SelectedIndex=scoreSource=="supplied_vsqx_phone_draft"?0:1;
        int placeholders=0;
        foreach(var note in data.GetProperty("notes").EnumerateArray())
        {
            double start=note.GetProperty("start_sec").GetDouble(),duration=note.GetProperty("duration_sec").GetDouble();
            int pitch=note.GetProperty("pitch").GetInt32();bool placeholder=note.GetProperty("placeholder").GetBoolean();if(placeholder)placeholders++;
            string phones=string.Join(" ",note.GetProperty("v4c_phonemes").EnumerateArray().Select(x=>x.GetString()));
            scoreRows.Add(new ScoreRow(start,duration,pitch,phones,placeholder));
        }
        Show(ScoreStatus, () => scoreSource=="supplied_vsqx_phone_draft"
            ?UiLanguage.Text($"已导入 {scoreRows.Count} 个音符；原 VSQX 另存只读备份，并生成独立音素草稿。音高、时值、歌词和轨道结构保持；机器音素按时间关联。", $"Imported {scoreRows.Count} notes; the original VSQX has a separate read-only backup and an independent phone draft. Pitch, timing, lyrics and track structure are retained; machine phones are linked by time.")
            :UiLanguage.Text($"{(scoreSource=="audio_game"?"GAME":"pYIN")}：{scoreRows.Count} 个音符，{placeholders} 个音符缺少可用发音而使用中性占位。VSQX 已生成，音高、起点和时值已自动读回核对。", $"{(scoreSource=="audio_game"?"GAME":"pYIN")}: {scoreRows.Count} notes; {placeholders} use neutral placeholders because usable pronunciation evidence is missing. VSQX generated; pitch, onset and duration were read back and checked automatically."));
        string session=Path.Combine(service.StateDirectory,"local-ui-session.json");
        await File.WriteAllTextAsync(session+".tmp",JsonSerializer.Serialize(new{result_path=resultPath,score_delivery=response}));
        File.Move(session+".tmp",session,true);
    }

    private async Task LoadResult(string file)
    {
        draftDebounce?.Cancel();rows.Clear();resultPath="";
        byte[] bytes=await File.ReadAllBytesAsync(file);using var doc=JsonDocument.Parse(bytes);var data=doc.RootElement;
        string source=data.GetProperty("input").GetProperty("path").GetString()!;
        await using(var sourceStream=File.OpenRead(source))
        {
            string sourceHash=Convert.ToHexString(await SHA256.HashDataAsync(sourceStream)).ToLowerInvariant();
            if(sourceHash!=data.GetProperty("input").GetProperty("sha256").GetString())
                throw new PresentationException(() => UiLanguage.Text("结果绑定的原音频已改变；拒绝将旧音素对应到新声音。", "The original audio bound to these results has changed; old phones cannot be linked to different audio."));
        }
        if(audio!=source)await SetAudio(source);
        double begin=data.GetProperty("input").GetProperty("start_sec").GetDouble();
        if(data.TryGetProperty("scope",out var scope) && scope.GetString()=="whole_file")
        {WholeMode.IsChecked=true;StartBox.Value=0;DurationBox.Value=Math.Min(20,audioDuration);}
        else
        {ClipMode.IsChecked=true;StartBox.Value=begin;DurationBox.Value=data.GetProperty("input").GetProperty("end_sec").GetDouble()-begin;}
        string digest=Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        var projection=await service.CallAsync("preview_xsampa",new{result_path=file,result_sha256=digest});
        var xsampas=projection.GetProperty("xsampa").EnumerateArray().Select(x=>x.GetString()!).ToArray();
        if(xsampas.Length!=data.GetProperty("segments").GetArrayLength())
            throw new PresentationException(() => UiLanguage.Text("X-SAMPA 导出预览数量不匹配；原始结果未改变。", "The X-SAMPA export preview count does not match; original results are unchanged."));
        int i=0;foreach(var s in data.GetProperty("segments").EnumerateArray())
        {
            string ipa=s.GetProperty("normalized_ipa").GetString()!,xs=xsampas[i];
            rows.Add(new PhoneRow{Index=i++,Start=s.GetProperty("start_sec").GetDouble(),End=s.GetProperty("end_sec").GetDouble(),MachineIpa=ipa,EditedIpa=ipa,XSampa=xs});
        }
        resultPath=file;resultHash=digest;outputDir=Path.GetDirectoryName(file)!;
        string draft=Path.Combine(outputDir,"review-draft.json");
        if(File.Exists(draft))
        {
            using var d=JsonDocument.Parse(await File.ReadAllTextAsync(draft));
            if(d.RootElement.GetProperty("result_sha256").GetString()==resultHash)
            {
                var edits=d.RootElement.GetProperty("edits").EnumerateArray().ToArray();
                if(edits.Length!=rows.Count || edits.Where((x,index)=>x.GetProperty("index").GetInt32()!=index).Any())
                    throw new PresentationException(() => UiLanguage.Text("草稿索引不连续或数量错配；未应用草稿。", "Draft indices or count do not match; the draft was not applied."));
                foreach(var edit in edits)rows[edit.GetProperty("index").GetInt32()].EditedIpa=edit.GetProperty("ipa").GetString()!;
            }
        }
        foreach(var row in rows)row.Changed=QueueDraft;
        int missing=projection.GetProperty("unmapped_phone_count").GetInt32();
        double missingRatio=projection.GetProperty("unmapped_phone_ratio").GetDouble();
        int missingTypes=projection.GetProperty("unmapped_label_counts").EnumerateObject().Count();
        string resultMode=data.GetProperty("mode").GetString()??"";
        Show(ResultStatus, () => UiLanguage.Text($"来源：{resultMode} · {rows.Count} 个音素 · X-SAMPA 导出缺失 {missing}/{rows.Count}（{missingRatio:P1}，{missingTypes} 类）。转换覆盖不代表识别准确率；时间为发射线索。", $"Source: {resultMode} · {rows.Count} phones · X-SAMPA export missing {missing}/{rows.Count} ({missingRatio:P1}, {missingTypes} types). Conversion coverage does not establish recognition accuracy; times are emission cues."));
        Show(DraftStatus, () => UiLanguage.Text("机器结果只读；修改将另存草稿", "Machine results are read-only; edits are saved as a separate draft"));UseDraftBox.IsChecked=false;dirty=false;
        scoreRows.Clear();scorePath="";scoreHash="";scoreSource="";Show(ScoreStatus, () => UiLanguage.Text("尚无音符数据；请分析音高/时值，或导入已有 VSQX。", "No note data yet; analyze pitch and duration, or import an existing VSQX."));
        string scoreLink=Path.Combine(outputDir,"score-link.json");
        if(File.Exists(scoreLink))
        {
            try{using var link=JsonDocument.Parse(await File.ReadAllTextAsync(scoreLink));await LoadScore(link.RootElement);}
            catch(Exception ex){Log(ex.Message);Show(ScoreStatus, () => UiLanguage.Text($"音符数据未恢复：{FriendlyError(ex)}", $"Note data was not restored: {FriendlyError(ex)}"));}
        }
        string session=Path.Combine(service.StateDirectory,"local-ui-session.json");
        Directory.CreateDirectory(Path.GetDirectoryName(session)!);
        await File.WriteAllTextAsync(session+".tmp",JsonSerializer.Serialize(new{result_path=file}));
        File.Move(session+".tmp",session,true);
    }
    private async void QueueDraft()
    {
        dirty=true;editVersion++;
        draftDebounce?.Cancel();var timer=draftDebounce=new CancellationTokenSource();Show(DraftStatus, () => UiLanguage.Text("待保存……", "Waiting to save…"));
        try{await Task.Delay(600,timer.Token);if(!closed)await SaveDraft();}
        catch(OperationCanceledException){}catch(Exception e){Error(e);Show(DraftStatus, () => UiLanguage.Text("保存失败", "Save failed"));}
    }
    private async Task SaveDraft()
    {
        if(string.IsNullOrEmpty(resultPath))return;
        await draftGate.WaitAsync();
        try
        {
            long version=editVersion;
            var edits=rows.Select(r=>new{index=r.Index,ipa=r.EditedIpa.Trim()}).ToArray();
            if(edits.Any(x=>string.IsNullOrWhiteSpace(x.ipa))){Show(DraftStatus, () => UiLanguage.Text("空音素未保存", "Empty phones were not saved"));throw new PresentationException(() => UiLanguage.Text("人工草稿包含空音素；请补全后保存或导出。机器结果未改变。", "The review draft contains empty phones; fill them in before saving or exporting. Machine results are unchanged."));}
            await service.CallAsync("draft",new{result_path=resultPath,result_sha256=resultHash,edits});if(version==editVersion)dirty=false;Show(DraftStatus, () => UiLanguage.Text("草稿已保存（非真值）", "Draft saved (not ground truth)"));
        }finally{draftGate.Release();}
    }
    private async void ExportDiagnostics_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var picker = new FolderPicker { CommitButtonText=UiLanguage.Text("导出诊断包", "Export diagnostics") }; picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker, App.WindowHandle);
            var folder = await picker.PickSingleFolderAsync(); if (folder is null) return;
            string file = Path.Combine(folder.Path, $"DIVA-diagnostics-{DateTime.UtcNow:yyyyMMdd-HHmmssfff}.zip");
            await service.ExportDiagnosticsAsync(file);
            StatusBar.Severity = InfoBarSeverity.Success;
            Title(() => UiLanguage.Text("诊断已导出", "Diagnostics exported"));
            Message(() => UiLanguage.Text("已保存到所选目录；不包含音频、原始日志或完整路径，未自动上传。", "Saved to the selected folder; excludes audio, raw logs and full paths. Nothing was uploaded."));
        }
        catch (Exception ex) { Error(ex); }
    }

    private async void CheckUpdate_Click(object sender, RoutedEventArgs e)
    {
        if (active is not null) return;
        if (sender is not Button button) return;
        button.IsEnabled = false;
        try
        {
            var update = await ReleaseUpdates.CheckAsync(CancellationToken.None);
            var dialog = new ContentDialog {
                XamlRoot = XamlRoot, Title = update.IsNewer ? UiLanguage.Text($"发现 {update.Tag}", $"Found {update.Tag}") : UiLanguage.Text("已是当前正式版", "Already on the current stable release"),
                Content = UiLanguage.Text($"本机 {ReleaseUpdates.CurrentTag}；公开正式版 {update.Tag}。旧安装和结果不会被覆盖。", $"Installed: {ReleaseUpdates.CurrentTag}; public stable release: {update.Tag}. Existing installations and results will not be overwritten."),
                CloseButtonText = UiLanguage.Text("关闭", "Close"), PrimaryButtonText = update.IsNewer ? UiLanguage.Text("打开下载页", "Open download page") : ""
            };
            if (await dialog.ShowAsync() == ContentDialogResult.Primary)
                await Launcher.LaunchUriAsync(update.Page);
        }
        catch (Exception ex) { service.Diagnostics.Record("update-check", "failed", ex);Error(new PresentationException(() => UiLanguage.Text("更新检查失败。请检查网络或代理后重试；程序和模型未改变。", "Update check failed. Check your network or proxy, then retry; the program and models are unchanged."))); }
        finally { button.IsEnabled = true; }
    }

    private async void Doctor_Click(object sender,RoutedEventArgs e)
    {if(active is not null)return;try{var d=await service.CallAsync("doctor",new{});Log(d.ToString());await RefreshModels();}catch(Exception ex){Error(ex);}}
    private async void Unload_Click(object sender,RoutedEventArgs e)
    {if(active is not null){Error(new PresentationException(() => UiLanguage.Text("请先取消或完成当前任务。", "Cancel or finish the current task first.")));return;}try{await service.CallAsync("unload",new{});Show(ModelStatus, () => UiLanguage.Text("已释放本应用的模型 worker 与显存；未结束其他进程。", "This app's model worker and GPU memory were released; other processes were left running."));}catch(Exception ex){Error(ex);}}
    private async Task Export(string format)
    {
        if(string.IsNullOrEmpty(resultPath))return;
        try
        {
            await SaveDraft();var picker=new FolderPicker { CommitButtonText=UiLanguage.Text("导出到此文件夹", "Export to this folder") };picker.FileTypeFilter.Add("*");WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();if(folder is null)return;
            string file=Path.Combine(folder.Path,$"phones-{DateTime.Now:yyyyMMdd-HHmmssfff}.{format}.txt");
            await service.CallAsync("export",new{result_path=resultPath,format,output=file,use_draft=UseDraftBox.IsChecked==true});Message(() => UiLanguage.Text($"已导出：{file}", $"Exported: {file}"));
        }catch(Exception e){Error(e);}
    }
    private async void ExportIpa_Click(object sender,RoutedEventArgs e)=>await Export("ipa");
    private async void ExportXSampa_Click(object sender,RoutedEventArgs e)=>await Export("xsampa");
    private async void OpenOutput_Click(object sender,RoutedEventArgs e)
    {try{if(Directory.Exists(outputDir))await Launcher.LaunchFolderAsync(await StorageFolder.GetFolderFromPathAsync(outputDir));}catch(Exception ex){Error(ex);}}
}
