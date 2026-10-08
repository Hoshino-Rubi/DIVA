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
    public string XSampa { get; init; } = "";
    private string editedIpa = "";
    public string EditedIpa
    {
        get => editedIpa;
        set { if (value == editedIpa) return; editedIpa=value; PropertyChanged?.Invoke(this,new(nameof(EditedIpa))); Changed?.Invoke(); }
    }
    public Action? Changed { get; set; }
    public event PropertyChangedEventHandler? PropertyChanged;
}
public sealed record ModelChoice(string Id, string Label, bool Available, string Detail);
public sealed record ScoreRow(double Start, double Duration, int Pitch, string Phones, bool Placeholder)
{
    public string Label => $"{Start:F2} s / {Duration:F2} s / MIDI {Pitch} / {(Placeholder ? "待定占位：" : "")}{Phones}";
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
    private long editVersion;
    private readonly Queue<string> log = new();
    public MainPage(LocalTaskService service)
    {
        this.service=service;
        InitializeComponent();
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
                    ModelSetupStatus.Text=$"{e.GetProperty("file").GetString()} · {done.GetInt64()/1048576d:F1}/{size.GetInt64()/1048576d:F1} MiB";
                }
                StatusBar.Message=stage=="model_ready"?"模型已安装并校验":"正在安装模型，可取消；已下载部分会保留";
            }
            else if(e.TryGetProperty("chunk_total",out var total) && total.GetInt32()>0)
            {
                var index=e.TryGetProperty("chunk_index",out var current)?current.GetInt32():0;
                if(e.TryGetProperty("processed_seconds",out var processed) && e.TryGetProperty("total_seconds",out var seconds) && seconds.GetDouble()>0)
                    WholeProgress.Value=Math.Clamp(processed.GetDouble()/seconds.GetDouble()*100,0,100);
                StatusBar.Message=$"{(stage is "pitch_analysis" or "note_transcription"?"音符识别":"整首识别")}：第 {index}/{total.GetInt32()} 段 · {stage}";
            }
            else StatusBar.Message=stage switch {"loading"=>"加载模型……","recognizing"=>"识别中……","saving"=>"保存与验证……","validating"=>"核验音频与运行包……",_=>stage};
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
        media.MediaFailed += (_,e)=>DispatcherQueue.TryEnqueue(()=>Error(new IOException("音频播放失败："+e.ErrorMessage)));
    }

    private async void Page_Loaded(object sender,RoutedEventArgs e)
    {
        if(initialized)return; initialized=true;
        App.Window.Closed += (_,_) => { closed=true; active?.Cancel(); draftDebounce?.Cancel(); playbackTimer.Stop(); media.Dispose(); };
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
        try
        {
            string session=Path.Combine(service.Root,".state","local-ui-session.json");
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
        }catch(Exception error){Log("上次会话未恢复："+error.Message);}
    }
    private void Log(string line) { log.Enqueue(line);while(log.Count>70)log.Dequeue();ActivityLog.Text=string.Join(Environment.NewLine,log); }
    private void Error(Exception e) { StatusBar.Severity=InfoBarSeverity.Error;StatusBar.Title="操作未完成";StatusBar.Message=e.Message;Log(e.Message); }
    private async Task RefreshModels()
    {
        try
        {
            var data=await service.CallAsync("models",new{});
            var choices=data.GetProperty("models").EnumerateArray().Select(m=>
            {
                bool ok=m.GetProperty("available").GetBoolean();string id=m.GetProperty("id").GetString()!;
                string detail=ok?m.GetProperty("quality").GetString()!:m.GetProperty("reason").GetString()!;
                return new ModelChoice(id,id+(ok?" · "+detail:" · 不可用"),ok,detail);
            }).ToList();
            string? selected=(ModelBox.SelectedItem as ModelChoice)?.Id;
            ModelBox.ItemsSource=choices;
            if(selected is not null)ModelBox.SelectedItem=choices.FirstOrDefault(m=>m.Id==selected);
            GpuStatus.Text=await LocalTaskService.GpuStatusAsync();
            ModelStatus.Text=choices.Count==0?"尚未配置真实模型；不会自动使用 fixture。":"运行包已核验；请选择模型。质量状态与可加载状态分开。";
        }catch(Exception e){Error(e);ModelStatus.Text="环境检查失败；未启动推理。";}
    }

    private string SetupPreferencesPath => Path.Combine(service.Root, ".state", "model-setup-preferences.json");

    private async Task InitializeModelSetup()
    {
        try
        {
            var info=await service.CallAsync("model_setup_status",new{});
            bool enabled=info.GetProperty("enabled").GetBoolean();
            InstallModelsButton.IsEnabled=ImportModelsButton.IsEnabled=enabled;
            AutoDownloadBox.IsChecked=info.GetProperty("auto_download").GetBoolean();
            if(File.Exists(SetupPreferencesPath))
            {
                using var doc=JsonDocument.Parse(await File.ReadAllTextAsync(SetupPreferencesPath));
                AutoDownloadBox.IsChecked=doc.RootElement.GetProperty("auto_download").GetBoolean();
                ModelProxyBox.Text=doc.RootElement.GetProperty("proxy").GetString() ?? "";
            }
            setupPreferencesReady=true;
            bool ready=info.GetProperty("ready").GetBoolean();
            ModelSetupStatus.Text=!enabled?"此开发版本未配置自动下载清单。":ready?"模型文件已完整核验；识别阶段不联网。":"缺少模型，首次安装需要约 2.4 GiB 下载空间。";
            if(enabled && !ready && AutoDownloadBox.IsChecked==true)await InstallModels();
        }
        catch(Exception ex){Error(ex);ModelSetupStatus.Text="模型安装检查失败："+ex.Message;}
    }

    private async Task SaveSetupPreferences()
    {
        if(!setupPreferencesReady)return;
        Directory.CreateDirectory(Path.GetDirectoryName(SetupPreferencesPath)!);
        string tmp=SetupPreferencesPath+".tmp";
        await File.WriteAllTextAsync(tmp,JsonSerializer.Serialize(new{auto_download=AutoDownloadBox.IsChecked==true,proxy=ModelProxyBox.Text.Trim()}));
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
        StatusBar.Severity=InfoBarSeverity.Informational;StatusBar.Title=offline is null?"安装模型":"导入离线模型";
        try
        {
            await SaveSetupPreferences();
            var p=new Dictionary<string,object>{{"proxy",ModelProxyBox.Text.Trim()}};
            if(offline is not null)p["offline_dir"]=offline;
            await service.CallAsync("install_models",p,active.Token);
            await RefreshModels();NoteSnapshotBox.Text="";await RefreshNoteRuntime();
            ModelDownloadProgress.Value=100;ModelSetupStatus.Text="模型安装完成，文件哈希均已核验。请选择真实模型后识别。";
            StatusBar.Severity=InfoBarSeverity.Success;StatusBar.Message="模型就绪。";
        }
        catch(OperationCanceledException)
        {ModelSetupStatus.Text="已取消安装，下载部分保留；点击“继续下载”可恢复。";StatusBar.Severity=InfoBarSeverity.Warning;StatusBar.Message=ModelSetupStatus.Text;}
        catch(Exception ex)
        {Error(ex);ModelSetupStatus.Text="安装未完成："+ex.Message+"。可配置代理后继续下载，或导入离线模型文件夹。";}
        finally{active.Dispose();active=null;SetBusy(false);}
    }

    private async void InstallModels_Click(object sender,RoutedEventArgs e)=>await InstallModels();
    private async void ImportModels_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        try
        {
            var picker=new FolderPicker();picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();
            if(folder is not null)await InstallModels(folder.Path);
        }
        catch(Exception ex){Error(ex);}
    }

    private void Model_Changed(object sender,SelectionChangedEventArgs e)
    { if(ModelBox.SelectedItem is ModelChoice m)ModelStatus.Text=m.Detail; }
    private FileOpenPicker OpenPicker(string extension)
    {
        var p=new FileOpenPicker();p.FileTypeFilter.Add(extension);
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
            scoreRows.Clear();scorePath="";scoreHash="";scoreSource="";ScoreStatus.Text="尚无音符数据。";
            ResultStatus.Text="已切换音频；旧结果保留在原目录，不与新音频混用。";DraftStatus.Text="";
        }
        var info=await service.CallAsync("inspect_audio",new{audio=file});
        var storage=await StorageFile.GetFileFromPathAsync(file);
        media.Source=MediaSource.CreateFromStorageFile(storage);
        audio=file;AudioPathBox.Text=file;audioDuration=info.GetProperty("duration").GetDouble();
        peaks=info.GetProperty("peaks").EnumerateArray().Select(x=>x.GetDouble()).ToArray();
        StartBox.Maximum=Math.Max(0,audioDuration-.1);StartBox.Value=0;DurationBox.Value=Math.Min(8,audioDuration);
        AudioStatus.Text=$"{audioDuration:F2} 秒 · {info.GetProperty("sample_rate").GetInt32()} Hz · 原文件只读";
        if(info.TryGetProperty("waveform_warning",out var warning))AudioStatus.Text+=" · "+warning.GetString();
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
            if(files.Count!=1 || files[0] is not StorageFile file || !file.FileType.Equals(".wav",StringComparison.OrdinalIgnoreCase))throw new IOException("请拖入一个 PCM WAV 文件。");
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
    {RunButton.IsEnabled=!busy;CancelButton.IsEnabled=busy;BusyRing.IsActive=busy;ModelBox.IsEnabled=!busy;DeviceBox.IsEnabled=!busy;PhoneList.IsEnabled=!busy;WholeMode.IsEnabled=!busy;ClipMode.IsEnabled=!busy;AnalyzeScoreButton.IsEnabled=!busy;ImportScoreButton.IsEnabled=!busy;ExportScoreButton.IsEnabled=!busy;ScoreTempo.IsEnabled=!busy;ImportNotesButton.IsEnabled=!busy;NoteSnapshotBox.IsEnabled=!busy;AutoVsqxBox.IsEnabled=!busy;InstallModelsButton.IsEnabled=!busy;ImportModelsButton.IsEnabled=!busy;AutoDownloadBox.IsEnabled=!busy;ModelProxyBox.IsEnabled=!busy;WholeProgress.Visibility=busy && WholeMode.IsChecked==true?Visibility.Visible:Visibility.Collapsed;if(!busy)WholeProgress.Value=0;}
    private async Task<bool> RunTask(string action,object parameters)
    {
        if(active is not null)return false;
        active=new CancellationTokenSource();SetBusy(true);StatusBar.Severity=InfoBarSeverity.Informational;StatusBar.Title=action=="recognize-whole"?"整首真实推理":action=="recognize"?"片段真实推理":action is "build_score" or "import_score"?"生成 VSQX":"显式回放（零推理）";
        try
        {
            await SaveDraft();
            var result=await service.CallAsync(action,parameters,active.Token);
            if(action is "build_score" or "import_score")await LoadScore(result);
            else await LoadResult(result.GetProperty("result_path").GetString()!);
            StatusBar.Severity=InfoBarSeverity.Success;StatusBar.Message="完成；结果已保存。";
            return true;
        }
        catch(OperationCanceledException){StatusBar.Severity=InfoBarSeverity.Warning;StatusBar.Message="任务已取消；没有自动重试。";}
        catch(Exception e){Error(e);}
        finally{active.Dispose();active=null;SetBusy(false);}
        return false;
    }
    private async void Recognize_Click(object sender,RoutedEventArgs e)
    {
        if(ModelBox.SelectedItem is not ModelChoice {Available:true} model){Error(new InvalidOperationException("请先明确选择可用的真实模型。"));return;}
        if(string.IsNullOrEmpty(audio)){Error(new InvalidOperationException("请先选择音频。"));return;}
        string device=(DeviceBox.SelectedItem as ComboBoxItem)?.Tag?.ToString()??"cuda";
        try
        {
            if(AutoVsqxBox.IsChecked==true)
            {
                var runtime=await service.CallAsync("note_runtime",NoteRuntimeParameters());
                if(!runtime.GetProperty("available").GetBoolean())throw new InvalidOperationException(runtime.GetProperty("reason").GetString());
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
        if(string.IsNullOrEmpty(resultPath)){Error(new InvalidOperationException("请先完成识别或回放，再分析音高和时值。"));return;}
        if(!double.IsFinite(ScoreTempo.Value)){Error(new InvalidOperationException("请输入有效 BPM。"));return;}
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
            NoteRuntimeStatus.Text=ready?"GAME 已配置 · CPU 四线程 · 运行时无需联网":info.GetProperty("reason").GetString();
        }
        catch(Exception ex){NoteRuntimeStatus.Text=ex.Message;}
    }

    private async void BrowseNoteModel_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        try
        {
            var picker=new FolderPicker();picker.FileTypeFilter.Add("*");
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
        if(string.IsNullOrEmpty(resultPath)){Error(new InvalidOperationException("先加载音素结果，再选择同一音频的 GAME 音符 JSON。"));return;}
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
        if(string.IsNullOrEmpty(resultPath)){Error(new InvalidOperationException("请先加载识别结果，再导入已有 VSQX。"));return;}
        try
        {
            var file=await OpenPicker(".vsqx").PickSingleFileAsync();if(file is null)return;
            await RunTask("import_score",new{result_path=resultPath,result_sha256=resultHash,template_path=file.Path});
        }
        catch(Exception ex){Error(ex);}
    }

    private async void ExportScore_Click(object sender,RoutedEventArgs e)
    {
        if(active is not null)return;
        if(string.IsNullOrEmpty(scorePath)){Error(new InvalidOperationException("请先分析音高/时值，或导入已有 VSQX。"));return;}
        try
        {
            var picker=new FolderPicker();picker.FileTypeFilter.Add("*");WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();if(folder is null)return;
            string output=Path.Combine(folder.Path,$"Any2Xsampa-{DateTime.Now:yyyyMMdd-HHmmssfff}.vsqx");
            var response=await service.CallAsync("export_score",new{score_path=scorePath,score_sha256=scoreHash,output});
            StatusBar.Severity=InfoBarSeverity.Success;StatusBar.Message="已导出 VSQX："+response.GetProperty("output").GetString();
        }
        catch(Exception ex){Error(ex);}
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
            throw new IOException("音符数据与识别结果版本不一致。");
        scoreRows.Clear();scorePath=file;scoreHash=digest;scoreSource=data.GetProperty("source").GetString()!;
        int placeholders=0;
        foreach(var note in data.GetProperty("notes").EnumerateArray())
        {
            double start=note.GetProperty("start_sec").GetDouble(),duration=note.GetProperty("duration_sec").GetDouble();
            int pitch=note.GetProperty("pitch").GetInt32();bool placeholder=note.GetProperty("placeholder").GetBoolean();if(placeholder)placeholders++;
            string phones=string.Join(" ",note.GetProperty("v4c_phonemes").EnumerateArray().Select(x=>x.GetString()));
            scoreRows.Add(new ScoreRow(start,duration,pitch,phones,placeholder));
        }
        ScoreStatus.Text=scoreSource=="supplied_vsqx_phone_draft"
            ?$"已导入 {scoreRows.Count} 个音符；原 VSQX 另存只读备份，并生成独立音素草稿。音高、时值、歌词和轨道结构保持；机器音素按时间关联。"
            :$"{(scoreSource=="audio_game"?"GAME":"pYIN")}：{scoreRows.Count} 个音符，{placeholders} 个音符缺少可用发音而使用中性占位。VSQX 已生成，音高、起点和时值已自动读回核对。";
        string session=Path.Combine(service.Root,".state","local-ui-session.json");
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
                throw new IOException("结果绑定的原音频已改变；拒绝将旧音素对应到新声音。");
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
            throw new IOException("X-SAMPA 导出预览数量不匹配；原始结果未改变。");
        int i=0;foreach(var s in data.GetProperty("segments").EnumerateArray())
        {
            string ipa=s.GetProperty("normalized_ipa").GetString()!,xs=xsampas[i];
            rows.Add(new PhoneRow{Index=i++,Start=s.GetProperty("start_sec").GetDouble(),End=s.GetProperty("end_sec").GetDouble(),MachineIpa=ipa,EditedIpa=ipa,XSampa=string.IsNullOrEmpty(xs)?"缺少转换规则":xs});
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
                    throw new IOException("草稿索引不连续或数量错配；未应用草稿。");
                foreach(var edit in edits)rows[edit.GetProperty("index").GetInt32()].EditedIpa=edit.GetProperty("ipa").GetString()!;
            }
        }
        foreach(var row in rows)row.Changed=QueueDraft;
        int missing=projection.GetProperty("unmapped_phone_count").GetInt32();
        double missingRatio=projection.GetProperty("unmapped_phone_ratio").GetDouble();
        int missingTypes=projection.GetProperty("unmapped_label_counts").EnumerateObject().Count();
        ResultStatus.Text=$"来源：{data.GetProperty("mode").GetString()} · {rows.Count} 个音素 · X-SAMPA 导出缺失 {missing}/{rows.Count}（{missingRatio:P1}，{missingTypes} 类）。转换覆盖不代表识别准确率；时间为发射线索。";
        DraftStatus.Text="机器结果只读；修改将另存草稿";UseDraftBox.IsChecked=false;dirty=false;
        scoreRows.Clear();scorePath="";scoreHash="";scoreSource="";ScoreStatus.Text="尚无音符数据；请分析音高/时值，或导入已有 VSQX。";
        string scoreLink=Path.Combine(outputDir,"score-link.json");
        if(File.Exists(scoreLink))
        {
            try{using var link=JsonDocument.Parse(await File.ReadAllTextAsync(scoreLink));await LoadScore(link.RootElement);}
            catch(Exception ex){ScoreStatus.Text="音符数据未恢复："+ex.Message;}
        }
        string session=Path.Combine(service.Root,".state","local-ui-session.json");
        Directory.CreateDirectory(Path.GetDirectoryName(session)!);
        await File.WriteAllTextAsync(session+".tmp",JsonSerializer.Serialize(new{result_path=file}));
        File.Move(session+".tmp",session,true);
    }
    private async void QueueDraft()
    {
        dirty=true;editVersion++;
        draftDebounce?.Cancel();var timer=draftDebounce=new CancellationTokenSource();DraftStatus.Text="待保存……";
        try{await Task.Delay(600,timer.Token);if(!closed)await SaveDraft();}
        catch(OperationCanceledException){}catch(Exception e){Error(e);DraftStatus.Text="保存失败";}
    }
    private async Task SaveDraft()
    {
        if(string.IsNullOrEmpty(resultPath))return;
        await draftGate.WaitAsync();
        try
        {
            long version=editVersion;
            var edits=rows.Select(r=>new{index=r.Index,ipa=r.EditedIpa.Trim()}).ToArray();
            if(edits.Any(x=>string.IsNullOrWhiteSpace(x.ipa))){DraftStatus.Text="空音素未保存";throw new InvalidOperationException("人工草稿包含空音素；请补全后保存或导出。机器结果未改变。");}
            await service.CallAsync("draft",new{result_path=resultPath,result_sha256=resultHash,edits});if(version==editVersion)dirty=false;DraftStatus.Text="草稿已保存（非真值）";
        }finally{draftGate.Release();}
    }
    private async void Doctor_Click(object sender,RoutedEventArgs e)
    {if(active is not null)return;try{var d=await service.CallAsync("doctor",new{});Log(d.ToString());await RefreshModels();}catch(Exception ex){Error(ex);}}
    private async void Unload_Click(object sender,RoutedEventArgs e)
    {if(active is not null){Error(new InvalidOperationException("请先取消或完成当前任务。"));return;}try{await service.CallAsync("unload",new{});ModelStatus.Text="已释放本应用的模型 worker 与显存；未结束其他进程。";}catch(Exception ex){Error(ex);}}
    private async Task Export(string format)
    {
        if(string.IsNullOrEmpty(resultPath))return;
        try
        {
            await SaveDraft();var picker=new FolderPicker();picker.FileTypeFilter.Add("*");WinRT.Interop.InitializeWithWindow.Initialize(picker,App.WindowHandle);
            var folder=await picker.PickSingleFolderAsync();if(folder is null)return;
            string file=Path.Combine(folder.Path,$"phones-{DateTime.Now:yyyyMMdd-HHmmssfff}.{format}.txt");
            await service.CallAsync("export",new{result_path=resultPath,format,output=file,use_draft=UseDraftBox.IsChecked==true});StatusBar.Message="已导出："+file;
        }catch(Exception e){Error(e);}
    }
    private async void ExportIpa_Click(object sender,RoutedEventArgs e)=>await Export("ipa");
    private async void ExportXSampa_Click(object sender,RoutedEventArgs e)=>await Export("xsampa");
    private async void OpenOutput_Click(object sender,RoutedEventArgs e)
    {try{if(Directory.Exists(outputDir))await Launcher.LaunchFolderAsync(await StorageFolder.GetFolderFromPathAsync(outputDir));}catch(Exception ex){Error(ex);}}
}
