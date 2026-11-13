using Any2Xsampa_Desktop.Services;
using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.System;

namespace Any2Xsampa_Desktop;

public sealed partial class MainPage
{
    // Null delegates the default models/cantonese4 location to the local service.
    private string? cantoneseBundleDir;
    private string cantoneseOutputDir = "";
    private bool cantoneseAvailable;
    private string cantoneseReason = "";

    private void WorkspaceTabs_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!ReferenceEquals(e.OriginalSource, WorkspaceTabs)) return;
        var visibility = CantoneseTab is not null && ReferenceEquals(WorkspaceTabs.SelectedItem, CantoneseTab)
            ? Visibility.Collapsed : Visibility.Visible;
        RunButton.Visibility = visibility;
        AutoVsqxBox.Visibility = visibility;
    }

    private void InitializeCantonese()
    {
        RenderCantoneseLanguage();
        Show(CantoneseStatus, () => UiLanguage.Text("正在检查模型……", "Checking model…"));
    }

    private void RenderCantoneseLanguage()
    {
        CantoneseTab.Header = UiLanguage.Text("粤语增强", "Cantonese");
        CantoneseModelButton.Content = UiLanguage.Text("选择文件夹", "Model folder");
        CantonesePrepareButton.Content = UiLanguage.Text("准备模型", "Prepare");
        CantoneseRunButton.Content = UiLanguage.Text("开始", "Start");
        CantoneseOpenButton.Content = UiLanguage.Text("打开结果", "Open results");
        CantoneseJyutpingBox.Header = UiLanguage.Text("粤拼 / Jyutping", "Jyutping");
    }

    private void SetCantoneseBusy(bool busy)
    {
        CantoneseModelButton.IsEnabled = !busy;
        CantonesePrepareButton.IsEnabled = !busy;
        CantoneseRunButton.IsEnabled = !busy;
        CantoneseOpenButton.IsEnabled = !busy && !string.IsNullOrEmpty(cantoneseOutputDir);
    }

    private void ShowCantoneseReady() => Show(CantoneseStatus, () => cantoneseAvailable
        ? UiLanguage.Text("模型就绪", "Model ready")
        : cantoneseReason == "separator_code_missing"
            ? UiLanguage.Text("需准备", "Prepare needed")
            : UiLanguage.Text("请选择模型文件夹", "Select a model folder"));

    private void ApplyCantoneseStatus(JsonElement status)
    {
        cantoneseAvailable = status.GetProperty("available").GetBoolean();
        cantoneseReason = status.TryGetProperty("reason", out var reason) ? reason.GetString() ?? "" : "";
        CantoneseBundlePath.Text = status.GetProperty("bundle_dir").GetString() ?? "models/cantonese4";
        ShowCantoneseReady();
    }

    private void ResetCantoneseResult()
    {
        cantoneseOutputDir = "";
        CantoneseJyutpingBox.Text = "";
        CantoneseIpaBox.Text = "";
        SetCantoneseBusy(active is not null);
        ShowCantoneseReady();
    }

    private async Task<bool> RefreshCantoneseStatus(CancellationToken token = default)
    {
        try
        {
            var status = await service.CallAsync("cantonese4_status", new { bundle_dir = cantoneseBundleDir }, token);
            ApplyCantoneseStatus(status);
            return cantoneseAvailable;
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            cantoneseAvailable = false;
            cantoneseReason = "";
            Log(ex.Message);
            Show(CantoneseStatus, () => UiLanguage.Text("模型检查失败", "Model check failed"));
            return false;
        }
    }

    private async void CantoneseModel_Click(object sender, RoutedEventArgs e)
    {
        if (active is not null) return;
        active = new CancellationTokenSource();
        SetBusy(true);
        try
        {
            var picker = new FolderPicker { CommitButtonText = UiLanguage.Text("选择模型文件夹", "Select model folder") };
            picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker, App.WindowHandle);
            var folder = await picker.PickSingleFolderAsync();
            active.Token.ThrowIfCancellationRequested();
            if (folder is null) return;
            cantoneseBundleDir = folder.Path;
            CantoneseBundlePath.Text = folder.Path;
            await RefreshCantoneseStatus(active.Token);
        }
        catch (OperationCanceledException) { ShowCantoneseCancelled(); }
        catch (Exception ex) { ShowCantoneseError(ex); }
        finally { active.Dispose(); active = null; SetBusy(false); }
    }

    // Explicit click only: status checks, folder selection and inference never prepare files.
    private async void CantonesePrepare_Click(object sender, RoutedEventArgs e)
    {
        if (active is not null) return;
        var parameters = new
        {
            bundle_dir = cantoneseBundleDir,
            proxy = string.IsNullOrWhiteSpace(ModelProxyBox.Text) ? null : ModelProxyBox.Text.Trim()
        };
        active = new CancellationTokenSource();
        SetBusy(true);
        StatusBar.Severity = InfoBarSeverity.Informational;
        Title(() => UiLanguage.Text("粤语增强", "Cantonese"));
        Show(CantoneseStatus, () => UiLanguage.Text("准备中……", "Preparing…"));
        Message(() => CantoneseStatus.Text);
        try
        {
            var status = await service.CallAsync("cantonese4_prepare", parameters, active.Token);
            ApplyCantoneseStatus(status);
            StatusBar.Severity = cantoneseAvailable ? InfoBarSeverity.Success : InfoBarSeverity.Warning;
            Message(() => CantoneseStatus.Text);
        }
        catch (OperationCanceledException) { ShowCantoneseCancelled(); }
        catch (Exception ex) { ShowCantoneseError(ex); }
        finally { active.Dispose(); active = null; SetBusy(false); }
    }

    private async void CantoneseRun_Click(object sender, RoutedEventArgs e)
    {
        if (active is not null) return;
        if (string.IsNullOrEmpty(audio))
        {
            Show(CantoneseStatus, () => UiLanguage.Text("请先选择 WAV", "Select a WAV first"));
            return;
        }
        if (!double.IsFinite(StartBox.Value) || StartBox.Value < 0 ||
            !double.IsFinite(DurationBox.Value) || DurationBox.Value <= 0)
        {
            Show(CantoneseStatus, () => UiLanguage.Text("请检查片段范围", "Check the clip range"));
            return;
        }
        // Snapshot the shared selection before the first await.
        var parameters = new
        {
            audio, bundle_dir = cantoneseBundleDir,
            scope = WholeMode.IsChecked == true ? "whole_file" : "clip",
            start = StartBox.Value, duration = DurationBox.Value
        };
        active = new CancellationTokenSource();
        SetBusy(true);
        StatusBar.Severity = InfoBarSeverity.Informational;
        Title(() => UiLanguage.Text("粤语增强", "Cantonese"));
        try
        {
            if (!await RefreshCantoneseStatus(active.Token))
            {
                StatusBar.Severity = InfoBarSeverity.Warning;
                Message(() => CantoneseStatus.Text);
                return;
            }
            Show(CantoneseStatus, () => UiLanguage.Text("处理中……", "Processing…"));
            Message(() => CantoneseStatus.Text);
            var result = await service.CallAsync("cantonese4", parameters, active.Token);
            // Read all fields before replacing the last successful, independent result.
            string jyutping = result.GetProperty("jyutping").GetString() ?? "";
            string ipa = result.GetProperty("ipa").GetString() ?? "";
            string directory = result.GetProperty("output_dir").GetString()!;
            CantoneseJyutpingBox.Text = jyutping;
            CantoneseIpaBox.Text = ipa;
            cantoneseOutputDir = directory;
            Show(CantoneseStatus, () => UiLanguage.Text("完成", "Done"));
            StatusBar.Severity = InfoBarSeverity.Success;
            Message(() => CantoneseStatus.Text);
        }
        catch (OperationCanceledException) { ShowCantoneseCancelled(); }
        catch (Exception ex) { ShowCantoneseError(ex); }
        finally { active.Dispose(); active = null; SetBusy(false); }
    }

    private void ShowCantoneseCancelled()
    {
        Show(CantoneseStatus, () => UiLanguage.Text("已取消", "Cancelled"));
        StatusBar.Severity = InfoBarSeverity.Warning;
        Message(() => CantoneseStatus.Text);
    }

    private void ShowCantoneseError(Exception ex)
    {
        Show(CantoneseStatus, () => UiLanguage.Text("未完成，请重试", "Failed; please retry"));
        Error(ex);
    }

    private async void CantoneseOpen_Click(object sender, RoutedEventArgs e)
    {
        if (active is not null || string.IsNullOrEmpty(cantoneseOutputDir)) return;
        try
        {
            await Launcher.LaunchFolderAsync(await StorageFolder.GetFolderFromPathAsync(cantoneseOutputDir));
        }
        catch (Exception ex) { ShowCantoneseError(ex); }
    }
}
