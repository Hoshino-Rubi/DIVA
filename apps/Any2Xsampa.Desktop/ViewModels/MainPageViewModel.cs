using Any2Xsampa_Desktop.Models;
using Any2Xsampa_Desktop.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Dispatching;
using System.Collections.ObjectModel;

namespace Any2Xsampa_Desktop.ViewModels;

public partial class MainPageViewModel : ObservableObject
{
    private readonly IAudioInferenceService _audioInferenceService;

    [ObservableProperty]
    private string audioPath = "samples/demo_real.wav";

    [ObservableProperty]
    private string templatePath = "tmp_validation/vocaloid_probe/phrase_probe_round2/miku_v4c_phrase_probe_round2.vsqx";

    [ObservableProperty]
    private string outputRoot = "tmp_validation/miku_v4c_product_draft/desktop_current";

    [ObservableProperty]
    private string lyrics = "喜歡你";

    [ObservableProperty]
    private string jyutpingOverride = string.Empty;

    [ObservableProperty]
    private string asrTranscriptPath = string.Empty;

    [ObservableProperty]
    private string pitchAnalysisPath = string.Empty;

    [ObservableProperty]
    private string qwen3AsrModel = string.Empty;

    [ObservableProperty]
    private string pitchesCsv = "64,67,69";

    [ObservableProperty]
    private double tempo = 128.0;

    [ObservableProperty]
    private bool useFakeAsr;

    [ObservableProperty]
    private bool timestamps;

    [ObservableProperty]
    private bool isBusy;

    [ObservableProperty]
    private string statusText = "Ready to create a Miku V4C draft VSQX.";

    [ObservableProperty]
    private string vsqxPath = string.Empty;

    [ObservableProperty]
    private string summaryPath = string.Empty;

    [ObservableProperty]
    private string productSummaryJson = string.Empty;

    public ObservableCollection<string> LogLines { get; } = new();

    public MainPageViewModel(IAudioInferenceService audioInferenceService)
    {
        _audioInferenceService = audioInferenceService;
        LogLines.Add("Desktop client initialized. Start the Python gRPC bridge on 127.0.0.1:50051 before inference.");
    }

    [RelayCommand(CanExecute = nameof(CanStartInference))]
    private async Task StartInferenceAsync()
    {
        IsBusy = true;
        StartInferenceCommand.NotifyCanExecuteChanged();
        AppendLog("Submitting InferDraftVsqx request to the local Python gRPC bridge...");
        StatusText = "Running draft inference...";

        try
        {
            var request = new DraftVsqxRequest(
                AudioPath,
                TemplatePath,
                OutputRoot,
                Lyrics,
                JyutpingOverride,
                AsrTranscriptPath,
                PitchAnalysisPath,
                Qwen3AsrModel,
                PitchesCsv,
                Tempo,
                UseFakeAsr,
                Timestamps,
                "Chinese",
                Lyrics,
                "miku_v4c_desktop_draft");

            var result = await _audioInferenceService.InferDraftVsqxAsync(request);
            await EnqueueUiAsync(() => ApplyResult(result));
        }
        catch (Exception ex)
        {
            await EnqueueUiAsync(() =>
            {
                StatusText = "gRPC bridge call failed.";
                AppendLog($"Error: {ex.Message}");
            });
        }
        finally
        {
            await EnqueueUiAsync(() =>
            {
                IsBusy = false;
                StartInferenceCommand.NotifyCanExecuteChanged();
            });
        }
    }

    [RelayCommand]
    private void ClearLog()
    {
        LogLines.Clear();
        ProductSummaryJson = string.Empty;
        StatusText = "Ready to create a Miku V4C draft VSQX.";
    }

    private bool CanStartInference() => !IsBusy;

    private void ApplyResult(DraftVsqxResult result)
    {
        VsqxPath = result.VsqxPath;
        SummaryPath = result.AudioProductSummaryPath;
        ProductSummaryJson = result.ProductSummaryJson;
        StatusText = result.Ok ? "Draft VSQX generated. Manual VOCALOID4FE playback is still required." : "Draft generation failed.";
        AppendLog($"Result: {(result.Ok ? "OK" : "FAILED")}; lyric_source={result.LyricSource}");
        if (!string.IsNullOrWhiteSpace(result.VsqxPath))
        {
            AppendLog($"VSQX: {result.VsqxPath}");
        }
        foreach (var error in result.Errors)
        {
            AppendLog($"Error: {error}");
        }
    }

    private void AppendLog(string message)
    {
        LogLines.Add($"{DateTime.Now:HH:mm:ss}  {message}");
    }

    private static Task EnqueueUiAsync(Action action)
    {
        var completion = new TaskCompletionSource();
        if (App.DispatcherQueue is null || App.DispatcherQueue.HasThreadAccess)
        {
            action();
            completion.SetResult();
            return completion.Task;
        }

        App.DispatcherQueue.TryEnqueue(DispatcherQueuePriority.Normal, () =>
        {
            action();
            completion.SetResult();
        });
        return completion.Task;
    }
}
