using System.Text.Json;
using Any2Xsampa_Desktop;
using Any2Xsampa_Desktop.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.System;

await MainPage.RunChecks();

namespace Any2Xsampa_Desktop
{
    public sealed partial class MainPage
    {
        public static async Task RunChecks()
        {
            int checks = 0;
            void Check(bool condition, string message)
            { if (!condition) throw new InvalidOperationException(message); checks++; }
            JsonElement Status(bool available, string? reason = null) => JsonSerializer.SerializeToElement(new
                { available, bundle_dir = "software/models/cantonese4", reason, model_loaded = false });
            JsonElement Result() => JsonSerializer.SerializeToElement(new
            {
                result_path = "c4/result.json", output_dir = "c4", jyutping = "nei5\nhou2",
                ipa = "nei˩˧\nhou˧˥", window_count = 2, unknown_count = 0
            });
            var page = new MainPage();
            var args = new RoutedEventArgs();
            page.service.Handler = (command, _) => command == "cantonese4_status" ? Status(true) : Result();
            page.InitializeCantonese();
            Check(!page.service.Calls.Any(), "Initialization must not start I/O or inference");
            page.WorkspaceTabs.SelectedItem = new Control();
            var tabChanged = new SelectionChangedEventArgs { OriginalSource = page.WorkspaceTabs };
            page.WorkspaceTabs_SelectionChanged(page.WorkspaceTabs, tabChanged);
            Check(page.RunButton.Visibility == Visibility.Visible && page.AutoVsqxBox.Visibility == Visibility.Visible,
                "Original startup actions remain visible on the initial workspace tab");
            page.WorkspaceTabs.SelectedItem = page.CantoneseTab;
            page.WorkspaceTabs_SelectionChanged(page.WorkspaceTabs, tabChanged);
            Check(page.RunButton.Visibility == Visibility.Collapsed && page.AutoVsqxBox.Visibility == Visibility.Collapsed,
                "Cantonese tab hides the original start and automatic VSQX controls");
            Check(page.WholeMode.Visibility == Visibility.Visible && page.ClipMode.Visibility == Visibility.Visible &&
                page.CancelButton.Visibility == Visibility.Visible && page.BusyRing.Visibility == Visibility.Visible,
                "Scope and cancellation indicators remain shared and visible");
            page.WorkspaceTabs.SelectedItem = new Control();
            page.WorkspaceTabs_SelectionChanged(page.WorkspaceTabs,
                new SelectionChangedEventArgs { OriginalSource = page.CantoneseJyutpingBox });
            Check(page.RunButton.Visibility == Visibility.Collapsed, "Bubbled child selection changes are ignored");
            page.WorkspaceTabs_SelectionChanged(page.WorkspaceTabs, tabChanged);
            Check(page.RunButton.Visibility == Visibility.Visible && page.AutoVsqxBox.Visibility == Visibility.Visible &&
                page.AutoVsqxBox.IsChecked == true && !page.service.Calls.Any(),
                "Leaving Cantonese restores visibility without changing VSQX preference or starting work");
            Check(await page.RefreshCantoneseStatus(), "Default model status");
            Check(page.service.Calls.Single().Command == "cantonese4_status", "Status must not load a model");
            Check(page.service.Calls[0].Parameters.GetProperty("bundle_dir").ValueKind == JsonValueKind.Null,
                "Default bundle location belongs to the service");
            page.service.Calls.Clear();
            page.CantoneseRun_Click(page, args);
            Check(page.service.Calls.Select(c => c.Command).SequenceEqual(new[] { "cantonese4_status", "cantonese4" }),
                "Run checks readiness then calls only Cantonese inference");
            var parameters = page.service.Calls.Last().Parameters;
            Check(parameters.GetProperty("audio").GetString() == "input.wav" &&
                parameters.GetProperty("scope").GetString() == "whole_file", "Shared WAV and whole-file scope");
            Check(parameters.GetProperty("start").GetDouble() == 2 && parameters.GetProperty("duration").GetDouble() == 8,
                "Shared range is forwarded");
            Check(page.CantoneseJyutpingBox.Text == "nei5\nhou2" && page.CantoneseIpaBox.Text == "nei˩˧\nhou˧˥",
                "Multiline Unicode outputs survive unchanged");
            Check(page.active is null && page.CantoneseRunButton.IsEnabled && page.CantoneseOpenButton.IsEnabled,
                "Completed task releases busy state");
            page.CantoneseOpen_Click(page, args);
            Check(Launcher.LastFolder == "c4", "Open uses the independent Cantonese result directory");

            string sandbox = Path.Combine(Path.GetTempPath(), "diva-c4-ui-" + Guid.NewGuid().ToString("N"));
            string? previousHome = Environment.GetEnvironmentVariable("ANY2XSAMPA_DATA_HOME");
            Environment.SetEnvironmentVariable("ANY2XSAMPA_DATA_HOME", sandbox);
            try
            {
                foreach (string language in new[] { "zh", "en" })
                {
                    UiLanguage.SetPreference(language);
                    page.RenderCantoneseLanguage();
                    foreach (var pair in page.presentations) pair.Key.Text = pair.Value();
                    Check(page.CantoneseTab.Header.ToString() == (language == "zh" ? "粤语增强" : "Cantonese") &&
                        page.CantoneseStatus.Text == (language == "zh" ? "完成" : "Done"), "Labels and status switch language");
                    Check(page.CantoneseJyutpingBox.Text == "nei5\nhou2" && page.cantoneseOutputDir == "c4",
                        "Language switch preserves output and result path");
                }
                page.service.Handler = (_, _) => Status(false, "separator_code_missing");
                page.service.Calls.Clear();
                await page.RefreshCantoneseStatus();
                foreach (string language in new[] { "zh", "en" })
                {
                    UiLanguage.SetPreference(language);
                    page.RenderCantoneseLanguage();
                    foreach (var pair in page.presentations) pair.Key.Text = pair.Value();
                    Check(page.CantoneseStatus.Text == (language == "zh" ? "需准备" : "Prepare needed") &&
                        page.CantonesePrepareButton.Content.ToString() == (language == "zh" ? "准备模型" : "Prepare"),
                        "Missing separator code and Prepare button use short localized text");
                }
                page.CantoneseRun_Click(page, args);
                Check(page.service.Calls.All(c => c.Command == "cantonese4_status"),
                    "Checking status and starting inference never automatically prepare files");
            }
            finally
            {
                Environment.SetEnvironmentVariable("ANY2XSAMPA_DATA_HOME", previousHome);
                if (Directory.Exists(sandbox)) Directory.Delete(sandbox, true);
            }

            page.service.Calls.Clear();
            page.service.Handler = (_, _) => Status(true);
            page.ModelProxyBox.Text = "  ";
            page.CantonesePrepare_Click(page, args);
            Check(page.service.Calls.Single().Command == "cantonese4_prepare" &&
                page.service.Calls[0].Parameters.GetProperty("bundle_dir").ValueKind == JsonValueKind.Null &&
                page.service.Calls[0].Parameters.GetProperty("proxy").ValueKind == JsonValueKind.Null,
                "Explicit Prepare supports the default model folder and an omitted proxy");
            Check(page.active is null && page.CantoneseStatus.Text == "Model ready" && page.CantonesePrepareButton.IsEnabled,
                "Prepare applies returned status without another request and releases controls");
            page.service.Handler = (command, _) => command == "cantonese4_status" ? Status(true) : Result();
            FolderPicker.NextFolder = new StorageFolder("optional/extracted-c4");
            page.CantoneseModel_Click(page, args);
            Check(page.cantoneseBundleDir == "optional/extracted-c4" &&
                page.service.Calls.Last().Parameters.GetProperty("bundle_dir").GetString() == "optional/extracted-c4",
                "Selected extracted model folder is forwarded");
            page.WholeMode.IsChecked = false;
            page.CantoneseRun_Click(page, args);
            Check(page.service.Calls.Last().Parameters.GetProperty("scope").GetString() == "clip", "Clip scope");

            page.service.Calls.Clear();
            page.ModelProxyBox.Text = " http://127.0.0.1:18080 ";
            page.service.Handler = (command, token) =>
            {
                Check(command == "cantonese4_prepare" && token.CanBeCanceled &&
                    !page.CantonesePrepareButton.IsEnabled && !page.CantoneseRunButton.IsEnabled &&
                    !page.CantoneseModelButton.IsEnabled && !page.CantoneseOpenButton.IsEnabled,
                    "Prepare uses the shared cancellation token and busy controls");
                return Status(true);
            };
            page.CantonesePrepare_Click(page, args);
            Check(page.service.Calls.Single().Parameters.GetProperty("bundle_dir").GetString() == "optional/extracted-c4" &&
                page.service.Calls[0].Parameters.GetProperty("proxy").GetString() == "http://127.0.0.1:18080",
                "Prepare forwards the selected model folder and optional proxy");
            Check(page.CantoneseJyutpingBox.Text == "nei5\nhou2" && page.cantoneseOutputDir == "c4",
                "Preparing files preserves the previous independent output");
            page.service.Handler = (_, token) =>
            {
                page.active!.Cancel();
                token.ThrowIfCancellationRequested();
                return Status(true);
            };
            page.CantonesePrepare_Click(page, args);
            Check(page.active is null && page.CantoneseStatus.Text == "Cancelled" && page.CantonesePrepareButton.IsEnabled,
                "Prepare cancellation releases controls without retries");
            page.service.Handler = (_, _) => throw new IOException("prepare failure");
            page.CantonesePrepare_Click(page, args);
            Check(page.errors.Last() == "prepare failure" && page.active is null && page.cantoneseOutputDir == "c4",
                "Prepare errors use existing error handling and preserve output");
            page.errors.Clear();
            page.service.Handler = (command, token) =>
            {
                if (command == "cantonese4_status") return Status(true);
                Check(!page.CantoneseRunButton.IsEnabled && !page.CantoneseModelButton.IsEnabled && !page.CantonesePrepareButton.IsEnabled &&
                    !page.CantoneseOpenButton.IsEnabled, "All Cantonese actions are disabled while busy");
                page.active!.Cancel();
                token.ThrowIfCancellationRequested();
                return Result();
            };
            page.CantoneseRun_Click(page, args);
            Check(page.active is null && page.CantoneseStatus.Text == "Cancelled" &&
                page.CantoneseJyutpingBox.Text == "nei5\nhou2" && page.CantoneseOpenButton.IsEnabled,
                "Cancellation preserves the last successful result and releases controls");
            page.service.Handler = (command, _) => command == "cantonese4_status" ? Status(true) : throw new IOException("test failure");
            page.CantoneseRun_Click(page, args);
            Check(page.errors.Count == 1 && page.active is null && page.cantoneseOutputDir == "c4",
                "Inference errors reach existing error handling without replacing output");
            page.service.Handler = (_, _) => Status(false);
            page.service.Calls.Clear();
            page.CantoneseRun_Click(page, args);
            Check(page.service.Calls.Count == 1 && page.CantoneseStatus.Text == "Select a model folder",
                "Unavailable models never launch inference");
            page.service.Calls.Clear();
            page.DurationBox.Value = double.NaN;
            page.CantoneseRun_Click(page, args);
            Check(!page.service.Calls.Any(), "Invalid clip values never reach JSON serialization");
            page.audio = "";
            page.CantoneseRun_Click(page, args);
            Check(!page.service.Calls.Any() && page.CantoneseStatus.Text == "Select a WAV first", "Missing WAV is handled locally");
            page.ResetCantoneseResult();
            Check(page.CantoneseJyutpingBox.Text == "" && page.CantoneseIpaBox.Text == "" &&
                !page.CantoneseOpenButton.IsEnabled, "Changing audio clears only the Cantonese display and path");
            Console.WriteLine(JsonSerializer.Serialize(new { status = "passed", checks, model_forwards = 0, desktop_builds = 0 }));
        }
    }
}
