using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Any2Xsampa_Desktop.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.Graphics.Imaging;

namespace Any2Xsampa_Desktop;

public sealed partial class MainPage
{
    private sealed record StartupUiChecks(bool Passed, List<object> Views, bool ModelsReady,
        bool LanguageSwitchPreservedState, bool StartupDownloadDisabled);

    // Explicit --startup-report only: render this application's own controls without desktop automation.
    private async Task<StartupUiChecks> CaptureStartupUiAsync(string report)
    {
        string originalPreference = UiLanguage.Preference;
        string originalAudio = audio, originalResult = resultPath;
        int originalRows = rows.Count;
        var views = new List<object>();
        bool passed = true;
        var modelInfo = await service.CallAsync("model_setup_status", new { });
        bool modelsReady = modelInfo.GetProperty("ready").GetBoolean();
        try
        {
            var cases = new[] {
                ("en", 0, 1240, 840), ("zh", 0, 1240, 840),
                ("en", 1, 1240, 840), ("zh", 1, 1240, 840),
                ("en", 2, 1240, 840), ("zh", 2, 1240, 840),
                ("en", 0, 1000, 720), ("zh", 0, 1000, 720),
                ("en", 0, 760, 720), ("zh", 0, 760, 720)
            };
            foreach (var (language, tab, width, height) in cases)
            {
                var window = App.Window.AppWindow;
                double scale = Math.Max(1, GetDpiForWindow(App.WindowHandle) / 96d);
                window.Resize(new Windows.Graphics.SizeInt32((int)(width * scale), (int)(height * scale)));
                LanguageBox.SelectedItem = LanguageBox.Items.OfType<ComboBoxItem>()
                    .First(item => item.Tag?.ToString() == language);
                WorkspaceTabs.SelectedIndex = tab;
                await Task.Delay(150);
                UpdateLayout();
                bool valid = UiLanguage.Language == language && RunButton.ActualWidth > 0
                    && PageLayout.ActualWidth > 0 && StatusBar.Severity != InfoBarSeverity.Error;
                if (tab == 0) valid &= PhoneList.ActualHeight > 60 && Player.ActualHeight >= 104;
                string expectedTitle = language == "zh" ? "本地音素" : "Local Phones";
                valid &= App.Window.Title.Contains(expectedTitle, StringComparison.Ordinal);
                var missing = new List<string>();
                CheckTranslations(PageLayout, language, missing);
                valid &= missing.Count == 0;
                var bitmap = new RenderTargetBitmap();
                await bitmap.RenderAsync(PageLayout);
                byte[] pixels = (await bitmap.GetPixelsAsync()).ToArray();
                string png = report + $".{language}-{tab}-{width}.png";
                using var output = File.Create(png).AsRandomAccessStream();
                var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, output);
                encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied,
                    (uint)bitmap.PixelWidth, (uint)bitmap.PixelHeight, 96, 96, pixels);
                await encoder.FlushAsync();
                passed &= valid;
                views.Add(new { language, tab, width, height, valid, missing_translations = missing,
                    content_width = PageLayout.ActualWidth, content_height = PageLayout.ActualHeight,
                    review_width = ReviewPanel.ActualWidth, phone_list_height = PhoneList.ActualHeight,
                    player_height = Player.ActualHeight,
                    screenshot = Path.GetFileName(png) });
            }
        }
        finally
        {
            LanguageBox.SelectedItem = LanguageBox.Items.OfType<ComboBoxItem>()
                .First(item => item.Tag?.ToString() == originalPreference);
        }
        bool retained = audio == originalAudio && resultPath == originalResult && rows.Count == originalRows;
        return new(passed && retained, views, modelsReady, retained,
            AutoDownloadBox.IsChecked != true && !AutoDownloadBox.IsEnabled);
    }

    private static void CheckTranslations(DependencyObject parent, string language, List<string> missing)
    {
        foreach (var property in new[] { UiText.TextProperty, UiText.ContentProperty, UiText.HeaderProperty,
            UiText.PlaceholderProperty, UiText.ToolTipProperty, UiText.AutomationNameProperty,
            UiText.TitleProperty, UiText.MessageProperty })
        {
            if (parent.GetValue(property) is not string source || source.Length == 0) continue;
            if (language == "en" && source.Any(c => c is >= '\u3400' and <= '\u9fff')
                && source != "中文" && !source.Contains("System", StringComparison.Ordinal)
                && UiText.Translate(source) == source) missing.Add(source);
        }
        for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            CheckTranslations(VisualTreeHelper.GetChild(parent, i), language, missing);
    }

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(nint window);
}
