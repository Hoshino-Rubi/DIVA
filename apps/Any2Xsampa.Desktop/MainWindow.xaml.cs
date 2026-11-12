using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;
using Any2Xsampa_Desktop.Services;
using Microsoft.UI.Windowing;
using System.Runtime.InteropServices;

namespace Any2Xsampa_Desktop;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        UpdateTitle();
        UiLanguage.Changed += UpdateTitle;
        Closed += (_, _) => UiLanguage.Changed -= UpdateTitle;
        double scale = Math.Max(1, GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(this)) / 96d);
        var workArea = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        AppWindow.Resize(new Windows.Graphics.SizeInt32(
            Math.Min((int)(1240 * scale), workArea.Width - 32),
            Math.Min((int)(840 * scale), workArea.Height - 32)));
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.SetIcon(System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", "AppIcon.ico"));
        RootFrame.Content = App.Services.GetRequiredService<MainPage>();
    }

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(nint window);

    private void UpdateTitle()
    {
        if (!DispatcherQueue.HasThreadAccess)
        {
            DispatcherQueue.TryEnqueue(UpdateTitle);
            return;
        }
        Title = UiLanguage.Text("DIVA · 本地音素与 VSQX 工作台", "DIVA · Local Phones & VSQX Workbench");
        AppTitleBar.Title = Title;
    }
}
