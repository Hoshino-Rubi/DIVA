using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;

namespace Any2Xsampa_Desktop;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        Title = "DIVA · 本地音素与 VSQX 工作台";
        AppWindow.Resize(new Windows.Graphics.SizeInt32(1320, 900));
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.SetIcon(System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", "AppIcon.ico"));
        RootFrame.Content = App.Services.GetRequiredService<MainPage>();
    }
}
