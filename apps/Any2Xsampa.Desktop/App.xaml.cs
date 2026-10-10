using Any2Xsampa_Desktop.Services;
using Any2Xsampa_Desktop.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;

namespace Any2Xsampa_Desktop;

public partial class App : Application
{
    public static Window Window { get; private set; } = null!;

    public static Microsoft.UI.Dispatching.DispatcherQueue DispatcherQueue { get; private set; } = null!;

    public static IServiceProvider Services { get; private set; } = null!;

    public static nint WindowHandle =>
        WinRT.Interop.WindowNative.GetWindowHandle(Window);

    internal static string? StartupReportPath { get; private set; }
    internal static void StartupTrace(string text)
    {
        if (StartupReportPath is string report)
            System.IO.File.AppendAllText(report + ".lifecycle.txt", text + Environment.NewLine);
    }

    public App()
    {
        UnhandledException += (_, e) =>
        {
            if (StartupReportPath is string report)
                System.IO.File.WriteAllText(report + ".exception.txt", e.Exception.ToString());
        };
        InitializeComponent();
        Services = ConfigureServices();
    }

    protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
    {
        var command = Environment.GetCommandLineArgs();
        if (command.Length == 3 && command[1] == "--startup-report")
        {
            StartupReportPath = System.IO.Path.GetFullPath(command[2]);
            if (System.IO.File.Exists(StartupReportPath))
                throw new System.IO.IOException("Startup report already exists");
        }
        Window = Services.GetRequiredService<MainWindow>();
        DispatcherQueue = Microsoft.UI.Dispatching.DispatcherQueue.GetForCurrentThread();
        Window.Closed += (_, _) => {
            StartupTrace("window-closed-service-dispose");
            Services.GetRequiredService<LocalTaskService>().Dispose();
            StartupTrace("service-disposed");
        };
        Window.Activate();
    }

    private static IServiceProvider ConfigureServices()
    {
        var services = new ServiceCollection();
        services.AddSingleton<LocalTaskService>();
        services.AddTransient<MainPage>();
        services.AddSingleton<MainWindow>();
        return services.BuildServiceProvider();
    }
}
