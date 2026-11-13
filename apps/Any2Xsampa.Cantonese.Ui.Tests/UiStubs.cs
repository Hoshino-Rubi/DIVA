// Compile the actual Cantonese page logic without WinUI, model loading, or desktop restore.
using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Microsoft.UI.Xaml
{
    public class RoutedEventArgs : EventArgs { public object? OriginalSource { get; set; } }
    public enum Visibility { Visible, Collapsed }
}

namespace Microsoft.UI.Xaml.Controls
{
    public class Control
    {
        public string Text { get; set; } = "";
        public object Header { get; set; } = "";
        public object Content { get; set; } = "";
        public bool IsEnabled { get; set; } = true;
        public bool? IsChecked { get; set; } = true;
        public double Value { get; set; }
        public InfoBarSeverity Severity { get; set; }
        public Visibility Visibility { get; set; } = Visibility.Visible;
        public object? SelectedItem { get; set; }
    }
    public sealed class SelectionChangedEventArgs : RoutedEventArgs { }
    public sealed class TextBlock : Control { }
    public enum InfoBarSeverity { Informational, Success, Warning, Error }
}

namespace Windows.Storage
{
    public sealed record StorageFolder(string Path)
    {
        public static Task<StorageFolder> GetFolderFromPathAsync(string path) => Task.FromResult(new StorageFolder(path));
    }
}

namespace Windows.Storage.Pickers
{
    public sealed class FolderPicker
    {
        public static Windows.Storage.StorageFolder? NextFolder;
        public string CommitButtonText { get; set; } = "";
        public List<string> FileTypeFilter { get; } = new();
        public Task<Windows.Storage.StorageFolder?> PickSingleFolderAsync() => Task.FromResult(NextFolder);
    }
}

namespace Windows.System
{
    public static class Launcher
    {
        public static string LastFolder = "";
        public static Task<bool> LaunchFolderAsync(Windows.Storage.StorageFolder folder)
        { LastFolder = folder.Path; return Task.FromResult(true); }
    }
}

namespace WinRT.Interop
{
    public static class InitializeWithWindow
    {
        public static void Initialize(object picker, nint handle) { }
    }
}

namespace Any2Xsampa_Desktop
{
    public static class App { public static nint WindowHandle => 0; }

    public sealed partial class MainPage
    {
        private readonly FakeService service = new();
        private CancellationTokenSource? active;
        private string audio = "input.wav";
        private readonly Control CantoneseTab = new(), CantoneseModelButton = new(), CantoneseRunButton = new(),
            CantoneseOpenButton = new(), CantonesePrepareButton = new(), ModelProxyBox = new(),
            CantoneseJyutpingBox = new(), CantoneseIpaBox = new(),
            WorkspaceTabs = new(), RunButton = new(), AutoVsqxBox = new(),
            ClipMode = new(), CancelButton = new(), BusyRing = new(),
            WholeMode = new(), StartBox = new() { Value = 2 }, DurationBox = new() { Value = 8 }, StatusBar = new();
        private readonly TextBlock CantoneseStatus = new(), CantoneseBundlePath = new();
        private readonly Dictionary<TextBlock, Func<string>> presentations = new();
        private readonly List<string> errors = new();
        private Func<string>? message;
        private void Show(TextBlock target, Func<string> text) { presentations[target] = text; target.Text = text(); }
        private void Title(Func<string> text) { }
        private void Message(Func<string> text) => message = text;
        private void Log(string text) { }
        private void Error(Exception ex) => errors.Add(ex.Message);
        private void SetBusy(bool busy) => SetCantoneseBusy(busy);
    }

    public sealed class FakeService
    {
        public readonly List<(string Command, JsonElement Parameters)> Calls = new();
        public Func<string, CancellationToken, JsonElement> Handler = (_, _) => throw new InvalidOperationException("Missing test response");
        public Task<JsonElement> CallAsync(string command, object parameters, CancellationToken token = default)
        {
            Calls.Add((command, JsonSerializer.SerializeToElement(parameters)));
            return Task.FromResult(Handler(command, token));
        }
    }
}
