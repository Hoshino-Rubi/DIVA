using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Data;

namespace Any2Xsampa_Desktop.Services;

/// <summary>Attached values are canonical Chinese literals; bound/dynamic fields retain ownership.</summary>
public static class UiText
{
    private sealed class Entry(DependencyObject target, DependencyProperty source, DependencyProperty destination)
    {
        public readonly WeakReference<DependencyObject> Target = new(target);
        public readonly DependencyProperty Source = source;
        public readonly DependencyProperty Destination = destination;
        public string? LastValue;
        public bool Applied;
    }

    private static readonly List<Entry> Entries = new();
    private static readonly Dictionary<string, string> English = LoadEnglish();

    static UiText() => UiLanguage.Changed += Refresh;

    public static string Translate(string source) => UiLanguage.Language == "en" && English.TryGetValue(source, out var english)
        ? english : source;

    public static readonly DependencyProperty TextProperty = Register("Text");
    public static readonly DependencyProperty ContentProperty = Register("Content");
    public static readonly DependencyProperty HeaderProperty = Register("Header");
    public static readonly DependencyProperty PlaceholderProperty = Register("Placeholder");
    public static readonly DependencyProperty ToolTipProperty = Register("ToolTip");
    public static readonly DependencyProperty AutomationNameProperty = Register("AutomationName");
    public static readonly DependencyProperty TitleProperty = Register("Title");
    public static readonly DependencyProperty MessageProperty = Register("Message");

    public static string GetText(DependencyObject target) => (string)target.GetValue(TextProperty);
    public static void SetText(DependencyObject target, string value) => target.SetValue(TextProperty, value);
    public static string GetContent(DependencyObject target) => (string)target.GetValue(ContentProperty);
    public static void SetContent(DependencyObject target, string value) => target.SetValue(ContentProperty, value);
    public static string GetHeader(DependencyObject target) => (string)target.GetValue(HeaderProperty);
    public static void SetHeader(DependencyObject target, string value) => target.SetValue(HeaderProperty, value);
    public static string GetPlaceholder(DependencyObject target) => (string)target.GetValue(PlaceholderProperty);
    public static void SetPlaceholder(DependencyObject target, string value) => target.SetValue(PlaceholderProperty, value);
    public static string GetToolTip(DependencyObject target) => (string)target.GetValue(ToolTipProperty);
    public static void SetToolTip(DependencyObject target, string value) => target.SetValue(ToolTipProperty, value);
    public static string GetAutomationName(DependencyObject target) => (string)target.GetValue(AutomationNameProperty);
    public static void SetAutomationName(DependencyObject target, string value) => target.SetValue(AutomationNameProperty, value);
    public static string GetTitle(DependencyObject target) => (string)target.GetValue(TitleProperty);
    public static void SetTitle(DependencyObject target, string value) => target.SetValue(TitleProperty, value);
    public static string GetMessage(DependencyObject target) => (string)target.GetValue(MessageProperty);
    public static void SetMessage(DependencyObject target, string value) => target.SetValue(MessageProperty, value);

    private static DependencyProperty Register(string name) => DependencyProperty.RegisterAttached(name,
        typeof(string), typeof(UiText), new PropertyMetadata("", SourceChanged));

    private static void SourceChanged(DependencyObject target, DependencyPropertyChangedEventArgs args)
    {
        DependencyProperty? destination = Destination(target, args.Property);
        if (destination is null) return;
        Entries.RemoveAll(entry => !entry.Target.TryGetTarget(out var element) ||
            (ReferenceEquals(element, target) && entry.Source == args.Property));
        var entry = new Entry(target, args.Property, destination);
        Entries.Add(entry);
        Apply(entry, target);
    }

    private static DependencyProperty? Destination(DependencyObject target, DependencyProperty source)
    {
        if (source == TextProperty) return target switch
        {
            TextBlock => TextBlock.TextProperty,
            TextBox => TextBox.TextProperty,
            _ => null
        };
        if (source == ContentProperty && target is ContentControl) return ContentControl.ContentProperty;
        if (source == HeaderProperty) return target switch
        {
            PivotItem => PivotItem.HeaderProperty,
            ComboBox => ComboBox.HeaderProperty,
            TextBox => TextBox.HeaderProperty,
            NumberBox => NumberBox.HeaderProperty,
            Slider => Slider.HeaderProperty,
            ToggleSwitch => ToggleSwitch.HeaderProperty,
            Expander => Expander.HeaderProperty,
            _ => null
        };
        if (source == PlaceholderProperty && target is TextBox) return TextBox.PlaceholderTextProperty;
        if (source == ToolTipProperty) return ToolTipService.ToolTipProperty;
        if (source == AutomationNameProperty) return AutomationProperties.NameProperty;
        if (source == TitleProperty) return target switch
        {
            InfoBar => InfoBar.TitleProperty,
            TitleBar => TitleBar.TitleProperty,
            ContentDialog => ContentDialog.TitleProperty,
            _ => null
        };
        if (source == MessageProperty && target is InfoBar) return InfoBar.MessageProperty;
        return null;
    }

    private static void Refresh()
    {
        foreach (var entry in Entries.ToArray())
        {
            if (!entry.Target.TryGetTarget(out var target)) { Entries.Remove(entry); continue; }
            var queue = target.DispatcherQueue;
            if (queue.HasThreadAccess) Apply(entry, target);
            else queue.TryEnqueue(() => { if (entry.Target.TryGetTarget(out var element)) Apply(entry, element); });
        }
    }

    private static void Apply(Entry entry, DependencyObject target)
    {
        if (target.ReadLocalValue(entry.Destination) is BindingExpression) return;
        string source = (string)target.GetValue(entry.Source);
        object? current = target.GetValue(entry.Destination);
        // Never replace a control subtree or text changed by a binding/event handler.
        if (current is not null && current is not string) return;
        if (entry.Applied && !Equals(current, entry.LastValue)) return;
        if (!entry.Applied && current is string value && value.Length > 0 && value != source && value != Translate(source)) return;
        string translated = Translate(source);
        target.SetValue(entry.Destination, translated);
        entry.LastValue = translated;
        entry.Applied = true;
    }

    private static Dictionary<string, string> LoadEnglish()
    {
        var assembly = typeof(UiText).Assembly;
        string? name = assembly.GetManifestResourceNames().FirstOrDefault(value =>
            value.EndsWith("Localization.StaticText.en.json", StringComparison.OrdinalIgnoreCase));
        if (name is null) return new(StringComparer.Ordinal);
        try
        {
            using var stream = assembly.GetManifestResourceStream(name);
            return stream is null ? new(StringComparer.Ordinal)
                : JsonSerializer.Deserialize<Dictionary<string, string>>(stream) ?? new(StringComparer.Ordinal);
        }
        catch (JsonException) { return new(StringComparer.Ordinal); }
    }
}
