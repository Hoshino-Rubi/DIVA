using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Security;
using System.Text.Json;

namespace Any2Xsampa_Desktop.Services;

/// <summary>UI language only; independent of recognition languages and model state.</summary>
public static class UiLanguage
{
    public static string Preference { get; private set; } = "system";
    public static string Language { get; private set; } = "en";
    public static event Action? Changed;

    public static string Text(string chinese, string english) => Language == "zh" ? chinese : english;

    public static string ResolveLanguage(string preference, IEnumerable<string> systemLanguages)
    {
        if (preference is "zh" or "en") return preference;
        foreach (string language in systemLanguages)
        {
            if (language.StartsWith("zh", StringComparison.OrdinalIgnoreCase)) return "zh";
            if (language.StartsWith("en", StringComparison.OrdinalIgnoreCase)) return "en";
        }
        return "en";
    }

    public static void Initialize()
    {
        string preference = "system";
        bool repair = false;
        try
        {
            string path = PreferencesPath();
            if (File.Exists(path))
            {
                using var json = JsonDocument.Parse(File.ReadAllText(path));
                if (json.RootElement.ValueKind == JsonValueKind.Object &&
                    json.RootElement.TryGetProperty("preference", out var value) &&
                    value.ValueKind == JsonValueKind.String && IsPreference(value.GetString()))
                    preference = value.GetString()!;
                else repair = true;
            }
        }
        catch (Exception error) when (IsStorageError(error)) { repair = true; }
        Apply(preference);
        if (repair) Save(preference);
    }

    public static void SetPreference(string preference)
    {
        preference = IsPreference(preference) ? preference : "system";
        Save(preference);
        Apply(preference);
    }

    private static bool IsPreference(string? value) => value is "system" or "zh" or "en";

    private static void Apply(string preference)
    {
        string language = ResolveLanguage(preference, SystemLanguages());
        bool changed = Preference != preference || Language != language;
        Preference = preference;
        Language = language;
        var culture = CultureInfo.GetCultureInfo(language == "zh" ? "zh-CN" : "en-US");
        CultureInfo.CurrentUICulture = culture;
        CultureInfo.DefaultThreadCurrentUICulture = culture;
        if (changed) Changed?.Invoke();
    }

    private static IEnumerable<string> SystemLanguages()
    {
        // Unlike CurrentUICulture, this is not affected by our own language switch.
        var languages = new List<string>();
        if (OperatingSystem.IsWindows())
        {
            uint size = 0;
            if (GetUserPreferredUILanguages(8, out _, null, ref size) && size > 0)
            {
                var buffer = new char[(int)size];
                if (GetUserPreferredUILanguages(8, out _, buffer, ref size))
                    languages.AddRange(new string(buffer).Split('\0', StringSplitOptions.RemoveEmptyEntries));
            }
        }
        languages.Add(CultureInfo.InstalledUICulture.Name);
        return languages;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetUserPreferredUILanguages(uint flags, out uint count,
        [Out] char[]? buffer, ref uint size);

    private static string PreferencesPath()
    {
        string root = AppContext.BaseDirectory;
        string mode = "portable_v1";
        string configPath = Path.Combine(root, "local-app.json");
        try
        {
            if (File.Exists(configPath))
            {
                using var config = JsonDocument.Parse(File.ReadAllText(configPath));
                if (config.RootElement.TryGetProperty("root", out var configuredRoot) &&
                    configuredRoot.ValueKind == JsonValueKind.String)
                    root = Path.GetFullPath(Path.Combine(root, configuredRoot.GetString()!));
                if (config.RootElement.TryGetProperty("storage_mode", out var storageMode) &&
                    storageMode.ValueKind == JsonValueKind.String)
                    mode = storageMode.GetString()!;
            }
        }
        catch (Exception error) when (IsStorageError(error)) { }
        string? dataHome = Environment.GetEnvironmentVariable("ANY2XSAMPA_DATA_HOME");
        if (!string.IsNullOrWhiteSpace(dataHome) && !Path.IsPathFullyQualified(dataHome)) dataHome = null;
        if (string.IsNullOrWhiteSpace(dataHome) && mode == "per_user_v1")
            dataHome = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Any2Xsampa");
        string state = string.IsNullOrWhiteSpace(dataHome) ? Path.Combine(root, ".state")
            : Path.Combine(Path.GetFullPath(dataHome), "state");
        return Path.Combine(state, "ui-preferences.json");
    }

    private static void Save(string preference)
    {
        string? temporary = null;
        try
        {
            string path = PreferencesPath();
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(new { preference }));
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception error) when (IsStorageError(error)) { /* Read-only media still allows session selection. */ }
        finally
        {
            if (temporary is not null)
                try { File.Delete(temporary); }
                catch (Exception error) when (IsStorageError(error)) { }
        }
    }

    private static bool IsStorageError(Exception error) => error is IOException or UnauthorizedAccessException
        or JsonException or ArgumentException or NotSupportedException or SecurityException or InvalidOperationException;
}
