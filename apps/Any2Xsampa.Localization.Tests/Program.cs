using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using Any2Xsampa_Desktop.Services;

int checks = 0;
void Check(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
    checks++;
}

void CheckSelection(string preference, string language)
{
    Check(UiLanguage.Preference == preference, $"Preference: expected {preference}, got {UiLanguage.Preference}");
    Check(UiLanguage.Language == language, $"Language: expected {language}, got {UiLanguage.Language}");
    Check(UiLanguage.Text("中文", "English") == (language == "zh" ? "中文" : "English"), "Text uses the selected UI language");
}

if (args.Length > 0)
{
    string dataHome = Environment.GetEnvironmentVariable("ANY2XSAMPA_DATA_HOME")
        ?? throw new InvalidOperationException("Child requires an isolated ANY2XSAMPA_DATA_HOME");
    Check(Path.IsPathFullyQualified(dataHome), "Preference directory must be absolute");
    UiLanguage.Initialize();
    if (args[0] == "write")
    {
        UiLanguage.SetPreference(args[1]);
        CheckSelection(args[1], args[1]);
        string file = Path.Combine(dataHome, "state", "ui-preferences.json");
        using var saved = JsonDocument.Parse(File.ReadAllText(file));
        Check(saved.RootElement.GetProperty("preference").GetString() == args[1], "Preference persisted in isolated data home");
    }
    else if (args[0] == "read") CheckSelection(args[1], args[1]);
    else if (args[0] == "system")
    {
        Check(UiLanguage.Preference == "system", "Absent/corrupt preference falls back to system");
        string systemLanguage = UiLanguage.Language;
        Check(systemLanguage is "zh" or "en", "System language resolves to a supported UI language");
        UiLanguage.SetPreference("zh");
        CheckSelection("zh", "zh");
        UiLanguage.SetPreference("en");
        CheckSelection("en", "en");
        UiLanguage.SetPreference("system");
        CheckSelection("system", systemLanguage);
        UiLanguage.SetPreference("unsupported");
        CheckSelection("system", systemLanguage);
    }
    else throw new ArgumentException("Unknown child check: " + args[0]);
    Console.WriteLine(JsonSerializer.Serialize(new { checks }));
    return;
}

var cases = new (string Preference, string[] Languages, string Expected)[]
{
    ("system", ["zh-CN"], "zh"),
    ("system", ["zh-TW"], "zh"),
    ("system", ["zh-Hant"], "zh"),
    ("system", ["zh-HK"], "zh"),
    ("system", ["zh-Hans-CN"], "zh"),
    ("system", ["ZH-tw"], "zh"),
    ("system", ["en"], "en"),
    ("system", ["en-US"], "en"),
    ("system", ["en-GB"], "en"),
    ("system", ["en-AU"], "en"),
    ("system", ["EN-ca"], "en"),
    ("system", ["ja-JP", "fr-FR", "de-DE"], "en"),
    ("system", [], "en"),
    ("system", ["", "ja-JP", "zh-TW", "en-US"], "zh"),
    ("system", ["fr-FR", "en-GB", "zh-CN"], "en"),
    ("system", ["zh-HK", "en-US"], "zh"),
    ("system", ["en-US", "zh-Hant"], "en"),
    ("zh", ["en-US"], "zh"),
    ("en", ["zh-TW"], "en"),
    ("zh", [], "zh"),
    ("en", [], "en"),
    ("unsupported", ["zh-Hant"], "zh"),
    ("", ["en-US"], "en"),
};
foreach (var item in cases)
    Check(UiLanguage.ResolveLanguage(item.Preference, item.Languages) == item.Expected,
        $"ResolveLanguage({item.Preference}, [{string.Join(",", item.Languages)}]) must be {item.Expected}");

// All preference writes, including corruption fixtures, stay in this test output tree.
string sandbox = Path.Combine(AppContext.BaseDirectory, ".language-tests-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(sandbox);
string? previousDataHome = Environment.GetEnvironmentVariable("ANY2XSAMPA_DATA_HOME");
Environment.SetEnvironmentVariable("ANY2XSAMPA_DATA_HOME", sandbox);
try
{
    string first = Path.Combine(sandbox, "用户 A");
    string second = Path.Combine(sandbox, "User B");
    await RunChild(first, "system");
    await RunChild(first, "write", "zh");
    await RunChild(first, "read", "zh");
    await RunChild(second, "system");
    await RunChild(second, "write", "en");
    await RunChild(second, "read", "en");
    await RunChild(first, "read", "zh");

    string[] corruptPreferences = ["{", "null", "[]", "{}", "{\"preference\":false}", "{\"preference\":\"fr\"}"];
    for (int i = 0; i < corruptPreferences.Length; i++)
    {
        string corruptHome = Path.Combine(sandbox, "corrupt-" + i);
        string state = Path.Combine(corruptHome, "state");
        Directory.CreateDirectory(state);
        File.WriteAllText(Path.Combine(state, "ui-preferences.json"), corruptPreferences[i]);
        await RunChild(corruptHome, "system");
    }
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        status = "passed", checks, language_cases = cases.Length,
        persistence = "fresh subprocesses, two isolated data homes", corrupt_preferences = corruptPreferences.Length,
        model_forwards = 0, network_requests = 0
    }));
}
finally
{
    Environment.SetEnvironmentVariable("ANY2XSAMPA_DATA_HOME", previousDataHome);
    Directory.Delete(sandbox, recursive: true);
}

async Task RunChild(string dataHome, params string[] arguments)
{
    var start = new ProcessStartInfo("dotnet")
    {
        UseShellExecute = false, CreateNoWindow = true,
        RedirectStandardOutput = true, RedirectStandardError = true
    };
    start.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
    foreach (string argument in arguments) start.ArgumentList.Add(argument);
    start.Environment["ANY2XSAMPA_DATA_HOME"] = dataHome;
    using var process = Process.Start(start) ?? throw new InvalidOperationException("Could not start preference check");
    Task<string> outputTask = process.StandardOutput.ReadToEndAsync();
    Task<string> errorTask = process.StandardError.ReadToEndAsync();
    using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
    try { await process.WaitForExitAsync(timeout.Token); }
    catch (OperationCanceledException) { process.Kill(entireProcessTree: true); throw; }
    string output = await outputTask;
    string error = await errorTask;
    Check(process.ExitCode == 0, $"Child {string.Join(' ', arguments)} failed:\n{output}\n{error}");
    using var result = JsonDocument.Parse(output);
    checks += result.RootElement.GetProperty("checks").GetInt32();
}
