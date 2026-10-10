using System.Diagnostics;
using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Any2Xsampa_Desktop.Services;

/// <summary>Allowlisted diagnostics; never exports request parameters or raw log lines.</summary>
public sealed class DiagnosticBundle
{
    private readonly object gate = new();
    private readonly Queue<object> events = new();
    private long? gpuUsedMiB, gpuTotalMiB;
    private static readonly HashSet<string> Commands = new(StringComparer.Ordinal) {
        "recognize", "recognize-whole", "replay", "models", "doctor", "note_runtime",
        "model_setup_status", "install_models", "inspect_audio", "draft", "export",
        "export_vsqx", "build_score", "import_score", "export_score", "preview_xsampa",
        "cancel", "unload", "capabilities", "desktop"
    };
    private static readonly HashSet<string> Codes = new(StringComparer.Ordinal) {
        "ValueError", "FileNotFoundError", "PermissionError", "OSError", "RuntimeError",
        "TimeoutError", "Cancelled", "JSONDecodeError", "KeyError", "TypeError",
        "InvalidDataException", "IOException", "FileNotFoundException",
        "UnauthorizedAccessException", "InvalidOperationException", "TimeoutException",
        "OperationCanceledException", "TaskCanceledException", "JsonException",
        "Win32Exception", "ObjectDisposedException", "ArgumentException"
    };

    public void Record(string command, string state, Exception? error = null, double elapsedMs = 0)
    {
        string code = error is LocalTaskException remote ? remote.Code : error?.GetType().Name ?? "";
        lock (gate)
        {
            events.Enqueue(new {
                at_utc = DateTimeOffset.UtcNow,
                command = Commands.Contains(command) ? command : "other",
                state = state is "started" or "completed" or "failed" or "cancelled" ? state : "other",
                error_code = Codes.Contains(code) ? code : error is null ? "" : "OtherError",
                elapsed_ms = double.IsFinite(elapsedMs) ? Math.Max(0, Math.Round(elapsedMs, 3)) : 0
            });
            while (events.Count > 256) events.Dequeue();
        }
    }

    public void RecordGpu(string csv)
    {
        // Only memory counters cross the export boundary; GPU names and stderr do not.
        var cells = csv.Split(',').Select(s => s.Trim()).ToArray();
        if (cells.Length == 3 && long.TryParse(cells[1], out var used)
            && long.TryParse(cells[2], out var total) && used >= 0 && total > 0 && used <= total)
            lock (gate) { gpuUsedMiB = used; gpuTotalMiB = total; }
    }

    private static JsonDocument? ReadSmall(string file)
    {
        if (!File.Exists(file) || new FileInfo(file).Length > 4 * 1024 * 1024) return null;
        return JsonDocument.Parse(File.ReadAllBytes(file), new JsonDocumentOptions { MaxDepth = 32 });
    }

    private static string? Digest(JsonElement e, string key)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String
            && Regex.IsMatch(v.GetString() ?? "", "^[a-f0-9]{64}$") ? v.GetString() : null;

    private static string ModelId(JsonElement model)
    {
        string? id = model.TryGetProperty("id", out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        return id is "phoneticxeus" or "game" or "guarded-candidate-v2" or "onnx-espeak-int8" ? id : "other";
    }

    private static string Inside(string root, string relative)
    {
        if (Path.IsPathRooted(relative)) throw new InvalidDataException();
        string full = Path.GetFullPath(Path.Combine(root, relative));
        if (!full.StartsWith(Path.TrimEndingDirectorySeparator(root) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException();
        return full;
    }

    private static object ModelSummary(string root)
    {
        var models = new List<object>();
        try
        {
            using var catalog = ReadSmall(Path.Combine(root, "model-downloads.json"));
            if (catalog is null) return new { state = "catalog_missing", models };
            foreach (var bundle in catalog.RootElement.GetProperty("bundles").EnumerateArray().Take(16))
            {
                var files = new List<object>();
                int index = 0;
                foreach (var row in bundle.GetProperty("files").EnumerateArray().Take(1024))
                {
                    string target = Inside(root, row.GetProperty("path").GetString()!);
                    if (!target.StartsWith(Path.Combine(root, "models") + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException();
                    long expected = row.GetProperty("bytes").GetInt64();
                    var info = new FileInfo(target);
                    files.Add(new { index = index++, expected_sha256 = Digest(row, "sha256"),
                        expected_bytes = expected, actual_bytes = info.Exists ? (long?)info.Length : null,
                        state = !info.Exists ? "missing" : info.Length != expected ? "size_mismatch" : "size_matches_not_hashed"
                    });
                }
                models.Add(new { id = ModelId(bundle), files });
            }
            return new { state = "read", models };
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException or InvalidOperationException
                                  or KeyNotFoundException or ArgumentException or FormatException or OverflowException)
        { return new { state = "catalog_invalid_or_unreadable", models = Array.Empty<object>() }; }
    }

    public Task ExportAsync(string output, string configPath)
        => Task.Run(() => Export(output, configPath));

    private void Export(string output, string configPath)
    {
        string root = Path.GetDirectoryName(Path.GetFullPath(configPath))!;
        string configState = "missing";
        bool pythonPresent = false;
        try
        {
            using var config = ReadSmall(configPath);
            if (config is not null)
            {
                root = Path.GetFullPath(Path.Combine(root, config.RootElement.GetProperty("root").GetString()!));
                pythonPresent = File.Exists(Path.GetFullPath(Path.Combine(root, config.RootElement.GetProperty("python").GetString()!)));
                configState = "read";
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException or InvalidOperationException
                                  or KeyNotFoundException or ArgumentException)
        { configState = "invalid_or_unreadable"; }
        object[] snapshot;
        long? used, total;
        lock (gate) { snapshot = events.ToArray(); used = gpuUsedMiB; total = gpuTotalMiB; }
        using var process = Process.GetCurrentProcess();
        var value = new {
            schema_version = "local_diagnostic_bundle_v1",
            release = "pre-1.0.1a",
            app_version = typeof(DiagnosticBundle).Assembly.GetName().Version?.ToString(),
            created_utc = DateTimeOffset.UtcNow,
            collection = "local_only_no_model_load_no_network",
            privacy = new {
                audio_included = false, raw_logs_included = false, paths_included = false,
                usernames_included = false, request_parameters_included = false,
                predictions_included = false, environment_variables_included = false
            },
            configuration = new { state = configState, core_python_present = pythonPresent },
            environment = new {
                os_version = Environment.OSVersion.Version.ToString(),
                architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
                dotnet_version = Environment.Version.ToString(),
                working_set_bytes = process.WorkingSet64,
                gpu_used_mib = used, gpu_total_mib = total,
                process_gpu_limit_bytes = 7L * 1024 * 1024 * 1024,
                card_gpu_limit_bytes = 14L * 1024 * 1024 * 1024
            },
            models = ModelSummary(root),
            events = snapshot,
            model_hashes_recomputed = false
        };
        // FileMode.CreateNew protects existing exports; clean up only this new file on failure.
        using var stream = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        try
        {
            using var archive = new ZipArchive(stream, ZipArchiveMode.Create, true);
            using (var writer = new StreamWriter(archive.CreateEntry("diagnostic.json").Open()))
                writer.Write(JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
            using (var writer = new StreamWriter(archive.CreateEntry("README.txt").Open()))
                writer.Write("Local diagnostics only. No audio, paths, usernames, raw logs, predictions or proxy credentials.\nModel hashes are expected catalog values; size_matches_not_hashed is not an integrity verification.\nEvents cover this application session (last 256). Nothing is uploaded automatically.\n");
        }
        catch { stream.Dispose(); File.Delete(output); throw; }
    }
}

public sealed class LocalTaskException(string code, string message) : InvalidOperationException(message)
{
    public string Code { get; } = code;
}
