using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace Any2Xsampa_Desktop.Services;

/// <summary>Port-free transport shared with the CLI engine. No silent retries.</summary>
public sealed class LocalTaskService : IDisposable
{
    public const string Protocol = "any2xsampa.local.v1";
    private readonly string configPath;
    private readonly SemaphoreSlim startGate = new(1), sendGate = new(1);
    private readonly ConcurrentDictionary<string, TaskCompletionSource<JsonElement>> pending = new();
    private Process? process;
    private OwnedProcessJob? job;
    private TaskCompletionSource<bool>? ready;
    private bool disposed;
    public event Action<string>? Log;
    public event Action<JsonElement>? Progress;
    public string Root { get; private set; } = "";
    public string StateDirectory { get; private set; } = "";
    public DiagnosticBundle Diagnostics { get; } = new();
    public Task ExportDiagnosticsAsync(string output) => Diagnostics.ExportAsync(output, configPath);

    public LocalTaskService() : this(Path.Combine(AppContext.BaseDirectory, "local-app.json")) { }
    public LocalTaskService(string configPath) { this.configPath = Path.GetFullPath(configPath); }

    private async Task EnsureStartedAsync(CancellationToken ct)
    {
        await startGate.WaitAsync(ct);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (process is { HasExited: false }) return;
            if (!File.Exists(configPath)) throw new FileNotFoundException("找不到本地运行配置 local-app.json；无需启动任何服务器。", configPath);
            using var config = JsonDocument.Parse(await File.ReadAllTextAsync(configPath, ct));
            Root = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(configPath)!, config.RootElement.GetProperty("root").GetString()!));
            string? dataHome = Environment.GetEnvironmentVariable("ANY2XSAMPA_DATA_HOME");
            if (!string.IsNullOrWhiteSpace(dataHome) && !Path.IsPathFullyQualified(dataHome))
                throw new InvalidDataException("ANY2XSAMPA_DATA_HOME must be an absolute directory");
            string mode = config.RootElement.TryGetProperty("storage_mode", out var storageMode)
                ? storageMode.GetString() ?? "" : "portable_v1";
            if (mode is not ("per_user_v1" or "portable_v1"))
                throw new InvalidDataException("Unsupported application storage mode");
            if (string.IsNullOrWhiteSpace(dataHome) && mode == "per_user_v1")
                dataHome = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Any2Xsampa");
            StateDirectory = string.IsNullOrWhiteSpace(dataHome) ? Path.Combine(Root, ".state")
                : Path.Combine(Path.GetFullPath(dataHome), "state");
            var python = Path.GetFullPath(Path.Combine(Root, config.RootElement.GetProperty("python").GetString()!));
            if (!File.Exists(python)) throw new FileNotFoundException("本地 Python 运行时不存在。", python);
            var start = new ProcessStartInfo(python) { UseShellExecute=false, CreateNoWindow=true, WorkingDirectory=Root,
                RedirectStandardInput=true, RedirectStandardOutput=true, RedirectStandardError=true,
                StandardInputEncoding=new UTF8Encoding(false), StandardOutputEncoding=Encoding.UTF8, StandardErrorEncoding=Encoding.UTF8 };
            foreach (var arg in new[] { "-X", "utf8", "-m", "any2xsampa.local.server" }) start.ArgumentList.Add(arg);
            start.Environment["ANY2XSAMPA_HOME"] = Root;
            if (!string.IsNullOrWhiteSpace(dataHome)) start.Environment["ANY2XSAMPA_DATA_HOME"] = Path.GetFullPath(dataHome);
            start.Environment["PYTHONPATH"] = Path.Combine(Root, "src");
            start.Environment["PYTHONUTF8"] = "1";
            start.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
            if (config.RootElement.TryGetProperty("models", out var models))
            {
                var registry = Path.GetFullPath(Path.Combine(Root, models.GetString()!));
                if (!File.Exists(registry)) throw new FileNotFoundException("找不到此版本的模型注册表。", registry);
                start.Environment["ANY2XSAMPA_MODELS"] = registry;
            }
            start.Environment.Remove("PYTHONHOME");
            start.Environment.Remove("ANY2XSAMPA_NOTES");
            start.Environment.Remove("ANY2XSAMPA_GAME_SNAPSHOT");
            foreach (var name in new[] { "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS" }) start.Environment[name] = "4";
            start.Environment["HF_HUB_OFFLINE"] = start.Environment["TRANSFORMERS_OFFLINE"] = "1";
            process = Process.Start(start) ?? throw new IOException("Cannot start local worker");
            try { job = new OwnedProcessJob(process); }
            catch { process.Kill(entireProcessTree:true); process.Dispose(); process=null; throw; }
            ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
            // Concurrent drains are mandatory: neither pipe waits for the other.
            _ = ReadMessagesAsync(process, ready);
            _ = ReadErrorsAsync(process);
            try { await ready.Task.WaitAsync(TimeSpan.FromSeconds(30), ct); }
            catch { Stop(); throw; }
        }
        finally { startGate.Release(); }
    }

    private async Task ReadMessagesAsync(Process child, TaskCompletionSource<bool> handshake)
    {
        try
        {
            while (await child.StandardOutput.ReadLineAsync() is { } line)
            {
                using var json = JsonDocument.Parse(line);
                var e = json.RootElement;
                if (e.GetProperty("protocol").GetString() != Protocol) throw new IOException("Worker protocol version mismatch");
                var kind = e.GetProperty("event").GetString();
                if (kind == "ready") { handshake.TrySetResult(true); continue; }
                var id = e.TryGetProperty("id", out var key) ? key.GetString() : null;
                if (kind == "progress") { Progress?.Invoke(e.GetProperty("data").Clone()); continue; }
                if (id is null || !pending.TryGetValue(id, out var promise)) continue;
                if (kind == "result") promise.TrySetResult(e.GetProperty("data").Clone());
                else if (kind == "error") promise.TrySetException(new LocalTaskException(e.GetProperty("code").GetString() ?? "OtherError", e.GetProperty("message").GetString() ?? "Worker failed"));
                else if (kind == "cancelled") promise.TrySetCanceled();
            }
            throw new IOException("本地 worker 已退出；未自动重跑任务。");
        }
        catch (Exception e)
        {
            handshake.TrySetException(e);
            if (ReferenceEquals(child, process)) FailPending(e);
        }
    }

    private async Task ReadErrorsAsync(Process child)
    {
        try { while (await child.StandardError.ReadLineAsync() is { } line) Log?.Invoke(line); }
        catch (Exception e) when (e is IOException or ObjectDisposedException or InvalidOperationException) { }
    }

    private async Task SendAsync(object command)
    {
        await sendGate.WaitAsync();
        try
        {
            if (process is not { HasExited:false }) throw new IOException("Local worker is not running");
            await process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(command));
            await process.StandardInput.FlushAsync();
        }
        finally { sendGate.Release(); }
    }

    public async Task<JsonElement> CallAsync(string command, object parameters, CancellationToken ct=default)
    {
        var timer = Stopwatch.StartNew();
        Diagnostics.Record(command, "started");
        try
        {
            var result = await CallCoreAsync(command, parameters, ct);
            Diagnostics.Record(command, "completed", elapsedMs:timer.Elapsed.TotalMilliseconds);
            return result;
        }
        catch (Exception error)
        {
            Diagnostics.Record(command, error is OperationCanceledException ? "cancelled" : "failed", error, timer.Elapsed.TotalMilliseconds);
            throw;
        }
    }

    private async Task<JsonElement> CallCoreAsync(string command, object parameters, CancellationToken ct)
    {
        await EnsureStartedAsync(ct);
        string id = Guid.NewGuid().ToString("N");
        var completion = new TaskCompletionSource<JsonElement>(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = completion;
        try
        {
            await SendAsync(new { protocol=Protocol, id, command, @params=parameters });
            // A whole-file task has a bounded per-window worker watchdog, but may
            // legitimately outlive the timeout for a single window.
            try { return command is "recognize-whole" or "build_score" or "install_models"
                    ? await completion.Task.WaitAsync(ct)
                    : await completion.Task.WaitAsync(TimeSpan.FromSeconds(960), ct); }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                await SendAsync(new { protocol=Protocol, id=Guid.NewGuid().ToString("N"), command="cancel", target=id });
                if (await Task.WhenAny(completion.Task, Task.Delay(6500)) != completion.Task) Stop();
                throw;
            }
            catch (TimeoutException) { Stop(); throw; }
        }
        finally { pending.TryRemove(id, out _); }
    }

    public static async Task<string> GpuStatusAsync(DiagnosticBundle? diagnostics = null)
    {
        var info=new ProcessStartInfo("nvidia-smi") { UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true };
        info.ArgumentList.Add("--query-gpu=name,memory.used,memory.total");
        info.ArgumentList.Add("--format=csv,noheader,nounits");
        try
        {
            using var probe=Process.Start(info) ?? throw new IOException("Cannot start nvidia-smi");
            var stdout=probe.StandardOutput.ReadToEndAsync();var stderr=probe.StandardError.ReadToEndAsync();
            using var limit=new CancellationTokenSource(TimeSpan.FromSeconds(8));
            try { await probe.WaitForExitAsync(limit.Token); }
            catch(OperationCanceledException){probe.Kill(entireProcessTree:true);return "GPU 状态查询超时；未运行模型";}
            var text=(await stdout).Trim();var error=(await stderr).Trim();
            if (probe.ExitCode == 0) diagnostics?.RecordGpu(text);
            return probe.ExitCode==0 ? "GPU 快照（名称 / 已用 MiB / 总 MiB）："+text+"；进程预算 7 GiB / 整卡 14 GiB" : "GPU 状态不可用："+error;
        }
        catch(Exception e) when(e is System.ComponentModel.Win32Exception or IOException) { return "GPU 状态不可用："+e.Message; }
    }

    private void FailPending(Exception error)
    { foreach (var promise in pending.Values) promise.TrySetException(error); }

    private void Stop()
    {
        var child = process; process = null;
        job?.Dispose(); job = null;
        if (child is not null)
        {
            try { if (!child.HasExited) child.Kill(entireProcessTree:true); }
            catch (InvalidOperationException) { }
            child.Dispose();
        }
        FailPending(new IOException("Owned local process stopped; no automatic retry"));
    }

    public void Dispose() { disposed=true; Stop(); }
}
