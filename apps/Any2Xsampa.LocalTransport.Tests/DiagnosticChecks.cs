using Any2Xsampa_Desktop.Services;
using System.IO.Compression;
using System.Text.Json;

static class DiagnosticChecks
{
    public static async Task Run(string output)
    {
        string root = Path.GetFullPath(output);
        if (Directory.Exists(root)) throw new IOException("Use a new diagnostic test directory");
        Directory.CreateDirectory(root);
        string personal = Path.Combine(root, "PRIVATE_USER_私密");
        Directory.CreateDirectory(personal);
        string model = Path.Combine(personal, "models", "private-track.weights");
        Directory.CreateDirectory(Path.GetDirectoryName(model)!);
        await File.WriteAllTextAsync(model, "123");
        string config = Path.Combine(personal, "local-app.json");
        await File.WriteAllTextAsync(config, JsonSerializer.Serialize(new {root=".", python="missing-private.exe"}));
        await File.WriteAllTextAsync(Path.Combine(personal, "model-downloads.json"), JsonSerializer.Serialize(new {
            bundles=new[]{new{id="phoneticxeus",name="SECRET_LYRICS",files=new[]{
                new{path="models/private-track.weights",bytes=4,sha256=new string('a',64)},
                new{path="models/missing.weights",bytes=9,sha256=new string('b',64)}
            }}}
        }));
        using var service = new LocalTaskService(config);
        try { await service.CallAsync("models", new { proxy="https://SECRET_PASSWORD@example.invalid" }); }
        catch (FileNotFoundException) { }
        service.Diagnostics.Record("PRIVATE_USER_私密", "failed", new LocalTaskException("ValueError", "SECRET_TOKEN " + personal));
        service.Diagnostics.RecordGpu("PRIVATE_GPU_NAME, 10, 100");
        string bundle = Path.Combine(root, "diagnostic.zip");
        await service.ExportDiagnosticsAsync(bundle);
        using var zip = ZipFile.OpenRead(bundle);
        if (zip.Entries.Count != 2) throw new Exception("Unexpected attachments");
        using var reader = new StreamReader(zip.GetEntry("diagnostic.json")!.Open());
        string json = await reader.ReadToEndAsync();
        foreach (var privateText in new[]{"PRIVATE_USER", "私密", "SECRET", "private-track", "PRIVATE_GPU_NAME", personal})
            if (json.Contains(privateText)) throw new Exception("Private content leaked");
        using var doc = JsonDocument.Parse(json);
        var data = doc.RootElement;
        if (data.GetProperty("configuration").GetProperty("core_python_present").GetBoolean()) throw new Exception("Missing runtime hidden");
        if (data.GetProperty("events").EnumerateArray().All(x=>x.GetProperty("error_code").GetString()!="FileNotFoundException"))
            throw new Exception("Startup failure not recorded");
        if (data.GetProperty("models").GetProperty("models")[0].GetProperty("files")[0].GetProperty("state").GetString()!="size_mismatch")
            throw new Exception("Corruption not recorded");
        if (data.GetProperty("environment").GetProperty("gpu_used_mib").GetInt64()!=10) throw new Exception("GPU counters missing");
        byte[] before = await File.ReadAllBytesAsync(bundle);
        try { await service.ExportDiagnosticsAsync(bundle); throw new Exception("Existing export overwritten"); }
        catch (IOException) { }
        if (!(await File.ReadAllBytesAsync(bundle)).SequenceEqual(before)) throw new Exception("Export changed");
        for (int i=0;i<300;i++)service.Diagnostics.Record("doctor", "completed");
        string second=Path.Combine(root,"bounded.zip");
        await service.ExportDiagnosticsAsync(second);
        using var zip2=ZipFile.OpenRead(second);
        using var reader2=new StreamReader(zip2.GetEntry("diagnostic.json")!.Open());
        using var doc2=JsonDocument.Parse(await reader2.ReadToEndAsync());
        if(doc2.RootElement.GetProperty("events").GetArrayLength()!=256)throw new Exception("Unbounded journal");
        await File.WriteAllTextAsync(config, "{");
        await service.ExportDiagnosticsAsync(Path.Combine(root,"broken-config.zip"));
        File.Delete(config);
        await service.ExportDiagnosticsAsync(Path.Combine(root,"missing-config.zip"));
        Console.WriteLine(JsonSerializer.Serialize(new{status="passed",model_forwards=0,checks=10,
            privacy="allowlisted",startup_failure_export=true}));
    }
}
