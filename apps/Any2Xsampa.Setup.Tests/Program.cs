using Any2Xsampa.Setup;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;

if (args.Length == 5 && args[0] == "--install")
{
    using var source = File.OpenRead(args[1]);
    var value = Installer.Read(source);
    string target = Installer.Install(value, args[2], args[3], null, CancellationToken.None);
    File.WriteAllText(args[4], JsonSerializer.Serialize(new { status = "passed", target }));
    return;
}
if (args.Length != 1) throw new ArgumentException("test output directory");
string workspace = Path.GetFullPath(args[0]);
if (Directory.Exists(workspace)) throw new IOException("Use a new test directory");
Directory.CreateDirectory(workspace);
int passed = 0;
string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
void Check(bool ok, string name) { if (!ok) throw new Exception(name); passed++; }
void Reject(Action action, string name)
{
    try { action(); } catch (Exception e) when (e is IOException or OperationCanceledException or InvalidDataException) { passed++; return; }
    throw new Exception("Accepted: " + name);
}
(ReleaseManifest manifest, string directory) Make(string id, string entryName = "Application/中文.txt")
{
    var directory = Path.Combine(workspace, id); Directory.CreateDirectory(directory);
    using var memory = new MemoryStream();
    using (var zip = new ZipArchive(memory, ZipArchiveMode.Create, true))
    {
        var entry = zip.CreateEntry(entryName);
        using var s = entry.Open(); s.Write(new byte[] { 1, 2, 3, 4 });
    }
    byte[] data = memory.ToArray(); int mid = data.Length / 2;
    byte[][] chunks = { data[..mid], data[mid..] };
    var parts = chunks.Select((c, i) => new Part("app.zip." + (i + 1).ToString("D3"), c.Length, Hash(c))).ToArray();
    for (int i = 0; i < parts.Length; i++) File.WriteAllBytes(Path.Combine(directory, parts[i].name), chunks[i]);
    return (new("portable_split_release_v1", "Application", data.Length, Hash(data), 4, 1, parts), directory);
}
var normal = Make("good");
string installed = Installer.Install(normal.manifest, normal.directory, Path.Combine(workspace, "中文 空格"), null, CancellationToken.None);
Check(File.ReadAllBytes(Path.Combine(installed, "中文.txt")).SequenceEqual(new byte[] { 1, 2, 3, 4 }), "exact extraction");
Check(!Directory.GetDirectories(Path.GetDirectoryName(installed)!, ".any2xsampa-install-*").Any(), "cleanup after success");
Reject(() => Installer.Install(normal.manifest, normal.directory, Path.GetDirectoryName(installed)!, null, CancellationToken.None), "existing target");
Check(File.Exists(Path.Combine(installed, "中文.txt")), "existing target preserved");
var bad = Make("bad");
string badPart = Path.Combine(bad.directory, bad.manifest.parts[0].name); byte[] corrupt = File.ReadAllBytes(badPart); corrupt[3] ^= 1; File.WriteAllBytes(badPart, corrupt);
string failed = Path.Combine(workspace, "failed");
Reject(() => Installer.Install(bad.manifest, bad.directory, failed, null, CancellationToken.None), "corrupt volume");
Check(!Directory.GetFileSystemEntries(failed).Any(), "cleanup after hash failure");
var traversal = Make("unsafe", "Application/../escape.txt");
Reject(() => Installer.Install(traversal.manifest, traversal.directory, Path.Combine(workspace, "unsafe-target"), null, CancellationToken.None), "zip path traversal");
var missing = Make("missing"); File.Delete(Path.Combine(missing.directory, missing.manifest.parts[1].name));
Reject(() => Installer.Install(missing.manifest, missing.directory, Path.Combine(workspace, "missing-target"), null, CancellationToken.None), "missing volume");
using var cancel = new CancellationTokenSource(); cancel.Cancel();
Reject(() => Installer.Install(normal.manifest, normal.directory, Path.Combine(workspace, "cancelled"), null, cancel.Token), "cancel before changes");
Check(!Directory.Exists(Path.Combine(workspace, "cancelled")), "cancel does not write");
Reject(() => Installer.Validate(normal.manifest with { archive_bytes = 12 }), "size binding");
Reject(() => Installer.Validate(normal.manifest with { parts = new[] { normal.manifest.parts[0], normal.manifest.parts[0] } }), "duplicate parts");
var fullMismatch = normal.manifest with { archive_sha256 = new string('0', 64) };
Reject(() => Installer.Install(fullMismatch, normal.directory, Path.Combine(workspace, "full-hash"), null, CancellationToken.None), "whole archive hash");
var report = new { status = "passed", checks = passed, model_forwards = 0 };
File.WriteAllText(Path.Combine(workspace, "RESULTS.json"), JsonSerializer.Serialize(report));
Console.WriteLine(JsonSerializer.Serialize(report));
