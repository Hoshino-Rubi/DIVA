using Any2Xsampa.Setup;
using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
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
var stable = normal.manifest with {
    schema_version = "portable_split_release_v2", release_tag = "v1.0.1",
    download_base_url = "https://github.com/Hoshino-Rubi/any2xsampa/releases/download/v1.0.1/"
};
Installer.Validate(stable); Check(true, "stable manifest accepted");
var diva = stable with { release_tag = "v1.0.2", download_base_url = "https://github.com/Hoshino-Rubi/DIVA/releases/download/v1.0.2/" };
Installer.Validate(diva); Check(true, "DIVA origin accepted alongside legacy origin");
foreach (string invalid in new[] {
    "https://github.com/Hoshino-Rubi/DIVA/releases/download/v1.0.1/",
    "https://github.com/Hoshino-Rubi/DIVA/releases/download/v1.0.2/?token=x",
    "https://github.com/Hoshino-Rubi/DIVA/releases/download/v1.0.2/../",
    "https://github.com/Hoshino-Rubi/DIVA-evil/releases/download/v1.0.2/",
    "https://github.com.evil.invalid/Hoshino-Rubi/DIVA/releases/download/v1.0.2/",
    "http://github.com/Hoshino-Rubi/DIVA/releases/download/v1.0.2/" })
    Reject(() => Installer.Validate(diva with { download_base_url = invalid }), "DIVA exact origin/tag validation");
Reject(() => Installer.Validate(stable with { download_base_url = "https://example.invalid/v1.0.1/" }), "untrusted download origin");
Reject(() => Installer.Validate(stable with { release_tag = "v1.0.2" }), "wrong tag download origin");
var fake = new FakeReleaseHandler(normal.directory, normal.manifest.parts);
using var http = new HttpClient(fake);
string cache = Path.Combine(workspace, "download-cache");
string emptyLocal = Path.Combine(workspace, "no-local"); Directory.CreateDirectory(emptyLocal);
await PartDownloader.EnsureAsync(stable, emptyLocal, cache, null, CancellationToken.None, http);
Check(fake.Calls == stable.parts.Length, "download every pinned part once");
foreach (var part in stable.parts)
    Check(File.ReadAllBytes(Path.Combine(cache, part.name)).SequenceEqual(File.ReadAllBytes(Path.Combine(normal.directory, part.name))), "downloaded exact part");
await PartDownloader.EnsureAsync(stable, emptyLocal, cache, null, CancellationToken.None, http);
Check(fake.Calls == stable.parts.Length, "verified cache avoids another network call");
string resumed = Path.Combine(workspace, "resume-cache"); Directory.CreateDirectory(resumed);
var firstPart = stable.parts[0];
File.WriteAllBytes(Path.Combine(resumed, firstPart.name + ".part"), File.ReadAllBytes(Path.Combine(normal.directory, firstPart.name))[..2]);
await PartDownloader.EnsureAsync(stable, emptyLocal, resumed, null, CancellationToken.None, http);
Check(fake.RangeRequests > 0, "exact Range resume");
string zeroCache = Path.Combine(workspace, "zero-byte-cache"); Directory.CreateDirectory(zeroCache);
File.WriteAllBytes(Path.Combine(zeroCache, firstPart.name + ".part"), Array.Empty<byte>());
await PartDownloader.EnsureAsync(stable, emptyLocal, zeroCache, null, CancellationToken.None, http);
Check(File.Exists(Path.Combine(zeroCache, firstPart.name)), "zero byte partial resumes without manual removal");
var refused = new FakeReleaseHandler(normal.directory, normal.manifest.parts) { RefuseRange = true };
using var refusedHttp = new HttpClient(refused);
string refusedCache = Path.Combine(workspace, "refused-cache"); Directory.CreateDirectory(refusedCache);
File.WriteAllBytes(Path.Combine(refusedCache, firstPart.name + ".part"), File.ReadAllBytes(Path.Combine(normal.directory, firstPart.name))[..2]);
try {
    await PartDownloader.EnsureAsync(stable, emptyLocal, refusedCache, null, CancellationToken.None, refusedHttp);
    throw new Exception("Range refusal accepted");
} catch (HttpRequestException) { passed++; }
Check(new FileInfo(Path.Combine(refusedCache, firstPart.name + ".part")).Length == 2, "partial retained on refusal");
string corruptCache = Path.Combine(workspace, "corrupt-cache"); Directory.CreateDirectory(corruptCache);
File.WriteAllBytes(Path.Combine(corruptCache, firstPart.name), new byte[firstPart.bytes]);
int beforeCorrupt = fake.Calls;
try {
    await PartDownloader.EnsureAsync(stable, emptyLocal, corruptCache, null, CancellationToken.None, http);
    throw new Exception("Bad cache accepted");
} catch (InvalidDataException) { passed++; }
Check(fake.Calls == beforeCorrupt, "corrupt cache did not trigger hidden retry");
string badDownloadCache = Path.Combine(workspace, "bad-download-cache");
var poisoned = new FakeReleaseHandler(normal.directory, normal.manifest.parts) { TamperBody = true };
using var poisonedHttp = new HttpClient(poisoned);
try {
    await PartDownloader.EnsureAsync(stable, emptyLocal, badDownloadCache, null, CancellationToken.None, poisonedHttp);
    throw new Exception("Tampered download accepted");
} catch (InvalidDataException) { passed++; }
Check(!File.Exists(Path.Combine(badDownloadCache, firstPart.name)), "tampered download not promoted");
using var cancelledDownload = new CancellationTokenSource(); cancelledDownload.Cancel();
try {
    await PartDownloader.EnsureAsync(stable, emptyLocal, Path.Combine(workspace, "cancel-download"), null, cancelledDownload.Token, http);
    throw new Exception("Cancelled download accepted");
} catch (OperationCanceledException) { passed++; }
string oldModels = Path.Combine(workspace, "previous-install");
string newModels = Path.Combine(workspace, "new-install");
Directory.CreateDirectory(Path.Combine(oldModels, "models", "test")); Directory.CreateDirectory(newModels);
byte[] priorWeight = { 4, 2, 8, 8 };
string priorFile = Path.Combine(oldModels, "models", "test", "weights.bin");
File.WriteAllBytes(priorFile, priorWeight);
void WriteCatalog(string relative) => File.WriteAllText(Path.Combine(newModels, "model-downloads.json"), JsonSerializer.Serialize(new {
    schema_version = "local_model_downloads_v1", bundles = new[] { new { files = new[] {
        new { path = relative, bytes = priorWeight.Length, sha256 = Hash(priorWeight) }
    } } }
}));
WriteCatalog("models/test/weights.bin");
var reuse = ModelReuse.CopyVerified(oldModels, newModels, CancellationToken.None);
Check(reuse.files == 1 && reuse.bytes == priorWeight.Length, "reuse verified model");
Check(File.ReadAllBytes(priorFile).SequenceEqual(priorWeight), "old model unchanged");
File.Delete(Path.Combine(newModels, "models", "test", "weights.bin"));
File.WriteAllBytes(priorFile, new byte[] { 0, 0, 0, 0 });
Check(ModelReuse.CopyVerified(oldModels, newModels, CancellationToken.None).files == 0, "corrupt old model skipped");
Check(!File.Exists(Path.Combine(newModels, "models", "test", "weights.bin")), "bad old model not copied");
WriteCatalog("models/../escape.bin");
Reject(() => ModelReuse.CopyVerified(oldModels, newModels, CancellationToken.None), "model catalog traversal");
string upgradeParts = Path.Combine(workspace, "upgrade-parts"); Directory.CreateDirectory(upgradeParts);
using var upgradeMemory = new MemoryStream();
byte[] reuseWeight = { 7, 7, 5 };
byte[] upgradeCatalog = JsonSerializer.SerializeToUtf8Bytes(new { schema_version = "local_model_downloads_v1",
    bundles = new[] { new { files = new[] { new { path = "models/test/weight.bin", bytes = reuseWeight.Length,
        sha256 = Hash(reuseWeight) } } } } });
using (var archive = new ZipArchive(upgradeMemory, ZipArchiveMode.Create, true))
{
    var entry = archive.CreateEntry("Application/model-downloads.json");
    using var stream = entry.Open(); stream.Write(upgradeCatalog);
}
byte[] upgradeZip = upgradeMemory.ToArray();
string upgradePart = "upgrade.zip.001"; File.WriteAllBytes(Path.Combine(upgradeParts, upgradePart), upgradeZip);
var upgradeManifest = new ReleaseManifest("portable_split_release_v1", "Application", upgradeZip.Length,
    Hash(upgradeZip), upgradeCatalog.Length, 1, new[] { new Part(upgradePart, upgradeZip.Length, Hash(upgradeZip)) });
string previous = Path.Combine(workspace, "prior-with-results");
Directory.CreateDirectory(Path.Combine(previous, "models", "test"));
File.WriteAllBytes(Path.Combine(previous, "models", "test", "weight.bin"), reuseWeight);
File.WriteAllText(Path.Combine(previous, "my-result.vsqx"), "keep");
string upgraded = Installer.Install(upgradeManifest, upgradeParts, Path.Combine(workspace, "upgrade-destination"),
    null, CancellationToken.None, previous);
Check(File.ReadAllBytes(Path.Combine(upgraded, "models", "test", "weight.bin")).SequenceEqual(reuseWeight),
    "installer reused verified weight");
Check(File.ReadAllText(Path.Combine(previous, "my-result.vsqx")) == "keep", "old result preserved");
File.WriteAllText(Path.Combine(upgraded, "user-result.vsqx"), "saved");
var removedInstall = Installer.Uninstall(upgradeManifest, upgraded);
Check(removedInstall.removed == 1, "uninstall removes only installed application file");
Check(File.ReadAllText(Path.Combine(upgraded, "user-result.vsqx")) == "saved", "uninstall retains unlisted result");
Check(File.ReadAllBytes(Path.Combine(upgraded, "models", "test", "weight.bin")).SequenceEqual(reuseWeight),
    "uninstall retains reused model");
File.WriteAllText(Path.Combine(installed, "中文.txt"), "user modified");
Check(Installer.Uninstall(normal.manifest, installed).preserved == 1, "uninstall preserves modified file");
Reject(() => Installer.Uninstall(upgradeManifest with { archive_sha256 = new string('0', 64) }, upgraded),
    "wrong version cannot uninstall");
var report = new { status = "passed", checks = passed, model_forwards = 0 };
File.WriteAllText(Path.Combine(workspace, "RESULTS.json"), JsonSerializer.Serialize(report));
Console.WriteLine(JsonSerializer.Serialize(report));

sealed class FakeReleaseHandler(string sourceDirectory, Part[] parts) : HttpMessageHandler
{
    public int Calls { get; private set; }
    public int RangeRequests { get; private set; }
    public bool RefuseRange { get; set; }
    public bool TamperBody { get; set; }
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Calls++;
        string name = Uri.UnescapeDataString(request.RequestUri!.Segments[^1]);
        if (!parts.Any(p => p.name == name)) throw new Exception("Unexpected asset request");
        byte[] whole = File.ReadAllBytes(Path.Combine(sourceDirectory, name));
        if (TamperBody) whole[0] ^= 1;
        long offset = request.Headers.Range?.Ranges.Single().From ?? 0;
        if (offset > 0) RangeRequests++;
        var code = offset > 0 && !RefuseRange ? HttpStatusCode.PartialContent : HttpStatusCode.OK;
        var response = new HttpResponseMessage(code) { RequestMessage = request,
            Content = new ByteArrayContent(code == HttpStatusCode.PartialContent ? whole[(int)offset..] : whole) };
        if (code == HttpStatusCode.PartialContent)
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(offset, whole.Length - 1, whole.Length);
        return Task.FromResult(response);
    }
}
