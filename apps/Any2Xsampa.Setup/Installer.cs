using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Any2Xsampa.Setup;

public sealed record Part(string name, long bytes, string sha256);
public sealed record ReleaseManifest(string schema_version, string root_directory, long archive_bytes,
    string archive_sha256, long expanded_bytes, int file_count, Part[] parts,
    string? release_tag = null, string? download_base_url = null);
public sealed record InstallProgress(int percent, string message);
public sealed record InstalledFile(string path, string sha256);
public sealed record InstallReceipt(string schema_version, string archive_sha256, InstalledFile[] files);

public static class Installer
{
    static bool HashValid(string value) => Regex.IsMatch(value ?? "", "^[0-9a-f]{64}$");
    static bool SimpleName(string value) => !string.IsNullOrWhiteSpace(value) && value is not "." and not ".."
        && value.IndexOfAny(Path.GetInvalidFileNameChars()) < 0 && !value.Contains('/') && !value.Contains('\\')
        && !value.EndsWith(' ') && !value.EndsWith('.');

    public static ReleaseManifest Read(Stream stream)
    {
        var value = JsonSerializer.Deserialize<ReleaseManifest>(stream)
            ?? throw new InvalidDataException("安装清单为空");
        Validate(value);
        return value;
    }

    public static void Validate(ReleaseManifest value)
    {
        if (value.schema_version is not ("portable_split_release_v1" or "portable_split_release_v2") || !SimpleName(value.root_directory)
            || value.archive_bytes <= 0 || value.expanded_bytes <= 0 || value.file_count <= 0
            || !HashValid(value.archive_sha256) || value.parts is not { Length: > 0 and <= 16 })
            throw new InvalidDataException("安装清单无效");
        if (value.schema_version == "portable_split_release_v2")
        {
            if (!Regex.IsMatch(value.release_tag ?? "", "^v[0-9]+\\.[0-9]+\\.[0-9]+$")
                || value.download_base_url != "https://github.com/Hoshino-Rubi/any2xsampa/releases/download/"
                    + value.release_tag + "/")
                throw new InvalidDataException("正式版下载地址必须绑定版本和指定仓库");
        }
        else if (value.release_tag is not null || value.download_base_url is not null)
            throw new InvalidDataException("预览版清单不应包含下载地址");
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long total = 0;
        foreach (var part in value.parts)
        {
            if (!SimpleName(part.name) || part.bytes <= 0 || part.bytes >= 2L * 1024 * 1024 * 1024
                || !HashValid(part.sha256) || !names.Add(part.name))
                throw new InvalidDataException("分卷清单无效");
            total = checked(total + part.bytes);
        }
        if (total != value.archive_bytes) throw new InvalidDataException("分卷大小与清单不符");
    }

    public static string Install(ReleaseManifest manifest, string partsDirectory, string parentDirectory,
        IProgress<InstallProgress>? progress, CancellationToken cancel, string? reuseModelsFrom = null)
    {
        Validate(manifest);
        cancel.ThrowIfCancellationRequested();
        string parent = Path.GetFullPath(parentDirectory);
        string target = Path.Combine(parent, manifest.root_directory);
        if (Directory.Exists(target) || File.Exists(target))
            throw new IOException("目标文件夹已存在，请选择其他安装位置；不会覆盖已有模型和结果。");
        foreach (var part in manifest.parts)
        {
            var file = new FileInfo(Path.Combine(partsDirectory, part.name));
            if (!file.Exists || file.Length != part.bytes)
                throw new InvalidDataException("分卷缺失或大小不符：" + part.name);
        }
        Directory.CreateDirectory(parent);
        var drive = new DriveInfo(Path.GetPathRoot(parent)!);
        if (drive.AvailableFreeSpace < checked(manifest.archive_bytes + manifest.expanded_bytes + 128L * 1024 * 1024))
            throw new IOException("磁盘空间不足，安装过程需要约 8 GiB 可用空间。");
        string staging = Path.Combine(parent, ".any2xsampa-install-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(staging);
        try
        {
            string archive = Path.Combine(staging, "application.zip");
            Join(manifest, partsDirectory, archive, progress, cancel);
            string payload = Path.Combine(staging, "payload");
            var installedFiles = Extract(manifest, archive, payload, progress, cancel);
            if (!string.IsNullOrWhiteSpace(reuseModelsFrom))
            {
                var reused = ModelReuse.CopyVerified(reuseModelsFrom, Path.Combine(payload, manifest.root_directory), cancel);
                progress?.Report(new(99, $"已复用 {reused.files} 个经过校验的旧版模型文件。"));
            }
            cancel.ThrowIfCancellationRequested();
            string receiptFile = Path.Combine(payload, manifest.root_directory, "installation-receipt.json");
            using (var receipt = new FileStream(receiptFile, FileMode.CreateNew, FileAccess.Write))
                JsonSerializer.Serialize(receipt, new InstallReceipt("any2xsampa_install_receipt_v1",
                    manifest.archive_sha256, installedFiles));
            Directory.Move(Path.Combine(payload, manifest.root_directory), target);
            progress?.Report(new(100, "安装完成。启动应用后将自动下载模型。"));
            return target;
        }
        finally
        {
            // Only this invocation's newly created staging tree can be removed.
            if (Directory.Exists(staging)) Directory.Delete(staging, true);
        }
    }

    static void Join(ReleaseManifest manifest, string directory, string destination,
        IProgress<InstallProgress>? progress, CancellationToken cancel)
    {
        byte[] buffer = new byte[1024 * 1024];
        long total = 0;
        using var fullHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var output = new FileStream(destination, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        foreach (var part in manifest.parts)
        {
            using var input = new FileStream(Path.Combine(directory, part.name), FileMode.Open, FileAccess.Read, FileShare.Read);
            using var partHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long count = 0;
            int read;
            while ((read = input.Read(buffer)) > 0)
            {
                cancel.ThrowIfCancellationRequested();
                count += read; total += read;
                if (count > part.bytes) throw new InvalidDataException("分卷大小变化：" + part.name);
                partHash.AppendData(buffer, 0, read); fullHash.AppendData(buffer, 0, read);
                output.Write(buffer, 0, read);
                progress?.Report(new((int)(total * 40 / manifest.archive_bytes), "正在校验 " + part.name));
            }
            if (count != part.bytes || Convert.ToHexString(partHash.GetHashAndReset()).ToLowerInvariant() != part.sha256)
                throw new InvalidDataException("分卷 SHA256 校验失败：" + part.name);
        }
        if (total != manifest.archive_bytes || Convert.ToHexString(fullHash.GetHashAndReset()).ToLowerInvariant() != manifest.archive_sha256)
            throw new InvalidDataException("完整安装包 SHA256 校验失败");
        output.Flush(true);
    }

    static InstalledFile[] Extract(ReleaseManifest manifest, string archivePath, string payload,
        IProgress<InstallProgress>? progress, CancellationToken cancel)
    {
        using var archive = ZipFile.OpenRead(archivePath);
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var entries = new List<(ZipArchiveEntry entry, string path)>();
        long expanded = 0;
        foreach (var entry in archive.Entries)
        {
            string name = entry.FullName;
            string[] parts = name.Split('/');
            if (parts.Length < 2 || parts[0] != manifest.root_directory || parts.Any(p => !SimpleName(p))
                || !names.Add(name) || ((entry.ExternalAttributes >> 16) & 0xF000) == 0xA000
                || (entry.ExternalAttributes & (int)FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException("不安全或重复的 ZIP 路径：" + name);
            string path = Path.GetFullPath(Path.Combine(payload, Path.Combine(parts)));
            if (!path.StartsWith(Path.GetFullPath(payload) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("ZIP 路径越界");
            expanded = checked(expanded + entry.Length);
            entries.Add((entry, path));
        }
        if (entries.Count != manifest.file_count || expanded != manifest.expanded_bytes)
            throw new InvalidDataException("ZIP 文件数量或展开大小与清单不符");
        byte[] buffer = new byte[1024 * 1024];
        var installed = new List<InstalledFile>();
        long done = 0;
        foreach (var (entry, path) in entries)
        {
            cancel.ThrowIfCancellationRequested();
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            using var input = entry.Open();
            using var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long count = 0; int read;
            while ((read = input.Read(buffer)) > 0)
            {
                cancel.ThrowIfCancellationRequested(); count += read;
                if (count > entry.Length) throw new InvalidDataException("ZIP 内容大小不符");
                digest.AppendData(buffer, 0, read);
                output.Write(buffer, 0, read);
            }
            if (count != entry.Length) throw new InvalidDataException("ZIP 内容不完整");
            installed.Add(new(entry.FullName[(manifest.root_directory.Length + 1)..],
                Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant()));
            done += count;
            progress?.Report(new(40 + (int)(done * 60 / manifest.expanded_bytes), "正在解压 " + entry.Name));
        }
        return installed.ToArray();
    }

    public static (int removed, int preserved) Uninstall(ReleaseManifest manifest, string target)
    {
        Validate(manifest);
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(target));
        if (Path.GetFileName(root) != manifest.root_directory
            || (File.GetAttributes(root) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("卸载目标与此版本不符");
        string receiptFile = Path.Combine(root, "installation-receipt.json");
        var receipt = JsonSerializer.Deserialize<InstallReceipt>(File.ReadAllBytes(receiptFile));
        if (receipt?.schema_version != "any2xsampa_install_receipt_v1"
            || receipt.archive_sha256 != manifest.archive_sha256 || receipt.files.Length != manifest.file_count)
            throw new InvalidDataException("卸载清单与此安装器不符");
        var targets = new List<(string path, string hash)>();
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int preserved = 0;
        foreach (var row in receipt.files)
        {
            string[] parts = row.path.Split('/');
            if (parts.Any(p => !SimpleName(p)) || !HashValid(row.sha256) || !names.Add(row.path))
                throw new InvalidDataException("卸载清单包含不安全路径");
            string full = Path.GetFullPath(Path.Combine(root, Path.Combine(parts)));
            if (!full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("卸载路径越界");
            if (parts[0].Equals("models", StringComparison.OrdinalIgnoreCase)
                || parts[0].Equals(".state", StringComparison.OrdinalIgnoreCase)) { preserved++; continue; }
            bool linked = false;
            for (string? path = full; path is not null && path.Length > root.Length; path = Path.GetDirectoryName(path))
                if (File.Exists(path) || Directory.Exists(path))
                    linked |= (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0;
            if (linked) throw new InvalidDataException("卸载路径经过链接，已停止");
            if (File.Exists(full)) targets.Add((full, row.sha256));
        }
        var removals = new List<string>();
        foreach (var (file, expectedHash) in targets)
        {
            // Pre-open every file exclusively: a running app aborts before anything is removed.
            using var input = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.None);
            if (Convert.ToHexString(SHA256.HashData(input)).Equals(expectedHash, StringComparison.OrdinalIgnoreCase))
                removals.Add(file);
            else preserved++;
        }
        foreach (string file in removals) File.Delete(file);
        // Keep receipt, model directories, unlisted user files and all per-user data.
        return (removals.Count, preserved);
    }
}
