using System.Security.Cryptography;
using System.Text.Json;

namespace Any2Xsampa.Setup;

/// <summary>Copies only weights matching the new package catalog; never moves or edits the old install.</summary>
public static class ModelReuse
{
    public static (int files, long bytes) CopyVerified(string oldRoot, string newRoot, CancellationToken cancel)
    {
        string catalog = Path.Combine(newRoot, "model-downloads.json");
        if (!Directory.Exists(oldRoot) || !File.Exists(catalog)) return (0, 0);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(catalog));
        if (doc.RootElement.GetProperty("schema_version").GetString() != "local_model_downloads_v1")
            throw new InvalidDataException("新版本模型目录清单格式错误");
        int files = 0; long bytes = 0;
        foreach (var bundle in doc.RootElement.GetProperty("bundles").EnumerateArray().Take(16))
        foreach (var row in bundle.GetProperty("files").EnumerateArray().Take(1024))
        {
            cancel.ThrowIfCancellationRequested();
            string relative = row.GetProperty("path").GetString() ?? "";
            if (!relative.StartsWith("models/", StringComparison.Ordinal)
                || relative.Split('/').Any(part => part is "" or "." or ".."))
                throw new InvalidDataException("不安全的模型复用路径");
            string oldFile = Path.GetFullPath(Path.Combine(oldRoot, relative.Replace('/', Path.DirectorySeparatorChar)));
            string newFile = Path.GetFullPath(Path.Combine(newRoot, relative.Replace('/', Path.DirectorySeparatorChar)));
            string oldBase = Path.TrimEndingDirectorySeparator(Path.GetFullPath(oldRoot));
            string newBase = Path.TrimEndingDirectorySeparator(Path.GetFullPath(newRoot));
            if (!oldFile.StartsWith(oldBase + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                || !newFile.StartsWith(newBase + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("模型复用路径越界");
            if (!File.Exists(oldFile) || File.Exists(newFile)
                || (File.GetAttributes(oldFile) & FileAttributes.ReparsePoint) != 0) continue;
            bool linkedParent = false;
            for (var directory = Path.GetDirectoryName(oldFile); directory is not null
                 && directory.Length > oldBase.Length; directory = Path.GetDirectoryName(directory))
                linkedParent |= (File.GetAttributes(directory) & FileAttributes.ReparsePoint) != 0;
            if (linkedParent) continue;
            long expectedBytes = row.GetProperty("bytes").GetInt64();
            string expectedHash = row.GetProperty("sha256").GetString() ?? "";
            if (expectedBytes < 0 || expectedHash.Length != 64 || new FileInfo(oldFile).Length != expectedBytes) continue;
            Directory.CreateDirectory(Path.GetDirectoryName(newFile)!);
            string temporary = newFile + ".reusing";
            using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long copied = 0;
            try
            {
                using (var input = File.OpenRead(oldFile))
                using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    byte[] buffer = new byte[1024 * 1024]; int read;
                    while ((read = input.Read(buffer)) > 0)
                    {
                        cancel.ThrowIfCancellationRequested();
                        copied += read;
                        if (copied > expectedBytes) break;
                        digest.AppendData(buffer, 0, read); output.Write(buffer, 0, read);
                    }
                }
                if (copied == expectedBytes
                    && Convert.ToHexString(digest.GetHashAndReset()).Equals(expectedHash, StringComparison.OrdinalIgnoreCase))
                {
                    File.Move(temporary, newFile); files++; bytes += copied;
                }
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
        return (files, bytes);
    }
}
