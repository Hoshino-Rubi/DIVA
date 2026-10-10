using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace Any2Xsampa.Setup;

/// <summary>Downloads only embedded, SHA-bound release volumes; never contacts a manifest URL.</summary>
public static class PartDownloader
{
    public static async Task<string> EnsureAsync(ReleaseManifest manifest, string localDirectory,
        string cacheDirectory, IProgress<InstallProgress>? progress, CancellationToken cancel,
        HttpClient? client = null)
    {
        Installer.Validate(manifest);
        cancel.ThrowIfCancellationRequested();
        if (manifest.schema_version != "portable_split_release_v2")
            throw new InvalidDataException("此安装器未配置自动下载；请使用同目录分卷安装。");
        Directory.CreateDirectory(cacheDirectory);
        if ((File.GetAttributes(cacheDirectory) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("下载缓存不能是链接目录");
        using var cacheLock = new FileStream(Path.Combine(cacheDirectory, "download.lock"),
            FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        bool ownsClient = client is null;
        client ??= new HttpClient();
        try
        {
            for (int index = 0; index < manifest.parts.Length; index++)
            {
                cancel.ThrowIfCancellationRequested();
                var part = manifest.parts[index];
                string complete = Path.Combine(cacheDirectory, part.name);
                if (File.Exists(complete))
                {
                    if (!await Matches(complete, part, cancel))
                        throw new InvalidDataException("缓存分卷校验失败，请删除该缓存分卷后显式重试：" + part.name);
                    continue;
                }
                string local = Path.Combine(localDirectory, part.name);
                if (File.Exists(local))
                {
                    if (!await Matches(local, part, cancel))
                        throw new InvalidDataException("本地分卷校验失败：" + part.name);
                    string copying = complete + ".copying";
                    try
                    {
                        await using (var source = File.OpenRead(local))
                        await using (var target = new FileStream(copying, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                            await source.CopyToAsync(target, cancel);
                        if (!await Matches(copying, part, cancel))
                            throw new InvalidDataException("复制后分卷校验失败：" + part.name);
                        File.Move(copying, complete);
                    }
                    finally { if (File.Exists(copying)) File.Delete(copying); }
                    continue;
                }
                await DownloadAsync(manifest, part, cacheDirectory, client, cancel, progress,
                    index, manifest.parts.Length);
            }
            return cacheDirectory;
        }
        finally { if (ownsClient) client.Dispose(); }
    }

    private static async Task<bool> Matches(string path, Part part, CancellationToken cancel)
    {
        var info = new FileInfo(path);
        if ((info.Attributes & FileAttributes.ReparsePoint) != 0 || info.Length != part.bytes) return false;
        await using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        string hash = Convert.ToHexString(await SHA256.HashDataAsync(input, cancel)).ToLowerInvariant();
        return hash == part.sha256;
    }

    private static async Task DownloadAsync(ReleaseManifest manifest, Part part, string cache,
        HttpClient client, CancellationToken cancel, IProgress<InstallProgress>? progress,
        int index, int count)
    {
        string partial = Path.Combine(cache, part.name + ".part");
        var partialInfo = new FileInfo(partial);
        if (partialInfo.Exists && (partialInfo.Attributes & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("缓存分卷不能是链接：" + part.name);
        long offset = partialInfo.Exists ? partialInfo.Length : 0;
        if (offset > part.bytes) throw new InvalidDataException("缓存分卷超出预期大小：" + part.name);
        if (offset == part.bytes)
        {
            if (!await Matches(partial, part, cancel)) throw new InvalidDataException("完整缓存分卷哈希错误：" + part.name);
            File.Move(partial, Path.Combine(cache, part.name));
            return;
        }
        var url = new Uri(manifest.download_base_url + Uri.EscapeDataString(part.name));
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.TryAddWithoutValidation("Accept-Encoding", "identity");
        request.Headers.UserAgent.ParseAdd("Any2Xsampa-Setup/" + manifest.release_tag!.TrimStart('v'));
        if (offset != 0) request.Headers.Range = new RangeHeaderValue(offset, null);
        using var headerLimit = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        headerLimit.CancelAfter(TimeSpan.FromSeconds(30));
        HttpResponseMessage headers;
        try { headers = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, headerLimit.Token); }
        catch (OperationCanceledException) when (!cancel.IsCancellationRequested)
        { throw new TimeoutException("分卷连接超时；保留续传缓存，没有自动重试。"); }
        using var response = headers;
        if (response.RequestMessage?.RequestUri?.Scheme != Uri.UriSchemeHttps)
            throw new InvalidDataException("下载被重定向到非 HTTPS 地址");
        if (offset == 0 ? response.StatusCode != HttpStatusCode.OK : response.StatusCode != HttpStatusCode.PartialContent)
            throw new HttpRequestException("发布分卷不可访问或服务器拒绝精确续传：" + part.name + " (" + (int)response.StatusCode + ")");
        if (offset != 0)
        {
            var range = response.Content.Headers.ContentRange;
            if (range?.From != offset || range.To != part.bytes - 1 || range.Length != part.bytes)
                throw new InvalidDataException("续传范围不等于固定分卷；缓存保持不变：" + part.name);
        }
        long remaining = part.bytes - offset;
        if (response.Content.Headers.ContentLength is long size && size != remaining)
            throw new InvalidDataException("下载大小与固定清单不符：" + part.name);
        await using var source = await response.Content.ReadAsStreamAsync(cancel);
        await using (var target = new FileStream(partial, partialInfo.Exists ? FileMode.Open : FileMode.CreateNew,
            FileAccess.Write, FileShare.None, 1024 * 1024, useAsync: true))
        {
            if (target.Length != offset) throw new InvalidDataException("续传缓存大小发生变化");
            target.Position = offset;
            var buffer = new byte[1024 * 1024];
            int read;
            while (true)
            {
                using var idleLimit = CancellationTokenSource.CreateLinkedTokenSource(cancel);
                idleLimit.CancelAfter(TimeSpan.FromSeconds(60));
                try { read = await source.ReadAsync(buffer, idleLimit.Token); }
                catch (OperationCanceledException) when (!cancel.IsCancellationRequested)
                { throw new TimeoutException("分卷下载无响应；保留续传缓存，没有自动重试。"); }
                if (read == 0) break;
                offset += read;
                if (offset > part.bytes) throw new InvalidDataException("下载分卷超出固定清单：" + part.name);
                await target.WriteAsync(buffer.AsMemory(0, read), cancel);
                progress?.Report(new(5 + (int)((index + (double)offset / part.bytes) * 30 / count),
                    "正在下载并校验 " + part.name));
            }
            await target.FlushAsync(cancel);
        }
        if (offset != part.bytes) throw new InvalidDataException("分卷下载不完整，保留续传缓存：" + part.name);
        if (!await Matches(partial, part, cancel))
            throw new InvalidDataException("下载 SHA256 不匹配，保留文件供排查：" + part.name);
        File.Move(partial, Path.Combine(cache, part.name));
    }
}
