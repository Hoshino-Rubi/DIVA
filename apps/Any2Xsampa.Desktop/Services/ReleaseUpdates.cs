using System.Net.Http;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Any2Xsampa_Desktop.Services;

public sealed record ReleaseUpdate(string Tag, bool IsNewer, Uri Page);

/// <summary>Explicit, read-only update check. It never downloads or replaces executables.</summary>
public static class ReleaseUpdates
{
    public static string CurrentTag => typeof(ReleaseUpdates).Assembly
        .GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion.Split('+')[0];

    public static ReleaseUpdate Parse(string json, string currentTag)
    {
        using var document = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = 16 });
        var release = document.RootElement;
        string tag = release.GetProperty("tag_name").GetString() ?? "";
        if (release.GetProperty("draft").GetBoolean() || release.GetProperty("prerelease").GetBoolean()
            || !Regex.IsMatch(tag, "^v[0-9]+\\.[0-9]+\\.[0-9]+$"))
            throw new InvalidDataException("返回的不是正式版");
        var current = Version.Parse(currentTag.TrimStart('v'));
        var incoming = Version.Parse(tag[1..]);
        return new(tag, incoming > current,
            new Uri("https://github.com/Hoshino-Rubi/any2xsampa/releases/tag/" + tag));
    }

    public static async Task<ReleaseUpdate> CheckAsync(CancellationToken cancel, HttpClient? client = null)
    {
        bool ownsClient = client is null;
        client ??= new HttpClient();
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        limit.CancelAfter(TimeSpan.FromSeconds(15));
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Get,
                "https://api.github.com/repos/Hoshino-Rubi/any2xsampa/releases/latest");
            request.Headers.UserAgent.ParseAdd("Any2Xsampa/" + CurrentTag.TrimStart('v'));
            using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, limit.Token);
            response.EnsureSuccessStatusCode();
            await using var input = await response.Content.ReadAsStreamAsync(limit.Token);
            using var output = new MemoryStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = await input.ReadAsync(buffer, limit.Token)) != 0)
            {
                if (output.Length + read > 1024 * 1024) throw new InvalidDataException("更新响应过大");
                output.Write(buffer, 0, read);
            }
            return Parse(System.Text.Encoding.UTF8.GetString(output.ToArray()), CurrentTag);
        }
        finally { if (ownsClient) client.Dispose(); }
    }
}
