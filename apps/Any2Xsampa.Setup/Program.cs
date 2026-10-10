using System.Diagnostics;
using System.Reflection;
using System.Text.Json;

namespace Any2Xsampa.Setup;

internal static class Program
{
    [STAThread]
    static int Main(string[] args)
    {
        try
        {
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("release-manifest.json")
                ?? throw new InvalidOperationException("此安装器未绑定发布清单，请使用正式 Release 安装器。");
            var manifest = Installer.Read(stream);
            if (args.Length is 4 or 5 && args[0] == "--install")
            {
                if (File.Exists(args[3])) throw new IOException("Report already exists");
                var target = Installer.Install(manifest, args[1], args[2], null, CancellationToken.None, args.Length == 5 ? args[4] : null);
                File.WriteAllText(args[3], JsonSerializer.Serialize(new { status = "passed", target,
                    archive_sha256 = manifest.archive_sha256, model_forwards = 0, downloads = 0 }));
                return 0;
            }
            if (args.Length == 3 && args[0] == "--uninstall")
            {
                if (File.Exists(args[2])) throw new IOException("Report already exists");
                var removed = Installer.Uninstall(manifest, args[1]);
                File.WriteAllText(args[2], JsonSerializer.Serialize(new { status = "passed",
                    removed_files = removed.removed, preserved_files = removed.preserved, user_data_preserved = true }));
                return 0;
            }
            if (args.Length != 0) throw new ArgumentException("Usage: --install <parts_directory> <destination_parent> <report.json> [previous_install] or --uninstall <installation> <report.json>");
            ApplicationConfiguration.Initialize();
            Application.Run(new SetupForm(manifest));
            return 0;
        }
        catch (Exception ex)
        {
            if (args.Length > 0)
                Console.Error.WriteLine(ex.Message);
            else MessageBox.Show(ex.Message, "Any2Xsampa 安装失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}

internal sealed class SetupForm : Form
{
    readonly ReleaseManifest manifest;
    readonly TextBox folder = new() { Dock = DockStyle.Fill };
    readonly TextBox oldFolder = new() { Dock = DockStyle.Fill };
    readonly Label status = new() { AutoSize = true, MaximumSize = new Size(690, 0) };
    readonly ProgressBar progress = new() { Dock = DockStyle.Fill, Height = 25 };
    readonly Button browse = new() { Text = "选择位置", AutoSize = true };
    readonly Button install = new() { Text = "安装", AutoSize = true };
    readonly Button cancel = new() { Text = "关闭", AutoSize = true };
    readonly Button open = new() { Text = "打开应用", AutoSize = true, Visible = false };
    readonly Button remove = new() { Text = "卸载此版本", AutoSize = true };
    readonly Button oldBrowse = new() { Text = "选择旧版", AutoSize = true };
    CancellationTokenSource? active;
    string? installed;

    public SetupForm(ReleaseManifest value)
    {
        manifest = value;
        Text = "Any2Xsampa " + (manifest.release_tag ?? "") + " 安装";
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(730, 445);
        MinimumSize = new Size(650, 480);
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Microsoft YaHei UI", 10);
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(20), RowCount = 9, ColumnCount = 1 };
        layout.Controls.Add(new Label { Text = "Any2Xsampa · 本地音素与 VSQX 工作台", Font = new Font(Font.FontFamily, 15, FontStyle.Bold), AutoSize = true });
        layout.Controls.Add(new Label { Text = manifest.schema_version == "portable_split_release_v2"
            ? "只需此安装器：将从固定 GitHub Release 下载分卷、校验 SHA256 后安装。\n已下载的分卷保留在本机缓存；也可把三个分卷放在安装器旁边离线安装。"
            : "请将安装器与三个 .zip.001 / .002 / .003 分卷放在同一文件夹。\n程序会校验 SHA256 并解压；无需另外安装 Python 或 .NET。", AutoSize = true });
        layout.Controls.Add(new Label { Text = "安装位置（将在下方位置创建 " + manifest.root_directory + "）", AutoSize = true });
        var location = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, AutoSize = true };
        location.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        location.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        folder.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs");
        location.Controls.Add(folder, 0, 0); location.Controls.Add(browse, 1, 0);
        layout.Controls.Add(location);
        layout.Controls.Add(new Label { Text = "可选：旧版安装目录（仅复用 SHA256 验证通过的模型，不搬动旧结果）", AutoSize = true });
        var previous = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, AutoSize = true };
        previous.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        previous.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        previous.Controls.Add(oldFolder, 0, 0); previous.Controls.Add(oldBrowse, 1, 0);
        layout.Controls.Add(previous);
        layout.Controls.Add(progress);
        status.Text = "首次自动下载加安装需约 12 GiB 临时可用空间。模型会在应用首次启动时下载，也可离线导入。";
        layout.Controls.Add(status);
        var actions = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, FlowDirection = FlowDirection.RightToLeft };
        actions.Controls.Add(cancel); actions.Controls.Add(open); actions.Controls.Add(install); actions.Controls.Add(remove);
        layout.Controls.Add(actions); Controls.Add(layout);
        browse.Click += (_, _) => { using var dialog = new FolderBrowserDialog { InitialDirectory = folder.Text, Description = "选择应用安装位置" }; if (dialog.ShowDialog(this) == DialogResult.OK) folder.Text = dialog.SelectedPath; };
        oldBrowse.Click += (_, _) => { using var dialog = new FolderBrowserDialog { Description = "选择旧版应用根目录（包含 model-downloads.json）" }; if (dialog.ShowDialog(this) == DialogResult.OK) oldFolder.Text = dialog.SelectedPath; };
        install.Click += async (_, _) => await InstallAsync();
        remove.Click += async (_, _) =>
        {
            using var dialog = new FolderBrowserDialog { Description = "选择此版本的安装目录；将保留模型和结果。" };
            if (dialog.ShowDialog(this) != DialogResult.OK) return;
            if (MessageBox.Show(this, "删除未修改的程序文件？模型、结果和共享缓存将保留。请先关闭应用。",
                "卸载 Any2Xsampa", MessageBoxButtons.OKCancel) != DialogResult.OK) return;
            install.Enabled = remove.Enabled = false;
            try { var result = await Task.Run(() => Installer.Uninstall(manifest, dialog.SelectedPath));
                status.Text = $"已删除 {result.removed} 个程序文件；模型和用户数据保留。"; }
            catch (Exception ex) { status.Text = "卸载停止：" + ex.Message; }
            finally { install.Enabled = remove.Enabled = true; }
        };
        cancel.Click += (_, _) => { if (active != null) active.Cancel(); else Close(); };
        open.Click += (_, _) => { if (installed != null) Process.Start(new ProcessStartInfo(Path.Combine(installed, "Any2Xsampa.Desktop.exe")) { UseShellExecute = true, WorkingDirectory = installed }); };
        FormClosing += (_, e) => { if (active != null) { active.Cancel(); e.Cancel = true; status.Text = "正在取消并清理临时安装文件…"; } };
    }

    async Task InstallAsync()
    {
        using var cancellation = new CancellationTokenSource();
        active = cancellation; install.Enabled = remove.Enabled = browse.Enabled = oldBrowse.Enabled = folder.Enabled = oldFolder.Enabled = false; cancel.Text = "取消";
        var timer = Stopwatch.StartNew();
        var updates = new Progress<InstallProgress>(p =>
        {
            if (timer.ElapsedMilliseconds < 100 && p.percent != 100) return;
            timer.Restart(); progress.Value = Math.Clamp(p.percent, 0, 100); status.Text = p.message;
        });
        try
        {
            string parent = folder.Text;
            string? prior = string.IsNullOrWhiteSpace(oldFolder.Text) ? null : Path.GetFullPath(oldFolder.Text.Trim());
            if (prior is not null && !Directory.Exists(prior)) throw new DirectoryNotFoundException("旧版安装目录不存在。");
            string partsDirectory = AppContext.BaseDirectory;
            if (manifest.schema_version == "portable_split_release_v2")
            {
                string cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "Any2Xsampa", "InstallerCache", manifest.release_tag!);
                if (Directory.Exists(Path.Combine(Path.GetFullPath(parent), manifest.root_directory)))
                    throw new IOException("此版本已经安装，请选择其他位置；原版本保持不变。");
                partsDirectory = await Task.Run(() => PartDownloader.EnsureAsync(manifest, AppContext.BaseDirectory,
                    cache, updates, cancellation.Token));
            }
            installed = await Task.Run(() => Installer.Install(manifest, partsDirectory, parent, updates, cancellation.Token, prior));
            progress.Value = 100; status.Text = "安装完成：" + installed + "\n点击“打开应用”即可自动下载模型。";
            open.Visible = true; install.Visible = false;
        }
        catch (OperationCanceledException) { status.Text = "已取消，原有文件不受影响。"; progress.Value = 0; }
        catch (Exception ex) { status.Text = "安装失败：" + ex.Message; }
        finally { active = null; install.Enabled = remove.Enabled = browse.Enabled = oldBrowse.Enabled = folder.Enabled = oldFolder.Enabled = true; cancel.Text = "关闭"; }
    }
}
