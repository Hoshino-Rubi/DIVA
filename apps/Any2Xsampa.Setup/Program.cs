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
            if (args.Length == 4 && args[0] == "--install")
            {
                var target = Installer.Install(manifest, args[1], args[2], null, CancellationToken.None);
                File.WriteAllText(args[3], JsonSerializer.Serialize(new { status = "passed", target,
                    archive_sha256 = manifest.archive_sha256, model_forwards = 0, downloads = 0 }));
                return 0;
            }
            if (args.Length != 0) throw new ArgumentException("Usage: --install <parts_directory> <destination_parent> <report.json>");
            ApplicationConfiguration.Initialize();
            Application.Run(new SetupForm(manifest));
            return 0;
        }
        catch (Exception ex)
        {
            if (args.Length == 4 && args[0] == "--install")
                File.WriteAllText(args[3], JsonSerializer.Serialize(new { status = "failed", error = ex.ToString() }));
            else MessageBox.Show(ex.Message, "Any2Xsampa 安装失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}

internal sealed class SetupForm : Form
{
    readonly ReleaseManifest manifest;
    readonly TextBox folder = new() { Dock = DockStyle.Fill };
    readonly Label status = new() { AutoSize = true, MaximumSize = new Size(690, 0) };
    readonly ProgressBar progress = new() { Dock = DockStyle.Fill, Height = 25 };
    readonly Button browse = new() { Text = "选择位置", AutoSize = true };
    readonly Button install = new() { Text = "安装", AutoSize = true };
    readonly Button cancel = new() { Text = "关闭", AutoSize = true };
    readonly Button open = new() { Text = "打开应用", AutoSize = true, Visible = false };
    CancellationTokenSource? active;
    string? installed;

    public SetupForm(ReleaseManifest value)
    {
        manifest = value;
        Text = "Any2Xsampa 安装";
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(730, 345);
        MinimumSize = new Size(650, 380);
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Microsoft YaHei UI", 10);
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(20), RowCount = 7, ColumnCount = 1 };
        layout.Controls.Add(new Label { Text = "Any2Xsampa · 本地音素与 VSQX 工作台", Font = new Font(Font.FontFamily, 15, FontStyle.Bold), AutoSize = true });
        layout.Controls.Add(new Label { Text = "请将安装器与三个 .zip.001 / .002 / .003 分卷放在同一文件夹。\n程序会校验 SHA256 并解压；无需另外安装 Python 或 .NET。", AutoSize = true });
        layout.Controls.Add(new Label { Text = "安装位置（将在下方位置创建 " + manifest.root_directory + "）", AutoSize = true });
        var location = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, AutoSize = true };
        location.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        location.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        folder.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs");
        location.Controls.Add(folder, 0, 0); location.Controls.Add(browse, 1, 0);
        layout.Controls.Add(location);
        layout.Controls.Add(progress);
        status.Text = "安装需约 8 GiB 临时可用空间。模型会在应用首次启动时下载，也可离线导入。";
        layout.Controls.Add(status);
        var actions = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, FlowDirection = FlowDirection.RightToLeft };
        actions.Controls.Add(cancel); actions.Controls.Add(open); actions.Controls.Add(install);
        layout.Controls.Add(actions); Controls.Add(layout);
        browse.Click += (_, _) => { using var dialog = new FolderBrowserDialog { InitialDirectory = folder.Text, Description = "选择应用安装位置" }; if (dialog.ShowDialog(this) == DialogResult.OK) folder.Text = dialog.SelectedPath; };
        install.Click += async (_, _) => await InstallAsync();
        cancel.Click += (_, _) => { if (active != null) active.Cancel(); else Close(); };
        open.Click += (_, _) => { if (installed != null) Process.Start(new ProcessStartInfo(Path.Combine(installed, "Any2Xsampa.Desktop.exe")) { UseShellExecute = true, WorkingDirectory = installed }); };
        FormClosing += (_, e) => { if (active != null) { active.Cancel(); e.Cancel = true; status.Text = "正在取消并清理临时安装文件…"; } };
    }

    async Task InstallAsync()
    {
        using var cancellation = new CancellationTokenSource();
        active = cancellation; install.Enabled = browse.Enabled = folder.Enabled = false; cancel.Text = "取消";
        var timer = Stopwatch.StartNew();
        var updates = new Progress<InstallProgress>(p =>
        {
            if (timer.ElapsedMilliseconds < 100 && p.percent != 100) return;
            timer.Restart(); progress.Value = Math.Clamp(p.percent, 0, 100); status.Text = p.message;
        });
        try
        {
            string parent = folder.Text;
            installed = await Task.Run(() => Installer.Install(manifest, AppContext.BaseDirectory, parent, updates, cancellation.Token));
            progress.Value = 100; status.Text = "安装完成：" + installed + "\n点击“打开应用”即可自动下载模型。";
            open.Visible = true; install.Visible = false;
        }
        catch (OperationCanceledException) { status.Text = "已取消，原有文件不受影响。"; progress.Value = 0; }
        catch (Exception ex) { status.Text = "安装失败：" + ex.Message; }
        finally { active = null; install.Enabled = browse.Enabled = folder.Enabled = true; cancel.Text = "关闭"; }
    }
}
