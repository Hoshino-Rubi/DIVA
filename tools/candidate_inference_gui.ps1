param([Parameter(Mandatory=$true)][string]$Bundle, [string]$ProjectRoot=(Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
$Bundle=(Resolve-Path -LiteralPath $Bundle).Path
$python=Join-Path $ProjectRoot '.venv/Scripts/python.exe'
function Quote-ProcessArg([string]$Value) {
    '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
$form=New-Object System.Windows.Forms.Form
$form.Text='any2xsampa · 候选音素识别'
$form.Size=New-Object System.Drawing.Size(860,570)
$form.StartPosition='CenterScreen'
$form.Font=New-Object System.Drawing.Font('Microsoft YaHei UI',11)
$form.AutoScaleMode='Dpi'
$title=New-Object System.Windows.Forms.Label
$title.Text='选择原唱 WAV，指定一段音频，运行并导出音素。'
$title.Location=New-Object System.Drawing.Point(24,24);$title.Size=New-Object System.Drawing.Size(780,35)
$form.Controls.Add($title)
$audio=New-Object System.Windows.Forms.TextBox
$audio.Location=New-Object System.Drawing.Point(24,75);$audio.Size=New-Object System.Drawing.Size(665,32)
$form.Controls.Add($audio)
$browse=New-Object System.Windows.Forms.Button
$browse.Text='选择 WAV';$browse.Location=New-Object System.Drawing.Point(704,72);$browse.Size=New-Object System.Drawing.Size(110,38)
$form.Controls.Add($browse)
$browse.Add_Click({
    $dialog=New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Filter='PCM WAV 音频 (*.wav)|*.wav';$dialog.Title='选择音频文件'
    if($dialog.ShowDialog() -eq 'OK'){$audio.Text=$dialog.FileName}
    $dialog.Dispose()
})
$startLabel=New-Object System.Windows.Forms.Label
$startLabel.Text='开始位置（秒）';$startLabel.Location=New-Object System.Drawing.Point(24,135);$startLabel.AutoSize=$true;$form.Controls.Add($startLabel)
$start=New-Object System.Windows.Forms.NumericUpDown
$start.Location=New-Object System.Drawing.Point(170,130);$start.Size=New-Object System.Drawing.Size(140,34);$start.Maximum=86400;$start.DecimalPlaces=3;$start.Increment=[decimal]0.1
$form.Controls.Add($start)
$durationLabel=New-Object System.Windows.Forms.Label
$durationLabel.Text='片段长度（秒）';$durationLabel.Location=New-Object System.Drawing.Point(350,135);$durationLabel.AutoSize=$true;$form.Controls.Add($durationLabel)
$duration=New-Object System.Windows.Forms.NumericUpDown
$duration.Location=New-Object System.Drawing.Point(510,130);$duration.Size=New-Object System.Drawing.Size(140,34);$duration.Minimum=[decimal]0.1;$duration.Maximum=20;$duration.Value=8;$duration.DecimalPlaces=3;$duration.Increment=[decimal]0.1
$form.Controls.Add($duration)
$note=New-Object System.Windows.Forms.Label
$note.Text='使用已冻结候选。长 WAV 只读取指定片段；原文件保持不变。结果含 IPA、X-SAMPA、JSON 和试听页面。'
$note.Location=New-Object System.Drawing.Point(24,185);$note.Size=New-Object System.Drawing.Size(780,60);$form.Controls.Add($note)
$runButton=New-Object System.Windows.Forms.Button
$runButton.Text='开始识别';$runButton.Location=New-Object System.Drawing.Point(24,260);$runButton.Size=New-Object System.Drawing.Size(170,48);$form.Controls.Add($runButton)
$resultButton=New-Object System.Windows.Forms.Button
$resultButton.Text='查看结果 / 试听';$resultButton.Location=New-Object System.Drawing.Point(210,260);$resultButton.Size=New-Object System.Drawing.Size(210,48);$resultButton.Enabled=$false;$form.Controls.Add($resultButton)
$status=New-Object System.Windows.Forms.TextBox
$status.Multiline=$true;$status.ReadOnly=$true;$status.ScrollBars='Vertical';$status.Location=New-Object System.Drawing.Point(24,330);$status.Size=New-Object System.Drawing.Size(790,170)
$status.Text='就绪。首次运行需加载模型，请等待状态变成完成。';$form.Controls.Add($status)
$state=@{Process=$null;Output=$null;OutLog=$null;ErrLog=$null}
$timer=New-Object System.Windows.Forms.Timer;$timer.Interval=400
$runButton.Add_Click({
    try {
        if(-not(Test-Path -LiteralPath $audio.Text -PathType Leaf)){throw '请先选择现有的 WAV 文件。'}
        $stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
        $state.Output=Join-Path $ProjectRoot ('.state/runs/candidate-use-'+$stamp)
        $state.OutLog=Join-Path $ProjectRoot ('.state/runs/candidate-launch-'+$stamp+'.stdout.log')
        $state.ErrLog=Join-Path $ProjectRoot ('.state/runs/candidate-launch-'+$stamp+'.stderr.log')
        $culture=[Globalization.CultureInfo]::InvariantCulture
        $argv=@('-X','utf8','-m','any2xsampa.cli','infer-candidate','--audio',$audio.Text,'--start',$start.Value.ToString($culture),'--duration',$duration.Value.ToString($culture),'--bundle',$Bundle,'--output-dir',$state.Output)
        $arguments=($argv | ForEach-Object {Quote-ProcessArg $_}) -join ' '
        $state.Process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $state.OutLog -RedirectStandardError $state.ErrLog -PassThru
        $null=$state.Process.Handle # Keep the Windows process handle for its eventual exit code.
        $runButton.Enabled=$false;$browse.Enabled=$false;$resultButton.Enabled=$false;$status.Text='正在识别。完成后可点击“查看结果 / 试听”。';$timer.Start()
    } catch {$status.Text=$_.Exception.Message}
})
$timer.Add_Tick({
    if($null -ne $state.Process -and $state.Process.HasExited){
        $timer.Stop();$state.Process.WaitForExit();$runButton.Enabled=$true;$browse.Enabled=$true
        $page=Join-Path $state.Output 'index.html'
        if($state.Process.ExitCode -eq 0 -and (Test-Path -LiteralPath $page)){
            $status.Text='完成。点击“查看结果 / 试听”。'+"`r`n保存位置："+$state.Output;$resultButton.Enabled=$true
        } else {
            $details=if(Test-Path -LiteralPath $state.ErrLog){Get-Content -LiteralPath $state.ErrLog -Raw -Encoding UTF8}else{'没有错误日志。'}
            $status.Text="运行未完成：`r`n"+$details
        }
        $state.Process.Dispose();$state.Process=$null
    }
})
$resultButton.Add_Click({if($state.Output){Start-Process -FilePath (Join-Path $state.Output 'index.html')}})
$form.Add_FormClosing({param($sender,$eventArgs)
    if($null -ne $state.Process -and -not $state.Process.HasExited){$eventArgs.Cancel=$true;$status.Text='识别任务仍在运行，请完成后关闭窗口。'}
})
[void]$form.ShowDialog()
$timer.Dispose();$form.Dispose()


