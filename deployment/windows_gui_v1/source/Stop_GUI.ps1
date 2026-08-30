$ErrorActionPreference = 'Stop'
$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$StatePath = Join-Path $Root 'runs\gui_server_state.json'

if (-not (Test-Path -LiteralPath $StatePath)) {
    exit 0
}

$State = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
$PidToStop = [int]$State.pid
$Process = Get-CimInstance Win32_Process -Filter "ProcessId = $PidToStop" -ErrorAction SilentlyContinue
if (-not $Process) {
    Remove-Item -LiteralPath $StatePath -Force
    exit 0
}

$ExpectedLauncher = [System.IO.Path]::GetFullPath((Join-Path $Root 'launcher.py'))
$CommandLine = [string]$Process.CommandLine
if ($Process.Name -notmatch '^pythonw?\.exe$' -or $CommandLine.IndexOf($ExpectedLauncher, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
    throw "Refusing to stop PID $PidToStop because it is not this package's launcher process."
}

Stop-Process -Id $PidToStop -Force
$Deadline = (Get-Date).AddSeconds(10)
do {
    Start-Sleep -Milliseconds 250
    $StillRunning = Get-Process -Id $PidToStop -ErrorAction SilentlyContinue
} while ($StillRunning -and (Get-Date) -lt $Deadline)

if ($StillRunning) {
    throw "The GUI process $PidToStop did not stop within ten seconds."
}
Remove-Item -LiteralPath $StatePath -Force -ErrorAction SilentlyContinue
exit 0

