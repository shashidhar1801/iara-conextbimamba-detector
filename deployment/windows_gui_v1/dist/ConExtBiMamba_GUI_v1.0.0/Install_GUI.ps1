param(
    [ValidateSet('Auto', 'CPU', 'CUDA130')]
    [string]$Runtime = 'Auto'
)

$ErrorActionPreference = 'Stop'
$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$Logs = Join-Path $Root 'logs'
$Venv = Join-Path $Root '.gui-venv'
$Log = Join-Path $Logs 'install.log'
$State = Join-Path $Root 'INSTALLATION.json'

New-Item -ItemType Directory -Force -Path $Logs | Out-Null
Start-Transcript -LiteralPath $Log -Append | Out-Null

try {
    Write-Output "ConExtBiMamba GUI installation started: $(Get-Date -Format o)"
    Write-Output "Package root: $Root"
    Write-Output "Requested runtime: $Runtime"

    $PythonArgs = @()
    $PythonCommand = $null
    $PyLauncher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        $PythonCommand = $PyLauncher.Source
        $PythonArgs = @('-3.12')
    } else {
        $Python = Get-Command 'python.exe' -ErrorAction SilentlyContinue
        if ($Python) {
            $PythonCommand = $Python.Source
        }
    }
    if (-not $PythonCommand) {
        throw 'Python 3.12 (64 bit) was not found. Install it from python.org and enable the Windows py launcher.'
    }

    $Version = & $PythonCommand @PythonArgs -c "import struct,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{struct.calcsize(\"P\")*8}')"
    if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '3.12|64') {
        throw "The deployment requires 64-bit Python 3.12. Detected: $Version"
    }

    if (-not (Test-Path -LiteralPath (Join-Path $Venv 'Scripts\python.exe'))) {
        & $PythonCommand @PythonArgs -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the private Python environment.' }
    }

    $VenvPython = Join-Path $Venv 'Scripts\python.exe'
    & $VenvPython -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Failed to update pip/setuptools/wheel.' }

    $SelectedRuntime = $Runtime
    if ($SelectedRuntime -eq 'Auto') {
        $SelectedRuntime = if (Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue) { 'CUDA130' } else { 'CPU' }
    }
    $TorchIndex = if ($SelectedRuntime -eq 'CUDA130') {
        'https://download.pytorch.org/whl/cu130'
    } else {
        'https://download.pytorch.org/whl/cpu'
    }

    Write-Output "Selected runtime: $SelectedRuntime"
    Write-Output "PyTorch wheel index: $TorchIndex"
    & $VenvPython -m pip install --disable-pip-version-check torch==2.12.1 torchaudio==2.11.0 --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the $SelectedRuntime PyTorch runtime." }

    & $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $Root 'requirements-app.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install the GUI application dependencies.' }

    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'The installed Python environment failed pip check.' }

    & $VenvPython (Join-Path $Root 'scripts\verify_deployment.py') --quick
    if ($LASTEXITCODE -ne 0) { throw 'The installed GUI failed the deployment verification.' }

    $TorchInfo = & $VenvPython -c "import json,torch,torchaudio; print(json.dumps({'torch':torch.__version__,'torchaudio':torchaudio.__version__,'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda}))"
    $Payload = [ordered]@{
        status = 'PASS'
        installed_utc = (Get-Date).ToUniversalTime().ToString('o')
        runtime = $SelectedRuntime
        python = '3.12|64'
        torch = ($TorchInfo | ConvertFrom-Json)
        package_root = $Root
    }
    $Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $State -Encoding UTF8
    Write-Output 'Installation and verification completed successfully.'
    exit 0
} catch {
    Write-Error $_
    $Payload = [ordered]@{
        status = 'FAIL'
        failed_utc = (Get-Date).ToUniversalTime().ToString('o')
        runtime_requested = $Runtime
        error = $_.Exception.Message
        package_root = $Root
    }
    $Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $State -Encoding UTF8
    exit 1
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}

