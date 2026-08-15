param(
    [string[]]$Models = @("naive", "moving_average", "arima", "lstm", "minimalist_transformer"),
    [int[]]$Seeds = @(20260723, 20260724, 20260725),
    [switch]$SkipExternal
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LocalPython = Join-Path $RepoRoot ".venv-baselines\Scripts\python.exe"
$TimeGnnPython = "D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe"

function Test-BaselineRuntime([string]$PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath)) { return $false }
    & $PythonPath -c "import numpy,pandas,sklearn,torch" 2>$null
    return $LASTEXITCODE -eq 0
}

if (Test-BaselineRuntime $LocalPython) {
    $Python = $LocalPython
} elseif (Test-BaselineRuntime $TimeGnnPython) {
    $Python = $TimeGnnPython
} else {
    throw "No usable PyTorch runtime found. Follow experiments/README.md to install .venv-baselines."
}

Write-Host "Using Python runtime: $Python"
& $Python (Join-Path $PSScriptRoot "run_all_baselines.py") --models @Models --seeds @Seeds
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $SkipExternal) {
    & $Python (Join-Path $PSScriptRoot "run_external_baselines.py") --seeds @Seeds
}
exit $LASTEXITCODE
