param(
  [string]$Python = "python",
  [string]$PipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv-baostock"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
  & $Python -m venv $Venv
}

& $VenvPython -m pip install -i $PipIndex --timeout 120 -r (Join-Path $PSScriptRoot "requirements-stage-a.txt")
Write-Host "Stage A environment ready: $VenvPython" -ForegroundColor Green
