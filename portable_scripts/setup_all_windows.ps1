param(
  [string]$Python = "python",
  [string]$PipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Zips = Join-Path $Root "source_zips"
$Runtime = Join-Path $Root "runtime"
$Sources = Join-Path $Runtime "sources"
$Venvs = Join-Path $Runtime "venvs"

function Info($msg) { Write-Host "[AI-Finance] $msg" -ForegroundColor Cyan }
function Run($cmd) { Info $cmd; Invoke-Expression $cmd }
function PyExe($venv) { Join-Path $venv "Scripts\python.exe" }
function Ensure-Dir($p) { New-Item -ItemType Directory -Force -Path $p | Out-Null }

Ensure-Dir $Runtime
Ensure-Dir $Sources
Ensure-Dir $Venvs

Info "Checking Python..."
& $Python --version

Info "Extracting source zips..."
Expand-Archive -LiteralPath (Join-Path $Zips "FreTS-main.zip") -DestinationPath $Sources -Force
Expand-Archive -LiteralPath (Join-Path $Zips "Time-GNN-main.zip") -DestinationPath $Sources -Force
Expand-Archive -LiteralPath (Join-Path $Zips "sep-main.zip") -DestinationPath $Sources -Force

$FreTS = Join-Path $Sources "FreTS-main"
$TimeGNN = Join-Path $Sources "Time-GNN-main"
$SEP = Join-Path $Sources "sep-main"

Info "Creating virtual environments..."
& $Python -m venv (Join-Path $Venvs "frets")
& $Python -m venv (Join-Path $Venvs "timegnn")
& $Python -m venv (Join-Path $Venvs "sep")

$FrePy = PyExe (Join-Path $Venvs "frets")
$TimePy = PyExe (Join-Path $Venvs "timegnn")
$SepPy = PyExe (Join-Path $Venvs "sep")

Info "Installing FreTS dependencies..."
& $FrePy -m pip install --upgrade pip
& $FrePy -m pip install -i $PipIndex numpy pandas scikit-learn matplotlib torch
$FreTools = Join-Path $FreTS "utils\tools.py"
(Get-Content -LiteralPath $FreTools) -replace 'np\.Inf','np.inf' | Set-Content -LiteralPath $FreTools -Encoding utf8

Info "Installing Time-GNN dependencies..."
& $TimePy -m pip install --upgrade pip
& $TimePy -m pip install -i $PipIndex numpy pandas matplotlib scikit-learn PyYAML torch torch_geometric
$TimeConfig = @"
dataset: weather
features: multi
seq_len: 96
horizon: 1
cut: 0.05
runs: 1
n_epochs: 1
val_interval: 1
output_dir: outputs/
"@
[System.IO.File]::WriteAllText((Join-Path $TimeGNN "Experiment_config.yaml"), $TimeConfig, [System.Text.UTF8Encoding]::new($false))
$TimeTrain = Join-Path $TimeGNN "TimeGNN_train.py"
(Get-Content -LiteralPath $TimeTrain) -replace 'yaml\.load_all\(f\)','yaml.safe_load_all(f)' | Set-Content -LiteralPath $TimeTrain -Encoding utf8

Info "Installing SEP dependencies. This is an environment/import deployment, not full model training."
& $SepPy -m pip install --upgrade pip
& $SepPy -m pip install -i $PipIndex numpy openai tenacity tiktoken pandas scikit-learn torch datasets sentencepiece peft evaluate trl protobuf accelerate transformers pyarrow_hotfix
& $SepPy -m pip install -i $PipIndex --force-reinstall --no-deps trl==0.7.1 transformers==4.34.1 peft==0.6.2 datasets==2.14.7 evaluate==0.4.1 huggingface-hub==0.17.3
Get-ChildItem -Recurse -LiteralPath $SEP -Filter "*.py" | ForEach-Object {
  (Get-Content -LiteralPath $_.FullName) -replace 'prepare_model_for_int8_training','prepare_model_for_kbit_training' | Set-Content -LiteralPath $_.FullName -Encoding utf8
}
$DepTable = Join-Path $Venvs "sep\Lib\site-packages\transformers\dependency_versions_table.py"
if (Test-Path $DepTable) {
  (Get-Content -LiteralPath $DepTable) -replace 'tokenizers>=0\.14,<0\.15','tokenizers>=0.14,<0.23' | Set-Content -LiteralPath $DepTable -Encoding utf8
}

Info "Writing local run shortcuts..."
$RunFreTS = @"
`$ErrorActionPreference='Stop'
Set-Location `"$FreTS`"
& `"$FrePy`" run_longExp.py --data covid --root_path ./dataset/ --data_path covid.csv --features M --enc_in 55 --dec_in 55 --c_out 55 --seq_len 36 --label_len 18 --pred_len 24 --train_epochs 3 --batch_size 4 --use_gpu False
"@
Set-Content -LiteralPath (Join-Path $Root "run_frets.ps1") -Value $RunFreTS -Encoding utf8

$RunTime = @"
`$ErrorActionPreference='Stop'
Set-Location `"$TimeGNN`"
& `"$TimePy`" TimeGNN_train.py
"@
Set-Content -LiteralPath (Join-Path $Root "run_timegnn.ps1") -Value $RunTime -Encoding utf8

$RunSep = @"
`$ErrorActionPreference='Stop'
Set-Location `"$SEP`"
& `"$SepPy`" -c "from exp.exp_model import Exp_Model; print('SEP Exp_Model import ok')"
"@
Set-Content -LiteralPath (Join-Path $Root "run_sep_check.ps1") -Value $RunSep -Encoding utf8

Info "Deployment complete."
Info "Run FreTS:     powershell -ExecutionPolicy Bypass -File .\run_frets.ps1"
Info "Run Time-GNN:  powershell -ExecutionPolicy Bypass -File .\run_timegnn.ps1"
Info "Run SEP check: powershell -ExecutionPolicy Bypass -File .\run_sep_check.ps1"
