param(
    [string]$TargetRoot = 'D:\ProjectArchive\importance',
    [string]$Version = 'v0.3.0-stage-c',
    [Parameter(Mandatory = $true)][string]$Commit
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$shortCommit = $Commit.Substring(0, [Math]::Min(7, $Commit.Length))
$archiveRoot = Join-Path $TargetRoot "AI_Finance_Prototype_${Version}_${shortCommit}"
if (Test-Path -LiteralPath $archiveRoot) {
    throw "Archive already exists: $archiveRoot"
}

function Copy-ArchiveFile {
    param([string]$SourceRelative, [string]$DestinationRelative)
    $source = Join-Path $repoRoot $SourceRelative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required archive source is missing: $source"
    }
    $destination = Join-Path $archiveRoot $DestinationRelative
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

$publicFiles = @(
    'README.md', 'CHANGELOG.md',
    'reports/STAGE_C_FINAL_REPORT.md',
    'reports/STAGE_C_C4_INDEPENDENT_SCREENING_REPORT.md',
    'reports/STAGE_C_C4_SCREENING_SUMMARY.json',
    'plans/STAGE_C_REVISED_COMPLETION_PLAN_AFTER_C4.md',
    'plans/STAGE_D_IMPLEMENTATION_PLAN.md'
)
foreach ($file in $publicFiles) {
    Copy-ArchiveFile $file (Join-Path '00_project_and_version' $file)
}

Get-ChildItem -LiteralPath (Join-Path $repoRoot 'stage_c') -Recurse -File | Where-Object {
    $_.FullName -notmatch '__pycache__|\.pyc$'
} | ForEach-Object {
    $relative = $_.FullName.Substring($repoRoot.Length + 1)
    Copy-ArchiveFile $relative (Join-Path '01_stage_c_source' $relative)
}
Copy-ArchiveFile 'experiments/governance/future_screening_manifest.json' '01_stage_c_source/experiments/governance/future_screening_manifest.json'

$reportPatterns = @('reports/STAGE_C_*', 'plans/STAGE_C_*', 'plans/STAGE_D_*')
foreach ($pattern in $reportPatterns) {
    Get-ChildItem -Path (Join-Path $repoRoot $pattern) -File | ForEach-Object {
        $relative = $_.FullName.Substring($repoRoot.Length + 1)
        Copy-ArchiveFile $relative (Join-Path '02_reports_and_plans' $relative)
    }
}

$seeds = @(20260723, 20260724, 20260725)
foreach ($seed in $seeds) {
    $candidateDir = "outputs/experiments/stage_c_30stocks_recommended_v2/fixed_control_ensemble_v2_seed$seed"
    foreach ($name in @('model_manifest.json', 'metrics.json')) {
        Copy-ArchiveFile "$candidateDir/$name" "03_models_and_configs/$candidateDir/$name"
    }
    foreach ($variant in @('temporal_only_control', 'fixed_temporal_graph_control')) {
        $runDir = "outputs/experiments/stage_c_30stocks_graph_stabilization/${variant}_seed$seed"
        foreach ($name in @('model.pt', 'model_metadata.json', 'resolved_config.json', 'seeds.json')) {
            Copy-ArchiveFile "$runDir/$name" "03_models_and_configs/$runDir/$name"
        }
    }
    foreach ($variant in @('frets_return_l4', 'minimalist_price_only_l8')) {
        $runDir = "outputs/experiments/stage_b_30stocks_bounded_ablations/${variant}_seed$seed"
        foreach ($name in @('model.pt', 'model_metadata.json', 'resolved_config.json', 'seeds.json')) {
            Copy-ArchiveFile "$runDir/$name" "03_models_and_configs/$runDir/$name"
        }
    }
    $graphDir = "outputs/experiments/stage_c_graph_frequency_v1/graph_frequency_v1_seed$seed"
    foreach ($name in @('model.pt', 'model_metadata.json', 'resolved_config.json', 'seeds.json')) {
        Copy-ArchiveFile "$graphDir/$name" "03_models_and_configs/$graphDir/$name"
    }
    $naiveDir = "outputs/experiments/stage_b_30stocks_baselines/naive_seed$seed"
    foreach ($name in @('model.json', 'resolved_config.json', 'seeds.json')) {
        Copy-ArchiveFile "$naiveDir/$name" "03_models_and_configs/$naiveDir/$name"
    }
}

$screeningRoot = Join-Path $repoRoot 'outputs/screening/stage_c_recommended_v2_c4_20230609_20240607'
Get-ChildItem -LiteralPath $screeningRoot -File | ForEach-Object {
    $relative = $_.FullName.Substring($repoRoot.Length + 1)
    Copy-ArchiveFile $relative (Join-Path '04_c4_screening_evidence' $relative)
}

$archiveNote = @"
# Stage C local importance archive

- Version: $Version
- Git commit: $Commit
- Purpose: source, required checkpoints, freeze records, and C-4 evidence
- Stage conclusion: engineering complete; independent performance acceptance failed
- Raw source data: excluded
- C-4 data and predictions must not be used for post-screening model tuning
"@
$notePath = Join-Path $archiveRoot '00_project_and_version/ARCHIVE_README.md'
New-Item -ItemType Directory -Path (Split-Path -Parent $notePath) -Force | Out-Null
Set-Content -LiteralPath $notePath -Value $archiveNote -Encoding utf8

$manifestPath = Join-Path $archiveRoot '00_project_and_version/SHA256_MANIFEST.csv'
$manifestRows = Get-ChildItem -LiteralPath $archiveRoot -Recurse -File | Where-Object {
    $_.FullName -ne $manifestPath
} | Sort-Object FullName | ForEach-Object {
    [pscustomobject]@{
        relative_path = $_.FullName.Substring($archiveRoot.Length + 1).Replace('\', '/')
        size_bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifestRows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8

Write-Output $archiveRoot
Write-Output "files=$((Get-ChildItem -LiteralPath $archiveRoot -Recurse -File).Count)"
