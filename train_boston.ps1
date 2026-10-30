param(
    [int]$Seed = 42,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$ExperimentPath = Join-Path $ProjectRoot "boston_mlp_experiment\experiment.py"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python from the project virtual environment was not found: $PythonPath"
}

if (-not (Test-Path -LiteralPath $ExperimentPath)) {
    throw "Boston experiment was not found: $ExperimentPath"
}

$ExperimentArguments = @($ExperimentPath, "--seed", $Seed)
if ($Quick) {
    $ExperimentArguments += @(
        "--surrogate-runs", "12",
        "--grid-candidates", "13",
        "--selection-epochs", "50",
        "--lr-finder-steps", "50",
        "--compare-epochs", "100"
    )
    Write-Host "Starting QUICK Boston MLP experiment..."
    Write-Warning "Quick mode overwrites the full models and results. Run full mode afterwards."
}
else {
    Write-Host "Starting FULL Boston MLP experiment..."
}

& $PythonPath @ExperimentArguments
if ($LASTEXITCODE -ne 0) {
    throw "Boston MLP experiment failed with exit code $LASTEXITCODE"
}

Write-Host "Boston MLP experiment completed successfully."
Write-Host "Metrics: $(Join-Path $ProjectRoot 'boston_mlp_experiment\results\metrics.json')"
