param(
    [ValidateSet("demo", "check", "regressions", "configure", "hands", "preview")]
    [string]$Mode = "demo",
    [string]$Python = "",
    [int]$Port = 8082,
    [double]$Duration = 0,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path $PSScriptRoot -Parent
$LocalConfig = Join-Path $PSScriptRoot ".local\python_path.txt"
$Probe = "import importlib.util,sys; required=('torch','mujoco','warp','viser','matplotlib','PIL','tensordict','rsl_rl','tyro'); sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in required) else 1)"

function Test-DemoPython([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    try {
        & $Candidate -c $Probe 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$Candidates = New-Object System.Collections.Generic.List[string]
if ($Mode -eq "configure" -and -not $Python) {
    $Python = (Read-Host "Full path to the Python executable in your existing mjlab environment").Trim().Trim('"')
}
if ($Python) {
    if (-not (Test-DemoPython $Python)) { throw "This Python does not have the required mjlab dependencies: $Python" }
    $Candidates.Add($Python)
} else {
    if ($env:MJLAB_PYTHON) { $Candidates.Add($env:MJLAB_PYTHON) }
    if (Test-Path -LiteralPath $LocalConfig) { $Candidates.Add((Get-Content -LiteralPath $LocalConfig -Raw).Trim()) }
    if ($env:VIRTUAL_ENV) { $Candidates.Add((Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")) }
    if ($env:CONDA_PREFIX) { $Candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe")) }
    $Candidates.Add((Join-Path $PSScriptRoot ".venv\Scripts\python.exe"))
    $Cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($Cmd) { $Candidates.Add($Cmd.Source) }
    $Py = Get-Command py -ErrorAction SilentlyContinue
    if ($Py) {
        foreach ($Version in @("-3.11", "-3.12", "-3.13", "-3.10")) {
            try {
                $Candidate = (& $Py.Source $Version -c "import sys; print(sys.executable)" 2>$null)
                if ($LASTEXITCODE -eq 0 -and $Candidate) { $Candidates.Add(($Candidate | Select-Object -Last 1).Trim()) }
            } catch {}
        }
    }
}
$SelectedPython = $null
foreach ($Candidate in $Candidates) {
    if (Test-DemoPython $Candidate) { $SelectedPython = $Candidate; break }
}
if (-not $SelectedPython) {
    Write-Host "No existing mjlab Python environment was found. No packages were downloaded."
    Write-Host "Run code/configure_python.bat and enter the full path to that environment's python.exe."
    exit 2
}
New-Item -ItemType Directory -Path (Split-Path $LocalConfig -Parent) -Force | Out-Null
[IO.File]::WriteAllText($LocalConfig, $SelectedPython, (New-Object Text.UTF8Encoding($false)))
Write-Host "Python: $SelectedPython"
if ($Mode -eq "configure") { Write-Host "Environment saved. Double-click start_demo.bat."; exit 0 }
$DemoArguments = @()
switch ($Mode) {
    "check" { $DemoArguments += "--check" }
    "regressions" { $DemoArguments += "--regressions" }
    "hands" { $DemoArguments += @("--script", "flexitac_demo.py") }
    "preview" { $DemoArguments += @("--script", "preview_robot.py", "--manual") }
    default {
        $DemoArguments += @("--port", "$Port")
        if ($Duration -gt 0) { $DemoArguments += @("--duration", "$Duration") }
        if ($NoBrowser) { $DemoArguments += "--no-browser" }
    }
}
Push-Location $BundleRoot
try {
    & $SelectedPython (Join-Path $PSScriptRoot "run_demo.py") @DemoArguments
    exit $LASTEXITCODE
} finally { Pop-Location }
