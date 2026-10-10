# JAT2026 overnight run: corrected open-loop run (iteration 0) on the pricing day,
# then evaluation of that plan on demand days (seeds) 1-5.
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File C:\work_UAM\uam_system_model\scripts\jat2026\overnight_iter0.ps1
# Logs: runs\jat\iter0_K30.log, runs\jat\eval_iter0_K30.log  (Claude can read these)
# Safe to re-run: finished steps are skipped.

param([string]$Python = "")

$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
# Python: -Python <path>, else the JAT_PYTHON environment variable, else <repo>\.venv, else python on PATH
$py = $null
if ($Python) { $py = $Python }
elseif ($env:JAT_PYTHON) { $py = $env:JAT_PYTHON }
elseif (Test-Path (Join-Path $repo ".venv\Scripts\python.exe")) { $py = Join-Path $repo ".venv\Scripts\python.exe" }
else { $py = "python" }
& $py -c "import gurobipy, pandas, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python '$py' cannot import gurobipy/pandas/numpy. Pass -Python <path to python.exe> or set JAT_PYTHON." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $py" -ForegroundColor Green
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force runs\jat | Out-Null

# Keep Windows awake for the whole run: ask Windows directly not to sleep while this
# script is alive (same mechanism video players use). Released automatically when the
# script ends. Note: closing the lid can still sleep the laptop unless the lid action is
# set to "do nothing" (see the warning below).
Add-Type -Namespace JAT -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000003"  # system + display (Modern Standby)
[void][JAT.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Write-Host "Sleep blocked while this script runs." -ForegroundColor Green
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
try {
    $bat = Get-CimInstance Win32_Battery -ErrorAction Stop
    if ($bat -and $bat.BatteryStatus -eq 1) { Write-Host "WARNING: running on battery. Plug in the charger." -ForegroundColor Yellow }
} catch {}
Write-Host "Keep the lid open, or set the lid action to do nothing when plugged in:" -ForegroundColor Yellow
Write-Host "  powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0; powercfg /setactive SCHEME_CURRENT" -ForegroundColor Yellow

$start = Get-Date
Write-Host "[$(Get-Date -Format HH:mm)] 1/2 corrected open-loop run (stage 1 + stage 2, seed 9)" -ForegroundColor Cyan
& $py scripts\jat2026\run_jat.py converge --fleet 30 --max-iter 1 2>&1 | Tee-Object runs\jat\iter0_K30.log
if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 1 failed (exit $LASTEXITCODE). See runs\jat\iter0_K30.log" -ForegroundColor Red
    exit 1
}

Write-Host "[$(Get-Date -Format HH:mm)] 2/2 evaluate the plan on seeds 1-5" -ForegroundColor Cyan
& $py scripts\jat2026\run_jat.py evaluate --fleet 30 --plan iter0 --seeds 1-5 2>&1 | Tee-Object runs\jat\eval_iter0_K30.log
if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 2 failed (exit $LASTEXITCODE). See runs\jat\eval_iter0_K30.log" -ForegroundColor Red
    exit 1
}

$hrs = [math]::Round(((Get-Date) - $start).TotalHours, 1)
"done $(Get-Date -Format s) after $hrs h" | Out-File runs\jat\overnight_iter0_DONE.txt
Write-Host "[$(Get-Date -Format HH:mm)] All done in $hrs h" -ForegroundColor Green
[void][JAT.Power]::SetThreadExecutionState($ES_CONTINUOUS)
