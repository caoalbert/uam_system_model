# JAT2026 full run: revised Algorithm 1 (hourly network caps that can rise + cost per seat-mile feedback),
# 30 aircraft, 9 vertiports, 24 h, pricing day seed 9, corrected bridge; then the final plan on demand days 1-5.
# Usage:
#   powershell -ExecutionPolicy Bypass -File C:\work_UAM\uam_system_model\scripts\jat2026\overnight_full.ps1 [-MaxIter 8] [-StepUp 2]
# Output: runs\jat_thin\hourly_cost_K30\ (convergence.csv, DONE.json, iterN\, eval_iterN\seedS\, evaluation_iterN.csv)
# Logs:   runs\jat_full_converge.log, runs\jat_full_evaluate.log      Safe to re-run: finished steps are reused.
param([int]$MaxIter = 8, [double]$StepUp = 2, [string]$Seeds = "1-5")

$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$py = "C:\work_python\wisk\wisk_ram_study\.venv\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force runs | Out-Null
$sub = "hourly_cost_K30"; if ($StepUp -ne 2) { $sub = "hourly_cost_K30_step$StepUp" }
$out = "runs\jat_thin\$sub"

# Keep Windows (system AND display) awake while this script runs; released when it ends.
Add-Type -Namespace JATFull -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][JATFull.Power]::SetThreadExecutionState([uint32]"0x80000003")
Write-Host "[full run] sleep blocked while this window runs. Output: $out" -ForegroundColor Green
$start = Get-Date

Write-Host "[$(Get-Date -Format HH:mm)] 1/2 converge (up to $MaxIter rounds, about 1 h each)" -ForegroundColor Cyan
& $py scripts\jat2026\run_jat.py converge --method hourly --fleet 30 --max-iter $MaxIter --step-up $StepUp --n-processes 10 --vrp-threads 2 --bridge thin 2>&1 | Tee-Object -FilePath runs\jat_full_converge.log -Append
if ($LASTEXITCODE -ne 0) {
    "FAILED in converge (exit $LASTEXITCODE) at $(Get-Date -Format s)" | Out-File runs\jat_full_FAILED.txt
    Write-Host "Converge failed. See runs\jat_full_converge.log. Re-run this same command to resume." -ForegroundColor Red
    [void][JATFull.Power]::SetThreadExecutionState([uint32]"0x80000000"); exit 1
}
& $py scripts\jat2026\toy_evidence.py $out --fleet 30 2>&1 | Tee-Object -FilePath runs\jat_full_converge.log -Append

Write-Host "[$(Get-Date -Format HH:mm)] 2/2 evaluate the final plan on seeds $Seeds" -ForegroundColor Cyan
& $py scripts\jat2026\run_jat.py evaluate --method hourly --fleet 30 --plan final --seeds $Seeds --step-up $StepUp --n-processes 10 --vrp-threads 2 --bridge thin 2>&1 | Tee-Object -FilePath runs\jat_full_evaluate.log -Append
if ($LASTEXITCODE -ne 0) {
    "FAILED in evaluate (exit $LASTEXITCODE) at $(Get-Date -Format s)" | Out-File runs\jat_full_FAILED.txt
    Write-Host "Evaluate failed. See runs\jat_full_evaluate.log. Re-run this same command to resume." -ForegroundColor Red
    [void][JATFull.Power]::SetThreadExecutionState([uint32]"0x80000000"); exit 1
}
$hrs = [math]::Round(((Get-Date) - $start).TotalHours, 1)
"done $(Get-Date -Format s) after $hrs h" | Out-File runs\jat_full_DONE.txt
Write-Host "[$(Get-Date -Format HH:mm)] All done in $hrs h" -ForegroundColor Green
[void][JATFull.Power]::SetThreadExecutionState([uint32]"0x80000000")
