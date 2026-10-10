# JAT2026 overnight toy run of one feedback method (12 h, LAX/DTLA/LGB, 10 aircraft, corrected bridge).
# Usage (one terminal per method):
#   powershell -ExecutionPolicy Bypass -File C:\work_UAM\uam_system_model\scripts\jat2026\overnight_toy.ps1 -Method market
#   powershell -ExecutionPolicy Bypass -File C:\work_UAM\uam_system_model\scripts\jat2026\overnight_toy.ps1 -Method hourly [-StepUp 3]
# market = original F2-7 plan (per-market caps with +2 rule, cost per seat-mile feedback)
# hourly = Algorithm 1 revised (network-wide hourly flight caps that can rise by -StepUp, cost per seat-mile feedback)
# Output: runs\jat_toy_7-19_LAX-DTLA-LGB_thin\hourly_cost_K10[_stepN]\ (or market_K10_v2)  (convergence.csv, DONE.json, iterN\...)
# Log:    runs\jat_toy_7-19_hourly_cost[_stepN].log      Safe to re-run: finished rounds are reused.
param([ValidateSet("market", "hourly")][string]$Method = "hourly", [int]$MaxIter = 20, [double]$StepUp = 2)

$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$py = "C:\work_python\wisk\wisk_ram_study\.venv\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force runs | Out-Null
$tag = $Method; if ($Method -eq "hourly") { $tag = "hourly_cost" }
if ($StepUp -ne 2) { $tag = "$($tag)_step$StepUp" }
$log = "runs\jat_toy_7-19_$tag.log"
$sub = "hourly_cost_K10"; if ($StepUp -ne 2) { $sub = "hourly_cost_K10_step$StepUp" }
if ($Method -eq "market") { $sub = "market_K10_v2" }
$out = "runs\jat_toy_7-19_LAX-DTLA-LGB_thin\$sub"

# Keep Windows (system AND display) awake while this script runs; released when it ends.
Add-Type -Namespace JATRun -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][JATRun.Power]::SetThreadExecutionState([uint32]"0x80000003")
Write-Host "[$Method] sleep blocked while this window runs. Log: $log" -ForegroundColor Green

$start = Get-Date
& $py scripts\jat2026\run_jat.py converge --method $Method --toy --toy-hours 7-19 --toy-vertiports LAX,DTLA,LGB --fleet 10 --max-iter $MaxIter --step-up $StepUp --n-processes 4 --vrp-threads 2 --bridge thin 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
& $py scripts\jat2026\toy_evidence.py $out --fleet 10 2>&1 | Tee-Object -FilePath $log -Append
$hrs = [math]::Round(((Get-Date) - $start).TotalHours, 1)
if ($code -ne 0) {
    "FAILED exit $code at $(Get-Date -Format s) after $hrs h" | Out-File "runs\jat_toy_7-19_$($tag)_FAILED.txt"
    Write-Host "[$Method] FAILED (exit $code) after $hrs h. See $log. Re-run this same command to resume." -ForegroundColor Red
} else {
    "done $(Get-Date -Format s) after $hrs h" | Out-File "runs\jat_toy_7-19_$($tag)_DONE.txt"
    Write-Host "[$Method] done in $hrs h" -ForegroundColor Green
}
[void][JATRun.Power]::SetThreadExecutionState([uint32]"0x80000000")
