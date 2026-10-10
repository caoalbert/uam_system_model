# Keep Windows awake while runs go in other windows (does not touch those runs).
# Usage: powershell -ExecutionPolicy Bypass -File C:\work_UAM\uam_system_model\scripts\jat2026\keep_awake.ps1 [-Hours 16]
# Leave this window open; close it (or wait) to allow sleep again.
param([double]$Hours = 16)
# Plugged in: never sleep, never hibernate, never turn the screen off, lid close does nothing.
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
Add-Type -Namespace JATAwake -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ms = (powercfg /a) -join " "
if ($ms -match "S0 Low Power Idle") { Write-Host "Modern Standby laptop: the screen is kept on on purpose (you can dim it)." -ForegroundColor Yellow }
try {
    $bat = Get-CimInstance Win32_Battery -ErrorAction Stop
    if ($bat -and $bat.BatteryStatus -eq 1) { Write-Host "WARNING: on battery. Plug in the charger." -ForegroundColor Yellow }
} catch {}
$until = (Get-Date).AddHours($Hours)
Write-Host "Keeping system AND display awake until $($until.ToString('ddd HH:mm')). Leave this window open." -ForegroundColor Green
# Re-assert every minute (0x80000003 = CONTINUOUS | SYSTEM_REQUIRED | DISPLAY_REQUIRED).
while ((Get-Date) -lt $until) {
    [void][JATAwake.Power]::SetThreadExecutionState([uint32]"0x80000003")
    Start-Sleep -Seconds 60
}
[void][JATAwake.Power]::SetThreadExecutionState([uint32]"0x80000000")
Write-Host "Done - sleep allowed again." -ForegroundColor Green
