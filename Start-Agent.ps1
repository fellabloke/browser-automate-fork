[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$Objective,

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$CdpPort = 9222,

    [Parameter()]
    [string]$Distro
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cdpEndpoint = "http://127.0.0.1:$CdpPort"
$cdpVersionUrl = "$cdpEndpoint/json/version"
if (-not $env:LOCALAPPDATA) {
    throw "Chrome startup failure: LOCALAPPDATA is unavailable, so the dedicated profile path cannot be resolved."
}
$chromeProfile = Join-Path $env:LOCALAPPDATA "AgentFirstBrowse\ChromeProfile"
$logDirectory = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$launcherLogPath = Join-Path $logDirectory ("windows_launcher_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $launcherLogPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "[Agent First Browse] Launcher log: $launcherLogPath" -ForegroundColor DarkGray
}
catch {
    Write-Warning "Could not create launcher transcript at ${launcherLogPath}: $($_.Exception.Message)"
}

try {

function Write-Step {
    param([string]$Message)
    Write-Host "[Agent First Browse] $Message" -ForegroundColor Cyan
}

function Get-CdpVersion {
    try {
        $version = Invoke-RestMethod -Uri $cdpVersionUrl -Method Get -TimeoutSec 2
        if ($version.webSocketDebuggerUrl -and
            ([string]$version.webSocketDebuggerUrl).StartsWith("ws")) {
            return $version
        }
    }
    catch {
        return $null
    }
    return $null
}

function Find-Chrome {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:PROGRAMFILES) {
        $candidates.Add((Join-Path $env:PROGRAMFILES "Google\Chrome\Application\chrome.exe"))
    }
    if (${env:ProgramFiles(x86)}) {
        $candidates.Add((Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"))
    }
    if ($env:LOCALAPPDATA) {
        $candidates.Add((Join-Path $env:LOCALAPPDATA "Google\Chrome\Application\chrome.exe"))
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    foreach ($registryPath in @(
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
    )) {
        try {
            $registryKey = Get-Item -LiteralPath $registryPath -ErrorAction Stop
            $registeredChrome = [string]$registryKey.GetValue("")
            if ($registeredChrome -and (Test-Path -LiteralPath $registeredChrome -PathType Leaf)) {
                return $registeredChrome
            }
        }
        catch {
            # Try the next standard location.
        }
    }

    $command = Get-Command chrome.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

function Get-PortProxyConflict {
    $netshPath = Join-Path $env:SystemRoot "System32\netsh.exe"
    if (-not (Test-Path -LiteralPath $netshPath -PathType Leaf)) {
        return $null
    }

    $rules = & $netshPath interface portproxy show v4tov4 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    foreach ($line in $rules) {
        if ([string]$line -match "^\s*(\S+)\s+$CdpPort\s+(\S+)\s+(\d+)\s*$") {
            return ([string]$line).Trim()
        }
    }
    return $null
}

function Invoke-WslCapture {
    param([string[]]$Arguments)

    # Windows PowerShell 5 wraps native stderr as ErrorRecord objects. Capture
    # it without allowing the script-level Stop preference to hide diagnostics.
    $ErrorActionPreference = "Continue"
    $output = (& wsl.exe @Arguments 2>&1 | Out-String).Trim()
    $exitCode = $LASTEXITCODE
    return [PSCustomObject]@{
        ExitCode = $exitCode
        Output = $output
    }
}

if ([string]::IsNullOrWhiteSpace($Objective)) {
    throw "The automation objective cannot be empty."
}

$cdpVersion = Get-CdpVersion
if ($cdpVersion) {
    Write-Step "Reusing the automation Chrome already listening at $cdpEndpoint."
}
else {
    $portProxyConflict = Get-PortProxyConflict
    if ($portProxyConflict) {
        throw @"
Port conflict: a Windows netsh port-proxy rule already owns port $CdpPort but is not serving Chrome CDP:
  $portProxyConflict

This obsolete rule must be removed from an Administrator PowerShell before Chrome can use the port:
  netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$CdpPort

The launcher did not remove the rule or terminate any process.
"@
    }

    $activeListener = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
        Where-Object { $_.Port -eq $CdpPort } |
        Select-Object -First 1
    if ($activeListener) {
        throw "Port conflict: $($activeListener.Address):$CdpPort is already listening but does not provide valid Chrome CDP. Identify it with 'netstat -ano | findstr :$CdpPort'; the launcher will not terminate it automatically."
    }

    $chromePath = Find-Chrome
    if (-not $chromePath) {
        throw "Chrome startup failure: chrome.exe was not found in the standard install locations or PATH."
    }

    New-Item -ItemType Directory -Path $chromeProfile -Force | Out-Null
    $chromeArgs = @(
        "--remote-debugging-port=$CdpPort",
        "--user-data-dir=`"$chromeProfile`"",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-crash-restore-bubble",
        "--disable-session-crashed-bubble",
        "--new-window",
        "about:blank"
    )

    Write-Step "Starting dedicated Chrome with profile: $chromeProfile"
    try {
        $chromeProcess = Start-Process -FilePath $chromePath -ArgumentList $chromeArgs -PassThru
    }
    catch {
        throw "Chrome startup failure: $($_.Exception.Message)"
    }

    Write-Step "Waiting for $cdpVersionUrl to return a webSocketDebuggerUrl..."
    $deadline = [DateTime]::UtcNow.AddSeconds(25)
    $handoffReported = $false
    do {
        Start-Sleep -Milliseconds 250
        $cdpVersion = Get-CdpVersion
        if ($cdpVersion) {
            break
        }
        if ($chromeProcess.HasExited) {
            if ($chromeProcess.ExitCode -ne 0) {
                throw "Chrome startup failure: dedicated Chrome exited with code $($chromeProcess.ExitCode) before CDP became ready."
            }
            if (-not $handoffReported) {
                Write-Step "Chrome starter exited normally; waiting for the browser process/CDP handoff."
                $handoffReported = $true
            }
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    if (-not $cdpVersion) {
        throw "CDP readiness failure: $cdpVersionUrl did not return a valid webSocketDebuggerUrl within 25 seconds. Port $CdpPort may be occupied or Chrome may have rejected the dedicated profile."
    }
    Write-Step "Windows Chrome CDP is ready."
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL startup failure: wsl.exe is unavailable. Install WSL and create the project's .venv inside the distribution."
}

# Repositories opened through \\wsl.localhost\Distro\... can be mapped without
# invoking a shell. Other Windows paths are translated by wslpath.
$wslProjectPath = $null
if ($PSScriptRoot -match '^\\\\wsl(?:\.localhost)?\\([^\\]+)\\(.*)$') {
    $pathDistro = $Matches[1]
    if ($Distro -and $Distro -ne $pathDistro) {
        throw "WSL path failure: this repository belongs to '$pathDistro', not the requested '$Distro' distribution."
    }
    $Distro = $pathDistro
    $wslProjectPath = "/" + ($Matches[2] -replace '\\', '/')
}

$wslBaseArgs = @()
if ($Distro) {
    $wslBaseArgs += @("--distribution", $Distro)
}

if (-not $wslProjectPath) {
    $convertArgs = $wslBaseArgs + @("--exec", "wslpath", "-a", "-u", $PSScriptRoot)
    $conversion = Invoke-WslCapture -Arguments $convertArgs
    if ($conversion.ExitCode -ne 0 -or -not $conversion.Output) {
        throw "WSL path failure: could not translate the project directory '$PSScriptRoot'. $($conversion.Output)"
    }
    $wslProjectPath = $conversion.Output
}

$venvCheckArgs = $wslBaseArgs + @(
    "--cd", $wslProjectPath,
    "--exec", "test", "-x", ".venv/bin/python"
)
$venvCheck = Invoke-WslCapture -Arguments $venvCheckArgs
if ($venvCheck.ExitCode -ne 0) {
    throw "WSL environment failure: .venv/bin/python is missing or not executable in $wslProjectPath. Create the WSL venv and install the project first. $($venvCheck.Output)"
}

# Fail before Playwright with a precise instruction when WSL can report that it
# is still using NAT networking. Older WSL releases without wslinfo fall through
# to the actual reachability probe below.
$networkModeArgs = $wslBaseArgs + @("--exec", "wslinfo", "--networking-mode")
$networkMode = Invoke-WslCapture -Arguments $networkModeArgs
if ($networkMode.ExitCode -eq 0 -and $networkMode.Output -notmatch "(?i)mirrored") {
    throw @"
WSL networking mode is '$($networkMode.Output.Trim())', not 'mirrored'.
Create %UserProfile%\.wslconfig with:

[wsl2]
networkingMode=mirrored

Then run 'wsl --shutdown' in PowerShell and retry.
"@
}

# Probe from WSL as well as Windows. This is what distinguishes a working Chrome
# from a missing mirrored-networking configuration.
$probeArgs = $wslBaseArgs + @(
    "--cd", $wslProjectPath,
    "--exec", "env", "LOCAL_CDP_ENDPOINT=$cdpEndpoint", "AGENT_NO_FILE_LOG=1",
    ".venv/bin/python", "-m", "agent_first_browse.cli", "probe-cdp"
)
$probe = Invoke-WslCapture -Arguments $probeArgs
if ($probe.ExitCode -ne 0) {
    throw @"
WSL networking failure: Windows can reach $cdpVersionUrl, but WSL cannot.
Enable mirrored networking in %UserProfile%\.wslconfig:

[wsl2]
networkingMode=mirrored

Then run 'wsl --shutdown' in PowerShell and retry. The launcher does not guess a transient gateway IP.
Probe output: $($probe.Output)
"@
}

Write-Step "WSL can reach Windows Chrome over 127.0.0.1. Starting canonical agent CLI..."
$objectiveBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Objective))
$agentArgs = $wslBaseArgs + @(
    "--cd", $wslProjectPath,
    "--exec", "env",
    "LOCAL_CDP_ENDPOINT=$cdpEndpoint",
    "BROWSER_MODE=LOCAL_CDP",
    "BROWSER_HEADLESS=false",
    "BROWSER_OS=Windows",
    "PYTHONUNBUFFERED=1",
    ".venv/bin/python", "-m", "agent_first_browse.cli", "run", "--objective-base64", $objectiveBase64
)

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & wsl.exe @agentArgs
    $agentExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($agentExitCode -ne 0) {
    Write-Host "The WSL Python agent exited with code $agentExitCode. Chrome CDP was ready; inspect the Python log for the Playwright/model failure." -ForegroundColor Red
}
}
finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # The host may already be closing after a terminating error.
        }
    }
}
exit $agentExitCode
