<#
.SYNOPSIS
    NER Logistics Sentinel launcher for Windows PowerShell.
.EXAMPLE
    .\run.ps1
    .\run.ps1 -Live
    .\run.ps1 -NoSimFleet
#>
[CmdletBinding()]
param(
    [switch]$Https,
    [switch]$Live,
    [switch]$NoSimFleet,
    [int]$Port = 8000,
    [int]$HttpsPort = 8443
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Live) { $env:SENTINEL_LIVE_WEATHER = "1" }
if ($NoSimFleet) { $env:SENTINEL_SIM_FLEET = "0" }

Write-Host "──────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "  NER LOGISTICS SENTINEL (PowerShell Launcher)" -ForegroundColor Cyan
Write-Host "──────────────────────────────────────────────────────────────" -ForegroundColor Cyan

$VenvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "  [+] Setting up Python virtual environment..." -ForegroundColor Yellow
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv .venv
        if (-not (Test-Path $VenvPy)) { & py -m venv .venv }
    } else {
        & python -m venv .venv
    }
    
    if (-not (Test-Path $VenvPy)) {
        Write-Error "Could not create Python virtual environment. Please install Python 3.11+."
        exit 1
    }
    
    Write-Host "  [+] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    $VenvPip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"
    & $VenvPip install -r requirements.txt
}

if (-not (Test-Path "data\artifacts\risk_clf.joblib")) {
    Write-Host "  First run - building dataset and training models (~2 minutes)...`n" -ForegroundColor Yellow
    & $VenvPy scripts\pipeline.py
    Write-Host ""
}

# Resolve local IP
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { 
    $_.InterfaceAlias -notmatch 'Loopback|vEthernet|WSL' -and $_.IPAddress -notlike '169.254.*'
} | Select-Object -First 1).IPAddress

if (-not $localIP) { $localIP = "127.0.0.1" }

if ($Https) {
    if (-not (Test-Path "data\certs")) { New-Item -ItemType Directory -Path "data\certs" | Out-Null }
    Write-Host "  Dashboard      https://localhost:$HttpsPort" -ForegroundColor Green
    Write-Host "  API Docs (UI)  https://localhost:$HttpsPort/docs        <- interactive Swagger" -ForegroundColor Green
    Write-Host "  Driver (GPS)   https://$($localIP):$HttpsPort/track     <- open on the phone" -ForegroundColor Green
    Write-Host "  Field reporter https://$($localIP):$HttpsPort/field" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Ctrl+C to stop"
    Write-Host "──────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    & $VenvPy -m uvicorn backend.app:app --host 0.0.0.0 --port $HttpsPort --ssl-keyfile data\certs\key.pem --ssl-certfile data\certs\cert.pem
} else {
    Write-Host "  Dashboard      http://localhost:$Port" -ForegroundColor Green
    Write-Host "  API Docs (UI)  http://localhost:$Port/docs        <- interactive Swagger" -ForegroundColor Green
    Write-Host "  Field reporter http://localhost:$Port/field" -ForegroundColor Green
    Write-Host "  Driver (GPS)   http://localhost:$Port/track" -ForegroundColor Green
    Write-Host "  API index      http://localhost:$Port/api" -ForegroundColor Green
    Write-Host ""
    Write-Host "  On the same Wi-Fi, the phone can reach   http://$($localIP):$Port" -ForegroundColor White
    Write-Host "  NOTE: phone GPS needs HTTPS — use .\run.ps1 -Https for that." -ForegroundColor DarkYellow
    Write-Host "  Ctrl+C to stop"
    Write-Host "──────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    & $VenvPy -m uvicorn backend.app:app --host 0.0.0.0 --port $Port
}
