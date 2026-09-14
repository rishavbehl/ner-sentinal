@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PORT=8000"
set "HTTPS_PORT=8443"
set "USE_HTTPS=0"
set "SENTINEL_LIVE_WEATHER=0"
set "SENTINEL_SIM_FLEET=1"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--https" set "USE_HTTPS=1"
if /i "%~1"=="--live" set "SENTINEL_LIVE_WEATHER=1"
if /i "%~1"=="--no-sim-fleet" set "SENTINEL_SIM_FLEET=0"
shift
goto parse_args
:args_done

echo ──────────────────────────────────────────────────────────────
echo   NER LOGISTICS SENTINEL (Windows Launcher)
echo ──────────────────────────────────────────────────────────────

REM Check if .venv exists, if not create using Python 3.11 or python
if not exist ".venv\Scripts\python.exe" (
    echo   [+] Virtual environment not found. Setting up .venv...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.11 -m venv .venv 2>nul
        if not exist ".venv\Scripts\python.exe" py -m venv .venv
    ) else (
        python -m venv .venv
    )
    if not exist ".venv\Scripts\python.exe" (
        echo   [!] Error: Could not create Python virtual environment.
        echo   Please ensure Python 3.11+ is installed.
        pause
        exit /b 1
    )
    echo   [+] Installing requirements...
    ".venv\Scripts\pip.exe" install -r requirements.txt
)

set "PY=.venv\Scripts\python.exe"

REM Preflight verification
if not exist "data\artifacts\risk_clf.joblib" (
    echo   First run - building dataset and training models (~2 minutes)...
    echo.
    "%PY%" scripts\pipeline.py
    echo.
)

REM Detect local IP address
for /f "tokens=4" %%a in ('route print ^| findstr 0.0.0.0.*0.0.0.0 ^| findstr /v "127.0.0.1"') do (
    if not defined IP set "IP=%%a"
)
if not defined IP set "IP=127.0.0.1"

if "%USE_HTTPS%"=="1" (
    if not exist "data\certs" mkdir data\certs
    echo   Dashboard      https://localhost:%HTTPS_PORT%
    echo   API Docs (UI)  https://localhost:%HTTPS_PORT%/docs        ^<- interactive Swagger
    echo   Driver (GPS)   https://%IP%:%HTTPS_PORT%/track     ^<- open on the phone
    echo   Field reporter https://%IP%:%HTTPS_PORT%/field
    echo.
    echo   Ctrl+C to stop
    echo ──────────────────────────────────────────────────────────────
    "%PY%" -m uvicorn backend.app:app --host 0.0.0.0 --port %HTTPS_PORT% --ssl-keyfile data\certs\key.pem --ssl-certfile data\certs\cert.pem
) else (
    echo   Dashboard      http://localhost:%PORT%
    echo   API Docs (UI)  http://localhost:%PORT%/docs        ^<- interactive Swagger
    echo   Field reporter http://localhost:%PORT%/field
    echo   Driver (GPS)   http://localhost:%PORT%/track
    echo   API index      http://localhost:%PORT%/api
    echo.
    echo   On the same Wi-Fi, the phone can reach   http://%IP%:%PORT%
    echo   NOTE: phone GPS needs HTTPS.
    echo   Ctrl+C to stop
    echo ──────────────────────────────────────────────────────────────
    "%PY%" -m uvicorn backend.app:app --host 0.0.0.0 --port %PORT%
)
