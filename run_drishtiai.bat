@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "MAIN_SCRIPT=gui\main_gui.py"
set "VENV_PY="

if exist ".venv311\Scripts\python.exe" set "VENV_PY=.venv311\Scripts\python.exe"
if not defined VENV_PY if exist ".venv310\Scripts\python.exe" set "VENV_PY=.venv310\Scripts\python.exe"
if not defined VENV_PY if exist ".venv\Scripts\python.exe" set "VENV_PY=.venv\Scripts\python.exe"

if not defined VENV_PY (
    echo [ERROR] No virtual environment found.
    echo [FIX] Run setup_drishtiai.bat once before running the app.
    pause
    exit /b 1
)

if not exist "%MAIN_SCRIPT%" (
    echo [ERROR] Entry file not found: "%MAIN_SCRIPT%"
    pause
    exit /b 1
)

echo [DrishtiAI Run] Using interpreter: "%VENV_PY%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $root = [System.IO.Path]::GetFullPath('%ROOT%'); Set-Location -LiteralPath $root; $env:PYTHONPATH = $root + ';' + $env:PYTHONPATH; & (Join-Path $root '%VENV_PY%') (Join-Path $root 'gui\\main_gui.py'); exit $LASTEXITCODE }"
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Application exited with code %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%
