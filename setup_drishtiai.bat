@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo [DrishtiAI Setup] Project root: "%ROOT%"

if not exist "gui\main_gui.py" (
    echo [ERROR] Entry file not found: "gui\main_gui.py"
    pause
    exit /b 1
)

set "PY_CMD="
set "PY_MM="

py -3.11 -c "import sys" >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3.11"
    set "PY_MM=3.11"
)

if not defined PY_CMD (
    py -3.10 -c "import sys" >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=py -3.10"
        set "PY_MM=3.10"
    )
)

if not defined PY_CMD (
    echo [ERROR] Python 3.10/3.11 not found via py launcher.
    echo [FIX] Install Python 3.11 and ensure the launcher command `py` works.
    pause
    exit /b 1
)

echo [DrishtiAI Setup] Selected interpreter: %PY_CMD% (Python %PY_MM%)

set "VENV_NAME=.venv311"
if "%PY_MM%"=="3.10" set "VENV_NAME=.venv310"

set "VENV_DIR=%ROOT%%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [DrishtiAI Setup] Creating virtual environment at "%VENV_DIR%"...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment at "%VENV_DIR%".
        pause
        exit /b 1
    )
) else (
    echo [DrishtiAI Setup] Reusing existing %VENV_NAME%.
)

echo [DrishtiAI Setup] Upgrading pip/setuptools/wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade packaging tools.
    pause
    exit /b 1
)

if not exist "requirements-runtime.txt" (
    echo [ERROR] Missing dependency file: "requirements-runtime.txt"
    pause
    exit /b 1
)

echo [DrishtiAI Setup] Installing project dependencies...
"%VENV_PY%" -m pip install -r "requirements-runtime.txt"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo [FIX] Install Microsoft C++ Build Tools if dlib/face-recognition build fails, then rerun setup.
    pause
    exit /b 1
)

echo [DrishtiAI Setup] Ensuring runtime folders exist...
for %%D in (attendance reports snapshots timetable known_faces) do (
    if not exist "%%D" mkdir "%%D"
)

if not exist "settings.json" echo [WARN] Optional file missing: settings.json
if not exist "gui\settings.json" echo [WARN] Optional file missing: gui\settings.json

if not exist "*.h5" echo [WARN] No model .h5 file found in project root.

echo [DrishtiAI Setup] Completed successfully.
pause
exit /b 0
