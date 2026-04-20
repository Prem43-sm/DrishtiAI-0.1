@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_NAME=face_env"
set "REQ_FILE=requirements-face_env-lock.txt"
set "MAIN_SCRIPT=gui\main_gui.py"
set "PY_CMD="
set "VENV_PY=%ROOT%%VENV_NAME%\Scripts\python.exe"

echo [1/7] Checking project files...
if not exist "%REQ_FILE%" (
    echo [ERROR] Missing "%REQ_FILE%".
    pause
    exit /b 1
)

if not exist "%MAIN_SCRIPT%" (
    echo [ERROR] Entry file not found: "%MAIN_SCRIPT%"
    pause
    exit /b 1
)

echo [2/7] Detecting Python 3.10...
py -3.10 -c "import sys" >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3.10"
)

if not defined PY_CMD (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo [ERROR] Python 3.10 was not found.
    echo [FIX] Install Python 3.10.x and make sure "py -3.10" or "python" works in terminal.
    pause
    exit /b 1
)

echo [DrishtiAI Setup] Using interpreter: %PY_CMD%

echo [3/7] Creating virtual environment (%VENV_NAME%)...
if exist "%VENV_PY%" (
    echo [DrishtiAI Setup] Reusing existing %VENV_NAME%.
) else (
    %PY_CMD% -m venv "%VENV_NAME%"
    if errorlevel 1 (
        echo [ERROR] Failed to create "%VENV_NAME%".
        pause
        exit /b 1
    )
)

"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] "%VENV_NAME%" is not using Python 3.10.
    echo [FIX] Delete "%VENV_NAME%" and run setup_drishtiai.bat again with Python 3.10 installed.
    pause
    exit /b 1
)

echo [4/7] Upgrading pip, setuptools, and wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade packaging tools.
    pause
    exit /b 1
)

echo [5/7] Installing locked dependencies...
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo [FIX] If tensorflow or dlib fails, verify Python 3.10 and install Microsoft C++ Build Tools.
    pause
    exit /b 1
)

echo [6/7] Ensuring runtime folders exist...
for %%D in (storage storage\attendance storage\reports storage\snapshots storage\snapshots\noise_alerts storage\timetable storage\known_faces models models\emotion models\emotion\legacy models\face_recognition tools archive datasets) do (
    if not exist "%%D" mkdir "%%D"
)

echo [7/7] Saving installed package snapshot...
"%VENV_PY%" -m pip list --format=freeze > "%VENV_NAME%\installed-freeze.txt"

echo.
echo [DrishtiAI Setup] face_env is ready.
echo [DrishtiAI Setup] Start the app with: run_drishtiai.bat
pause
exit /b 0
