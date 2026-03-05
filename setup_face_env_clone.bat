@echo off
setlocal enabledelayedexpansion

REM Recreate a close copy of the original face_env on another Windows machine.
REM Run from project root:
REM   setup_face_env_clone.bat

set "VENV_NAME=face_env"
set "REQ_FILE=requirements-face_env-lock.txt"
set "PY_CMD="

echo [1/6] Checking project files...
if not exist "%REQ_FILE%" (
    echo ERROR: Missing %REQ_FILE%.
    echo Keep this BAT and %REQ_FILE% in the project root, then run again.
    exit /b 1
)

echo [2/6] Detecting Python...
py -3.10 -V >nul 2>&1
if %errorlevel%==0 (
    set "PY_CMD=py -3.10"
) else (
    py -3 -V >nul 2>&1
    if %errorlevel%==0 (
        set "PY_CMD=py -3"
    ) else (
        python -V >nul 2>&1
        if %errorlevel%==0 (
            set "PY_CMD=python"
        ) else (
            echo ERROR: Python not found.
            echo Install Python 3.10.x and ensure "py" or "python" works in terminal.
            exit /b 1
        )
    )
)
echo Using: %PY_CMD%

echo [3/6] Creating virtual environment (%VENV_NAME%)...
if exist "%VENV_NAME%\Scripts\python.exe" (
    echo Existing %VENV_NAME% found. Reusing it.
) else (
    %PY_CMD% -m venv "%VENV_NAME%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
)

echo [4/6] Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate %VENV_NAME%.
    exit /b 1
)

echo [5/6] Installing pinned packages (this can take several minutes)...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    exit /b 1
)

pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo ERROR: Package install failed.
    echo If tensorflow or dlib fails, verify Python is 3.10 and MSVC build tools are installed.
    exit /b 1
)

echo [6/6] Verification...
python -V
python -m pip list --format=freeze > "%VENV_NAME%\installed-freeze.txt"

echo.
echo Environment ready.
echo Activate later with:
echo   call %VENV_NAME%\Scripts\activate.bat
echo Freeze snapshot saved to:
echo   %VENV_NAME%\installed-freeze.txt

endlocal
exit /b 0
