@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "MAIN_SCRIPT=gui\main_gui.py"
set "VENV_PY="

if exist "face_env\Scripts\python.exe" set "VENV_PY=face_env\Scripts\python.exe"
if not defined VENV_PY if exist ".venv311\Scripts\python.exe" set "VENV_PY=.venv311\Scripts\python.exe"
if not defined VENV_PY if exist ".venv310\Scripts\python.exe" set "VENV_PY=.venv310\Scripts\python.exe"
if not defined VENV_PY if exist ".venv\Scripts\python.exe" set "VENV_PY=.venv\Scripts\python.exe"

if not defined VENV_PY (
    echo [ERROR] No virtual environment was found.
    echo [FIX] Run setup_drishtiai.bat first to create face_env.
    pause
    exit /b 1
)

if not exist "%MAIN_SCRIPT%" (
    echo [ERROR] Entry file not found: "%MAIN_SCRIPT%"
    pause
    exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%"
)

echo [DrishtiAI Run] Using interpreter: "%VENV_PY%"
"%VENV_PY%" "%MAIN_SCRIPT%"
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Application exited with code %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%
