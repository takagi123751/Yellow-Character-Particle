@echo off
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating the isolated Python environment...
    py -3 -m venv "%VENV%"
    if errorlevel 1 python -m venv "%VENV%"
    if errorlevel 1 goto :error
)

"%VENV%\Scripts\python.exe" -c "import numpy, moderngl, glfw" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    "%VENV%\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 goto :error
)

"%VENV%\Scripts\python.exe" "%~dp0yellow_character.py"
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Failed to start. Please check the error message above.
pause
exit /b 1
