@echo off
REM TEGR Genesis - double-click to open (Windows)
cd /d "%~dp0"

REM 1. Check for bundled local portable Python
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" bootstrap.py
    if errorlevel 1 pause
    exit /b
)

REM 2. Check for system python
where python >nul 2>nul
if %errorlevel%==0 (
    python bootstrap.py
    if errorlevel 1 pause
    exit /b
)

where py >nul 2>nul
if %errorlevel%==0 (
    py bootstrap.py
    if errorlevel 1 pause
    exit /b
)

echo.
echo ====================================================
echo       Python 3 was not found on your system
echo ====================================================
echo TEGR Genesis needs Python to run.
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo (Important: Check the box "Add Python to PATH" during installation)
echo.
pause
