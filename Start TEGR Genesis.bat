@echo off
REM TEGR Genesis - double-click to open (Windows)
cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw main.py
    exit /b
)

REM Fallback: no pythonw found, use python and keep the window open if it fails
python main.py
if errorlevel 1 (
    echo.
    echo TEGR Genesis could not start. See the message above.
    pause
)
