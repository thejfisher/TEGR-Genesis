@echo off
REM TEGR Genesis — Double-Slit Quantum Interference Launcher
chcp 65001 >nul 2>nul
cd /d "%~dp0"

echo ========================================================================
echo       TEGR Genesis — Double-Slit Quantum Interference Benchmark
echo ========================================================================
echo.

REM 1. Check for bundled local portable Python
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" run_double_slit.py
    goto FINISH
)

REM 2. Check for system Python
where python >nul 2>nul
if %errorlevel%==0 (
    python run_double_slit.py
    goto FINISH
)

REM 3. Check for Python launcher (py)
where py >nul 2>nul
if %errorlevel%==0 (
    py run_double_slit.py
    goto FINISH
)

echo.
echo ========================================================================
echo                Python 3 was not found on your system
echo ========================================================================
echo This benchmark requires Python 3.10+ with PyTorch and Matplotlib.
echo Please install Python 3.10+ from: https://www.python.org/downloads/
echo (Important: Check the box "Add Python to PATH" during installation)
echo.
pause
exit /b 1

:FINISH
if %errorlevel%==0 (
    echo.
    echo ========================================================================
    echo Simulation complete! Opening results and interference plot...
    echo ========================================================================
    if exist "%~dp0results\double_slit\double_slit_results.txt" (
        start "" notepad "%~dp0results\double_slit\double_slit_results.txt"
    )
    if exist "%~dp0results\double_slit\double_slit_histogram.png" (
        start "" "%~dp0results\double_slit\double_slit_histogram.png"
    )
) else (
    echo.
    echo ========================================================================
    echo [ERROR] The simulation encountered an error during execution.
    echo Please make sure all dependencies are installed:
    echo    pip install -r requirements.txt
    echo ========================================================================
)

echo.
pause
