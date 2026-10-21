@echo off
REM TEGR Genesis — Duda Hydrogen Bound State Launcher
chcp 65001 >nul 2>nul
cd /d "%~dp0"

echo ========================================================================
echo       TEGR Genesis — Duda Hydrogen 1s Bound State Benchmark
echo ========================================================================
echo.

REM 1. Check for bundled local portable Python
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" run_duda_hydrogen.py
    goto FINISH
)

REM 2. Check for system Python
where python >nul 2>nul
if %errorlevel%==0 (
    python run_duda_hydrogen.py
    goto FINISH
)

REM 3. Check for Python launcher (py)
where py >nul 2>nul
if %errorlevel%==0 (
    py run_duda_hydrogen.py
    goto FINISH
)

echo.
echo ========================================================================
echo                Python 3 was not found on your system
echo ========================================================================
echo This benchmark requires Python 3.10+ with PyTorch and PySINDy.
echo Please install Python 3.10+ from: https://www.python.org/downloads/
echo (Important: Check the box "Add Python to PATH" during installation)
echo.
pause
exit /b 1

:FINISH
if %errorlevel%==0 (
    echo.
    echo ========================================================================
    echo Simulation complete! Opening results file...
    echo ========================================================================
    if exist "%~dp0results\duda_hydrogen\duda_hydrogen_results.txt" (
        start "" notepad "%~dp0results\duda_hydrogen\duda_hydrogen_results.txt"
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
