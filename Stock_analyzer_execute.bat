@echo off
title US Stock Analyzer Launcher
cd /d "%~dp0"

echo Starting US Stock Analyzer...

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [Error] Cannot find the .venv virtual environment.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run "Stock_analyzer_2026_17_rate_limit_safe.py"

pause