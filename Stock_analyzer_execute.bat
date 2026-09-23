@echo off
title US Stock Analyzer Launcher
echo [알림] 가상환경을 활성화하고 미국 주식 분석기 앱을 실행합니다...
cd /d "%~dp0"

:: 1. 가상환경 활성화
call .\stock_env\Scripts\activate.bat

:: 2. Streamlit 앱 실행
streamlit run "Stock_analyzer_2026_17_rate_limit_safe.py"

pause