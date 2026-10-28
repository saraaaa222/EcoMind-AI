@echo off
color 0A
title Smart Home Energy Dashboard - AI Lab Project
cls

echo.
echo ================================================
echo    SMART HOME ENERGY MANAGEMENT DASHBOARD
echo    AI Lab Project - Neural Networks + Fuzzy Logic
echo ================================================
echo.
echo [INFO] Starting the dashboard...
echo [INFO] Your browser will open automatically
echo.
echo [WARNING] DO NOT CLOSE THIS WINDOW!
echo           The application runs in this window.
echo.
echo ================================================
echo.

cd /d "%~dp0"
python -m streamlit run dashboard.py --server.port 8501 --server.headless true

echo.
echo [ERROR] Application stopped or crashed.
echo.
pause