@echo off
chcp 65001 >nul
cd /d "%~dp0code"
"C:\Users\Sasha Bray\AppData\Local\Python\pythoncore-3.14-64\python.exe" app_gui.py
echo.
echo --- приложение закрыто (это окно можно закрыть) ---
pause
