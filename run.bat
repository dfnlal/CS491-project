@echo off
rem Double-click to start the 3-Sweep app with a sample photo.
cd /d "%~dp0"
python main.py samples\synthetic_cylinder.png
if errorlevel 1 (
    echo.
    echo The app could not start. If a library is missing, run:
    echo     pip install -r requirements.txt
    pause
)
