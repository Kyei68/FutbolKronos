@echo off
cd /d "%~dp0"
echo Abriendo Kronos... (no cierres esta ventana mientras uses la aplicacion)
python3 kronos_gui.py
if errorlevel 1 pause
