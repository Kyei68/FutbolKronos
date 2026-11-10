@echo off
cd /d "%~dp0"
echo Abriendo FutbolKronos... (no cierres esta ventana mientras uses la aplicacion)
python3 futbolkronos_gui.py
if errorlevel 1 pause
