@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo FutbolKronos: prediciendo la proxima jornada de LaLiga...
echo.
python3 predecir_jornada.py %*
echo.
pause
