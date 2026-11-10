@echo off
rem Modelo de produccion de FutbolKronos (todos los partidos, calibrado), con la MISMA configuracion
rem que entrenar_europa.bat. Ejecutalo despues de este. Registro en datos\log_produccion.txt
cd /d "%~dp0"
..\venv-gpu\Scripts\python.exe entrenar.py --dispositivo cuda --datos ligas --produccion --calibrar --hilos 4 --lr-plan coseno --cabeza poisson --rasgos > datos\log_produccion.txt 2>&1
echo FIN %errorlevel% >> datos\log_produccion.txt
