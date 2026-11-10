@echo off
rem Evaluacion de FutbolKronos (configuracion ganadora de noviembre 2026):
rem ligas + Europa, cabeza Poisson-Dixon-Coles, rasgos numericos, LR coseno.
rem Valida en todas las ligas 2018-2021 y prueba en LaLiga 2021-2027. Registro en datos\log_entrenamiento_europa.txt
cd /d "%~dp0"
python3 crear_secuencias_ligas.py --europa > datos\log_entrenamiento_europa.txt 2>&1
..\venv-gpu\Scripts\python.exe entrenar.py --dispositivo cuda --datos ligas --modelo reducido --hilos 4 --lr-plan coseno --cabeza poisson --rasgos >> datos\log_entrenamiento_europa.txt 2>&1
echo FIN %errorlevel% >> datos\log_entrenamiento_europa.txt
