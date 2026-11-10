@echo off
rem Compara variantes de FutbolKronos en validacion (unos 40 minutos con la grafica).
rem Resultados en datos\experimentos.txt; progreso en datos\log_experimentos.txt
cd /d "%~dp0"
python3 -u experimentos.py > datos\log_experimentos.txt 2>&1
echo FIN %errorlevel% >> datos\log_experimentos.txt
