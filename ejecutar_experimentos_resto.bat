@echo off
rem Variantes que faltaron cuando se paro la primera tanda (base incluida como referencia).
cd /d "%~dp0"
python3 -u experimentos.py --solo base,rojas_descanso,historial40,historial10 > datos\log_experimentos2.txt 2>&1
echo FIN %errorlevel% >> datos\log_experimentos2.txt
