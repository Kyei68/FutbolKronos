@echo off
rem Tanda de noviembre 2026: cabezas nuevas (Poisson, hibrido), LR coseno y variantes pendientes.
rem Solo validacion (todas las ligas 2018-2021), 2 semillas. Unos 45 minutos con la grafica.
rem Resultados en datos\experimentos.txt; progreso en datos\log_experimentos_mejoras.txt
cd /d "%~dp0"
python3 -u experimentos.py --solo base,coseno,poisson,hibrido,poisson_hibrido,descanso,rojas_descanso,historial10,historial40 > datos\log_experimentos_mejoras.txt 2>&1
echo FIN %errorlevel% >> datos\log_experimentos_mejoras.txt
