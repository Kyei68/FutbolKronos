@echo off
chcp 65001 > nul
rem Evaluacion y despues produccion, seguidas (unos 20-25 minutos con la grafica).
rem No cierres esta ventana: cuando termine lo dira y se quedara abierta.
cd /d "%~dp0"
echo FutbolKronos: reentrenando (unos 20-25 minutos). No cierres esta ventana.
echo.
echo [1/2] Evaluacion... empezada a las %time:~0,5%  (progreso en datos\log_entrenamiento_europa.txt)
call entrenar_europa.bat
findstr /b /c:"FIN 0" datos\log_entrenamiento_europa.txt > nul
if errorlevel 1 (
  echo.
  echo ERROR en la evaluacion: revisa datos\log_entrenamiento_europa.txt. No se ha tocado el modelo de produccion.
  goto fin
)
echo [2/2] Produccion... empezada a las %time:~0,5%  (progreso en datos\log_produccion.txt)
call entrenar_produccion.bat
findstr /b /c:"FIN 0" datos\log_produccion.txt > nul
if errorlevel 1 (
  echo.
  echo ERROR en produccion: revisa datos\log_produccion.txt.
  goto fin
)
echo.
echo ============================================================
echo   TERMINADO a las %time:~0,5%. Modelo nuevo listo en pesos\
echo ============================================================
:fin
echo.
pause
