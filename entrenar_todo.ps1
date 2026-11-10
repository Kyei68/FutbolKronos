# Evaluacion y despues produccion de FutbolKronos (configuracion de noviembre 2026:
# ligas + Europa, cabeza Poisson-Dixon-Coles, rasgos numericos, LR coseno; sin cuotas).
# Lo mismo que entrenar_todo.bat, para lanzarlo desde PowerShell:
#   powershell -ExecutionPolicy Bypass -File entrenar_todo.ps1
# Registros (UTF-8): datos\log_secuencias.txt, datos\log_entrenamiento_europa.txt, datos\log_produccion.txt
Set-Location -Path $PSScriptRoot
$gpu = (Resolve-Path "..\venv-gpu\Scripts\python.exe").Path
$op = "--dispositivo cuda --datos ligas --modelo reducido --hilos 4 --lr-plan coseno --cabeza poisson --rasgos"

function Paso($exe, $argumentos, $log) {
    $p = Start-Process -FilePath $exe -ArgumentList $argumentos -NoNewWindow -Wait -PassThru `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    "FIN $($p.ExitCode)" | Out-File -Append -Encoding utf8 $log
    return $p.ExitCode
}

if ((Paso "python3" "-u crear_secuencias_ligas.py --europa" "datos\log_secuencias.txt") -ne 0) { exit 1 }
if ((Paso $gpu "-u entrenar.py $op" "datos\log_entrenamiento_europa.txt") -ne 0) { exit 1 }
Paso $gpu "-u entrenar.py $op --produccion --calibrar" "datos\log_produccion.txt" | Out-Null
