@echo off
setlocal
REM Descarga Kronos (codigo + Kronos-base + tokenizador) en la carpeta donde este este archivo.
REM Uso: guardalo en Desktop\Kronos y haz doble clic.

cd /d "%~dp0"
echo Carpeta de destino: %cd%
echo.

where curl >nul 2>nul
if errorlevel 1 (
    echo ERROR: no se encuentra curl. Necesitas Windows 10 o superior.
    goto fin
)

echo [1/3] Descargando el codigo del proyecto desde GitHub...
if exist "Kronos-code" (
    echo    Ya existe Kronos-code, lo salto.
) else (
    curl -L --fail -o kronos.zip https://github.com/shiyu-coder/Kronos/archive/refs/heads/master.zip
    if errorlevel 1 ( echo ERROR descargando el codigo. & goto fin )
    tar -xf kronos.zip
    if errorlevel 1 ( echo ERROR descomprimiendo el codigo. & goto fin )
    ren "Kronos-master" "Kronos-code"
    del kronos.zip
)
echo.

echo [2/3] Descargando Kronos-base y su tokenizador desde Hugging Face...
for %%R in (Kronos-base Kronos-Tokenizer-base) do (
    if not exist "models\%%R" mkdir "models\%%R"
    for %%F in (config.json model.safetensors README.md) do (
        echo    %%R/%%F
        curl -L --fail -o "models\%%R\%%F" "https://huggingface.co/NeoQuasar/%%R/resolve/main/%%F"
        if errorlevel 1 echo    AVISO: no se pudo descargar %%R/%%F
    )
)
echo.

echo [3/3] Terminado. Contenido de la carpeta:
dir /s /b models
echo.
echo  Kronos-code\                    codigo del proyecto
echo  models\Kronos-base\             modelo (102M parametros)
echo  models\Kronos-Tokenizer-base\   tokenizador

:fin
echo.
pause
