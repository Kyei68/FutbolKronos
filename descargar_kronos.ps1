# Descarga Kronos (código + Kronos-base + tokenizador) en la carpeta donde se ejecute este script.
# Uso: clic derecho sobre el archivo > "Ejecutar con PowerShell" (déjalo dentro de Desktop\Kronos)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "1/3 Descargando el código del proyecto desde GitHub..."
if (Get-Command git -ErrorAction SilentlyContinue) {
    if (-not (Test-Path "Kronos-code")) { git clone https://github.com/shiyu-coder/Kronos.git Kronos-code }
} else {
    Invoke-WebRequest "https://github.com/shiyu-coder/Kronos/archive/refs/heads/master.zip" -OutFile "kronos.zip"
    Expand-Archive "kronos.zip" -DestinationPath "." -Force
    Rename-Item "Kronos-master" "Kronos-code"
    Remove-Item "kronos.zip"
}

Write-Host "2/3 Descargando Kronos-base y su tokenizador desde Hugging Face..."
$repos = @("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base")
$files = @("config.json", "model.safetensors", "README.md")
foreach ($repo in $repos) {
    $name = $repo.Split("/")[1]
    $dest = Join-Path "models" $name
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    foreach ($f in $files) {
        $url = "https://huggingface.co/$repo/resolve/main/$f"
        Write-Host "   $repo/$f"
        try { Invoke-WebRequest $url -OutFile (Join-Path $dest $f) } catch { Write-Host "   (no encontrado: $f)" }
    }
}

Write-Host "3/3 Listo. Estructura:"
Write-Host "   Kronos-code\        -> código del proyecto"
Write-Host "   models\Kronos-base\ -> modelo (102M parámetros)"
Write-Host "   models\Kronos-Tokenizer-base\ -> tokenizador"
Write-Host ""
Write-Host "Para cargarlo en Python desde Kronos-code:"
Write-Host '   tokenizer = KronosTokenizer.from_pretrained("../models/Kronos-Tokenizer-base")'
Write-Host '   model = Kronos.from_pretrained("../models/Kronos-base")'
Read-Host "Pulsa Enter para cerrar"
