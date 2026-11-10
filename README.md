# FutbolKronos

Predicción de partidos de fútbol con la arquitectura de [Kronos](https://github.com/shiyu-coder/Kronos)
(un transformer de series temporales) adaptada al fútbol. Simula un partido miles de veces y
muestra la distribución completa: 1X2, marcadores exactos, goles por equipo, más/menos goles,
ambos marcan y la banda de incertidumbre del modelo. Si el partido ya se jugó, lo compara con
el resultado real y con las cuotas.

El repositorio incluye los modelos ya entrenados (`pesos/`) y los datos procesados (`datos/`),
así que la aplicación funciona sin entrenar nada. Los detalles técnicos están en
[ESQUEMA.md](ESQUEMA.md).

## 1. Instalación

Requisitos: Windows y Python 3.11 o superior.

```bash
git clone https://github.com/Kyei68/FutbolKronos.git
cd FutbolKronos
python -m pip install -e .
```

Eso instala las dependencias de `pyproject.toml` (torch, numpy, pandas, matplotlib,
safetensors, einops, huggingface_hub).

> Los `.bat` llaman a `python3`. Si en tu Windows ese comando no existe o abre la Microsoft
> Store, instala Python desde python.org o cambia `python3` por `python` en los `.bat`.

## 2. Usar la aplicación (sin claves)

Doble clic en **`Abrir FutbolKronos.bat`** (o `python futbolkronos_gui.py`).

## 3. Claves de API (solo para descargar datos nuevos)

Las claves son personales y **no se incluyen en el repositorio**: cada persona necesita las
suyas. Las dos son gratuitas. Crea cada archivo en esta carpeta, con la clave sola en una línea:

| Archivo | Dónde se consigue | Lo usan |
|---|---|---|
| `football_data_clave.txt` | https://www.football-data.org/client/register | `predecir_jornada.py`, `Predecir jornada.bat`, `entrenadores_actuales.py`, `crear_europa.py` |
| `api_football_clave.txt` | https://dashboard.api-football.com (también vale la variable de entorno `API_FOOTBALL_KEY`) | `api_football.py` |

El `.gitignore` excluye `*_clave.txt`, así que nunca se suben al hacer push.

## 4. Predecir una jornada

```bash
python predecir_jornada.py                         # próxima jornada de LaLiga
python predecir_jornada.py --jornada 13            # una jornada concreta
python predecir_jornada.py --evaluar               # acierto de las predicciones pasadas
python predecir_jornada.py --competicion champions # Champions (antes: python crear_europa.py)
```

## 5. Reconstruir los datos (opcional)

`datos/originales/game_lineups.csv.gz` (120 MB) no está en el repositorio porque supera el
límite de GitHub. Solo lo necesita `crear_alineaciones.py`. Para volver a descargarlo todo:

```bash
python descargar_datos.py
python descargar_ligas.py
```

## 6. Reentrenar con la GPU (opcional)

Los scripts de entrenamiento (`entrenar_produccion.bat`, `entrenar_europa.bat`,
`entrenar_todo.ps1`, `experimentos.py`) esperan un entorno virtual con PyTorch para CUDA en
**`..\venv-gpu`**, es decir, en la carpeta que contiene a esta:

```bash
cd ..
python -m venv venv-gpu
venv-gpu\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
cd FutbolKronos
..\venv-gpu\Scripts\python.exe -m pip install -e .
```

Elige en https://pytorch.org el comando de PyTorch que corresponda a tu versión de CUDA.
Sin GPU se puede entrenar en CPU con `python entrenar.py` (mucho más lento).

## Licencia

El código de la arquitectura (`kronos_futbol/`) procede de Kronos, con licencia MIT; ver
[LICENSE_Kronos.txt](LICENSE_Kronos.txt).
