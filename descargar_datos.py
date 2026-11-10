"""
Paso 2: descargar los datos de LaLiga (Primera División).

Fuentes (gratuitas, sin registro):
  1. Transfermarkt Datasets  (github.com/dcaribou/transfermarkt-datasets, licencia CC0)
     -> equipos, jugadores, entrenadores, partidos, alineaciones, eventos
     Cubre LaLiga desde 2012/13 hasta julio de 2026 (sus actualizaciones están pausadas).
  2. football-data.co.uk
     -> resultados, estadísticas (tiros, córners, tarjetas) y cuotas desde 1993/94,
        incluida la temporada actual.

Guarda tablas limpias en la carpeta "datos". Los archivos originales se quedan en
"datos/originales": si vuelves a ejecutar el script, no se descargan otra vez
(salvo la temporada actual de football-data, que se actualiza).

Uso (desde esta carpeta):  python3 descargar_datos.py
"""
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import date

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datos")
RAW = os.path.join(DATA, "originales")

LIGA = "ES1"  # código de LaLiga en Transfermarkt
TM_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{}.csv.gz"
TM_TABLES = ["clubs", "players", "games", "appearances", "game_lineups", "game_events"]
FD_URL = "https://www.football-data.co.uk/mmz4281/{}/SP1.csv"
FD_FIRST_SEASON = 1993
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def download(url, dest, label, force=False, optional=False):
    """Descarga url -> dest mostrando el progreso. Devuelve True si el archivo está disponible."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        print(f"   {label}: ya descargado")
        return True
    tmp = dest + ".part"
    try:
        try:
            _download_python(url, tmp, label)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            _download_curl(url, tmp, label, first_error=e)
        except Exception as e:
            _download_curl(url, tmp, label, first_error=e)
        os.replace(tmp, dest)
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"\r   {label}: no se pudo actualizar, uso la copia anterior")
            return True
        _raise_or_skip(e, url, label, optional)
        return False


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download_python(url, tmp, label):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done, last = 0, 0.0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 0.5:
                    pct = f" ({done * 100 // total} %)" if total else ""
                    print(f"\r   {label}: {done / 1e6:.1f} MB{pct}   ", end="", flush=True)
                    last = time.time()
    print(f"\r   {label}: {done / 1e6:.1f} MB  ✓          ")


def _download_curl(url, tmp, label, first_error):
    """Plan B: curl.exe viene incluido en Windows 10 y 11."""
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise first_error
    print(f"\r   {label}: reintentando con curl…          ", end="", flush=True)
    res = subprocess.run([curl, "-L", "--fail", "--silent", "--show-error", "--retry", "2",
                          "-A", UA["User-Agent"], "-o", tmp, url], capture_output=True, text=True)
    if res.returncode != 0:
        if "404" in (res.stderr or ""):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        raise RuntimeError(f"Python: {first_error} | curl: {(res.stderr or '').strip()}")
    print(f"\r   {label}: {os.path.getsize(tmp) / 1e6:.1f} MB  ✓ (curl)          ")


def _raise_or_skip(e, url, label, optional):
    """Las temporadas que aún no existen (404) se saltan en silencio; el resto de fallos se avisan."""
    if isinstance(e, urllib.error.HTTPError) and e.code == 404 and optional:
        return
    if optional:
        print(f"\r   {label}: no disponible ({e})")
        return
    detail = f"error HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else str(e)
    raise RuntimeError(f"No se pudo descargar {label} ({detail})\nURL: {url}")


def tm_path(name):
    return os.path.join(RAW, f"{name}.csv.gz")


def read_tm(name):
    return pd.read_csv(tm_path(name), compression="gzip", low_memory=False)


def read_tm_filtered(name, column, keep_values):
    """Lee una tabla grande por trozos y se queda solo con las filas de LaLiga."""
    parts = []
    for chunk in pd.read_csv(tm_path(name), compression="gzip", low_memory=False, chunksize=400_000):
        ids = pd.to_numeric(chunk[column], errors="coerce")
        parts.append(chunk[ids.isin(keep_values)])
    return pd.concat(parts, ignore_index=True)


def keep_cols(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()


def current_season_start():
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


# --------------------------------------------------------------------------- #
# 1. Transfermarkt
# --------------------------------------------------------------------------- #
def process_transfermarkt():
    print("\n[2/4] Filtrando LaLiga en los datos de Transfermarkt…")

    games = read_tm("games")
    games = games[games["competition_id"] == LIGA].copy()
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    for c in ["game_id", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]:
        games[c] = pd.to_numeric(games[c], errors="coerce")
    games = games.dropna(subset=["game_id", "home_club_id", "away_club_id"]).sort_values(["date", "game_id"])
    games[["game_id", "home_club_id", "away_club_id"]] = games[["game_id", "home_club_id", "away_club_id"]].astype(int)
    game_ids = set(games["game_id"])
    print(f"   partidos de LaLiga: {len(games):,}")

    partidos = keep_cols(games, [
        "game_id", "season", "round", "date",
        "home_club_id", "home_club_name", "away_club_id", "away_club_name",
        "home_club_goals", "away_club_goals", "home_club_position", "away_club_position",
        "home_club_manager_name", "away_club_manager_name",
        "home_club_formation", "away_club_formation",
        "stadium", "attendance", "referee", "url",
    ])

    apariciones = read_tm_filtered("appearances", "game_id", game_ids)
    alineaciones = read_tm_filtered("game_lineups", "game_id", game_ids)
    eventos = read_tm_filtered("game_events", "game_id", game_ids)
    print(f"   apariciones: {len(apariciones):,}   alineaciones: {len(alineaciones):,}   eventos: {len(eventos):,}")

    # --- equipos
    clubs = read_tm("clubs")
    clubs["club_id"] = pd.to_numeric(clubs["club_id"], errors="coerce")
    long_games = pd.concat([
        games[["season", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
        games[["season", "away_club_id"]].rename(columns={"away_club_id": "club_id"}),
    ])
    seasons = long_games.groupby("club_id")["season"].agg(
        temporadas_en_laliga="nunique", primera_temporada="min", ultima_temporada="max").reset_index()
    equipos = seasons.merge(clubs, on="club_id", how="left")
    equipos["en_laliga_actualmente"] = equipos["domestic_competition_id"] == LIGA
    equipos = keep_cols(equipos, [
        "club_id", "name", "club_code", "en_laliga_actualmente", "temporadas_en_laliga",
        "primera_temporada", "ultima_temporada", "coach_name", "stadium_name", "stadium_seats",
        "squad_size", "average_age", "foreigners_number", "national_team_players",
        "total_market_value", "net_transfer_record", "url",
    ]).sort_values(["en_laliga_actualmente", "temporadas_en_laliga"], ascending=False)

    # --- jugadores (todos los que han jugado en LaLiga + los de las plantillas actuales)
    players = read_tm("players")
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    apariciones["date"] = pd.to_datetime(apariciones["date"], errors="coerce")
    stats = apariciones.groupby("player_id").agg(
        partidos_laliga=("game_id", "nunique"),
        minutos_laliga=("minutes_played", "sum"),
        goles_laliga=("goals", "sum"),
        asistencias_laliga=("assists", "sum"),
        amarillas_laliga=("yellow_cards", "sum"),
        rojas_laliga=("red_cards", "sum"),
        primer_partido_laliga=("date", "min"),
        ultimo_partido_laliga=("date", "max"),
    ).reset_index()
    stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce")
    in_current = players["current_club_domestic_competition_id"] == LIGA \
        if "current_club_domestic_competition_id" in players.columns else False
    ids = set(stats["player_id"]) | set(players.loc[in_current, "player_id"])
    jugadores = players[players["player_id"].isin(ids)].merge(stats, on="player_id", how="left")
    jugadores["en_plantilla_laliga"] = jugadores["current_club_domestic_competition_id"] == LIGA
    for c in ["partidos_laliga", "minutos_laliga", "goles_laliga", "asistencias_laliga",
              "amarillas_laliga", "rojas_laliga"]:
        jugadores[c] = jugadores[c].fillna(0).astype(int)
    jugadores = keep_cols(jugadores, [
        "player_id", "name", "first_name", "last_name", "date_of_birth", "country_of_citizenship",
        "country_of_birth", "position", "sub_position", "foot", "height_in_cm",
        "current_club_id", "current_club_name", "en_plantilla_laliga", "contract_expiration_date",
        "market_value_in_eur", "highest_market_value_in_eur",
        "partidos_laliga", "minutos_laliga", "goles_laliga", "asistencias_laliga",
        "amarillas_laliga", "rojas_laliga", "primer_partido_laliga", "ultimo_partido_laliga",
        "international_caps", "international_goals", "agent_name", "url",
    ]).sort_values("partidos_laliga", ascending=False)

    # --- entrenadores (a partir del entrenador de cada equipo en cada partido)
    def side(prefix, other):
        d = games[["season", "date", f"{prefix}_club_id", f"{prefix}_club_name", f"{prefix}_club_manager_name",
                   f"{prefix}_club_goals", f"{other}_club_goals"]].copy()
        d.columns = ["season", "date", "club_id", "club_name", "entrenador", "gf", "gc"]
        return d
    m = pd.concat([side("home", "away"), side("away", "home")]).dropna(subset=["entrenador"])
    m["victoria"] = (m["gf"] > m["gc"]).astype(int)
    m["empate"] = (m["gf"] == m["gc"]).astype(int)
    m["derrota"] = (m["gf"] < m["gc"]).astype(int)
    entrenadores = m.groupby(["entrenador", "club_id", "club_name", "season"]).agg(
        partidos=("date", "count"), victorias=("victoria", "sum"), empates=("empate", "sum"),
        derrotas=("derrota", "sum"), goles_favor=("gf", "sum"), goles_contra=("gc", "sum"),
        primer_partido=("date", "min"), ultimo_partido=("date", "max"),
    ).reset_index()
    entrenadores["puntos_por_partido"] = ((3 * entrenadores["victorias"] + entrenadores["empates"])
                                          / entrenadores["partidos"]).round(2)
    entrenadores = entrenadores.sort_values(["season", "club_name", "primer_partido"])

    return {
        "equipos": equipos, "jugadores": jugadores, "entrenadores": entrenadores,
        "partidos": partidos, "alineaciones": alineaciones, "eventos": eventos,
        "apariciones": apariciones,
    }


# --------------------------------------------------------------------------- #
# 2. football-data.co.uk
# --------------------------------------------------------------------------- #
FD_COLS = [
    "temporada", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA",
    "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5",
]


def download_football_data():
    print("\n[3/4] Descargando resultados y cuotas de football-data.co.uk…")
    cur = current_season_start()
    frames = []
    for y in range(FD_FIRST_SEASON, cur + 1):
        code = f"{y % 100:02d}{(y + 1) % 100:02d}"
        label = f"Temporada {y}/{(y + 1) % 100:02d}"
        dest = os.path.join(RAW, f"football-data_SP1_{code}.csv")
        if not download(FD_URL.format(code), dest, label, force=(y == cur), optional=True):
            continue
        try:
            df = pd.read_csv(dest, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print(f"   {label}: no se pudo leer ({e})")
            continue
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])
        df.insert(0, "temporada", f"{y}/{(y + 1) % 100:02d}")
        frames.append(df)
    if not frames:
        return None
    fd = pd.concat(frames, ignore_index=True)
    fd["Date"] = pd.to_datetime(fd["Date"], dayfirst=True, format="mixed", errors="coerce")
    fd = keep_cols(fd, FD_COLS)
    for c in ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if c in fd.columns:
            fd[c] = pd.to_numeric(fd[c], errors="coerce").round().astype("Int64")
    return fd.rename(columns={"Date": "fecha"}).sort_values("fecha")


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RAW, exist_ok=True)
    print("=" * 64)
    print(" FutbolKronos · Descarga de datos de LaLiga")
    print("=" * 64)

    print("\n[1/4] Descargando Transfermarkt Datasets (unos 150 MB la primera vez)…")
    for t in TM_TABLES:
        download(TM_URL.format(t), tm_path(t), t)

    tables = process_transfermarkt()
    fd = download_football_data()
    if fd is not None:
        tables["resultados_y_cuotas"] = fd

    print("\n[4/4] Guardando tablas en la carpeta datos…")
    summary = []
    for name, df in tables.items():
        path = os.path.join(DATA, f"{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: se abre bien en Excel
        summary.append((name, len(df), len(df.columns)))
        print(f"   {name + '.csv':<26} {len(df):>9,} filas  {len(df.columns):>3} columnas")

    p = tables["partidos"]
    lines = [
        "FutbolKronos · Resumen de datos de LaLiga",
        f"Generado: {date.today():%d/%m/%Y}",
        "",
        f"Transfermarkt: {p['season'].nunique()} temporadas "
        f"({int(p['season'].min())}/{(int(p['season'].min()) + 1) % 100:02d} a "
        f"{int(p['season'].max())}/{(int(p['season'].max()) + 1) % 100:02d}), "
        f"último partido {p['date'].max():%d/%m/%Y}",
    ]
    if fd is not None:
        lines.append(f"football-data.co.uk: {fd['temporada'].nunique()} temporadas "
                     f"({fd['temporada'].iloc[0]} a {fd['temporada'].iloc[-1]}), "
                     f"último partido {fd['fecha'].max():%d/%m/%Y}")
    lines += ["", "Tablas:"] + [f"  {n + '.csv':<26} {r:>9,} filas  {c:>3} columnas" for n, r, c in summary]
    with open(os.path.join(DATA, "RESUMEN.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[3:5]))
    print(f"\nListo. Todo está en: {DATA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
