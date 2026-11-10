"""
Paso 2b: descargar resultados de otras ligas europeas (football-data.co.uk).

Sirven para que el modelo aprenda "cómo funciona el fútbol" con muchos más partidos.
El modelo no memoriza equipos (los juzga por sus últimos partidos y su Elo), así que un
partido de la Premier o de la Bundesliga le enseña lo mismo que uno de LaLiga.
La evaluación sigue haciéndose solo con LaLiga.

Incluye Segunda División (SP2): así los recién ascendidos llegan a Primera con su
historial real.

Salida:  datos/ligas_resultados.csv   un partido por fila, todas las divisiones
         datos/originales/football-data_<DIV>_<temporada>.csv   archivos originales

Uso (desde esta carpeta):  python3 descargar_ligas.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import descargar_datos as dd

URL = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"
FIRST_SEASON = 1993

# código football-data: (país, división 1 = primera, nombre)
DIVISIONES = {
    "SP1": ("ESP", 1, "España · Primera"), "SP2": ("ESP", 2, "España · Segunda"),
    "E0": ("ENG", 1, "Inglaterra · Premier League"), "E1": ("ENG", 2, "Inglaterra · Championship"),
    "E2": ("ENG", 3, "Inglaterra · League One"), "E3": ("ENG", 4, "Inglaterra · League Two"),
    "EC": ("ENG", 5, "Inglaterra · National League"),
    "D1": ("GER", 1, "Alemania · Bundesliga"), "D2": ("GER", 2, "Alemania · 2. Bundesliga"),
    "I1": ("ITA", 1, "Italia · Serie A"), "I2": ("ITA", 2, "Italia · Serie B"),
    "F1": ("FRA", 1, "Francia · Ligue 1"), "F2": ("FRA", 2, "Francia · Ligue 2"),
    "N1": ("NED", 1, "Países Bajos · Eredivisie"), "B1": ("BEL", 1, "Bélgica · Pro League"),
    "P1": ("POR", 1, "Portugal · Primeira Liga"), "T1": ("TUR", 1, "Turquía · Süper Lig"),
    "G1": ("GRE", 1, "Grecia · Super League"),
    "SC0": ("SCO", 1, "Escocia · Premiership"), "SC1": ("SCO", 2, "Escocia · Championship"),
    "SC2": ("SCO", 3, "Escocia · League One"), "SC3": ("SCO", 4, "Escocia · League Two"),
}
RENOMBRAR = {"HT": "HomeTeam", "AT": "AwayTeam", "Home": "HomeTeam", "Away": "AwayTeam",
             "HG": "FTHG", "AG": "FTAG"}
COLS = ["liga", "temporada", "temporada_inicio", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "HS", "AS", "HST", "AST", "HR", "AR", "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A",
        "mercado_local", "mercado_empate", "mercado_visitante", "mercado_casas"]
NO_CASAS = {"Avg", "BbAv", "Max", "BbMx"}          # medias y máximos: se usan las casas una a una


def mercado(df):
    """Probabilidad del mercado antes del partido: media de las casas disponibles, cada una sin
    su margen (normalizada a 1). Se excluyen máximos, medias ya hechas y cuotas de cierre
    (terminan en C), porque al predecir un partido futuro solo se tienen las de antes."""
    cols = set(df.columns)
    casas = [c[:-1] for c in df.columns if c.endswith("H") and {c[:-1] + "D", c[:-1] + "A"} <= cols]
    casas = [c for c in casas if c not in NO_CASAS and not (c.endswith("C") and c[:-1] in casas + list(NO_CASAS))]
    suma, n = 0.0, 0
    for c in casas:
        q = df[[c + "H", c + "D", c + "A"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / q
            tot = inv.sum(1, keepdims=True)
            ok = (q > 1.0).all(1) & (tot[:, 0] > 0.98) & (tot[:, 0] < 1.45)
            p = np.where(ok[:, None], inv / tot, 0.0)
        suma, n = suma + p, n + ok
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(np.asarray(n)[:, None] > 0, suma / np.asarray(n)[:, None], np.nan) if casas else             np.full((len(df), 3), np.nan)
    df = df.copy()
    df["mercado_local"], df["mercado_empate"], df["mercado_visitante"] = m[:, 0], m[:, 1], m[:, 2]
    df["mercado_casas"] = n if casas else 0
    return df


def main():
    os.makedirs(dd.RAW, exist_ok=True)
    print(f"Paso 2b · Descargar {len(DIVISIONES)} divisiones de football-data.co.uk\n")
    cur = dd.current_season_start()
    frames = []
    for div, (_, _, nombre) in DIVISIONES.items():
        n_files, n_rows = 0, 0
        for y in range(FIRST_SEASON, cur + 1):
            code = f"{y % 100:02d}{(y + 1) % 100:02d}"
            dest = os.path.join(dd.RAW, f"football-data_{div}_{code}.csv")
            existed = os.path.exists(dest)
            ok = dd.download(URL.format(code, div), dest, f"{div} {y}/{(y + 1) % 100:02d}",
                             force=(y == cur), optional=True)
            if not ok:
                continue
            if not existed or y == cur:
                time.sleep(0.3)                     # sin prisas con el servidor
            try:
                df = pd.read_csv(dest, encoding="latin-1", on_bad_lines="skip")
            except Exception as e:
                print(f"   {div} {code}: no se pudo leer ({e})")
                continue
            # Cabeceras alternativas de algunos archivos antiguos
            df = df.rename(columns={k: v for k, v in RENOMBRAR.items() if v not in df.columns})
            if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(df.columns):
                print(f"   {div} {code}: formato no reconocido, se omite")
                continue
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            df = df.assign(liga=div, temporada=f"{y}/{(y + 1) % 100:02d}", temporada_inicio=y)
            frames.append(dd.keep_cols(mercado(df), COLS))
            n_files += 1
            n_rows += len(df)
        print(f"   {nombre:<34} {n_files:>2} temporadas  {n_rows:>7,} partidos")

    out = pd.concat(frames, ignore_index=True)
    out["fecha"] = pd.to_datetime(out.pop("Date"), dayfirst=True, format="mixed", errors="coerce")
    out = out.dropna(subset=["fecha"])
    for c in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HR", "AR"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.rename(columns={"HomeTeam": "local", "AwayTeam": "visitante", "FTHG": "goles_local",
                              "FTAG": "goles_visitante", "HS": "tiros_local", "AS": "tiros_visitante",
                              "HST": "tiros_puerta_local", "AST": "tiros_puerta_visitante",
                              "HR": "rojas_local", "AR": "rojas_visitante"})
    out["local"] = out["local"].str.strip()
    out["visitante"] = out["visitante"].str.strip()
    out = out.sort_values(["fecha", "liga", "local"]).reset_index(drop=True)
    destino = os.path.join(dd.DATA, "ligas_resultados.csv")
    out.to_csv(destino + ".tmp", index=False, encoding="utf-8-sig")
    os.replace(destino + ".tmp", destino)           # de golpe: otros procesos pueden estar leyéndolo
    con_tiros = out["tiros_puerta_local"].notna().mean()
    print(f"\n   Total: {len(out):,} partidos ({con_tiros * 100:.0f} % con tiros a puerta), "
          f"{out['fecha'].min():%d/%m/%Y} – {out['fecha'].max():%d/%m/%Y}")
    print(f"   {out['mercado_local'].notna().mean() * 100:.0f} % con probabilidades del mercado (casas de apuestas)")
    print("   Guardado en datos/ligas_resultados.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
