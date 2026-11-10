"""
Paso 4d: partidos de otras competiciones (Copa, Champions, Europa League…) de cada equipo.

football-data.co.uk solo tiene la liga. Para calcular los días de descanso de verdad hacen
falta también los partidos entre semana de copa y competiciones europeas. Transfermarkt los
tiene desde 2012/13 para los clubes de las 11 primeras divisiones que cubre; para el resto de
equipos y años, el descanso se calcula solo con los partidos de liga.

Entrada:  datos/originales/games.csv.gz (Transfermarkt), datos/ligas_resultados.csv
Salida:   datos/partidos_otras_competiciones.csv   equipo_id, fecha, competicion

Uso (desde esta carpeta):  python3 crear_calendario.py
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
from crear_alineaciones import TM_A_FD, load_games, map_clubs
from crear_secuencias_ligas import team_ids

DATA = os.path.join(BASE, "datos")
RAW = os.path.join(DATA, "originales")


def main():
    print("Paso 4d · Partidos de copa y competiciones europeas (Transfermarkt)\n")
    fd = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["liga", "fecha", "local", "visitante", "goles_local", "goles_visitante"])
    fd["fecha"] = pd.to_datetime(fd["fecha"])
    fd = fd[fd["liga"].isin(TM_A_FD.values())]
    mapa = map_clubs(load_games(), fd)                 # (liga, club_id) -> nombre de football-data
    clubes = pd.DataFrame([{"liga": l, "club_id": c, "local": n, "visitante": n} for (l, c), n in mapa.items()])
    clubes = team_ids(clubes)[["club_id", "local_id"]].rename(columns={"local_id": "equipo_id"})
    clubes = clubes.drop_duplicates("club_id")

    g = pd.read_csv(os.path.join(RAW, "games.csv.gz"),
                    usecols=["competition_id", "competition_type", "date", "home_club_id", "away_club_id"])
    g = g[g["competition_type"] != "domestic_league"]           # la liga ya está en football-data
    g["fecha"] = pd.to_datetime(g["date"])
    filas = pd.concat([g[["home_club_id", "fecha", "competition_id"]].rename(columns={"home_club_id": "club_id"}),
                       g[["away_club_id", "fecha", "competition_id"]].rename(columns={"away_club_id": "club_id"})])
    out = filas.merge(clubes, on="club_id")[["equipo_id", "fecha", "competition_id"]]
    out = out.rename(columns={"competition_id": "competicion"}).drop_duplicates(["equipo_id", "fecha"])
    out = out.sort_values(["equipo_id", "fecha"])
    out["fecha"] = out["fecha"].dt.date
    out.to_csv(os.path.join(DATA, "partidos_otras_competiciones.csv"), index=False, encoding="utf-8-sig")
    print(f"   {len(out):,} partidos de {out['equipo_id'].nunique()} equipos "
          f"({pd.to_datetime(out['fecha']).min():%Y} – {pd.to_datetime(out['fecha']).max():%Y})")
    print("   competiciones más frecuentes:", ", ".join(f"{c} {n:,}" for c, n in out["competicion"].value_counts().head(6).items()))
    print("\nGuardado en datos/partidos_otras_competiciones.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
