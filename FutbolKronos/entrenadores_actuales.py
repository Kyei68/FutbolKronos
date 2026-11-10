"""
Paso 2d: entrenadores de LaLiga de la temporada en curso (football-data.org, plan gratuito).

Transfermarkt está pausado desde julio de 2026, así que los entrenadores de 2026/27 se sacan
de football-data.org, que da el entrenador ACTUAL de cada equipo (no el de cada partido).

- Si coincide con el último que registró Transfermarkt, ha dirigido toda la temporada.
- Si no, hubo un cambio: se supone que fue al empezar la temporada y se marca "revisar"
  para que corrijas la fecha a mano si fue a mitad de temporada.
- Si se ejecuta más adelante y un equipo ha cambiado de entrenador, se añade una fila nueva
  con la fecha de hoy (también marcada "revisar").

Todo se guarda en datos/entrenadores_<temporada>.csv (equipo, entrenador, desde, nota), que
puedes editar: las filas que ya existen no se tocan. unificar_equipos.py lo usa para rellenar
los entrenadores que falten.

Clave gratuita: https://www.football-data.org/client/register → football_data_clave.txt
Gasta 2 consultas (equipos y partidos; el límite gratuito es 10 por minuto).

Uso (desde esta carpeta):  python3 entrenadores_actuales.py [--temporada 2026]
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import date

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
from api_football import same_coach, tm_coach_names

DATA = os.path.join(BASE, "datos")
CACHE = os.path.join(DATA, "originales", "football_data_org")
URL = "https://api.football-data.org/v4"


def get(ruta, key):
    req = urllib.request.Request(f"{URL}/{ruta}", headers={"X-Auth-Token": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def clave():
    path_key = os.path.join(BASE, "football_data_clave.txt")
    key = os.environ.get("FOOTBALL_DATA_KEY") or (open(path_key, encoding="utf-8").read().strip()
                                                   if os.path.exists(path_key) else None)
    if not key:
        sys.exit("ERROR: falta la clave de football-data.org en football_data_clave.txt")
    return key


def fecha_local(utc):
    return pd.Timestamp(utc).tz_convert("Europe/Madrid").tz_localize(None).normalize()


def mapear_equipos(matches, t):
    """Id de equipo de football-data.org -> nombre de football-data.co.uk, por votación
    (partidos terminados de la temporada t con misma fecha ±1 y mismo resultado)."""
    fd = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["liga", "temporada_inicio", "fecha", "local", "visitante", "goles_local", "goles_visitante"])
    fd = fd[(fd["liga"] == "SP1") & (fd["temporada_inicio"] == t)].copy()
    fd["fecha"] = pd.to_datetime(fd["fecha"])
    votos = {}
    for m in matches["matches"]:
        if m["status"] != "FINISHED":
            continue
        f = fecha_local(m["utcDate"])
        gh, ga = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        c = fd[((fd["fecha"] - f).dt.days.abs() <= 1) & (fd["goles_local"] == gh) & (fd["goles_visitante"] == ga)]
        for x in c.itertuples(index=False):
            votos.setdefault(m["homeTeam"]["id"], Counter())[x.local] += 1
            votos.setdefault(m["awayTeam"]["id"], Counter())[x.visitante] += 1
    return {k: c.most_common(1)[0][0] for k, c in votos.items() if c.most_common(1)[0][1] >= 2}


def last_tm_coach():
    """Último entrenador que registró Transfermarkt para cada equipo (nombre de football-data)."""
    pu = pd.read_csv(os.path.join(DATA, "partidos_unificados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["fecha", "local_id", "visitante_id", "entrenador_local", "entrenador_visitante"])
    eq = pd.read_csv(os.path.join(DATA, "equivalencias_equipos.csv"), encoding="utf-8-sig")
    a_fd = dict(zip(eq["equipo_id"], eq["nombre_football_data"]))
    filas = pd.concat([pu[["fecha", "local_id", "entrenador_local"]].set_axis(["fecha", "eq", "ent"], axis=1),
                       pu[["fecha", "visitante_id", "entrenador_visitante"]].set_axis(["fecha", "eq", "ent"], axis=1)])
    filas = filas.dropna().sort_values("fecha")
    return {a_fd.get(e, e): g["ent"].iloc[-1] for e, g in filas.groupby("eq")}


def main():
    ap = argparse.ArgumentParser(description="Entrenadores actuales de LaLiga")
    ap.add_argument("--temporada", type=int, default=2026)
    args = ap.parse_args()
    key = clave()
    os.makedirs(CACHE, exist_ok=True)
    t = args.temporada
    print(f"Paso 2d · Entrenadores de LaLiga {t}/{(t + 1) % 100:02d} (football-data.org)\n")

    teams = get(f"competitions/PD/teams?season={t}", key)
    matches = get(f"competitions/PD/matches?season={t}&status=FINISHED", key)
    hoy = date.today().isoformat()
    for nombre, d in [("equipos", teams), ("partidos", matches)]:
        with open(os.path.join(CACHE, f"{nombre}_{t}_{hoy}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    mapa = mapear_equipos(matches, t)

    tm = tm_coach_names()
    ultimo_tm = last_tm_coach()
    out_path = os.path.join(DATA, f"entrenadores_{t}.csv")
    prev = (pd.read_csv(out_path, encoding="utf-8-sig") if os.path.exists(out_path)
            else pd.DataFrame(columns=["equipo", "entrenador", "desde", "nota"]))
    filas = prev.to_dict("records")
    inicio = f"{t}-07-01"
    for team in teams["teams"]:
        equipo = mapa.get(team["id"])
        coach = (team.get("coach") or {}).get("name")
        if not equipo or not coach:
            print(f"   AVISO: sin emparejar o sin entrenador: {team['name']}")
            continue
        coach = same_coach(coach, tm.get(equipo, set())) or coach     # mismo nombre que Transfermarkt
        suyas = prev[prev["equipo"] == equipo].sort_values("desde")
        if len(suyas) and suyas["entrenador"].iloc[-1] == coach:
            continue                                                   # nada nuevo
        if len(suyas):
            filas.append({"equipo": equipo, "entrenador": coach, "desde": hoy,
                          "nota": "revisar: cambio detectado hoy, corrige la fecha de llegada"})
        elif same_coach(coach, [ultimo_tm.get(equipo, "")]):
            filas.append({"equipo": equipo, "entrenador": coach, "desde": inicio, "nota": "sigue desde la temporada pasada"})
        else:
            filas.append({"equipo": equipo, "entrenador": coach, "desde": inicio,
                          "nota": "revisar: nuevo respecto a Transfermarkt; si llegó a mitad de temporada, corrige la fecha"})
    out = pd.DataFrame(filas).sort_values(["equipo", "desde"])
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"   {len(mapa)} de {len(teams['teams'])} equipos emparejados\n")
    for r in out.itertuples(index=False):
        print(f"   {r.equipo:<14} {r.entrenador:<26} desde {r.desde}   {r.nota}")
    print(f"\nGuardado en datos/entrenadores_{t}.csv. Revisa las filas marcadas \"revisar\".")
    print("Siguiente: unificar_equipos.py → crear_velas.py → crear_secuencias.py → crear_secuencias_ligas.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
