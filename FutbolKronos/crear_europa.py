"""
Paso 2e: partidos de competiciones europeas (Champions, Europa League, Conference y previas).

- 2012/13–2025/26: Transfermarkt (datos/originales/games.csv.gz, ya descargado).
- Temporada en curso de la Champions: football-data.org (plan gratuito, 1 consulta), con los
  partidos jugados y los que faltan por jugar.

Cada equipo recibe el MISMO identificador que en las ligas (así su historial mezcla liga y
Europa). Los clubes de países que no están en nuestras ligas (Shakhtar, Slavia, Bodø/Glimt…)
reciben un identificador "EUR:<nombre>" y su historial es solo europeo.

Salida:  datos/europa_resultados.csv   partidos jugados (mismo formato que ligas_resultados.csv
                                        más local_id / visitante_id)
         datos/europa_pendientes.csv   partidos de Champions por jugar (football-data.org)
         datos/europa_equipos_fdorg.csv  cómo se ha emparejado cada equipo de football-data.org
                                        (revísalo si algún nombre sale mal)

Uso (desde esta carpeta):  python3 crear_europa.py [--sin-api]
"""
import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
import urllib.request

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
from crear_alineaciones import TM_A_FD, load_games, map_clubs
from crear_secuencias_ligas import team_ids

DATA = os.path.join(BASE, "datos")
RAW = os.path.join(DATA, "originales")

# competición de Transfermarkt -> (código propio, nombre)
EUROPA = {"CL": ("CL", "Champions League"), "CLQ": ("CLQ", "Champions League · previas"),
          "EL": ("EL", "Europa League"), "ELQ": ("ELQ", "Europa League · previas"),
          "UCOL": ("ECL", "Conference League"), "ECLQ": ("ECLQ", "Conference League · previas")}
ARTICULOS = {"fc", "cf", "sk", "fk", "ac", "as", "ssc", "sc", "afc", "cd", "rc", "sl", "kv", "pae", "rcd",
             "ud", "sv", "vfb", "vfl", "tsg", "bsc", "if", "bk", "aik", "club", "de", "fútbol", "futbol",
             "football", "calcio", "1907", "1899", "1909", "1900", "04", "05", "07", "08", "09"}


ALIAS = {"internazionale": "inter", "munchen": "munich", "praha": "prague", "osc": "", "olympique": "",
         "lyonnais": "lyon", "fotballklubben": "", "sporting clube portugal": "sporting cp", "1907": ""}


def normal(nombre):
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = " ".join(t for t in s.split() if t not in ARTICULOS)
    for a, b in ALIAS.items():
        s = re.sub(rf"{a}", b, s)
    return " ".join(s.split())


def parecido(a, b):
    """Parecido entre nombres normalizados: letras en común o palabras compartidas."""
    ta, tb = set(a.split()), set(b.split())
    contiene = len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0
    return max(difflib.SequenceMatcher(None, a, b).ratio(), 0.9 * contiene)


def club_a_equipo():
    """club_id de Transfermarkt -> identificador de equipo de las ligas (si juega en una de las 11)."""
    fd = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["liga", "fecha", "local", "visitante", "goles_local", "goles_visitante"])
    fd["fecha"] = pd.to_datetime(fd["fecha"])
    fd = fd[fd["liga"].isin(TM_A_FD.values())]
    mapa = map_clubs(load_games(), fd)
    c = pd.DataFrame([{"liga": l, "club_id": k, "local": n, "visitante": n} for (l, k), n in mapa.items()])
    c = team_ids(c)
    return dict(zip(c["club_id"], c["local_id"]))


def main():
    ap = argparse.ArgumentParser(description="Partidos de competiciones europeas")
    ap.add_argument("--sin-api", action="store_true", help="no consultar football-data.org")
    ap.add_argument("--temporada", type=int, default=2026)
    args = ap.parse_args()
    print("Paso 2e · Competiciones europeas\n")

    g = pd.read_csv(os.path.join(RAW, "games.csv.gz"),
                    usecols=["competition_id", "season", "round", "date", "home_club_id", "away_club_id",
                             "home_club_name", "away_club_name", "home_club_goals", "away_club_goals"])
    g = g[g["competition_id"].isin(EUROPA)].dropna(subset=["home_club_goals", "away_club_goals"]).copy()
    a_equipo = club_a_equipo()
    nombres = {}
    for cid, n in list(zip(g["home_club_id"], g["home_club_name"])) + list(zip(g["away_club_id"], g["away_club_name"])):
        if isinstance(n, str):
            nombres[cid] = n
    ident = lambda cid: a_equipo.get(cid) or f"EUR:{nombres.get(cid, cid)}"
    t = g["season"].astype(int)
    tm = pd.DataFrame({
        "liga": g["competition_id"].map(lambda c: EUROPA[c][0]), "temporada": [f"{s}/{(s + 1) % 100:02d}" for s in t],
        "temporada_inicio": t, "fecha": pd.to_datetime(g["date"]),
        "local": g["home_club_id"].map(nombres), "visitante": g["away_club_id"].map(nombres),
        "local_id": g["home_club_id"].map(ident), "visitante_id": g["away_club_id"].map(ident),
        "goles_local": g["home_club_goals"], "goles_visitante": g["away_club_goals"], "ronda": g["round"]})
    enlazados = pd.concat([tm["local_id"], tm["visitante_id"]])
    print(f"   Transfermarkt: {len(tm):,} partidos europeos ({tm['temporada_inicio'].min()}–{tm['temporada_inicio'].max()}); "
          f"{(~enlazados.str.startswith('EUR:')).mean() * 100:.0f} % de las apariciones son de equipos de nuestras ligas")

    pendientes = pd.DataFrame()
    if not args.sin_api:
        try:
            from entrenadores_actuales import clave, fecha_local
            req = urllib.request.Request(
                f"https://api.football-data.org/v4/competitions/CL/matches?season={args.temporada}",
                headers={"X-Auth-Token": clave()})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            with open(os.path.join(RAW, f"football_data_org_CL_{args.temporada}.json"), "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            # Emparejar nombres de football-data.org con clubes de Transfermarkt (por parecido de nombre,
            # entre los clubes que han jugado en Europa las últimas temporadas)
            recientes = g[g["season"] >= args.temporada - 4]
            # candidatos: (nombre, identificador, ¿es de nuestras ligas?)
            candidatos = [(nombres[cid], ident(cid), False) for cid in
                          set(recientes["home_club_id"]) | set(recientes["away_club_id"]) if cid in nombres]
            ligas = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False,
                                usecols=["liga", "temporada_inicio", "local", "visitante"])
            ligas = ligas[(ligas["temporada_inicio"] >= args.temporada - 1)
                          & ligas["liga"].isin(TM_A_FD.values())].drop_duplicates(["liga", "local"])
            ligas["visitante"] = ligas["local"]
            ligas = team_ids(ligas)
            candidatos += [(n, i, True) for n, i in zip(ligas["local"], ligas["local_id"])]
            norm_cand = [(normal(n), n, i, dom) for n, i, dom in candidatos]
            filas_map, mapa_fd = [], {}
            equipos_fd = {m[s]["id"]: m[s] for m in d["matches"] for s in ("homeTeam", "awayTeam") if m[s]["id"]}
            for tid, team in equipos_fd.items():
                mejor, pts = None, 0.0
                for variante in {team["name"], team.get("shortName") or team["name"]}:
                    n = normal(variante)
                    for nc, nombre_c, id_c, dom in norm_cand:
                        r = parecido(n, nc) + (0.05 if dom else 0.0)       # a igualdad, el de nuestras ligas
                        if r > pts:
                            mejor, pts = (nombre_c, id_c), r
                ok = pts >= 0.8
                ident_fd = mejor[1] if ok else f"EUR:{team['shortName'] or team['name']}"
                nombre = mejor[0] if ok else team["name"]
                mapa_fd[tid] = (ident_fd, nombre)
                filas_map.append({"equipo_football_data_org": team["name"], "candidato": mejor[0] if mejor else None,
                                  "parecido": round(min(pts, 1), 2), "emparejado": ok, "equipo_id": ident_fd})
            pd.DataFrame(filas_map).sort_values("parecido").to_csv(
                os.path.join(DATA, "europa_equipos_fdorg.csv"), index=False, encoding="utf-8-sig")
            nuevos, pend = [], []
            for m in d["matches"]:
                if not m["homeTeam"]["id"] or not m["awayTeam"]["id"]:
                    continue
                (lid, ln), (vid, vn) = mapa_fd[m["homeTeam"]["id"]], mapa_fd[m["awayTeam"]["id"]]
                base = {"liga": "CL", "temporada": f"{args.temporada}/{(args.temporada + 1) % 100:02d}",
                        "temporada_inicio": args.temporada, "fecha": fecha_local(m["utcDate"]),
                        "local": ln, "visitante": vn, "local_id": lid, "visitante_id": vid,
                        "ronda": m.get("stage")}
                if m["status"] == "FINISHED":
                    nuevos.append({**base, "goles_local": m["score"]["fullTime"]["home"],
                                   "goles_visitante": m["score"]["fullTime"]["away"]})
                elif m["status"] in ("SCHEDULED", "TIMED"):
                    pend.append({**base, "fixture": m["id"], "jornada": m.get("matchday"),
                                 "hora": pd.Timestamp(m["utcDate"]).tz_convert("Europe/Madrid").strftime("%H:%M")})
            tm = pd.concat([tm, pd.DataFrame(nuevos)], ignore_index=True)
            pendientes = pd.DataFrame(pend)
            sin = [f["equipo_football_data_org"] for f in filas_map if not f["emparejado"]]
            print(f"   football-data.org: Champions {args.temporada}/{(args.temporada + 1) % 100:02d}: "
                  f"{len(nuevos)} jugados, {len(pend)} por jugar; {len(equipos_fd) - len(sin)} de {len(equipos_fd)} "
                  f"equipos emparejados con Transfermarkt")
            if sin:
                print(f"   Sin pareja (sin historial previo): {', '.join(sin)}")
        except Exception as e:
            print(f"   (football-data.org no disponible: {e})")

    tm = tm.sort_values(["fecha", "liga"]).reset_index(drop=True)
    tm["fecha"] = tm["fecha"].dt.date
    tm.to_csv(os.path.join(DATA, "europa_resultados.csv"), index=False, encoding="utf-8-sig")
    if len(pendientes):
        pendientes["fecha"] = pd.to_datetime(pendientes["fecha"]).dt.date
        pendientes.to_csv(os.path.join(DATA, "europa_pendientes.csv"), index=False, encoding="utf-8-sig")
    print(f"\n   Por competición: " + ", ".join(f"{k} {v:,}" for k, v in tm["liga"].value_counts().items()))
    print("   Guardado en datos/europa_resultados.csv" + (" y datos/europa_pendientes.csv" if len(pendientes) else ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
