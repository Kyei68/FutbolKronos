"""
Paso 2c: completar LaLiga 2026/27 con API-Football (Transfermarkt está pausado desde julio de 2026).

De cada partido terminado descarga la alineación, que incluye el once titular y el
entrenador de cada equipo. Con eso se completan:
  - los entrenadores de 2026/27 (unificar_equipos.py los usa para rellenar huecos), y
  - las ausencias de titulares habituales (crear_alineaciones.py).

Plan gratuito: 100 consultas al día. Cada ejecución gasta 1 consulta en la lista de partidos
y 1 por cada alineación que aún no esté guardada, hasta --max-consultas (90 por defecto).
Lo descargado se guarda en datos/originales/api_football/ y no se vuelve a pedir: si un día
no da para todo, basta con volver a ejecutarlo al día siguiente.

Clave: regístrate en https://dashboard.api-football.com y guarda tu clave en el archivo
api_football_clave.txt (en esta carpeta) o en la variable de entorno API_FOOTBALL_KEY.

Salida:  datos/api_football_laliga.csv   fecha, local, visitante (nombres de football-data),
                                          entrenadores y titulares de cada partido

Uso (desde esta carpeta):  python3 api_football.py [--temporada 2026] [--max-consultas 90]
"""
import argparse
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datos")
CACHE = os.path.join(DATA, "originales", "api_football")
URL = "https://v3.football.api-sports.io"
LALIGA = 140                                   # id de LaLiga en API-Football
TERMINADO = {"FT", "AET", "PEN"}


def api_key():
    key = os.environ.get("API_FOOTBALL_KEY")
    path = os.path.join(BASE, "api_football_clave.txt")
    if not key and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    if not key:
        sys.exit("ERROR: falta la clave de API-Football. Guárdala en api_football_clave.txt "
                 "(o en la variable API_FOOTBALL_KEY).")
    return key


class Api:
    def __init__(self, key, max_consultas):
        self.key, self.restantes, self.usadas = key, max_consultas, 0

    def get(self, ruta, **params):
        if self.restantes <= 0:
            raise RuntimeError("límite de consultas de esta ejecución alcanzado")
        q = "&".join(f"{k}={v}" for k, v in params.items())
        req = urllib.request.Request(f"{URL}/{ruta}?{q}", headers={"x-apisports-key": self.key})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
            dia = r.headers.get("x-ratelimit-requests-remaining")
        self.restantes -= 1
        self.usadas += 1
        if body.get("errors"):
            raise RuntimeError(f"API-Football: {body['errors']}")
        if dia is not None and int(dia) <= 0:
            self.restantes = 0
        time.sleep(0.25)
        return body["response"]


def map_teams(fx, fd):
    """Equipo de API-Football -> nombre de football-data, por votación (misma fecha ±1 y resultado)."""
    votos = {}
    for r in fx.itertuples(index=False):
        c = fd[(abs((fd["fecha"] - r.fecha).dt.days) <= 1)
               & (fd["goles_local"] == r.goles_local) & (fd["goles_visitante"] == r.goles_visitante)]
        for x in c.itertuples(index=False):
            votos.setdefault(r.local_api, Counter())[x.local] += 1
            votos.setdefault(r.visitante_api, Counter())[x.visitante] += 1
    return {k: c.most_common(1)[0][0] for k, c in votos.items() if c.most_common(1)[0][1] >= 2}


def tm_coach_names():
    """Último entrenador conocido (Transfermarkt) de cada equipo, por nombre de football-data."""
    pu = pd.read_csv(os.path.join(DATA, "partidos_unificados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["fecha", "local_id", "visitante_id", "entrenador_local", "entrenador_visitante"])
    eq = pd.read_csv(os.path.join(DATA, "equivalencias_equipos.csv"), encoding="utf-8-sig")
    id_a_fd = dict(zip(eq["equipo_id"], eq["nombre_football_data"]))
    filas = pd.concat([pu[["fecha", "local_id", "entrenador_local"]].set_axis(["fecha", "eq", "ent"], axis=1),
                       pu[["fecha", "visitante_id", "entrenador_visitante"]].set_axis(["fecha", "eq", "ent"], axis=1)])
    filas = filas.dropna().sort_values("fecha")
    return {id_a_fd.get(e): set(g["ent"].tail(60)) for e, g in filas.groupby("eq")}


def _tokens(nombre):
    sin_tildes = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    return [t for t in sin_tildes.replace(".", " ").lower().split() if len(t) > 1]


def same_coach(api_name, tm_names):
    """'H. Flick' ≈ 'Hansi Flick', 'Claudio' ≈ 'Claudio Giráldez', 'Inigo Perez' ≈ 'Íñigo Pérez',
    'José Bordalás' ≈ 'Pepe Bordalás'; pero 'Luis García' ≠ 'Xavier García Pimienta'.
    Sin tener en cuenta las tildes."""
    a = _tokens(api_name)
    for n in tm_names:
        t = _tokens(n)
        if not a or not t:
            continue
        if len(a) == 1 or len(t) == 1:                       # solo un nombre o un apellido
            if a[-1] in t or t[-1] in a:
                return n
        elif a[-1] == t[-1] or (a[-1] in t and a[0][0] == t[0][0]):
            return n
    return None


def main():
    ap = argparse.ArgumentParser(description="Completar LaLiga con API-Football")
    ap.add_argument("--temporada", type=int, default=2026, help="año de inicio de la temporada")
    ap.add_argument("--max-consultas", type=int, default=90)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    api = Api(api_key(), args.max_consultas)
    print(f"Paso 2c · API-Football · LaLiga {args.temporada}/{(args.temporada + 1) % 100:02d}\n")

    fixtures = api.get("fixtures", league=LALIGA, season=args.temporada)
    with open(os.path.join(CACHE, f"fixtures_{args.temporada}.json"), "w", encoding="utf-8") as f:
        json.dump(fixtures, f)
    fx = pd.DataFrame([{
        "fixture": x["fixture"]["id"],
        "fecha": pd.Timestamp(x["fixture"]["date"]).tz_convert("Europe/Madrid").tz_localize(None).normalize(),
        "estado": x["fixture"]["status"]["short"],
        "local_api": x["teams"]["home"]["name"], "visitante_api": x["teams"]["away"]["name"],
        "goles_local": x["goals"]["home"], "goles_visitante": x["goals"]["away"]} for x in fixtures])
    fx = fx[fx["estado"].isin(TERMINADO)].sort_values("fecha").reset_index(drop=True)
    print(f"   partidos terminados: {len(fx)}")

    pendientes = [f for f in fx["fixture"] if not os.path.exists(os.path.join(CACHE, f"lineups_{f}.json"))]
    print(f"   alineaciones por descargar: {len(pendientes)} (consultas disponibles hoy: {api.restantes})")
    for fid in pendientes:
        try:
            lu = api.get("fixtures/lineups", fixture=fid)
        except (RuntimeError, urllib.error.URLError) as e:
            print(f"   parada: {e}")
            break
        if lu:
            with open(os.path.join(CACHE, f"lineups_{fid}.json"), "w", encoding="utf-8") as f:
                json.dump(lu, f)

    fd = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["liga", "temporada_inicio", "fecha", "local", "visitante", "goles_local", "goles_visitante"])
    fd = fd[(fd["liga"] == "SP1") & (fd["temporada_inicio"] == args.temporada)].copy()
    fd["fecha"] = pd.to_datetime(fd["fecha"])
    mapa = map_teams(fx, fd)
    sin = sorted(set(fx["local_api"]) - set(mapa))
    if sin:
        print(f"   AVISO: equipos sin equivalencia en football-data: {', '.join(sin)}")
    tm = tm_coach_names()

    filas = []
    for r in fx.itertuples(index=False):
        path = os.path.join(CACHE, f"lineups_{r.fixture}.json")
        if not os.path.exists(path) or r.local_api not in mapa or r.visitante_api not in mapa:
            continue
        with open(path, encoding="utf-8") as f:
            lu = {x["team"]["name"]: x for x in json.load(f)}
        fila = {"fecha": r.fecha.date(), "local": mapa[r.local_api], "visitante": mapa[r.visitante_api],
                "fixture": r.fixture}
        for lado, eq_api in [("local", r.local_api), ("visitante", r.visitante_api)]:
            x = lu.get(eq_api, {})
            coach = (x.get("coach") or {}).get("name")
            if coach:
                coach = same_coach(coach, tm.get(fila[lado], set())) or coach
            fila[f"entrenador_{lado}"] = coach
            fila[f"titulares_{lado}"] = ";".join(str(p["player"]["id"]) for p in x.get("startXI", []))
        filas.append(fila)
    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(DATA, "api_football_laliga.csv"), index=False, encoding="utf-8-sig")
    print(f"\n   {len(out)} partidos con alineación y entrenador · consultas usadas: {api.usadas}")
    if len(out) < len(fx):
        print("   Faltan alineaciones: vuelve a ejecutarlo mañana para completar.")
    print("   Guardado en datos/api_football_laliga.csv")
    print("\n   Siguiente: unificar_equipos.py → crear_velas.py → crear_secuencias.py → "
          "crear_alineaciones.py → crear_secuencias_ligas.py → entrenar.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
