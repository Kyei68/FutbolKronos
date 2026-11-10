"""
Paso 4c: indicadores de alineación por partido (Transfermarkt), para las 11 primeras
divisiones que cubre Transfermarkt desde 2012/13.

Para cada equipo y partido calcula, solo con lo que se sabe al publicarse las alineaciones
(una hora antes del partido, como las cuotas de cierre de las casas):

  ausencias   cuántos de sus 11 titulares habituales (los que más han sido titulares en sus
              10 partidos de liga anteriores) NO salen de inicio: lesiones, sanciones, rotaciones.
  valor_xi    suma del valor de mercado de los 11 titulares en esa fecha (última tasación
              anterior al partido). Solo si existe datos/originales/player_valuations.csv.gz.

Los partidos de Transfermarkt se emparejan con los de football-data: cada club se asigna al
nombre de football-data con el que más veces coincide (misma división, fecha y resultado).

Entrada:  datos/originales/games.csv.gz, game_lineups.csv.gz, [player_valuations.csv.gz]
          datos/ligas_resultados.csv   (descargar_ligas.py)
Salida:   datos/alineaciones_ligas.csv   liga, fecha, local, visitante + indicadores

Uso (desde esta carpeta):  python3 crear_alineaciones.py
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datos")
RAW = os.path.join(DATA, "originales")

TM_A_FD = {"ES1": "SP1", "GB1": "E0", "L1": "D1", "IT1": "I1", "FR1": "F1", "NL1": "N1",
           "BE1": "B1", "PO1": "P1", "TR1": "T1", "GR1": "G1", "SC1": "SC0"}
N_PREVIOS = 10          # partidos anteriores para decidir quiénes son los titulares habituales


def load_games():
    g = pd.read_csv(os.path.join(RAW, "games.csv.gz"),
                    usecols=["game_id", "competition_id", "date", "home_club_id", "away_club_id",
                             "home_club_goals", "away_club_goals"])
    g = g[g["competition_id"].isin(TM_A_FD)].copy()
    g["liga"] = g["competition_id"].map(TM_A_FD)
    g["fecha"] = pd.to_datetime(g["date"])
    return g.dropna(subset=["home_club_goals", "away_club_goals"])


def map_clubs(g, fd):
    """Club de Transfermarkt -> nombre de football-data, por votación."""
    votos = {}
    cand = g.merge(fd, on=["liga", "fecha"], suffixes=("", "_fd"))
    cand = cand[(cand["home_club_goals"] == cand["goles_local"]) & (cand["away_club_goals"] == cand["goles_visitante"])]
    for r in cand.itertuples(index=False):
        votos.setdefault((r.liga, r.home_club_id), Counter())[r.local] += 1
        votos.setdefault((r.liga, r.away_club_id), Counter())[r.visitante] += 1
    mapa = {}
    for k, c in votos.items():
        nombre, n = c.most_common(1)[0]
        if n >= 3 and n > 0.5 * sum(c.values()):     # mayoría clara (descarta coincidencias casuales)
            mapa[k] = nombre
    return mapa


def ausencias_de(historia, xi):
    """(titulares habituales que faltan, parte de las titularidades recientes que falta).
    La segunda pondera por importancia: quien fue titular en los 10 últimos partidos pesa 10
    veces más que quien lo fue en uno, sin importar su edad ni su valor de mercado."""
    cuenta = Counter(p for s in historia[-N_PREVIOS:] for p in s)
    hab = {p for p, _ in cuenta.most_common(11)}
    total = sum(cuenta.values())
    return len(hab - xi), sum(n for p, n in cuenta.items() if p not in xi) / total


def habituales_missing(lineups, g):
    """Para cada (partido, club): ausencias (conteo y ponderada) respecto a sus partidos previos."""
    st = lineups[lineups["type"] == "starting_lineup"][["game_id", "club_id", "player_id"]]
    st = st.merge(g[["game_id", "fecha"]], on="game_id")
    res = {}
    for club, gc in st.groupby("club_id"):
        partidos = gc.groupby(["fecha", "game_id"])["player_id"].apply(set).sort_index()
        historia = []
        for (fecha, gid), xi in partidos.items():
            if len(historia) >= 3:
                res[(gid, club)] = ausencias_de(historia, xi)
            historia.append(xi)
    return res


def valor_xi(lineups, g):
    """Valor del once: (actual, corregido por edad). El corregido usa, para cada titular, su
    valor MÁXIMO de los últimos 3 años: un veterano que valía 40 M€ hace dos años y ahora 5 M€
    conserva buena parte de su nivel. Solo se usan tasaciones anteriores al partido."""
    path = os.path.join(RAW, "player_valuations.csv.gz")
    if not os.path.exists(path):
        print("   (sin player_valuations.csv.gz: se omite el valor del once)")
        return {}, {}
    v = pd.read_csv(path, usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"])
    v = v.dropna().sort_values(["player_id", "date"]).reset_index(drop=True)
    v["max3"] = (v.groupby("player_id").rolling("1095D", on="date")["market_value_in_eur"].max()
                 .reset_index(drop=True))
    v = v.sort_values("date")
    st = lineups[lineups["type"] == "starting_lineup"][["game_id", "club_id", "player_id"]]
    st = st.merge(g[["game_id", "fecha"]], on="game_id").sort_values("fecha")
    st = pd.merge_asof(st, v.rename(columns={"date": "fecha"}), on="fecha", by="player_id",
                       direction="backward", allow_exact_matches=False)
    # jugadores sin tasación previa: valor bajo (suelen ser canteranos)
    for c in ["market_value_in_eur", "max3"]:
        st[c] = st[c].fillna(100_000)
    tot = st.groupby(["game_id", "club_id"]).agg(actual=("market_value_in_eur", "sum"),
                                                 max3=("max3", "sum"), n=("player_id", "size"))
    tot = tot[tot["n"] >= 10]
    return tot["actual"].to_dict(), tot["max3"].to_dict()


def api_ausencias():
    """Ausencias de LaLiga 2026/27 con las alineaciones de api_football.py (mismo cálculo).
    Los ids de jugador de API-Football no coinciden con los de Transfermarkt, así que los
    habituales se calculan solo con partidos de API-Football (desde el 4.º de cada equipo)."""
    path = os.path.join(DATA, "api_football_laliga.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    api = pd.read_csv(path, encoding="utf-8-sig", dtype={"titulares_local": str, "titulares_visitante": str})
    api["fecha"] = pd.to_datetime(api["fecha"])
    filas = pd.concat([
        api[["fixture", "fecha", "local", "titulares_local"]].set_axis(["fixture", "fecha", "eq", "xi"], axis=1).assign(lado="local"),
        api[["fixture", "fecha", "visitante", "titulares_visitante"]].set_axis(["fixture", "fecha", "eq", "xi"], axis=1).assign(lado="visitante"),
    ]).sort_values("fecha")
    aus = {}
    for eq, g in filas.groupby("eq"):
        historia = []
        for r in g.itertuples(index=False):
            xi = set(str(r.xi).split(";")) if isinstance(r.xi, str) and r.xi else set()
            if len(xi) < 10:
                continue
            if len(historia) >= 3:
                aus[(r.fixture, r.lado)] = ausencias_de(historia, xi)
            historia.append(xi)
    return pd.DataFrame({
        "liga": "SP1", "fecha": api["fecha"].dt.date, "local": api["local"], "visitante": api["visitante"],
        **columnas(api["fixture"], api["fixture"], aus, lambda f: (f, "local"), lambda f: (f, "visitante"))})


def columnas(ids_l, ids_v, aus, key_l, key_v, val_actual=None, val_max3=None):
    """Columnas de indicadores para local y visitante."""
    out = {}
    for lado, ids, key in [("local", ids_l, key_l), ("visitante", ids_v, key_v)]:
        ids = list(ids)
        a = [aus.get(key(i)) for i in ids]
        out[f"ausencias_{lado}"] = [x[0] if x else None for x in a]
        out[f"ausencia_ponderada_{lado}"] = [round(x[1], 4) if x else None for x in a]
        out[f"valor_xi_{lado}"] = [(val_max3 or {}).get(key(i)) for i in ids]
        out[f"valor_xi_actual_{lado}"] = [(val_actual or {}).get(key(i)) for i in ids]
    return out


def main():
    print("Paso 4c · Indicadores de alineación (Transfermarkt)\n")
    g = load_games()
    fd = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False,
                     usecols=["liga", "fecha", "local", "visitante", "goles_local", "goles_visitante"])
    fd["fecha"] = pd.to_datetime(fd["fecha"])
    fd = fd[fd["liga"].isin(TM_A_FD.values())]

    mapa = map_clubs(g, fd)
    g["local"] = [mapa.get((l, c)) for l, c in zip(g["liga"], g["home_club_id"])]
    g["visitante"] = [mapa.get((l, c)) for l, c in zip(g["liga"], g["away_club_id"])]
    g = g.dropna(subset=["local", "visitante"])
    # la fecha puede variar un día entre fuentes: se busca en fecha exacta y ±1
    m = g.merge(fd[["liga", "local", "visitante", "fecha"]].rename(columns={"fecha": "fecha_fd"}),
                on=["liga", "local", "visitante"])
    m["dif"] = (m["fecha_fd"] - m["fecha"]).abs()
    m = m[m["dif"] <= pd.Timedelta(days=1)].sort_values("dif").drop_duplicates("game_id")
    g = m.drop(columns="dif").reset_index(drop=True)

    print("   Leyendo alineaciones…")
    lineups = pd.read_csv(os.path.join(RAW, "game_lineups.csv.gz"), usecols=["game_id", "club_id", "player_id", "type"])
    lineups = lineups[lineups["game_id"].isin(g["game_id"])]
    aus = habituales_missing(lineups, g)
    val_actual, val_max3 = valor_xi(lineups, g)

    out = pd.DataFrame({
        "liga": g["liga"], "fecha": g["fecha_fd"].dt.date, "local": g["local"], "visitante": g["visitante"],
        **columnas(zip(g["game_id"], g["home_club_id"]), zip(g["game_id"], g["away_club_id"]), aus,
                   lambda k: k, lambda k: k, val_actual, val_max3)})
    api = api_ausencias()
    if len(api):
        out = pd.concat([out, api], ignore_index=True)
        print(f"   añadidos {len(api)} partidos de LaLiga desde API-Football")
    out.to_csv(os.path.join(DATA, "alineaciones_ligas.csv"), index=False, encoding="utf-8-sig")

    total = fd[fd["fecha"].dt.year >= 2012].groupby("liga").size()
    hechos = out.groupby("liga").size()
    print(f"   Partidos emparejados con football-data ({len(mapa)} clubes asignados):")
    for l in TM_A_FD.values():
        print(f"     {l:<4} {hechos.get(l, 0):>6,}  (de {total.get(l, 0):,} en football-data desde 2012)")
    a = pd.concat([out["ausencias_local"], out["ausencias_visitante"]]).dropna()
    print(f"\n   Ausencias de titulares habituales: media {a.mean():.2f}; "
          + ", ".join(f"{k}: {v * 100:.0f} %" for k, v in a.clip(upper=5).value_counts(normalize=True).sort_index().items()))
    p = pd.concat([out["ausencia_ponderada_local"], out["ausencia_ponderada_visitante"]]).dropna()
    print(f"   Ausencia ponderada (parte de las titularidades recientes que falta): "
          f"media {p.mean():.2f}, percentiles 10/50/90: {p.quantile(0.1):.2f} / {p.median():.2f} / {p.quantile(0.9):.2f}")
    if val_max3:
        print(f"   Valor del once (máx. 3 años): mediana {np.nanmedian(out['valor_xi_local']) / 1e6:.0f} M€ · "
              f"actual: {np.nanmedian(out['valor_xi_actual_local']) / 1e6:.0f} M€")
    print("\nGuardado en datos/alineaciones_ligas.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
