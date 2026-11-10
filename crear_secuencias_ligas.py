"""
Paso 4b: velas, Elo y secuencias de TODAS las ligas (para preentrenar con más datos).

Igual que crear_velas.py + crear_secuencias.py, pero con las 22 divisiones de
descargar_ligas.py. Diferencias:

- Elo por país: las divisiones de un mismo país comparten escala (los equipos que suben
  y bajan llevan su rating), pero cada división solo "olvida" hacia su propia media.
- El nivel Elo que ve el modelo (contexto) es RELATIVO a su división y temporada:
  1.500 = la media de su liga. Así "ser de los buenos de tu liga" significa lo mismo en
  Escocia que en España.
- El s2 del token [INICIO] dice de qué liga es el partido (0-21).
- Los equipos españoles usan los mismos identificadores que el resto del proyecto, y las
  velas de Primera conservan el dato del entrenador (Transfermarkt).
- Los partidos de Primera se emparejan con secuencias_partidos.csv para traer las
  referencias (Elo de crear_velas.py y casas de apuestas): la evaluación no cambia.

Entrada:  datos/ligas_resultados.csv (descargar_ligas.py), datos/equivalencias_equipos.csv,
          datos/velas.csv y datos/secuencias_partidos.csv (pasos 3 y 4)
Salida:   datos/secuencias_ligas.npz, datos/secuencias_ligas_partidos.csv,
          datos/velas_ligas.csv

- En el token [PARTIDO], la primera casilla del contexto (la del entrenador) lleva la
  diferencia de Elo local − visitante en escalones de 20 puntos (1-59; 30 = igualados).
  El dato del entrenador sigue en las velas del historial. Con --sin-elo-diferencia se
  generan las secuencias como en la primera versión.
- Alineaciones (opcional; datos/alineaciones_ligas.csv, de crear_alineaciones.py; primeras
  divisiones desde 2012/13). Por defecto NO se usan: probadas en noviembre de 2026, ninguna
  variante mejoró la validación (ver ESQUEMA.md, paso 8). Si se activan:
  --ausencias ponderada|conteo: la casilla del día de la semana lleva las ausencias de
  titulares de ese equipo en ese partido (0 = desconocido, 1-6 = de menos a más). "ponderada"
  es la parte de las titularidades recientes que falta en el once (quien juega siempre pesa
  más, tenga la edad y el valor que tenga); "conteo", cuántos de los 11 habituales faltan.
  En [PARTIDO], la del local va en esa casilla y la del visitante en la del mes.
  --valor max3|actual: la primera casilla de [PARTIDO] lleva log(valor local / visitante) del
  once titular en escalones de 0,1 (1-59; 30 = iguales; 0 = desconocido) en lugar de la
  diferencia de Elo. "max3" usa el máximo de los últimos 3 años de cada titular (corrige que
  los veteranos tasen poco); "actual", la última tasación.

Uso (desde esta carpeta):
  python3 crear_secuencias_ligas.py [--ausencias no|ponderada|conteo] [--valor no|max3|actual]
                                    [--sin-elo-diferencia]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import futbol_tokenizer as tk
from crear_secuencias import N_HIST, SEQ_LEN
N_HIST_DEF = N_HIST
from crear_velas import goal_mult
from descargar_ligas import DIVISIONES

DATA = os.path.join(BASE, "datos")
# Competiciones europeas (crear_europa.py): no tienen país; su "liga" es la competición
EUROPA = {"CL": "Champions League", "CLQ": "Champions League · previas", "EL": "Europa League",
          "ELQ": "Europa League · previas", "ECL": "Conference League", "ECLQ": "Conference League · previas"}
LIGAS = list(DIVISIONES) + list(EUROPA)       # posición = id de liga en el s2 de [INICIO]
ELO_START, TIER_GAP = 1500.0, 150.0           # una división más abajo empieza 150 puntos por debajo
K, HFA, REGRESS = 20.0, 60.0, 0.1
ELO_DIFF_STEP = 20.0                          # escalón de la diferencia de Elo en [PARTIDO]
EURO_START, EURO_K = 1400.0, 30.0             # equipos sin liga en los datos: Elo solo europeo
OFFSET_SEMIVIDA = 3.0                         # temporadas: peso de los cruces europeos antiguos


def pais_de(liga):
    return DIVISIONES[liga][0] if liga in DIVISIONES else "EUR"


def nombre_competicion(liga):
    return DIVISIONES[liga][2] if liga in DIVISIONES else EUROPA[liga]


def team_ids(df):
    """Identificador de equipo: el de equivalencias para España, "<país>:<nombre>" para el resto.
    Las filas que ya traen identificador (competiciones europeas) se respetan."""
    eq = pd.read_csv(os.path.join(DATA, "equivalencias_equipos.csv"), encoding="utf-8-sig")
    esp = dict(zip(eq["nombre_football_data"].str.strip(), eq["equipo_id"]))
    pais = df["liga"].map(pais_de)
    for side in ["local", "visitante"]:
        nombre = df[side]
        mapped = nombre.map(esp)
        nuevo = np.where((pais == "ESP") & mapped.notna(), mapped, pais + ":" + nombre)
        if f"{side}_id" in df:
            nuevo = np.where(df[f"{side}_id"].notna(), df[f"{side}_id"], nuevo)
        df[f"{side}_id"] = nuevo
    return df


def ajustar_desfases(registros, t):
    """Desfase de la escala Elo de cada país, ajustado con los cruces europeos de temporadas
    anteriores a t (los recientes pesan más). Media cero: solo importan las diferencias."""
    r = [x for x in registros if x[0] < t]
    if not r:
        return {}
    paises = sorted({x[3] for x in r} | {x[4] for x in r})
    ix = {p: i for i, p in enumerate(paises)}
    tt, rh, ra, ch, ca, sc = (np.array(v) for v in zip(*r))
    ih, ia = np.array([ix[c] for c in ch]), np.array([ix[c] for c in ca])
    w = 0.5 ** ((t - 1 - tt.astype(float)) / OFFSET_SEMIVIDA)
    off = np.zeros(len(paises))
    for _ in range(400):                          # descenso de gradiente sobre el error cuadrático
        d = rh.astype(float) + off[ih] + HFA - ra.astype(float) - off[ia]
        e = 1 / (1 + 10 ** (-d / 400))
        g = w * (e - sc.astype(float)) * e * (1 - e) * np.log(10) / 400
        grad = np.bincount(ih, g, len(paises)) - np.bincount(ia, g, len(paises))
        off -= 4e5 * grad / w.sum() + 1e-3 * off
        off -= off.mean()
    return dict(zip(paises, off))


def run_elo(df):
    """Elo por país. Devuelve rating antes del partido (local, visitante) y la media de la
    división en esa temporada (para el Elo relativo).
    Partidos europeos: no cambian el Elo de liga. Cada equipo usa su Elo de liga + el desfase de
    su país (ajustado con cruces europeos anteriores); los equipos de países que no están en
    los datos llevan un Elo propio que solo se mueve con sus partidos europeos."""
    pais = df["liga"].map(pais_de).to_numpy()
    teams = {}                                   # (liga, temporada) -> equipos
    for (liga, t), g in df.groupby(["liga", "temporada_inicio"]):
        teams[(liga, t)] = set(g["local_id"]) | set(g["visitante_id"])
    divs_of = {}                                 # (país, temporada) -> divisiones
    for (liga, t) in teams:
        if liga in DIVISIONES:
            divs_of.setdefault((DIVISIONES[liga][0], t), []).append(liga)

    ratings, last_div = {}, {}
    estado = {"desfase": {}, "temporada": None}
    euro, registros = {}, []
    div_mean = {}
    started = set()
    pre_h, pre_a = np.empty(len(df)), np.empty(len(df))

    def start_season(p, t):
        divs = divs_of[(p, t)]
        # equipos que salieron de cada división hacia abajo (para colocar a los nuevos)
        bajaron = {}
        for liga in divs:
            prev = teams.get((liga, t - 1), set())
            fuera = [x for x in prev if x in ratings and x not in teams[(liga, t)]
                     and last_div.get(x) == liga]
            if fuera:
                bajaron[liga] = np.mean(sorted(ratings[x] for x in fuera)[:max(1, len(fuera) // 2)])
        for liga in divs:
            base = bajaron.get(liga, ELO_START - TIER_GAP * (DIVISIONES[liga][1] - 1))
            for x in teams[(liga, t)]:
                if x not in ratings:
                    ratings[x] = base
                elif last_div.get(x) != liga and DIVISIONES[liga][1] < DIVISIONES[last_div[x]][1]:
                    ratings[x] = min(ratings[x], base)              # recién ascendido
            m = np.mean([ratings[x] for x in teams[(liga, t)]])
            for x in teams[(liga, t)]:
                ratings[x] += REGRESS * (m - ratings[x])
                last_div[x] = liga
            div_mean[(liga, t)] = float(np.mean([ratings[x] for x in teams[(liga, t)]]))

    def elo_europeo(x):
        if x in ratings and x in last_div:
            c = DIVISIONES[last_div[x]][0]
            return ratings[x] + estado["desfase"].get(c, 0.0), c
        return euro.setdefault(x, EURO_START), None

    for i, (p, t, h, a, gh, ga) in enumerate(zip(pais, df["temporada_inicio"], df["local_id"],
                                                 df["visitante_id"], df["goles_local"], df["goles_visitante"])):
        if p == "EUR":
            if t != estado["temporada"]:
                estado["desfase"], estado["temporada"] = ajustar_desfases(registros, t), t
            (rh, ch), (ra, ca) = elo_europeo(h), elo_europeo(a)
            pre_h[i], pre_a[i] = rh, ra
            if gh != gh or ga != ga:
                continue
            score_h = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
            exp_h = 1 / (1 + 10 ** ((ra - (rh + HFA)) / 400))
            delta = EURO_K * goal_mult(gh - ga) * (score_h - exp_h)
            if ch is None:
                euro[h] = rh + delta
            if ca is None:
                euro[a] = ra - delta
            if ch is not None and ca is not None and ch != ca:
                d = estado["desfase"]
                registros.append((t, rh - d.get(ch, 0.0), ra - d.get(ca, 0.0), ch, ca, score_h))
            continue
        if (p, t) not in started:
            start_season(p, t)
            started.add((p, t))
        rh, ra = ratings[h], ratings[a]
        pre_h[i], pre_a[i] = rh, ra
        if gh != gh or ga != ga:                     # por jugar: no cambia el Elo
            continue
        exp_h = 1 / (1 + 10 ** ((ra - (rh + HFA)) / 400))
        score_h = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        delta = K * goal_mult(gh - ga) * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    # media de cada competición europea y temporada (para el Elo relativo)
    eur = pais == "EUR"
    if eur.any():
        e = pd.DataFrame({"l": df["liga"].to_numpy()[eur], "t": df["temporada_inicio"].to_numpy()[eur],
                          "v": (pre_h[eur] + pre_a[eur]) / 2})
        div_mean.update(e.groupby(["l", "t"])["v"].mean().to_dict())
    run_elo.desfases = estado["desfase"]
    mean = np.array([div_mean[(l, t)] for l, t in zip(df["liga"], df["temporada_inicio"])])
    return pre_h, pre_a, mean


PONDERADA_EDGES = [0.22, 0.29, 0.35, 0.42, 0.52]   # ~ percentiles 10/30/50/70/90


def aus_code(aus, ponderada):
    """Ausencias -> casilla de 7 valores (0 = desconocido, 1-6 de menos a más)."""
    aus = np.asarray(aus, dtype=float)
    nivel = np.searchsorted(PONDERADA_EDGES, aus) if ponderada else np.clip(aus, 0, 5)
    return np.where(np.isnan(aus), 0, 1 + nivel).astype(np.int8)


def leer_resultados(europa=False):
    """Partidos de ligas_resultados.csv y, con europa=True, de europa_resultados.csv (crear_europa.py)."""
    df = pd.read_csv(os.path.join(DATA, "ligas_resultados.csv"), encoding="utf-8-sig", low_memory=False)
    path = os.path.join(DATA, "europa_resultados.csv")
    if europa and os.path.exists(path):
        eu = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df = pd.concat([df, eu.drop(columns=["ronda"], errors="ignore")], ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


REST_EDGES = [2, 3, 4, 6, 13]       # días: ≤2 | 3 | 4 | 5-6 | 7-13 | ≥14 (o primer partido)


def descanso_code(velas):
    """Días desde el partido anterior de cada equipo, contando copa y Europa cuando se conocen."""
    path = os.path.join(DATA, "partidos_otras_competiciones.csv")
    fechas = velas[["equipo_id", "fecha"]]
    if os.path.exists(path):
        otras = pd.read_csv(path, encoding="utf-8-sig", usecols=["equipo_id", "fecha"])
        otras["fecha"] = pd.to_datetime(otras["fecha"])
        fechas = pd.concat([fechas, otras])
    fechas = fechas.drop_duplicates().rename(columns={"fecha": "anterior"}).sort_values("anterior")
    q = velas[["equipo_id", "fecha"]].reset_index().sort_values("fecha")
    q = pd.merge_asof(q, fechas, left_on="fecha", right_on="anterior", by="equipo_id",
                      allow_exact_matches=False).sort_values("index")
    dias = (q["fecha"] - q["anterior"]).dt.days.to_numpy(dtype=float)
    nivel = np.where(np.isnan(dias), len(REST_EDGES), np.searchsorted(REST_EDGES, dias))
    return (1 + nivel).astype(np.int8)                 # 1-6 (0 queda para "desconocido")


def rojas_code(favor, contra):
    code = np.where(np.isnan(favor) | np.isnan(contra), 0, 1)
    code = np.where((favor > 0) & ~(contra > 0), 2, code)
    code = np.where(~(favor > 0) & (contra > 0), 3, code)
    code = np.where((favor > 0) & (contra > 0), 4, code)
    return code.astype(np.int8)


def construir(df, ausencias="no", valor="no", elo_diferencia=True, verbose=True,
              rojas=False, descanso=False, n_hist=N_HIST, europa=False):
    """Velas y secuencias de todos los partidos de df. Los partidos sin goles (por jugar) se
    tratan como el [PARTIDO] a predecir: no actualizan el Elo y su [MARCADOR] queda en 0-0.
    rojas: la casilla del mes de cada vela del historial dice si hubo expulsiones
           (0 sin datos, 1 ninguna, 2 propia, 3 del rival, 4 ambas). Nunca la del partido a predecir.
    descanso: la casilla del día de la semana lleva los días de descanso antes de cada partido
           (contando copa y Europa desde 2012/13); en [PARTIDO], el del local, y el del
           visitante va en la casilla del mes.
    n_hist: velas de cada equipo que ve el modelo (20 por defecto).
    europa: solo informativo (los partidos europeos entran si df los trae: leer_resultados(europa=True)).
    Devuelve (S1, S2, ST, df, velas, n_hist), con df ya ordenado como las secuencias."""
    N_HIST, SEQ_LEN = n_hist, 2 * n_hist + 4
    log = print if verbose else (lambda *a, **k: None)
    ponderada = ausencias == "ponderada"
    df = df.sort_values(["fecha", "liga", "local"]).reset_index(drop=True)
    df = team_ids(df)
    # un mismo equipo no puede jugar dos veces el mismo día (filas duplicadas en origen)
    df = df.drop_duplicates(["fecha", "local_id", "visitante_id"]).reset_index(drop=True)
    jugado = df["goles_local"].notna() & df["goles_visitante"].notna()
    df["resultado"] = np.where(jugado, np.select([df["goles_local"] > df["goles_visitante"],
                                                  df["goles_local"] == df["goles_visitante"]], ["1", "X"], "2"), None)
    log(f"   {len(df):,} partidos de {df['liga'].nunique()} divisiones"
        f"{f' ({(~jugado).sum()} por jugar)' if (~jugado).any() else ''}")

    ali_path = os.path.join(DATA, "alineaciones_ligas.csv")
    if ausencias != "no" or valor != "no":
        if os.path.exists(ali_path):
            ali = pd.read_csv(ali_path, encoding="utf-8-sig")
            ali["fecha"] = pd.to_datetime(ali["fecha"])
            df = df.merge(ali.drop_duplicates(["liga", "fecha", "local", "visitante"]),
                          on=["liga", "fecha", "local", "visitante"], how="left")
            log(f"   con alineación: {df['ausencias_local'].notna().sum():,} partidos")
    aus_col = "ausencia_ponderada" if ponderada else "ausencias"
    val_col = {"max3": "valor_xi", "actual": "valor_xi_actual", "no": None}[valor]
    for lado in ["local", "visitante"]:
        df[f"aus_{lado}"] = df[f"{aus_col}_{lado}"] if f"{aus_col}_{lado}" in df else np.nan
        df[f"val_{lado}"] = df[f"{val_col}_{lado}"] if val_col and f"{val_col}_{lado}" in df else np.nan
    con_valor = df["val_local"].notna().any()

    pre_h, pre_a, mean = run_elo(df)
    df["elo_local_antes"], df["elo_visitante_antes"], df["elo_media_division"] = pre_h, pre_a, mean

    # Velas: dos por partido
    def side(o, r, is_home):
        return pd.DataFrame({
            "partido": df.index, "fecha": df["fecha"], "liga": df["liga"],
            "temporada_inicio": df["temporada_inicio"], "equipo_id": df[f"{o}_id"], "rival_id": df[f"{r}_id"],
            "es_local": int(is_home), "goles_favor": df[f"goles_{o}"], "goles_contra": df[f"goles_{r}"],
            "tiros_puerta_favor": df[f"tiros_puerta_{o}"], "tiros_puerta_contra": df[f"tiros_puerta_{r}"],
            "elo_antes": df[f"elo_{o}_antes"], "elo_rival_antes": df[f"elo_{r}_antes"],
            "elo_relativo": 1500 + df[f"elo_{o}_antes"] - df["elo_media_division"],
            "ausencias": df[f"aus_{o}"],
            "rojas_favor": df[f"rojas_{o}"] if f"rojas_{o}" in df else np.nan,
            "rojas_contra": df[f"rojas_{r}"] if f"rojas_{r}" in df else np.nan})
    velas = pd.concat([side("local", "visitante", True), side("visitante", "local", False)], ignore_index=True)
    velas = velas.sort_values(["equipo_id", "fecha"]).reset_index(drop=True)
    velas["pos"] = velas.groupby("equipo_id").cumcount()
    velas["partido_n_temporada"] = velas.groupby(["equipo_id", "liga", "temporada_inicio"]).cumcount() + 1

    # Entrenador: solo se conoce en Primera (Transfermarkt, desde 2012/13)
    v_es = pd.read_csv(os.path.join(DATA, "velas.csv"), encoding="utf-8-sig", low_memory=False,
                       usecols=["fecha", "equipo_id", "partido_n_entrenador"])
    v_es["fecha"] = pd.to_datetime(v_es["fecha"])
    velas = velas.merge(v_es.drop_duplicates(["fecha", "equipo_id"]), on=["fecha", "equipo_id"], how="left")
    velas.loc[velas["liga"] != "SP1", "partido_n_entrenador"] = np.nan
    # partidos por jugar de Primera: el entrenador sigue (su partido siguiente)
    por_jugar = (velas["liga"] == "SP1") & velas["goles_favor"].isna()
    previo = velas.groupby("equipo_id")["partido_n_entrenador"].shift()
    velas.loc[por_jugar, "partido_n_entrenador"] = previo[por_jugar] + 1

    log("   Tokenizando velas…")
    n = len(velas)
    s1 = np.array([tk.score_id(a, b) if a == a and b == b else 0                 # por jugar: 0-0
                   for a, b in zip(velas["goles_favor"], velas["goles_contra"])], dtype=np.int8)
    rival = np.array([tk.rival_level(a, b) for a, b in zip(velas["elo_antes"], velas["elo_rival_antes"])], dtype=np.int8)
    dom = np.array([tk.dominance(a, b) for a, b in zip(velas["tiros_puerta_favor"], velas["tiros_puerta_contra"])],
                   dtype=np.int8)
    s2 = (velas["es_local"].to_numpy() * tk.N_RIVAL * tk.N_DOM + rival * tk.N_DOM + dom).astype(np.int8)
    st = np.array([tk.stamp(c, e, f.weekday(), j, f.month) for c, e, f, j in
                   zip(velas["partido_n_entrenador"], velas["elo_relativo"], velas["fecha"],
                       velas["partido_n_temporada"])], dtype=np.int8)
    if ausencias != "no":
        st[:, 2] = aus_code(velas["ausencias"].to_numpy(), ponderada)          # casilla del día de la semana
    mes = st[:, 4].copy()
    if descanso:
        st[:, 2] = descanso_code(velas)
    if rojas:
        st[:, 4] = rojas_code(velas["rojas_favor"].to_numpy(dtype=float), velas["rojas_contra"].to_numpy(dtype=float))

    by_team = {t: g.index.to_numpy() for t, g in velas.groupby("equipo_id")}
    home_row = np.empty(len(df), dtype=np.int64)
    away_row = np.empty(len(df), dtype=np.int64)
    is_home = velas["es_local"].to_numpy() == 1
    home_row[velas.loc[is_home, "partido"].to_numpy()] = np.where(is_home)[0]
    away_row[velas.loc[~is_home, "partido"].to_numpy()] = np.where(~is_home)[0]
    pos = velas["pos"].to_numpy()
    team = velas["equipo_id"].to_numpy()

    log("   Construyendo secuencias…")
    M = len(df)
    S1 = np.full((M, SEQ_LEN), tk.S1_SIN_HISTORIAL, dtype=np.int8)
    S2 = np.full((M, SEQ_LEN), tk.S2_ESPECIAL, dtype=np.int8)
    ST = np.zeros((M, SEQ_LEN, 5), dtype=np.int8)
    S1[:, 0] = tk.S1_INICIO
    S2[:, 0] = df["liga"].map({l: i for i, l in enumerate(LIGAS)}).to_numpy()
    S1[:, N_HIST + 1] = tk.S1_SEP
    n_hist = np.zeros((M, 2), dtype=np.int16)
    for m in range(M):
        for j, r in enumerate((home_row[m], away_row[m])):
            idx = by_team[team[r]][max(0, pos[r] - N_HIST):pos[r]]
            end = 1 + N_HIST + j * (N_HIST + 1)
            S1[m, end - len(idx):end] = s1[idx]
            S2[m, end - len(idx):end] = s2[idx]
            ST[m, end - len(idx):end] = st[idx]
            n_hist[m, j] = len(idx)
    il = home_row
    S1[:, -2] = tk.S1_PARTIDO
    S2[:, -2] = tk.s2_id(1, 0, tk.DOM_POR_JUGAR) + rival[il] * tk.N_DOM
    ST[:, -2] = st[il]
    ST[:, -2, 4] = mes[il]                 # las rojas del propio partido son futuro: fuera
    if descanso:
        ST[:, -2, 4] = st[away_row, 2]     # descanso del visitante (el del local ya está en la casilla 2)
    if con_valor:
        ratio = np.log(df["val_local"].to_numpy() / df["val_visitante"].to_numpy())
        code = np.clip(np.round(ratio / 0.1) + 30, 1, 59)
        ST[:, -2, 0] = np.where(np.isnan(code), 0, code).astype(np.int8)
    elif elo_diferencia:
        diff = df["elo_local_antes"].to_numpy() - df["elo_visitante_antes"].to_numpy()
        ST[:, -2, 0] = np.clip(np.round(diff / ELO_DIFF_STEP) + 30, 1, 59).astype(np.int8)
    if ausencias != "no":
        ST[:, -2, 4] = aus_code(df["aus_visitante"].to_numpy(), ponderada)     # el local ya está en la casilla 2
    S1[:, -1] = s1[il]
    S2[:, -1] = s2[il]
    ST[:, -1] = st[il]
    return S1, S2, ST, df, velas, n_hist


def main():
    ap = argparse.ArgumentParser(description="Velas y secuencias de todas las ligas")
    ap.add_argument("--sin-elo-diferencia", action="store_true",
                    help="no meter la diferencia de Elo en el token [PARTIDO] (primera versión)")
    ap.add_argument("--ausencias", choices=["no", "ponderada", "conteo"], default="no")
    ap.add_argument("--valor", choices=["no", "max3", "actual"], default="no")
    ap.add_argument("--rojas", action="store_true", help="expulsiones en el contexto de cada vela")
    ap.add_argument("--descanso", action="store_true", help="días de descanso en el contexto")
    ap.add_argument("--europa", action="store_true", help="incluir Champions, Europa League y Conference")
    ap.add_argument("--historial", type=int, default=N_HIST_DEF, help="velas de cada equipo (20 por defecto)")
    ap.add_argument("--sufijo", default="",
                    help="se añade al nombre de los archivos de salida (p. ej. _exp), para que los experimentos "
                         "no sobrescriban las secuencias de producción")
    args = ap.parse_args()
    suf = args.sufijo
    print("Paso 4b · Velas y secuencias de todas las ligas "
          f"(rojas: {'sí' if args.rojas else 'no'}, descanso: {'sí' if args.descanso else 'no'}, "
          f"historial: {args.historial}, Europa: {'sí' if args.europa else 'no'})\n")
    df = leer_resultados(europa=args.europa)
    df = df.dropna(subset=["goles_local", "goles_visitante"])
    S1, S2, ST, df, velas, n_hist = construir(df, args.ausencias, args.valor, not args.sin_elo_diferencia,
                                              rojas=args.rojas, descanso=args.descanso, n_hist=args.historial)
    M = len(df)
    n_h, seq_len = args.historial, 2 * args.historial + 4

    # Meta + referencias de LaLiga
    meta = df[["liga", "fecha", "temporada", "temporada_inicio", "local_id", "visitante_id",
               "goles_local", "goles_visitante", "resultado", "elo_local_antes", "elo_visitante_antes"]
              + [c for c in ["mercado_local", "mercado_empate", "mercado_visitante"] if c in df]].copy()
    ref = pd.read_csv(os.path.join(DATA, "secuencias_partidos.csv"), encoding="utf-8-sig")
    ref["fecha"] = pd.to_datetime(ref["fecha"])
    ref_cols = ["prob_local", "prob_empate", "prob_visitante", "elo_prob_local", "elo_prob_empate",
                "elo_prob_visitante"]
    ref = ref[["fecha", "local_id", "visitante_id"] + ref_cols].assign(liga="SP1")
    meta = meta.merge(ref, on=["liga", "fecha", "local_id", "visitante_id"], how="left")
    assert len(meta) == M
    meta["historial_local"], meta["historial_visitante"] = n_hist[:, 0], n_hist[:, 1]

    sp1 = meta["liga"] == "SP1"
    emparejados = meta.loc[sp1, "elo_prob_local"].notna().sum()
    print(f"   Partidos de Primera emparejados con las referencias: {emparejados:,} de {len(ref):,}")

    import futbol_rasgos
    R = futbol_rasgos.calcular(velas, M)                    # para el modelo híbrido (entrenar.py --rasgos)
    np.savez_compressed(os.path.join(DATA, f"secuencias_ligas{suf}.npz"), s1=S1, s2=S2, stamp=ST, rasgos=R)
    meta["fecha"] = meta["fecha"].dt.date
    meta.to_csv(os.path.join(DATA, f"secuencias_ligas{suf}_partidos.csv"), index=False, encoding="utf-8-sig")
    velas.drop(columns=["pos"]).to_csv(os.path.join(DATA, f"velas_ligas{suf}.csv"), index=False, encoding="utf-8-sig")
    tk.save_config(os.path.join(DATA, f"vocabulario_ligas{suf}.json"), n_historial=n_h, longitud_secuencia=seq_len,
                   opciones={"rojas": args.rojas, "descanso": args.descanso, "ausencias": args.ausencias,
                             "europa": args.europa,
                             "valor": args.valor},
                   ligas={i: f"{l} · {nombre_competicion(l)}" for i, l in enumerate(LIGAS)})

    print(f"\n   secuencias: {M:,}  ·  tokens: {M * seq_len:,}  "
          f"({M * seq_len / 558_404:.0f} veces los de solo LaLiga)")
    print("   por división:")
    for l, c in meta["liga"].value_counts().reindex(LIGAS).items():
        print(f"     {l:<4} {nombre_competicion(l):<34} {c:>7,}")
    # Elo relativo actual de Primera, para comprobar que tiene sentido
    last = velas[velas["liga"] == "SP1"].sort_values("fecha").groupby("equipo_id").tail(1)
    last = last[last["temporada_inicio"] == last["temporada_inicio"].max()].sort_values("elo_antes", ascending=False)
    print("\n   Elo (escala España) antes del último partido de Primera:")
    for _, r in last.head(6).iterrows():
        print(f"     {r['equipo_id']:<15} {r['elo_antes']:>7.0f}")
    print("\nListo.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
