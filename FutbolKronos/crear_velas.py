"""
Paso 3b: convertir cada partido en dos "velas" (una por equipo), como las velas de Kronos.

Para cada equipo y partido guarda: si jugó en casa, goles a favor y en contra, tiros,
tiros a puerta, córners, tarjetas, lo que daban las casas de apuestas, su rating Elo y
el del rival ANTES del partido, y si estrenaba entrenador.

También calcula un rating Elo (que da más peso a lo reciente) y lo evalúa como primer
modelo de referencia: el modelo que entrenemos tendrá que mejorarlo.

Entrada:  datos/partidos_unificados.csv   (de unificar_equipos.py)
Salida:   datos/velas.csv, datos/elo_partidos.csv, datos/referencia_elo.txt

Uso (desde esta carpeta):  python3 crear_velas.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datos")

ELO_START = 1500.0
TUNE_SEASONS = (2000, 2014)   # temporadas para ajustar el Elo (la 93-99 es el "calentamiento")
TEST_FROM = 2015              # temporadas para evaluar: el Elo nunca las usa para ajustarse


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
def goal_mult(gd):
    """Una goleada mueve más el rating que una victoria por la mínima (como el Elo mundial de selecciones)."""
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def run_elo(df, k=20.0, hfa=60.0, regress=0.25):
    """Recorre los partidos en orden y devuelve el Elo de cada equipo ANTES de cada partido.
    - k: cuánto cambia el rating tras cada partido.
    - hfa: ventaja de jugar en casa, en puntos Elo.
    - regress: al empezar cada temporada, cada equipo se acerca esa fracción a la media
      (olvida parte del pasado: lo reciente pesa más).
    - Un equipo que llega a Primera empieza con la media de los que descendieron.
    """
    ratings = {}
    pre_h = np.empty(len(df))
    pre_a = np.empty(len(df))
    season_teams = df.groupby("temporada_inicio")["local_id"].apply(set).to_dict()
    seasons = sorted(season_teams)
    current = None
    for i, (season, h, a, gh, ga) in enumerate(zip(df["temporada_inicio"], df["local_id"], df["visitante_id"],
                                                   df["goles_local"], df["goles_visitante"])):
        if season != current:
            prev = seasons[seasons.index(season) - 1] if seasons.index(season) > 0 else None
            if prev is not None:
                relegated = season_teams[prev] - season_teams[season]
                base = np.mean([ratings[t] for t in relegated if t in ratings]) if relegated else ELO_START - 100
                for t in season_teams[season]:
                    if t not in season_teams[prev]:          # recién ascendido
                        ratings[t] = min(ratings.get(t, base), base)
                for t in season_teams[season]:
                    ratings[t] += regress * (ELO_START - ratings[t])
            current = season
        rh = ratings.setdefault(h, ELO_START)
        ra = ratings.setdefault(a, ELO_START)
        pre_h[i], pre_a[i] = rh, ra
        if pd.isna(gh) or pd.isna(ga):
            continue
        exp_h = 1 / (1 + 10 ** ((ra - (rh + hfa)) / 400))
        score_h = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        delta = k * goal_mult(gh - ga) * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return pre_h, pre_a


def elo_diff(df, pre_h, pre_a, hfa):
    return (pre_h + hfa) - pre_a


class BucketModel:
    """Traduce la diferencia de Elo a probabilidades de 1, X y 2, mirando qué pasó
    históricamente en partidos con diferencias parecidas (solo con datos de entrenamiento)."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def fit(self, diff, result):
        self.edges = np.unique(np.quantile(diff, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        b = np.searchsorted(self.edges, diff)
        self.table = np.zeros((len(self.edges) + 1, 3))
        for j, r in enumerate(["1", "X", "2"]):
            self.table[:, j] = np.bincount(b[result == r], minlength=len(self.edges) + 1)
        self.table = (self.table + 1) / (self.table + 1).sum(axis=1, keepdims=True)   # suavizado
        return self

    def predict(self, diff):
        return self.table[np.searchsorted(self.edges, diff)]


def log_loss(probs, result):
    idx = pd.Series(result).map({"1": 0, "X": 1, "2": 2}).to_numpy()
    p = np.clip(probs[np.arange(len(idx)), idx], 1e-9, 1)
    return float(-np.mean(np.log(p)))


def brier(probs, result):
    y = np.zeros_like(probs)
    y[np.arange(len(result)), pd.Series(result).map({"1": 0, "X": 1, "2": 2}).to_numpy()] = 1
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def accuracy(probs, result):
    return float(np.mean(np.array(["1", "X", "2"])[probs.argmax(axis=1)] == np.asarray(result)))


def tune_elo(df):
    tune = df["temporada_inicio"].between(*TUNE_SEASONS).to_numpy()
    best = None
    for k, hfa, reg in itertools.product([15, 20, 25, 30, 35], [0, 20, 40, 60, 80, 100], [0.0, 0.05, 0.1, 0.25, 0.4]):
        ph, pa = run_elo(df, k, hfa, reg)
        d = elo_diff(df, ph, pa, hfa)
        # Ajuste justo: la tabla se aprende con la primera mitad del periodo y se mide en la segunda
        first = tune & (df["temporada_inicio"] < 2007).to_numpy()
        second = tune & (df["temporada_inicio"] >= 2007).to_numpy()
        bm = BucketModel().fit(d[first], df["resultado"].to_numpy()[first])
        ll = log_loss(bm.predict(d[second]), df["resultado"].to_numpy()[second])
        if best is None or ll < best[0]:
            best = (ll, k, hfa, reg)
    return best[1:]


# --------------------------------------------------------------------------- #
def main():
    print("Paso 3b · Crear velas y rating Elo\n")
    df = pd.read_csv(os.path.join(DATA, "partidos_unificados.csv"), encoding="utf-8-sig", low_memory=False)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "local_id"]).reset_index(drop=True)
    played = df["resultado"].notna()

    # 1. Elo con parámetros ajustados en temporadas antiguas
    print("   Ajustando el Elo con las temporadas 2000/01 a 2014/15…")
    k, hfa, reg = tune_elo(df[played].reset_index(drop=True))
    print(f"   Mejor configuración: K={k}, ventaja de local={hfa} puntos, olvido entre temporadas={reg:.0%}")
    pre_h, pre_a = run_elo(df, k, hfa, reg)
    df["elo_local_antes"] = pre_h.round(1)
    df["elo_visitante_antes"] = pre_a.round(1)
    diff = elo_diff(df, pre_h, pre_a, hfa)

    # 2. Evaluación en temporadas que el Elo nunca ha visto
    train = (played & df["temporada_inicio"].between(*TUNE_SEASONS)).to_numpy()
    test = (played & (df["temporada_inicio"] >= TEST_FROM)).to_numpy()
    res = df["resultado"].to_numpy()
    bm = BucketModel().fit(diff[train], res[train])
    p_elo = bm.predict(diff)
    df[["elo_prob_local", "elo_prob_empate", "elo_prob_visitante"]] = p_elo.round(4)

    freq = np.array([(res[train] == r).mean() for r in ["1", "X", "2"]])
    p_naive = np.tile(freq, (len(df), 1))
    odds = test & df["prob_local"].notna().to_numpy()
    p_mkt = df[["prob_local", "prob_empate", "prob_visitante"]].to_numpy(dtype=float)

    rows = []
    for name, p, mask in [("Siempre lo mismo (frecuencias históricas)", p_naive, odds),
                          ("Rating Elo", p_elo, odds),
                          ("Casas de apuestas", p_mkt, odds)]:
        rows.append((name, log_loss(p[mask], res[mask]), brier(p[mask], res[mask]), accuracy(p[mask], res[mask])))
    first_s, last_s = df.loc[test, "temporada"].iloc[0], df.loc[test, "temporada"].iloc[-1]
    report = [
        "FutbolKronos · Modelos de referencia",
        f"Evaluados en {odds.sum():,} partidos de {first_s} a {last_s} (no usados para ajustar nada).",
        "Pérdida logarítmica y Brier: cuanto MÁS BAJO, mejor. Acierto: % de veces que el resultado",
        "más probable fue el que ocurrió.",
        "",
        f"{'Modelo':<44}{'Pérdida log':>12}{'Brier':>9}{'Acierto':>10}",
    ] + [f"{n:<44}{ll:>12.4f}{b:>9.4f}{a * 100:>9.1f}%" for n, ll, b, a in rows] + [
        "",
        f"Elo: K={k}, ventaja de local={hfa}, olvido entre temporadas={reg:.0%}",
        "Objetivo del modelo FutbolKronos: bajar la pérdida del Elo y acercarse a las casas de apuestas.",
    ]
    with open(os.path.join(DATA, "referencia_elo.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n   " + "\n   ".join(report[1:]))

    # 3. Velas: dos por partido (una desde cada equipo)
    def side(own, opp, is_home):
        o, r = ("local", "visitante") if own == "local" else ("visitante", "local")
        v = pd.DataFrame({
            "fecha": df["fecha"].dt.date, "temporada": df["temporada"], "temporada_inicio": df["temporada_inicio"],
            "equipo_id": df[f"{o}_id"], "rival_id": df[f"{r}_id"], "es_local": int(is_home),
            "goles_favor": df[f"goles_{o}"], "goles_contra": df[f"goles_{r}"],
            "tiros_favor": df[f"tiros_{o}"], "tiros_contra": df[f"tiros_{r}"],
            "tiros_puerta_favor": df[f"tiros_puerta_{o}"], "tiros_puerta_contra": df[f"tiros_puerta_{r}"],
            "corners_favor": df[f"corners_{o}"], "corners_contra": df[f"corners_{r}"],
            "amarillas": df[f"amarillas_{o}"], "rojas": df[f"rojas_{o}"],
            "prob_ganar_casas": df[f"prob_{o}"], "prob_empate_casas": df["prob_empate"],
            "prob_perder_casas": df[f"prob_{r}"],
            "elo_antes": df[f"elo_{o}_antes"], "elo_rival_antes": df[f"elo_{r}_antes"],
            "elo_prob_ganar": df[f"elo_prob_{o}"], "elo_prob_perder": df[f"elo_prob_{r}"],
            "entrenador": df[f"entrenador_{o}"],
        })
        return v

    velas = pd.concat([side("local", "visitante", True), side("visitante", "local", False)], ignore_index=True)
    velas = velas.sort_values(["equipo_id", "fecha"]).reset_index(drop=True)
    velas["resultado"] = np.select([velas["goles_favor"] > velas["goles_contra"],
                                    velas["goles_favor"] == velas["goles_contra"]], ["V", "E"], "D")
    velas.loc[velas["goles_favor"].isna(), "resultado"] = pd.NA
    prev_coach = velas.groupby("equipo_id")["entrenador"].shift()
    velas["cambio_entrenador"] = (velas["entrenador"].notna() & prev_coach.notna()
                                  & (velas["entrenador"] != prev_coach)).astype(int)
    velas["partido_n_entrenador"] = velas.groupby(
        ["equipo_id", (velas["entrenador"] != prev_coach).groupby(velas["equipo_id"]).cumsum()]).cumcount() + 1
    velas.loc[velas["entrenador"].isna(), "partido_n_entrenador"] = pd.NA
    velas["partido_n_equipo"] = velas.groupby("equipo_id").cumcount() + 1

    velas.to_csv(os.path.join(DATA, "velas.csv"), index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(DATA, "elo_partidos.csv"), index=False, encoding="utf-8-sig")

    print(f"\n   velas.csv: {len(velas):,} velas de {velas['equipo_id'].nunique()} equipos "
          f"({velas['partido_n_equipo'].max()} como máximo por equipo)")
    last = (velas.sort_values("fecha").groupby("equipo_id").tail(1)
            .query("temporada == @df['temporada'].iloc[-1]")
            .sort_values("elo_antes", ascending=False))
    print("\n   Elo actual (antes de su último partido):")
    for _, r in last.head(20).iterrows():
        print(f"     {r['equipo_id']:<15} {r['elo_antes']:>7.0f}")
    print("\nListo.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: falta {e.filename}. Ejecuta antes unificar_equipos.py.", file=sys.stderr)
        sys.exit(1)
