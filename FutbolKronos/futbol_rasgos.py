"""
Rasgos numéricos de cada partido para el modelo híbrido (entrenar.py --rasgos).

Complementan a las secuencias de tokens con valores continuos que el tokenizador redondea:
promedios móviles de los últimos 5, 10 y 20 partidos de cada equipo (goles a favor y en contra,
tiros a puerta a favor y en contra, puntos), días de descanso, Elo relativo a su división, Elo
absoluto y diferencia de Elo. Todo se calcula con partidos ANTERIORES (shift), así que vale
igual para un partido ya jugado que para uno por jugar.

Se calcula a partir de las velas que devuelve crear_secuencias_ligas.construir(), de modo que el
entrenamiento, predecir_jornada.py y el simulador usan exactamente la misma función.
"""
import numpy as np
import pandas as pd

BASE_COLS = {"gf": "goles_favor", "gc": "goles_contra", "tf": "tiros_puerta_favor",
             "tc": "tiros_puerta_contra", "pts": "pts"}
VENTANAS = (5, 10, 20)
POR_EQUIPO = [f"{n}{k}" for n in BASE_COLS for k in VENTANAS] + ["descanso", "elo_rel", "elo"]
NOMBRES = [f"{c}_local" for c in POR_EQUIPO] + [f"{c}_visitante" for c in POR_EQUIPO] + ["elo_dif"]
N_RASGOS = len(NOMBRES)                                     # 37


def calcular(velas, n_partidos):
    """Matriz (n_partidos, 37) en el orden de los partidos de construir() (columna 'partido')."""
    v = velas[["partido", "equipo_id", "fecha", "es_local", "goles_favor", "goles_contra",
               "tiros_puerta_favor", "tiros_puerta_contra", "elo_antes", "elo_relativo"]].copy()
    v = v.sort_values(["equipo_id", "fecha", "partido"], kind="stable").reset_index(drop=True)
    jugado = v["goles_favor"].notna() & v["goles_contra"].notna()
    v["pts"] = np.where(jugado, np.select([v["goles_favor"] > v["goles_contra"],
                                           v["goles_favor"] == v["goles_contra"]], [3.0, 1.0], 0.0), np.nan)
    g = v.groupby("equipo_id", sort=False)
    for nombre, col in BASE_COLS.items():
        previo = g[col].shift(1)
        for k in VENTANAS:
            v[f"{nombre}{k}"] = (previo.groupby(v["equipo_id"], sort=False)
                                 .rolling(k, min_periods=1).mean().reset_index(level=0, drop=True))
    v["descanso"] = (v["fecha"] - g["fecha"].shift(1)).dt.days.clip(upper=21).fillna(21) / 7.0
    v["elo_rel"] = (v["elo_relativo"] - 1500.0) / 400.0
    v["elo"] = (v["elo_antes"] - 1500.0) / 400.0
    X = np.full((n_partidos, N_RASGOS), np.nan, dtype=np.float32)
    n = len(POR_EQUIPO)
    for lado, desde in ((1, 0), (0, n)):
        s = v[v["es_local"] == lado]
        X[s["partido"].to_numpy(), desde:desde + n] = s[POR_EQUIPO].to_numpy(dtype=np.float32)
    X[:, -1] = X[:, POR_EQUIPO.index("elo")] - X[:, n + POR_EQUIPO.index("elo")]
    return X
