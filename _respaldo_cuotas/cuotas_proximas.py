"""
Cuotas de los próximos partidos, para el modelo con --cuotas.

football-data.co.uk publica fixtures.csv con las cuotas de los partidos de los próximos días
(normalmente martes y viernes) de las mismas ligas y con los mismos nombres de equipo que los
históricos. Se convierten igual que en el entrenamiento (descargar_ligas.mercado): media de las
casas, cada una sin su margen.

Para partidos que no salen ahí (Champions, o si aún no se han publicado) se pueden poner las
cuotas a mano (desde_cuotas). Sin cuotas, el modelo predice igualmente, pero peor.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import descargar_datos as dd
from descargar_ligas import mercado

URL = "https://www.football-data.co.uk/fixtures.csv"
DESTINO = os.path.join(dd.RAW, "fixtures.csv")
COLS = ["mercado_local", "mercado_empate", "mercado_visitante"]


def cargar(max_horas=6, log=print):
    """Próximos partidos con su probabilidad de mercado (se vuelve a descargar si tiene más de max_horas)."""
    viejo = not os.path.exists(DESTINO) or time.time() - os.path.getmtime(DESTINO) > max_horas * 3600
    if viejo:
        try:
            dd.download(URL, DESTINO, "cuotas de los próximos partidos", force=True, optional=True)
        except Exception as e:                      # sin conexión: se usa la copia anterior si la hay
            log(f"(no se pudieron descargar las cuotas: {e})")
    if not os.path.exists(DESTINO):
        return pd.DataFrame(columns=["liga", "fecha", "local", "visitante"] + COLS)
    f = pd.read_csv(DESTINO, encoding="utf-8-sig", on_bad_lines="skip")
    f.columns = [c.strip() for c in f.columns]
    f = mercado(f)
    f["fecha"] = pd.to_datetime(f["Date"], dayfirst=True, errors="coerce")
    f = f.rename(columns={"Div": "liga", "HomeTeam": "local", "AwayTeam": "visitante"})
    f["local"], f["visitante"] = f["local"].str.strip(), f["visitante"].str.strip()
    return f[["liga", "fecha", "local", "visitante"] + COLS].dropna(subset=["mercado_local"])


def buscar(tabla, liga, local, visitante, fecha, dias=3):
    """{mercado_local, mercado_empate, mercado_visitante} del partido, o None si no está."""
    if tabla is None or not len(tabla):
        return None
    t = tabla[(tabla["liga"] == liga) & (tabla["local"] == local) & (tabla["visitante"] == visitante)]
    t = t[(t["fecha"] - pd.Timestamp(fecha)).dt.days.abs() <= dias]
    return None if t.empty else {c: float(t[c].iloc[0]) for c in COLS}


def desde_cuotas(c1, cx, c2):
    """Cuotas decimales 1/X/2 (las de cualquier casa) -> probabilidades sin margen."""
    inv = 1 / np.array([c1, cx, c2], dtype=float)
    if not np.all(np.isfinite(inv)) or (np.array([c1, cx, c2]) <= 1).any():
        raise ValueError("las cuotas deben ser números mayores que 1")
    p = inv / inv.sum()
    return dict(zip(COLS, map(float, p)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t = cargar(max_horas=0)
    print(f"{len(t)} partidos con cuotas")
    print(t.groupby("liga").size().to_string())
