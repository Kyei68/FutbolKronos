"""
Adaptador de dominio de FutbolKronos: de partidos de fútbol a probabilidades de marcador.

Sustituye al KronosPredictor de Kronos, que trabajaba con velas financieras (apertura,
máximo, mínimo, cierre, volumen) y generaba el futuro token a token. Aquí no hay precios:
cada equipo es una serie de partidos y cada partido es un vector de características de
eventos deportivos, ya discreto (futbol_tokenizer.py):

  s1        marcador del partido (goles a favor 0-4+ × en contra 0-4+) o token especial
  s2        contexto: casa/fuera × fuerza del rival (Elo) × dominio (tiros a puerta)
  casillas  5 datos de contexto (entrenador, nivel Elo, día, jornada, mes; o las variantes
            elegidas al crear las secuencias: descanso, rojas…)

La secuencia de un partido es  [INICIO] historial del local [SEP] historial del visitante
[PARTIDO]  y basta UNA pasada del transformer: lo que predice en [PARTIDO] es la probabilidad
de cada uno de los 25 marcadores, de la que salen 1X2, goles, ambos marcan, etc.

Uso:
    from futbol_predictor import FutbolKronosPredictor
    pred = FutbolKronosPredictor()                      # modelo de producción (semillas)
    tabla = pred.predecir(historial, partidos)          # DataFrames de partidos
    tabla[["local", "visitante", "p_local", "p_empate", "p_visitante", "mas_2_5"]]
"""
import contextlib
import glob
import io
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import futbol_tokenizer as tk
import futbol_rasgos
from kronos_futbol import cargar, logits_marcador

BASE = os.path.dirname(os.path.abspath(__file__))
GOLES = np.array([tk.score_from_id(i) for i in range(tk.N_SCORES)])      # 25 × (local, visitante)


def mercados(p):
    """Probabilidades de los 25 marcadores (local-visitante) -> mercados habituales."""
    gl, gv = GOLES[:, 0], GOLES[:, 1]
    top = np.argsort(-p)[:3]
    return {"p_local": p[gl > gv].sum(), "p_empate": p[gl == gv].sum(), "p_visitante": p[gl < gv].sum(),
            "mas_2_5": p[gl + gv >= 3].sum(), "ambos_marcan": p[(gl >= 1) & (gv >= 1)].sum(),
            "marcadores": ", ".join(f"{etiqueta(i)} ({p[i] * 100:.0f} %)" for i in top)}


def etiqueta(i):
    gl, gv = GOLES[i]
    return f"{gl}{'+' if gl == tk.MAX_GOALS else ''}-{gv}{'+' if gv == tk.MAX_GOALS else ''}"


class FutbolKronosPredictor:
    """Carga las semillas de un modelo y predice partidos a partir de su historial."""

    def __init__(self, modelo="futbol-reducido-ligas", dispositivo="cpu"):
        rutas = sorted(glob.glob(os.path.join(BASE, "pesos", f"{modelo}-produccion", "semilla_*")))
        self.aviso_modelo = ""
        if not rutas:
            rutas = [os.path.join(BASE, "pesos", f"{modelo}-entrenado")]
            self.aviso_modelo = ("No hay modelo de producción: se usa el de evaluación (entrenado hasta 2020/21). "
                                 "Créalo con: python3 entrenar.py --datos ligas --produccion")
        self.rutas = rutas
        self.ruta_modelo = os.path.dirname(rutas[0]) if "semilla_" in rutas[0] else rutas[0]
        try:
            with open(os.path.join(self.ruta_modelo, "entrenamiento.json"), encoding="utf-8") as f:
                self.info = json.load(f)
        except FileNotFoundError:
            self.info = {}
        self.opciones = dict(self.info.get("secuencias", {}))        # cómo reconstruir las secuencias
        if self.opciones.pop("cuotas", False):
            raise ValueError(f"{self.ruta_modelo} se entrenó con cuotas como entrada, opción que ya no existe: "
                             "reentrénalo (entrenar_todo.bat).")
        self.n_hist = self.opciones.get("n_hist", 20)
        self.temperatura = (self.info.get("temperatura_calibrada", 1.0)
                            if self.info.get("usar_calibracion") else 1.0)
        self.dispositivo = torch.device(dispositivo)
        self.modelos = []
        for r in rutas:
            with contextlib.redirect_stdout(io.StringIO()):           # la librería avisa al cargar cada semilla
                m = cargar(r)                                       # FutbolKronos o Kronos antiguo
            self.modelos.append(m.to(self.dispositivo).eval())
        # modelo híbrido: necesita los rasgos numéricos de cada partido (futbol_rasgos.py)
        self.usa_rasgos = any(getattr(m, "n_rasgos", 0) for m in self.modelos)
        self.cabeza = getattr(self.modelos[0], "cabeza", "softmax")

    # ------------------------------------------------------------------ red
    @torch.no_grad()
    def probabilidades(self, S1, S2, ST, pasadas_dropout=0, media=True, lote=2048, R=None):
        """Probabilidad de los 25 marcadores de cada secuencia (arrays de construir()).
        Las secuencias incluyen la última posición [MARCADOR], que se descarta.
        pasadas_dropout > 0: además de cada semilla, esas pasadas con dropout activo (incertidumbre
        del propio modelo, como hace el simulador).
        media=True -> (N, 25), media de las versiones; media=False -> (versiones, N, 25).
        R: rasgos numéricos (N, 37), obligatorios si el modelo es híbrido."""
        if self.usa_rasgos and R is None:
            raise ValueError("Este modelo es híbrido: pasa R = futbol_rasgos.calcular(velas, n)[sel]")
        Rt = None if R is None else torch.from_numpy(np.asarray(R, dtype=np.float32))
        a = [torch.from_numpy(np.ascontiguousarray(X[:, :-1]).astype(np.int64)) for X in (S1, S2, ST)]
        k = max(1, pasadas_dropout)
        rep = lambda x: x.repeat(k, *([1] * (x.dim() - 1)))
        lote = max(1, lote // k)
        versiones = []
        for m in self.modelos:
            m.train(pasadas_dropout > 0)             # dropout activo: cada copia es una versión distinta
            out = []
            for i in range(0, len(a[0]), lote):
                s1, s2, st = (x[i:i + lote].to(self.dispositivo) for x in a)
                r = None if Rt is None else rep(Rt[i:i + lote].to(self.dispositivo))
                b = len(s1)
                logits = logits_marcador(m, rep(s1), rep(s2), rep(st), r)
                p = F.softmax(logits / self.temperatura, dim=-1)
                out.append(p.view(k, b, tk.N_SCORES).cpu())
            m.eval()
            versiones.append(torch.cat(out, 1).numpy().astype(np.float64))
        P = np.concatenate(versiones)                # (semillas × pasadas, N, 25)
        return P.mean(0) if media else P

    # ------------------------------------------------------------------ partidos
    def predecir(self, historial, partidos):
        """historial: partidos jugados (formato de crear_secuencias_ligas.leer_resultados).
        partidos: los que se quieren predecir (liga, temporada, temporada_inicio, fecha, local,
        visitante y, para Europa, local_id/visitante_id); se pueden añadir columnas propias, que
        se conservan. Devuelve una fila por partido con historial disponible, P de los 25
        marcadores (sc0…sc24) y los mercados (p_local, p_empate, p_visitante, mas_2_5…)."""
        from crear_secuencias_ligas import construir      # aquí para no cargar pandas pesado al importar
        hist = historial.dropna(subset=["goles_local", "goles_visitante"])
        nuevos = partidos.copy()
        nuevos["goles_local"], nuevos["goles_visitante"] = np.nan, np.nan
        nuevos["_predecir"] = True
        df = pd.concat([hist, nuevos], ignore_index=True)
        S1, S2, ST, dfc, velas, n_hist = construir(df, verbose=False, **self.opciones)
        sel = np.where(dfc["_predecir"].fillna(False).to_numpy(dtype=bool))[0]
        R = futbol_rasgos.calcular(velas, len(dfc))[sel] if self.usa_rasgos else None
        P = self.probabilidades(S1[sel], S2[sel], ST[sel], R=R)
        out = dfc.iloc[sel].drop(columns=["_predecir"]).reset_index(drop=True)
        out["historial_local"], out["historial_visitante"] = n_hist[sel, 0], n_hist[sel, 1]
        out[[f"sc{i}" for i in range(tk.N_SCORES)]] = P
        merc = pd.DataFrame([mercados(p) for p in P])
        return pd.concat([out, merc], axis=1)
