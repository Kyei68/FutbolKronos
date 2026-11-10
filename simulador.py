"""
Motor de simulación de partidos de FutbolKronos (lo usa futbolkronos_gui.py).

Para un partido Local–Visitante en una fecha:
1. Construye su secuencia exactamente como en el entrenamiento: las 20 últimas velas de cada
   equipo antes de esa fecha, su Elo y su entrenador (crear_secuencias_ligas.construir).
2. El modelo da la probabilidad de cada uno de los 25 marcadores (0-4+ × 0-4+).
   Se hace con cada semilla del modelo de producción y, además, con el "dropout" activado
   (Monte Carlo dropout): cada pasada es una versión ligeramente distinta del modelo, así que
   las simulaciones recogen también la incertidumbre del propio modelo, no solo la del partido.
3. Cada simulación elige una de esas versiones y sortea un marcador con sus probabilidades
   (con temperatura, como Kronos: <1 más conservador, >1 más abierto).

Si el partido ya se jugó, se simula con los datos anteriores a esa fecha y se devuelve el
resultado real para comparar (como el modo prueba de Kronos).
"""
import contextlib
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import futbol_tokenizer as tk
from crear_secuencias_ligas import EUROPA, construir, leer_resultados, nombre_competicion, pais_de
from descargar_ligas import DIVISIONES
from futbol_predictor import FutbolKronosPredictor

GOLES = np.array([tk.score_from_id(i) for i in range(tk.N_SCORES)])      # 25 × (local, visitante)
PASADAS_DROPOUT = 100                                                     # versiones por semilla


def temporada_de(fecha):
    return fecha.year if fecha.month >= 7 else fecha.year - 1


class Simulador:
    def __init__(self, modelo="futbol-reducido-ligas", log=print):
        self.log = log
        log("Cargando el modelo…")
        self.pred = FutbolKronosPredictor(modelo)
        self.modelos, self.info, self.ruta_modelo = self.pred.modelos, self.pred.info, self.pred.ruta_modelo
        self.aviso_modelo, self.opciones = self.pred.aviso_modelo, self.pred.opciones
        self.n_hist, self.temperatura = self.pred.n_hist, self.pred.temperatura
        self.europa = bool(self.opciones.get("europa"))
        log("Cargando resultados…")
        df = leer_resultados(europa=self.europa)
        self.df = df.dropna(subset=["goles_local", "goles_visitante"]).reset_index(drop=True)
        self.proximos = []
        eq = pd.read_csv(os.path.join(BASE, "datos", "equivalencias_equipos.csv"), encoding="utf-8-sig")
        self.nombre_equipo = dict(zip(eq["equipo_id"], eq["nombre"]))       # ids españoles -> nombre
        self._completar_con_api()
        if self.europa:
            self._proximos_champions()
        log("Listo.")

    # ---------------------------------------------------------------- datos
    def _completar_con_api(self):
        """Con la clave de football-data.org: añade los últimos resultados de LaLiga que
        football-data.co.uk aún no tiene y carga los próximos partidos. Si falla, se sigue sin ello."""
        try:
            from entrenadores_actuales import clave, fecha_local, mapear_equipos
            from predecir_jornada import PENDIENTE, calendario, completar_resultados
            if not os.path.exists(os.path.join(BASE, "football_data_clave.txt")):
                return
            t = temporada_de(pd.Timestamp.today())
            self.log("Consultando football-data.org…")
            with contextlib.redirect_stdout(io.StringIO()):
                matches = calendario(t, clave())
                mapa = mapear_equipos(matches, t)
                self.df = completar_resultados(self.df, matches, mapa, t)
            for m in matches["matches"]:
                loc, vis = mapa.get(m["homeTeam"]["id"]), mapa.get(m["awayTeam"]["id"])
                if m["status"] in PENDIENTE and loc and vis:
                    self.proximos.append({"liga": "SP1", "jornada": m["matchday"], "fecha": fecha_local(m["utcDate"]),
                                          "hora": pd.Timestamp(m["utcDate"]).tz_convert("Europe/Madrid").strftime("%H:%M"),
                                          "local": loc, "visitante": vis})
            self.proximos.sort(key=lambda x: (x["fecha"], x["hora"]))
        except Exception as e:                                  # sin conexión, sin clave, límite…
            self.log(f"(football-data.org no disponible: {e})")

    def _proximos_champions(self):
        """Partidos de Champions por jugar (crear_europa.py, de football-data.org)."""
        path = os.path.join(BASE, "datos", "europa_pendientes.csv")
        if not os.path.exists(path):
            return
        p = pd.read_csv(path, encoding="utf-8-sig")
        p = p[pd.to_datetime(p["fecha"]) >= pd.Timestamp.today().normalize()]
        for r in p.itertuples(index=False):
            self.proximos.append({"liga": "CL", "jornada": r.jornada, "fecha": pd.Timestamp(r.fecha), "hora": r.hora,
                                  "local": r.local, "visitante": r.visitante})
        self.proximos.sort(key=lambda x: (x["fecha"], x["hora"]))

    def ligas(self):
        """Ligas que se pueden simular (y la Champions, si el modelo se entrenó con Europa)."""
        out = [(d, info[2]) for d, info in DIVISIONES.items() if (self.df["liga"] == d).any()]
        if self.europa and (self.df["liga"] == "CL").any():
            out.insert(0, ("CL", EUROPA["CL"]))
        return out

    def equipos(self, liga):
        """Equipos de la última temporada disponible de esa división (o de la Champions)."""
        d = self.df[self.df["liga"] == liga]
        d = d[d["temporada_inicio"] == d["temporada_inicio"].max()]
        return sorted(set(d["local"]) | set(d["visitante"]))

    def _id_equipo(self, liga, nombre):
        """Identificador de un equipo de una competición europea (viene ya en los datos)."""
        d = self.df[self.df["liga"] == liga]
        for lado in ["local", "visitante"]:
            x = d[d[lado] == nombre]
            if len(x):
                return x[f"{lado}_id"].iloc[-1]
        return None

    # ---------------------------------------------------------------- partido
    def preparar(self, liga, local, visitante, fecha):
        """Secuencia del partido y contexto (forma reciente, Elo, resultado real si ya se jugó)."""
        fecha = pd.Timestamp(fecha).normalize()
        pais = pais_de(liga)
        if pais == "EUR" or self.europa:
            # Con un modelo entrenado con Europa, el historial de cada club mezcla liga y Europa y el
            # Elo europeo necesita a todos los países: hay que construir con TODOS los partidos, como en
            # el entrenamiento y en predecir_jornada.py. (Antes, para un partido de liga se usaban solo
            # los del país, y en el 60 % de los partidos de LaLiga el historial salía distinto.)
            sub = self.df.copy()
        else:
            sub = self.df[self.df["liga"].map(pais_de) == pais].copy()
        # ¿ya se jugó? -> modo prueba: se quita y se simula con lo anterior
        jugado = sub[(sub["liga"] == liga) & (sub["local"] == local) & (sub["visitante"] == visitante)
                     & ((sub["fecha"] - fecha).dt.days.abs() <= 1)]
        real = None
        if len(jugado):
            r = jugado.iloc[0]
            real = (int(r["goles_local"]), int(r["goles_visitante"]))
            fecha = r["fecha"]
            sub = sub.drop(jugado.index)
        sub = sub[sub["fecha"] < fecha] if real else sub                 # en modo prueba, nada posterior
        t = temporada_de(fecha)
        nuevo = {"liga": liga, "temporada": f"{t}/{(t + 1) % 100:02d}", "temporada_inicio": t, "fecha": fecha,
                 "local": local, "visitante": visitante, "goles_local": np.nan, "goles_visitante": np.nan,
                 "simular": True}
        if pais == "EUR":
            nuevo["local_id"], nuevo["visitante_id"] = self._id_equipo(liga, local), self._id_equipo(liga, visitante)
        sub = pd.concat([sub, pd.DataFrame([nuevo])], ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()):
            S1, S2, ST, dfc, velas, n_hist = construir(sub, verbose=False, **self.opciones)
        i = int(np.where(dfc["simular"].fillna(False).to_numpy(dtype=bool))[0][0])

        def forma(equipo_id):
            v = velas[(velas["equipo_id"] == equipo_id) & velas["goles_favor"].notna()].tail(5)
            out = []
            for r in v.itertuples(index=False):
                res = "V" if r.goles_favor > r.goles_contra else "E" if r.goles_favor == r.goles_contra else "D"
                rival = self.nombre_equipo.get(r.rival_id, r.rival_id.split(":")[-1])
                comp = "" if r.liga in DIVISIONES else f" [{r.liga}]"
                out.append((r.fecha.date(), "casa" if r.es_local else "fuera", rival + comp,
                            int(r.goles_favor), int(r.goles_contra), res))
            return out

        fila = dfc.iloc[i]
        R = None
        if self.pred.usa_rasgos:
            import futbol_rasgos
            R = futbol_rasgos.calcular(velas, len(dfc))[i:i + 1]
        return {
            "S1": S1[i:i + 1], "S2": S2[i:i + 1], "ST": ST[i:i + 1], "R": R,
            "liga": liga, "local": local, "visitante": visitante, "fecha": fecha, "real": real,
            "elo_local": float(fila["elo_local_antes"]), "elo_visitante": float(fila["elo_visitante_antes"]),
            "historial": (int(n_hist[i, 0]), int(n_hist[i, 1])),
            "forma_local": forma(fila["local_id"]), "forma_visitante": forma(fila["visitante_id"]),
        }

    def distribuciones(self, p, dropout=True):
        """Probabilidades de los 25 marcadores de cada versión del modelo: (versiones, 25)."""
        P = self.pred.probabilidades(p["S1"], p["S2"], p["ST"], R=p.get("R"),
                                     pasadas_dropout=PASADAS_DROPOUT if dropout else 0, media=False)
        return P[:, 0, :]

    # ---------------------------------------------------------------- simulación
    @staticmethod
    def simular(dist, n, temperatura=1.0, semilla=None):
        """n simulaciones: cada una elige una versión del modelo y sortea un marcador."""
        rng = np.random.default_rng(semilla)
        P = dist ** (1.0 / temperatura)
        P /= P.sum(1, keepdims=True)
        cdf = np.cumsum(P, axis=1)
        version = rng.integers(0, len(P), n)
        u = rng.random(n)
        marcador = np.empty(n, dtype=np.int64)
        for a in range(0, n, 20_000):                           # por bloques, para no gastar memoria
            b = min(n, a + 20_000)
            marcador[a:b] = (u[a:b, None] > cdf[version[a:b]]).sum(1)
        return np.minimum(marcador, tk.N_SCORES - 1), P

    @staticmethod
    def resumir(marcador, P):
        n = len(marcador)
        gl, gv = GOLES[marcador, 0], GOLES[marcador, 1]
        cuenta = np.bincount(marcador, minlength=tk.N_SCORES)
        res = np.where(gl > gv, 0, np.where(gl == gv, 1, 2))
        # incertidumbre del modelo: P(1/X/2) de cada versión, percentiles 10–90
        pv = np.stack([P[:, GOLES[:, 0] > GOLES[:, 1]].sum(1), P[:, GOLES[:, 0] == GOLES[:, 1]].sum(1),
                       P[:, GOLES[:, 0] < GOLES[:, 1]].sum(1)], axis=1)
        tot = gl + gv
        return {
            "n": n,
            "p": np.bincount(res, minlength=3) / n,
            "p_banda": np.percentile(pv, [10, 90], axis=0),
            "matriz": cuenta.reshape(tk.MAX_GOALS + 1, tk.MAX_GOALS + 1) / n,      # [local, visitante]
            "cuenta": cuenta,
            "goles_local": np.bincount(gl, minlength=tk.MAX_GOALS + 1) / n,
            "goles_visitante": np.bincount(gv, minlength=tk.MAX_GOALS + 1) / n,
            "media_local": gl.mean(), "media_visitante": gv.mean(),
            "mas": {k + 0.5: (tot > k).mean() for k in range(5)},
            "ambos": ((gl >= 1) & (gv >= 1)).mean(),
            "porteria_cero_local": (gv == 0).mean(), "porteria_cero_visitante": (gl == 0).mean(),
            "convergencia": np.cumsum(np.eye(3)[res], axis=0) / np.arange(1, n + 1)[:, None],
        }


def etiqueta(i):
    gl, gv = GOLES[i]
    return f"{gl}{'+' if gl == tk.MAX_GOALS else ''}-{gv}{'+' if gv == tk.MAX_GOALS else ''}"


if __name__ == "__main__":                                          # prueba rápida desde la terminal
    sys.stdout.reconfigure(encoding="utf-8")
    sim = Simulador()
    p = sim.preparar("SP1", "Ath Madrid", "Barcelona", pd.Timestamp.today() + pd.Timedelta(days=6))
    d = sim.distribuciones(p)
    m, P = sim.simular(d, 50_000, semilla=0)
    r = sim.resumir(m, P)
    print(f"{p['local']} – {p['visitante']}: 1 {r['p'][0]:.1%}  X {r['p'][1]:.1%}  2 {r['p'][2]:.1%}"
          f"  (banda 1: {r['p_banda'][0, 0]:.1%}–{r['p_banda'][1, 0]:.1%})")
    top = np.argsort(-r["cuenta"])[:5]
    print("marcadores:", ", ".join(f"{etiqueta(i)} {r['cuenta'][i] / r['n']:.1%}" for i in top))
    print("forma local:", p["forma_local"][-3:])
