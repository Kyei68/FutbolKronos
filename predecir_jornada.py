"""
Predecir una jornada de LaLiga con FutbolKronos.

1. Pide a football-data.org el calendario de la temporada (1 consulta, plan gratuito).
2. Si football-data.co.uk aún no tiene los últimos resultados, los completa con los de
   football-data.org (solo goles), para que el historial de cada equipo esté al día.
3. Construye la secuencia de cada partido exactamente como en el entrenamiento
   (crear_secuencias_ligas.construir) y la pasa por el modelo de producción.
4. Del token [PARTIDO] salen las probabilidades de los 25 marcadores (0-4+ × 0-4+); de ahí
   P(1), P(X), P(2), más/menos de 2,5 goles, ambos marcan y los marcadores más probables.
   Se promedian las semillas del modelo (como las simulaciones de Kronos).

Cada predicción se guarda en datos/predicciones/ y se añade a datos/predicciones_historial.csv.
Con --evaluar se mide el acierto de las predicciones ya resueltas: es la prueba de verdad,
con partidos que el modelo no había visto.

Requisitos: football_data_clave.txt y el modelo de producción
            (python3 entrenar.py --datos ligas --produccion).

Uso (desde esta carpeta):
  python3 predecir_jornada.py                 la próxima jornada
  python3 predecir_jornada.py --jornada 13    una jornada concreta
  python3 predecir_jornada.py --evaluar       acierto de las predicciones pasadas (LaLiga)
  python3 predecir_jornada.py --competicion champions   la próxima jornada de la Champions
                                              (antes: python3 crear_europa.py para actualizarla)
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
from crear_secuencias_ligas import leer_resultados
from entrenadores_actuales import clave, fecha_local, mapear_equipos
from futbol_predictor import FutbolKronosPredictor, mercados

DATA = os.path.join(BASE, "datos")
PRED_DIR = os.path.join(DATA, "predicciones")
HISTORIAL = os.path.join(DATA, "predicciones_historial.csv")
URL = "https://api.football-data.org/v4"
PENDIENTE = {"SCHEDULED", "TIMED"}
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def calendario(t, key):
    req = urllib.request.Request(f"{URL}/competitions/PD/matches?season={t}", headers={"X-Auth-Token": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def cargar_modelos(nombre):
    """Semillas del modelo de producción (o, si no existe, el de evaluación), vía futbol_predictor."""
    pred = FutbolKronosPredictor(nombre)
    if pred.aviso_modelo:
        print(f"   AVISO: {pred.aviso_modelo}")
    return pred


resumen = mercados          # compatibilidad: antes se definía aquí


def completar_resultados(df, matches, mapa, t):
    """Añade los partidos terminados que football-data.co.uk todavía no tiene."""
    sp1 = df[(df["liga"] == "SP1") & (df["temporada_inicio"] == t)]
    nuevos = []
    for m in matches["matches"]:
        if m["status"] != "FINISHED":
            continue
        loc, vis = mapa.get(m["homeTeam"]["id"]), mapa.get(m["awayTeam"]["id"])
        if not loc or not vis:
            continue
        f = fecha_local(m["utcDate"])
        ya = sp1[(sp1["local"] == loc) & (sp1["visitante"] == vis) & ((sp1["fecha"] - f).dt.days.abs() <= 1)]
        if len(ya) == 0:
            nuevos.append({"liga": "SP1", "temporada": f"{t}/{(t + 1) % 100:02d}", "temporada_inicio": t,
                           "fecha": f, "local": loc, "visitante": vis,
                           "goles_local": m["score"]["fullTime"]["home"],
                           "goles_visitante": m["score"]["fullTime"]["away"]})
    if nuevos:
        print(f"   {len(nuevos)} resultados recientes añadidos desde football-data.org "
              f"(football-data.co.uk aún no los tiene)")
        df = pd.concat([df, pd.DataFrame(nuevos)], ignore_index=True)
    return df


def evaluar(matches):
    if not os.path.exists(HISTORIAL):
        sys.exit("Aún no hay predicciones guardadas.")
    h = pd.read_csv(HISTORIAL, encoding="utf-8-sig")
    fin = {m["id"]: (m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"])
           for m in matches["matches"] if m["status"] == "FINISHED"}
    h = h[h["fixture"].isin(fin)].copy()
    h = h[pd.to_datetime(h["fecha_prediccion"]) < pd.to_datetime(h["fecha"])]       # solo antes del partido
    h = h.sort_values("fecha_prediccion").groupby("fixture").tail(1)
    if h.empty:
        sys.exit("Ninguna predicción guardada se ha resuelto todavía.")
    gl = np.array([fin[f][0] for f in h["fixture"]])
    gv = np.array([fin[f][1] for f in h["fixture"]])
    y = np.where(gl > gv, 0, np.where(gl == gv, 1, 2))
    p = h[["p_local", "p_empate", "p_visitante"]].to_numpy()
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    brier = ((p - np.eye(3)[y]) ** 2).sum(1).mean()
    acierto = (p.argmax(1) == y).mean()
    o = h["mas_2_5"].to_numpy()
    ll_o = -np.log(np.clip(np.where(gl + gv >= 3, o, 1 - o), 1e-12, 1)).mean()
    print(f"Predicciones resueltas: {len(h)} partidos\n")
    print(f"   1X2:  pérdida log {ll:.4f} · Brier {brier:.4f} · acierto {acierto * 100:.1f} %")
    print(f"   más/menos de 2,5 goles: pérdida log {ll_o:.4f} · acierto {((o > 0.5) == (gl + gv >= 3)).mean() * 100:.1f} %")
    print("\n   Referencia en la prueba 2021–2026: FutbolKronos 0,977 · Elo 0,987 · casas 0,965")
    if len(h) < 100:
        print(f"   (con {len(h)} partidos el resultado aún es muy variable: hacen falta unos cientos)")


def partidos_liga(args, t):
    """Próxima jornada de LaLiga (football-data.org) y resultados recientes que falten."""
    matches = calendario(t, clave())
    pendientes = [m for m in matches["matches"] if m["status"] in PENDIENTE]
    if not pendientes:
        sys.exit("No quedan partidos por jugar esta temporada.")
    jornada = args.jornada or min(m["matchday"] for m in pendientes)
    partidos = [m for m in pendientes if m["matchday"] == jornada]
    if not partidos:
        sys.exit(f"La jornada {jornada} no tiene partidos por jugar.")
    mapa = mapear_equipos(matches, t)
    filas = []
    for m in partidos:
        loc, vis = mapa.get(m["homeTeam"]["id"]), mapa.get(m["awayTeam"]["id"])
        if not loc or not vis:
            print(f"   AVISO: sin equivalencia, se omite {m['homeTeam']['name']} – {m['awayTeam']['name']}")
            continue
        filas.append({"liga": "SP1", "temporada": f"{t}/{(t + 1) % 100:02d}", "temporada_inicio": t,
                      "fecha": fecha_local(m["utcDate"]), "local": loc, "visitante": vis,
                      "goles_local": np.nan, "goles_visitante": np.nan, "fixture": m["id"],
                      "hora": pd.Timestamp(m["utcDate"]).tz_convert("Europe/Madrid").strftime("%H:%M"),
                      "nombre_local": m["homeTeam"]["shortName"], "nombre_visitante": m["awayTeam"]["shortName"]})
    return jornada, filas, lambda df: completar_resultados(df, matches, mapa, t)


def partidos_champions(args, t):
    """Próxima jornada de Champions (datos/europa_pendientes.csv, de crear_europa.py)."""
    path = os.path.join(DATA, "europa_pendientes.csv")
    if not os.path.exists(path):
        sys.exit("Falta datos/europa_pendientes.csv: ejecuta antes python3 crear_europa.py")
    p = pd.read_csv(path, encoding="utf-8-sig")
    p = p[pd.to_datetime(p["fecha"]) >= pd.Timestamp.today().normalize()]
    if p.empty:
        sys.exit("No hay partidos de Champions pendientes (¿has ejecutado crear_europa.py hace poco?).")
    jornada = args.jornada or int(p["jornada"].min())
    p = p[p["jornada"] == jornada]
    filas = [{"liga": "CL", "temporada": f"{t}/{(t + 1) % 100:02d}", "temporada_inicio": t,
              "fecha": pd.Timestamp(r.fecha), "local": r.local, "visitante": r.visitante,
              "local_id": r.local_id, "visitante_id": r.visitante_id,
              "goles_local": np.nan, "goles_visitante": np.nan, "fixture": r.fixture, "hora": r.hora,
              "nombre_local": r.local, "nombre_visitante": r.visitante} for r in p.itertuples(index=False)]
    return jornada, filas, lambda df: df


def main():
    ap = argparse.ArgumentParser(description="Predecir una jornada de LaLiga o de la Champions")
    ap.add_argument("--competicion", choices=["laliga", "champions"], default="laliga")
    ap.add_argument("--jornada", type=int, help="por defecto, la próxima con partidos por jugar")
    ap.add_argument("--temporada", type=int, default=2026)
    ap.add_argument("--modelo", default="futbol-reducido-ligas", help="nombre del modelo en pesos/")
    ap.add_argument("--evaluar", action="store_true", help="acierto de las predicciones de LaLiga ya resueltas")
    args = ap.parse_args()
    t = args.temporada
    if args.evaluar:
        return evaluar(calendario(t, clave()))

    pred = cargar_modelos(args.modelo)
    opciones = pred.opciones
    champions = args.competicion == "champions"
    if champions and not opciones.get("europa"):
        sys.exit("Este modelo se entrenó sin competiciones europeas. Crea las secuencias con --europa "
                 "y reentrena (entrenar.py --datos ligas y --produccion).")
    jornada, filas, completar = (partidos_champions if champions else partidos_liga)(args, t)
    nombre_comp = "Champions League" if champions else "LaLiga"
    print(f"FutbolKronos · {nombre_comp} {t}/{(t + 1) % 100:02d} · jornada {jornada} ({len(filas)} partidos)\n")

    df = leer_resultados(europa=bool(opciones.get("europa")))
    df = df.dropna(subset=["goles_local", "goles_visitante"])
    df = completar(df)
    tabla = pred.predecir(df, pd.DataFrame(filas))

    out = []
    for _, r in tabla.iterrows():
        out.append({"fecha_prediccion": date.today().isoformat(), "jornada": jornada, "fixture": int(r["fixture"]),
                    "fecha": r["fecha"].date().isoformat(), "hora": r["hora"],
                    "local": r["nombre_local"], "visitante": r["nombre_visitante"],
                    "historial_local": int(r["historial_local"]), "historial_visitante": int(r["historial_visitante"]),
                    **{c: r[c] for c in ["p_local", "p_empate", "p_visitante", "mas_2_5", "ambos_marcan", "marcadores"]}})
    out = pd.DataFrame(out).sort_values(["fecha", "hora"])

    for r in out.itertuples(index=False):
        dia = DIAS[pd.Timestamp(r.fecha).weekday()]
        print(f"   {dia} {pd.Timestamp(r.fecha):%d/%m} {r.hora}  {r.local} – {r.visitante}")
        print(f"      1: {r.p_local * 100:4.1f} %   X: {r.p_empate * 100:4.1f} %   2: {r.p_visitante * 100:4.1f} %"
              f"   ·  más de 2,5: {r.mas_2_5 * 100:.0f} %  ·  ambos marcan: {r.ambos_marcan * 100:.0f} %")
        print(f"      marcadores más probables: {r.marcadores}")
        if min(r.historial_local, r.historial_visitante) < opciones.get("n_hist", 20):
            print("      (historial incompleto de algún equipo: predicción menos fiable)")
    print(f"\n   Modelo: {len(pred.modelos)} semilla(s) de {os.path.relpath(pred.ruta_modelo, BASE)}")

    os.makedirs(PRED_DIR, exist_ok=True)
    num = ["p_local", "p_empate", "p_visitante", "mas_2_5", "ambos_marcan"]
    out[num] = out[num].round(4)
    pref = "champions_" if champions else ""
    path = os.path.join(PRED_DIR, f"{pref}jornada_{jornada:02d}_{date.today().isoformat()}.csv")
    historial = HISTORIAL.replace(".csv", "_champions.csv") if champions else HISTORIAL
    out.to_csv(path, index=False, encoding="utf-8-sig")
    out.to_csv(historial, mode="a", index=False, header=not os.path.exists(historial), encoding="utf-8-sig")
    print(f"   Guardado en {os.path.relpath(path, BASE)} y añadido a {os.path.relpath(historial, BASE)}")
    print("\n   Recuerda: son probabilidades, no certezas. Las casas de apuestas siguen siendo más precisas.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
