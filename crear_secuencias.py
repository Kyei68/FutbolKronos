"""
Paso 4: convertir todos los partidos en secuencias de tokens listas para entrenar.

Para cada partido Local–Visitante construye:

  [INICIO] últimas N velas del local [SEP] últimas N velas del visitante [PARTIDO] [MARCADOR]

- Las velas de cada equipo son solo de partidos ANTERIORES (nada del futuro se cuela).
- [PARTIDO] lleva lo que se sabe antes de jugar: casa/fuera, fuerza del rival, Elo,
  entrenador, jornada y fecha.
- [MARCADOR] es lo que el modelo tiene que aprender a predecir.
- Si un equipo tiene menos de N partidos previos, el hueco se rellena con [sin historial],
  para que todas las secuencias midan lo mismo (Kronos no admite longitudes distintas).

Entrada:  datos/velas.csv, datos/elo_partidos.csv   (de crear_velas.py)
Salida:   datos/secuencias.npz          tokens (s1, s2 y contexto) de cada partido
          datos/secuencias_partidos.csv  qué partido es cada secuencia (+ cuotas y Elo para evaluar)
          datos/vocabulario.json         definición del vocabulario
          datos/secuencias_ejemplo.txt   algunas secuencias traducidas a texto, para revisarlas

Uso (desde esta carpeta):  python3 crear_secuencias.py
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import futbol_tokenizer as tk

DATA = os.path.join(BASE, "datos")
N_HIST = 20                       # velas de cada equipo que ve el modelo
SEQ_LEN = 2 * N_HIST + 4          # INICIO + N + SEP + N + PARTIDO + MARCADOR


def main():
    print(f"Paso 4 · Crear secuencias ({N_HIST} partidos previos de cada equipo, {SEQ_LEN} tokens por secuencia)\n")
    velas = pd.read_csv(os.path.join(DATA, "velas.csv"), encoding="utf-8-sig", low_memory=False)
    partidos = pd.read_csv(os.path.join(DATA, "elo_partidos.csv"), encoding="utf-8-sig", low_memory=False)

    velas["fecha"] = pd.to_datetime(velas["fecha"])
    velas = velas.sort_values(["equipo_id", "fecha"]).reset_index(drop=True)
    velas["pos"] = velas.groupby("equipo_id").cumcount()
    velas["partido_n_temporada"] = velas.groupby(["equipo_id", "temporada"]).cumcount() + 1

    # 1. Tokenizar cada vela
    s1 = np.empty(len(velas), dtype=np.int8)
    s2 = np.empty(len(velas), dtype=np.int8)
    st = np.empty((len(velas), 5), dtype=np.int8)
    rival = np.empty(len(velas), dtype=np.int8)
    for i, r in enumerate(velas.itertuples(index=False)):
        rival[i] = tk.rival_level(r.elo_antes, r.elo_rival_antes)
        s1[i] = tk.score_id(r.goles_favor, r.goles_contra)
        s2[i] = tk.s2_id(r.es_local, rival[i], tk.dominance(r.tiros_puerta_favor, r.tiros_puerta_contra))
        st[i] = tk.stamp(r.partido_n_entrenador, r.elo_antes, r.fecha.weekday(), r.partido_n_temporada, r.fecha.month)

    by_team = {t: g.index.to_numpy() for t, g in velas.groupby("equipo_id")}
    key = {(e, f, rv): i for i, (e, f, rv) in enumerate(zip(velas["equipo_id"], velas["fecha"], velas["rival_id"]))}

    def history(team, pos):
        """Índices de las N velas anteriores (las más antiguas primero) + huecos."""
        idx = by_team[team][max(0, pos - N_HIST):pos]
        return idx, N_HIST - len(idx)

    # 2. Construir una secuencia por partido
    partidos["fecha"] = pd.to_datetime(partidos["fecha"])
    partidos = partidos.dropna(subset=["goles_local", "goles_visitante"]).reset_index(drop=True)
    M = len(partidos)
    S1 = np.full((M, SEQ_LEN), tk.S1_SIN_HISTORIAL, dtype=np.int8)
    S2 = np.full((M, SEQ_LEN), tk.S2_ESPECIAL, dtype=np.int8)
    ST = np.zeros((M, SEQ_LEN, 5), dtype=np.int8)
    n_hist = np.zeros((M, 2), dtype=np.int16)

    for m, p in enumerate(partidos.itertuples(index=False)):
        il = key[(p.local_id, p.fecha, p.visitante_id)]
        iv = key[(p.visitante_id, p.fecha, p.local_id)]
        pos = 0
        S1[m, pos] = tk.S1_INICIO
        pos += 1
        for j, (team, i_match) in enumerate([(p.local_id, il), (p.visitante_id, iv)]):
            idx, gaps = history(team, velas.at[i_match, "pos"])
            n_hist[m, j] = len(idx)
            pos += gaps                                    # huecos ya rellenos con [sin historial]
            S1[m, pos:pos + len(idx)] = s1[idx]
            S2[m, pos:pos + len(idx)] = s2[idx]
            ST[m, pos:pos + len(idx)] = st[idx]
            pos += len(idx)
            if j == 0:
                S1[m, pos] = tk.S1_SEP
                pos += 1
        # [PARTIDO]: solo información previa al partido, desde el punto de vista del local
        S1[m, pos] = tk.S1_PARTIDO
        S2[m, pos] = tk.s2_id(1, rival[il], tk.DOM_POR_JUGAR)
        ST[m, pos] = st[il]
        pos += 1
        # [MARCADOR]: el objetivo
        S1[m, pos] = s1[il]
        S2[m, pos] = s2[il]
        ST[m, pos] = st[il]
        assert pos == SEQ_LEN - 1

    # 3. Guardar
    np.savez_compressed(os.path.join(DATA, "secuencias.npz"), s1=S1, s2=S2, stamp=ST)
    meta = partidos[["fecha", "temporada", "temporada_inicio", "local_id", "visitante_id",
                     "goles_local", "goles_visitante", "resultado",
                     "prob_local", "prob_empate", "prob_visitante",
                     "elo_prob_local", "elo_prob_empate", "elo_prob_visitante"]].copy()
    meta["fecha"] = meta["fecha"].dt.date
    meta["historial_local"] = n_hist[:, 0]
    meta["historial_visitante"] = n_hist[:, 1]
    meta.to_csv(os.path.join(DATA, "secuencias_partidos.csv"), index=False, encoding="utf-8-sig")
    tk.save_config(os.path.join(DATA, "vocabulario.json"), n_historial=N_HIST, longitud_secuencia=SEQ_LEN)

    # 4. Ejemplos legibles
    def render(m):
        p = partidos.iloc[m]
        lines = [f"{p['fecha']:%d/%m/%Y}  {p['local_id']} {int(p['goles_local'])}-{int(p['goles_visitante'])} "
                 f"{p['visitante_id']}   (temporada {p['temporada']})"]
        for t in range(SEQ_LEN):
            lab = tk.describe(S1[m, t], S2[m, t])
            note = ""
            if t == SEQ_LEN - 1:
                note = "   ← lo que el modelo debe predecir"
            elif S1[m, t] == tk.S1_PARTIDO:
                note = "   ← lo que se sabe antes de jugar"
            ctx = [int(x) for x in ST[m, t]]
            extra = (f"   [entrenador: partido {ctx[0]}, Elo≈{tk.ELO_MIN + ctx[1] * tk.ELO_STEP}, "
                     f"jornada≈{round(ctx[3] * 38 / 31)}]") if S1[m, t] < tk.N_SCORES or S1[m, t] == tk.S1_PARTIDO else ""
            lines.append(f"  {t:>2}  {lab}{extra}{note}")
        return "\n".join(lines)

    examples = [M - 1]
    first_full = int(np.argmax((n_hist >= N_HIST).all(axis=1)))
    examples.append(first_full)
    with open(os.path.join(DATA, "secuencias_ejemplo.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(render(m) for m in examples) + "\n")

    # 5. Resumen
    full = (n_hist >= N_HIST).all(axis=1)
    targets = Counter(S1[:, -1].tolist())
    top = ", ".join(f"{tk.describe(k, tk.S2_ESPECIAL)} {v / M * 100:.1f}%" for k, v in targets.most_common(6))
    used = len(set(zip(S1.ravel().tolist(), S2.ravel().tolist())))
    print(f"   secuencias: {M:,}  ({full.mean() * 100:.1f} % con historial completo de ambos equipos)")
    print(f"   tokens distintos usados: {used} de {2 ** tk.S1_BITS * 2 ** tk.S2_BITS:,} posibles")
    print(f"   marcadores más frecuentes (local-visitante): {top}")
    print(f"   tokens de entrenamiento: {M * SEQ_LEN:,}")
    print("\n   Ejemplo (último partido):\n")
    print("   " + render(M - 1).replace("\n", "\n   "))
    print("\nListo. Revisa más ejemplos en datos/secuencias_ejemplo.txt")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: falta {e.filename}. Ejecuta antes crear_velas.py.", file=sys.stderr)
        sys.exit(1)
