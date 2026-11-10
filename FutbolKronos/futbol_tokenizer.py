"""
Tokenizador de FutbolKronos: convierte cada vela (un equipo en un partido) en un token
de dos partes, como Kronos:

  s1 (5 bits) · el MARCADOR: goles a favor 0-4+ × goles en contra 0-4+ = 25 valores,
                más tokens especiales.
  s2 (6 bits) · el CONTEXTO: casa/fuera × fuerza del rival × dominio (tiros a puerta).

Además, cada token lleva 5 datos de contexto que se meten en la capa que en Kronos
codifica la fecha (TemporalEmbedding), reutilizando sus 5 casillas:

  casilla Kronos   tamaño   en FutbolKronos
  minuto            60      nº de partido del entrenador actual (1-59; 0 = desconocido)
  hora              24      nivel Elo del equipo (24 escalones de 25 puntos, 1250-1850)
  día de semana      7      día de la semana del partido (0 = lunes)
  día del mes       32      momento de la temporada: jornada 1-38 reescalada a 1-31
  mes               13      mes del partido (1-12)

Así el modelo de Kronos se usa sin tocar su código.
"""
import json

import numpy as np

# ----------------------------- s1: marcador ----------------------------------
MAX_GOALS = 4                      # 4 significa "4 o más"
N_SCORES = (MAX_GOALS + 1) ** 2    # 25
S1_INICIO = 25                     # comienzo de secuencia
S1_SEP = 26                        # separa las velas del local y las del visitante
S1_PARTIDO = 27                    # el partido que hay que predecir
S1_SIN_HISTORIAL = 28              # hueco: el equipo no tenía tantos partidos previos
S1_BITS = 5                        # 32 valores posibles

# ----------------------------- s2: contexto ----------------------------------
RIVAL_EDGES = [-100, 0, 100]       # Elo rival - Elo propio
RIVAL_LABELS = ["rival mucho más débil", "rival más débil", "rival más fuerte", "rival mucho más fuerte"]
DOM_EDGES = [-3, -1, 1, 3]         # tiros a puerta a favor - en contra (≤-3 | -2,-1 | 0 | 1,2 | ≥3)
DOM_LABELS = ["muy dominado", "dominado", "igualado", "dominante", "muy dominante",
              "sin datos de tiros", "por jugar"]
DOM_SIN_DATOS = 5
DOM_POR_JUGAR = 6
N_RIVAL = len(RIVAL_LABELS)        # 4
N_DOM = len(DOM_LABELS)            # 7
S2_ESPECIAL = 63                   # s2 de los tokens especiales
S2_BITS = 6                        # 64 valores posibles (se usan 2 × 4 × 7 = 56 + 1)

# ----------------------------- contexto (stamp) ------------------------------
STAMP_SIZES = [60, 24, 7, 32, 13]
ELO_MIN, ELO_STEP = 1250, 25


def score_id(gf, gc):
    return int(min(gf, MAX_GOALS)) * (MAX_GOALS + 1) + int(min(gc, MAX_GOALS))


def score_from_id(i):
    return divmod(int(i), MAX_GOALS + 1)


def rival_level(elo_propio, elo_rival):
    return int(np.searchsorted(RIVAL_EDGES, elo_rival - elo_propio, side="right"))


def dominance(tp_favor, tp_contra):
    if tp_favor is None or tp_contra is None or np.isnan(tp_favor) or np.isnan(tp_contra):
        return DOM_SIN_DATOS
    d = tp_favor - tp_contra
    if d <= DOM_EDGES[0]:
        return 0
    if d <= DOM_EDGES[1]:
        return 1
    if d < DOM_EDGES[2]:
        return 2
    if d < DOM_EDGES[3]:
        return 3
    return 4


def s2_id(es_local, rival, dom):
    return int(es_local) * N_RIVAL * N_DOM + int(rival) * N_DOM + int(dom)


def s2_parts(i):
    i = int(i)
    if i == S2_ESPECIAL:
        return None
    es_local, rest = divmod(i, N_RIVAL * N_DOM)
    rival, dom = divmod(rest, N_DOM)
    return es_local, rival, dom


def stamp(partido_n_entrenador, elo, weekday, partido_n_temporada, month):
    coach = 0 if partido_n_entrenador is None or np.isnan(partido_n_entrenador) else int(min(partido_n_entrenador, 59))
    elo_lvl = int(np.clip((elo - ELO_MIN) // ELO_STEP, 0, 23))
    jornada = int(np.clip(round(partido_n_temporada * 31 / 38), 1, 31))   # 38 jornadas -> 1-31
    return [coach, elo_lvl, int(weekday), jornada, int(month)]


SPECIAL_STAMP = [0, 0, 0, 0, 0]


def describe(s1, s2):
    """Texto legible de un token (para revisar secuencias)."""
    names = {S1_INICIO: "[INICIO]", S1_SEP: "[SEP]", S1_PARTIDO: "[PARTIDO]", S1_SIN_HISTORIAL: "[sin historial]"}
    head = names.get(int(s1))
    if head is None:
        gf, gc = score_from_id(s1)
        head = f"{gf}{'+' if gf == MAX_GOALS else ''}-{gc}{'+' if gc == MAX_GOALS else ''}"
    parts = s2_parts(s2)
    if parts is None:
        return head
    es_local, rival, dom = parts
    return f"{head} ({'casa' if es_local else 'fuera'}, {RIVAL_LABELS[rival]}, {DOM_LABELS[dom]})"


def outcome_matrix():
    """Matriz 25×3 para pasar de probabilidades de marcador a P(1), P(X), P(2)
    (desde el punto de vista del equipo local)."""
    m = np.zeros((N_SCORES, 3))
    for i in range(N_SCORES):
        gf, gc = score_from_id(i)
        m[i, 0 if gf > gc else 1 if gf == gc else 2] = 1
    return m


def config():
    return {
        "s1_bits": S1_BITS, "s2_bits": S2_BITS, "max_goals": MAX_GOALS, "n_scores": N_SCORES,
        "s1_especiales": {"INICIO": S1_INICIO, "SEP": S1_SEP, "PARTIDO": S1_PARTIDO,
                          "SIN_HISTORIAL": S1_SIN_HISTORIAL},
        "s2_especial": S2_ESPECIAL,
        "rival_umbrales_elo": RIVAL_EDGES, "rival_niveles": RIVAL_LABELS,
        "dominio_umbrales_tiros_puerta": DOM_EDGES, "dominio_niveles": DOM_LABELS,
        "stamp": {"casillas": ["partido_n_entrenador", "nivel_elo", "dia_semana", "partido_n_temporada", "mes"],
                  "tamaños": STAMP_SIZES, "elo_min": ELO_MIN, "elo_escalon": ELO_STEP},
    }


def save_config(path, **extra):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**config(), **extra}, f, ensure_ascii=False, indent=2)
