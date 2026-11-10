"""
Paso 5: entrenar FutbolKronos y compararlo con el Elo y las casas de apuestas.

Cada secuencia es  [INICIO] 20 velas del local [SEP] 20 del visitante [PARTIDO] [MARCADOR].
Como Kronos, el modelo aprende a predecir el siguiente token en todas las posiciones
(pérdida s1 + s2), pero lo que importa es la posición [PARTIDO]: lo que predice ahí es la
probabilidad de cada uno de los 25 marcadores, y sumándolos salen P(1), P(X) y P(2).

División por temporadas (nada del futuro se usa para ajustar nada):
  entrenamiento  hasta 2017/18
  validación     2018/19 – 2020/21   → elegir semivida y nº de épocas
  prueba         2021/22 – 2026/27   → resultado final, contra Elo y casas de apuestas

Fase 1: para cada semivida, entrena con "entrenamiento" y para cuando deja de mejorar en
        "validación". Gana la semivida con menor pérdida logarítmica en validación.
Fase 2: reentrena con entrenamiento + validación, con esa semivida y ese nº de épocas,
        y evalúa una única vez en "prueba".

Peso de cada partido al entrenar:  0,5 ^ (temporadas de antigüedad / semivida).

Entrada:  datos/secuencias.npz, datos/secuencias_partidos.csv   (de crear_secuencias.py)
Salida:   pesos/futbol-<modelo>-entrenado/        el modelo final
          datos/resultados_entrenamiento.txt      tabla comparativa
          datos/predicciones_prueba.csv           P(1), P(X), P(2) de cada partido de prueba

Uso (desde esta carpeta):
  python3 entrenar.py                       modelo reducido (~0,6 M parámetros)
  python3 entrenar.py --modelo mini         tamaños completos de Kronos-mini
  python3 entrenar.py --semividas inf,4,2   otras semividas (en temporadas; inf = sin olvido)
  python3 entrenar.py --rapido              prueba exprés (pocas épocas, no guarda nada)
  python3 entrenar.py --datos ligas --produccion
                                            modelo para predecir: misma configuración que la
                                            última evaluación, pero con TODOS los partidos

Con la gráfica (entorno ..\\venv-gpu, con PyTorch para CUDA):
  ..\\venv-gpu\\Scripts\\python.exe entrenar.py --dispositivo cuda

Preentrenar con las 22 divisiones europeas y ajustar después solo con LaLiga
(validación y prueba siguen siendo los mismos partidos de LaLiga):
  ..\\venv-gpu\\Scripts\\python.exe entrenar.py --dispositivo cuda --datos ligas --ajuste 30
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.nn.functional as F

import futbol_rasgos
import futbol_tokenizer as tk
from kronos_futbol import FutbolKronos, Kronos, logits_marcador

DATA = os.path.join(BASE, "datos")
FIN_ENTRENAMIENTO = 2017          # última temporada (año de inicio) de entrenamiento
FIN_VALIDACION = 2020             # última temporada de validación

MODELOS = {
    # d_model, capas, cabezas, ff_dim
    "reducido": dict(n_layers=3, d_model=128, n_heads=4, ff_dim=256),
    "mini": dict(n_layers=4, d_model=256, n_heads=4, ff_dim=512),
}


def model_config(nombre):
    return dict(s1_bits=tk.S1_BITS, s2_bits=tk.S2_BITS, **MODELOS[nombre],
                ffn_dropout_p=0.2, attn_dropout_p=0.0, resid_dropout_p=0.2,
                token_dropout_p=0.0, learn_te=True)


# ------------------------------------------------------------------ datos
def load(datos, secuencias=""):
    sufijo = ("_ligas" if datos == "ligas" else "") + secuencias
    d = np.load(os.path.join(DATA, f"secuencias{sufijo}.npz"))
    meta = pd.read_csv(os.path.join(DATA, f"secuencias{sufijo}_partidos.csv"), encoding="utf-8-sig",
                       low_memory=False)
    if "liga" not in meta.columns:
        meta["liga"] = "SP1"
    s1 = torch.from_numpy(d["s1"].astype(np.int64))
    s2 = torch.from_numpy(d["s2"].astype(np.int64))
    st = torch.from_numpy(d["stamp"].astype(np.int64))
    R = torch.from_numpy(d["rasgos"].astype(np.float32)) if "rasgos" in d.files else None
    y = meta["resultado"].astype(str).map({"1": 0, "X": 1, "2": 2}).to_numpy()
    # Se evalúa solo en Primera, en los partidos que tienen las referencias (Elo y casas)
    laliga = (meta["liga"] == "SP1").to_numpy()
    evaluable = laliga & meta["elo_prob_local"].notna().to_numpy()
    return s1, s2, st, R, meta, y, laliga, evaluable


def sel(R, idx):
    return None if R is None else R[idx]


def crear(cfg, args):
    """Kronos a secas (como siempre) o FutbolKronos si se pide cabeza Poisson o rasgos."""
    if args.cabeza == "softmax" and not args.rasgos:
        return Kronos(**cfg)
    media, escala = args.r_stats if args.rasgos else (None, None)
    return FutbolKronos(kronos=cfg, cabeza=args.cabeza, n_rasgos=len(media) if args.rasgos else 0,
                        rasgos_media=media, rasgos_escala=escala)


def stats_rasgos(R, idx):
    """Media y escala de los rasgos en los partidos de entrenamiento (se guardan con el modelo)."""
    x = R[idx].numpy()
    media = np.nanmean(x, 0)
    escala = np.nanstd(x, 0) + 1e-6
    return [float(v) for v in np.nan_to_num(media)], [float(v) for v in np.nan_to_num(escala, nan=1.0)]


def recency_weights(seasons, ref, half_life):
    if math.isinf(half_life):
        w = np.ones(len(seasons))
    else:
        w = 0.5 ** ((ref - seasons) / half_life)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ------------------------------------------------------------------ modelo
OUTCOME = torch.tensor(tk.outcome_matrix(), dtype=torch.float32)       # 25 × 3


def loss_fn(model, s1, s2, st, w, peso_marcador, R=None):
    """Pérdida de Kronos ((s1 + s2) / 2 por posición), ponderada por antigüedad.
    Las posiciones cuyo objetivo es un hueco [sin historial] no cuentan; el marcador final
    pesa `peso_marcador` veces lo que pesa la media del resto de posiciones."""
    x1, x2, xs = s1[:, :-1], s2[:, :-1], st[:, :-1]
    y1, y2 = s1[:, 1:], s2[:, 1:]
    if isinstance(model, FutbolKronos):
        l1, ctx = model.m.decode_s1(x1, x2, xs)
        l2 = model.m.decode_s2(ctx, y1)                                       # = forward con teacher forcing
        ce1 = F.cross_entropy(l1.transpose(1, 2), y1, reduction="none")
        # en [PARTIDO] el marcador sale de la cabeza propia (Poisson y/o rasgos)
        ce1[:, -1] = F.cross_entropy(model.marcador(ctx[:, -1], l1[:, -1], R), y1[:, -1], reduction="none")
    else:
        l1, l2 = model(x1, x2, xs, use_teacher_forcing=True, s1_targets=y1)
        ce1 = F.cross_entropy(l1.transpose(1, 2), y1, reduction="none")
    ce = (ce1 + F.cross_entropy(l2.transpose(1, 2), y2, reduction="none")) / 2     # [B, T]
    aux = (y1 != tk.S1_SIN_HISTORIAL).float()
    aux[:, -1] = 0
    aux_loss = (ce * aux).sum(1) / aux.sum(1).clamp(min=1)
    per_seq = aux_loss + peso_marcador * ce[:, -1]
    return (per_seq * w).sum() / w.sum()


@torch.no_grad()
def predict_1x2(model, s1, s2, st, lote=1024, temperatura=1.0, R=None):
    """P(1), P(X), P(2) de cada secuencia, a partir de la predicción del token [PARTIDO].
    Con temperatura != 1 se recalibran las probabilidades de los 25 marcadores."""
    model.eval()
    out = []
    for i in range(0, len(s1), lote):
        sl = slice(i, i + lote)
        # La entrada acaba en [PARTIDO]: su última posición predice el marcador
        logits = logits_marcador(model, s1[sl, :-1], s2[sl, :-1], st[sl, :-1], sel(R, sl))
        p_score = F.softmax(logits / temperatura, dim=-1)                         # solo los 25 marcadores
        out.append(p_score @ OUTCOME.to(p_score.device))
    return torch.cat(out).cpu().numpy()


def fit_temperature(model, s1, s2, st, y, R=None):
    """Temperatura que minimiza la pérdida logarítmica 1X2 en validación (calibración)."""
    rejilla = np.round(np.arange(0.70, 1.501, 0.01), 2)
    perdidas = [metrics(predict_1x2(model, s1, s2, st, temperatura=t, R=R), y)["logloss"] for t in rejilla]
    return float(rejilla[int(np.argmin(perdidas))])


def fit_blend(p_fk, p_elo, y):
    """Combinación FutbolKronos + Elo ajustada en validación: softmax(a·log p_fk + b·log p_elo + c)."""
    lf = torch.tensor(np.log(np.clip(p_fk, 1e-9, 1)), dtype=torch.float32)
    le = torch.tensor(np.log(np.clip(p_elo, 1e-9, 1)), dtype=torch.float32)
    Y = torch.tensor(y)
    a, b = torch.ones(1, requires_grad=True), torch.zeros(1, requires_grad=True)
    c = torch.zeros(3, requires_grad=True)
    opt = torch.optim.LBFGS([a, b, c], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(a * lf + b * le + c, Y) + 1e-3 * (c ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.detach()), float(b.detach()), [float(x) for x in c.detach()]


def apply_blend(p_fk, p_elo, params):
    a, b, c = params
    z = a * np.log(np.clip(p_fk, 1e-9, 1)) + b * np.log(np.clip(p_elo, 1e-9, 1)) + np.array(c)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def elo_logistico(diff_tr, y_tr, diff_te):
    """Referencia "Elo europeo": P(1/X/2) = softmax(w·diferencia de Elo + b), ajustada con
    partidos anteriores al periodo de prueba."""
    X = torch.tensor(diff_tr / 400, dtype=torch.float32)[:, None]
    lin = torch.nn.Linear(1, 3)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(X), torch.tensor(y_tr))
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return F.softmax(lin(torch.tensor(diff_te / 400, dtype=torch.float32)[:, None]), -1).numpy()


def metrics(p, y):
    p = np.clip(p, 1e-12, 1)
    onehot = np.eye(3)[y]
    return {"logloss": float(-np.log(p[np.arange(len(y)), y]).mean()),
            "brier": float(((p - onehot) ** 2).sum(1).mean()),
            "acierto": float((p.argmax(1) == y).mean())}


def train(cfg, data, idx_train, w_train, args, idx_val=None, y_val=None, epocas=None, semilla=0, etiqueta="",
          model=None, lr=None, best_inicial=float("inf")):
    """Entrena un modelo. Con idx_val usa parada temprana; si no, entrena `epocas` épocas.
    Con `model` continúa entrenando ese modelo (ajuste fino); `best_inicial` es su pérdida en
    validación de partida, y si el ajuste no la mejora se devuelve el modelo tal cual (época 0)."""
    s1, s2, st, R = data
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    if model is None:
        model = crear(cfg, args).to(s1.device)
    w_train = w_train.to(s1.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    n = len(idx_train)
    pasos_epoca = math.ceil(n / args.lote)
    max_epocas = epocas if epocas is not None else (args.epocas_plan or args.epocas)
    if args.lr_plan == "coseno":
        # calentamiento corto y bajada en coseno hasta la última época prevista
        total = (args.epocas_plan or max_epocas) * pasos_epoca
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr or args.lr, total_steps=total + 1, pct_start=0.15)
    else:
        warmup = pasos_epoca
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warmup))
    best, best_ep, best_state, sin_mejora = best_inicial, 0, None, 0
    if idx_val is not None and math.isfinite(best_inicial):
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    historia = []
    for ep in range(1, max_epocas + 1):
        model.train()
        t0 = time.time()
        perm = idx_train[torch.randperm(n)]
        total = 0.0
        for i in range(0, n, args.lote):
            b = perm[i:i + args.lote]
            st_b = st[b]
            loss = loss_fn(model, s1[b], s2[b], st_b, w_train[b], args.peso_marcador, sel(R, b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        linea = f"   {etiqueta}época {ep:>2}  pérdida entrenamiento {total / n:.4f}"
        if idx_val is not None:
            m = metrics(predict_1x2(model, s1[idx_val], s2[idx_val], st[idx_val], R=sel(R, idx_val)), y_val)
            historia.append(m["logloss"])
            linea += f"  ·  validación: pérdida log {m['logloss']:.4f}, acierto {m['acierto'] * 100:.1f} %"
            if m["logloss"] < best - 1e-4:
                best, best_ep, sin_mejora = m["logloss"], ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                linea += "  *"
            else:
                sin_mejora += 1
        print(f"{linea}  ({time.time() - t0:.1f} s)", flush=True)
        if idx_val is not None and sin_mejora >= args.paciencia:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_ep, historia


# ------------------------------------------------------------------ programa
def parse_half_lives(txt):
    return [float("inf") if v.strip().lower() in ("inf", "infinito") else float(v) for v in txt.split(",")]


def fmt_hl(h):
    return "sin olvido" if math.isinf(h) else f"{h:g} temporada{'s' if h != 1 else ''}"


def opciones_secuencias(secuencias=""):
    """Opciones con las que se crearon las secuencias (para reconstruirlas igual al predecir)."""
    try:
        with open(os.path.join(DATA, f"vocabulario_ligas{secuencias}.json"), encoding="utf-8") as f:
            v = json.load(f)
        o = v.get("opciones", {})
        return {"rojas": o.get("rojas", False), "descanso": o.get("descanso", False),
                "ausencias": o.get("ausencias", "no"), "valor": o.get("valor", "no"),
                "europa": o.get("europa", False),
                "n_hist": v.get("n_historial", 20)}
    except FileNotFoundError:
        return {}


def produccion(args, cfg, data, season, laliga):
    """Modelo para predecir partidos futuros: la configuración elegida en la última evaluación
    (semivida, épocas, ajuste), pero entrenando con todos los partidos hasta hoy.
    Guarda cada semilla en pesos/futbol-<modelo>[-ligas]-produccion/semilla_<k>."""
    nombre = f"futbol-{args.modelo}{'-ligas' if args.datos == 'ligas' else ''}"
    with open(os.path.join(BASE, "pesos", f"{nombre}-entrenado", "entrenamiento.json"), encoding="utf-8") as f:
        ev = json.load(f)
    h = float("inf") if ev["semivida"] is None else ev["semivida"]
    idx = torch.arange(len(season))
    # Mismas épocas que en la evaluación: con más partidos cada época ya trae más pasos, así que
    # alargarlas además (como se hacía antes) entrenaba de más (4 épocas cuando la validación eligió 3)
    epocas = max(1, ev["epocas"])
    idx_es = idx[torch.tensor(laliga)]
    epocas_aj = ev.get("epocas_ajuste", 0)
    for op, defecto in (("cabeza", "softmax"), ("rasgos", False)):
        if ev.get("args", {}).get(op, defecto) != getattr(args, op):
            print(f"AVISO: la evaluación usó --{op} {ev.get('args', {}).get(op, defecto)} y ahora {getattr(args, op)}.")
    plan_eval = ev.get("args", {}).get("lr_plan", "constante")
    if plan_eval != args.lr_plan:
        print(f"AVISO: la evaluación usó --lr-plan {plan_eval} y ahora se pide {args.lr_plan}: "
              f"usa el mismo para que el número de épocas elegido valga.")
    w = recency_weights(season, season.max(), h)
    print(f"Modo producción · semivida {fmt_hl(h)}, {epocas} épocas"
          f"{f' + {epocas_aj} de ajuste en LaLiga' if epocas_aj else ''}, {len(idx):,} partidos "
          f"(hasta {season.max()}/{(season.max() + 1) % 100:02d}), {args.semillas} semilla(s)\n")
    out = os.path.join(BASE, "pesos", f"{nombre}-produccion")
    for k in range(args.semillas):
        model, _, _, _ = train(cfg, data, idx, w, args, epocas=epocas, semilla=100 + k, etiqueta=f"semilla {k + 1} · ")
        if epocas_aj:
            model, _, _, _ = train(cfg, data, idx_es, w, args, epocas=epocas_aj, semilla=100 + k, model=model,
                                   lr=args.lr_ajuste, etiqueta=f"semilla {k + 1} · ajuste · ")
        model.save_pretrained(os.path.join(out, f"semilla_{k + 1}"))
    with open(os.path.join(out, "entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump({**ev, "secuencias": opciones_secuencias(args.secuencias), "produccion": True,
                   "usar_calibracion": bool(args.calibrar), "epocas": epocas, "epocas_ajuste": epocas_aj,
                   "entrenado_hasta": str(season.max()), "partidos": len(idx), "semillas": args.semillas,
                   "fecha": pd.Timestamp.today().strftime("%Y-%m-%d")}, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: pesos/{nombre}-produccion (semilla_1 … semilla_{args.semillas})")


def main():
    ap = argparse.ArgumentParser(description="Entrenar FutbolKronos")
    ap.add_argument("--modelo", choices=list(MODELOS), default="reducido")
    ap.add_argument("--datos", choices=["laliga", "ligas"], default="laliga",
                    help="laliga = solo Primera; ligas = preentrenar con las 22 divisiones europeas")
    ap.add_argument("--ajuste", type=int, default=0,
                    help="máximo de épocas de ajuste fino solo con LaLiga tras el preentrenamiento (0 = sin ajuste)")
    ap.add_argument("--semividas", default="inf,8,4,2,1", help="en temporadas; inf = todas pesan igual")
    ap.add_argument("--epocas", type=int, default=60, help="máximo de épocas en la fase 1")
    ap.add_argument("--paciencia", type=int, default=6, help="épocas sin mejorar antes de parar")
    ap.add_argument("--lote", type=int, default=0, help="partidos por lote (0 = 128 con laliga, 512 con ligas)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-plan", choices=["constante", "coseno"], default="constante",
                    help="ritmo de aprendizaje: constante tras 1 época de calentamiento, o bajada en coseno")
    ap.add_argument("--epocas-coseno", type=int, default=6,
                    help="con --lr-plan coseno: longitud del plan (épocas) en todas las fases")
    ap.add_argument("--cabeza", choices=["softmax", "poisson"], default="softmax",
                    help="marcador: 25 probabilidades libres (softmax) o Poisson bivariante de Dixon-Coles")
    ap.add_argument("--rasgos", action="store_true",
                    help="híbrido: añade rasgos numéricos (promedios móviles, descanso, Elo; futbol_rasgos.py)")
    ap.add_argument("--validacion", choices=["todas", "laliga"], default="todas",
                    help="partidos para elegir semivida, épocas y temperatura: todas las ligas 2018–2021 "
                         "(~22.000, mucho menos ruido) o solo LaLiga (1.140). La prueba sigue siendo LaLiga.")
    ap.add_argument("--secuencias", default="",
                    help="sufijo de los archivos de secuencias (crear_secuencias_ligas.py --sufijo)")
    ap.add_argument("--lr-ajuste", type=float, default=3e-4, help="ritmo de aprendizaje del ajuste fino")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--peso-marcador", type=float, default=1.0,
                    help="peso del marcador final frente a la media del resto de posiciones")
    ap.add_argument("--semillas", type=int, default=3, help="modelos finales a promediar (como las simulaciones de Kronos)")
    ap.add_argument("--hilos", type=int, default=0, help="hilos del procesador (0 = automático)")
    ap.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="cpu",
                    help="cpu = procesador, cuda = gráfica NVIDIA, auto = la gráfica si está disponible")
    ap.add_argument("--rapido", action="store_true", help="prueba exprés: 2 semividas, 3 épocas, 1 semilla, no guarda")
    ap.add_argument("--solo-validacion", action="store_true",
                    help="solo la fase 1 (para comparar variantes sin mirar la prueba)")
    ap.add_argument("--semilla-base", type=int, default=0, help="semilla de la fase 1")
    ap.add_argument("--calibrar", action="store_true",
                    help="con --produccion: aplicar al predecir la temperatura calibrada en validación")
    ap.add_argument("--produccion", action="store_true",
                    help="entrenar con todos los partidos para predecir (usa la configuración de la última evaluación)")
    args = ap.parse_args()
    # con coseno el plan dura siempre lo mismo (fase 1, fase 2 y producción): solo cambia dónde se para
    args.epocas_plan = args.epocas_coseno if args.lr_plan == "coseno" else 0
    if args.rapido:
        args.semividas, args.epocas, args.paciencia, args.semillas = "inf,2", 3, 3, 1
        args.epocas_plan = min(args.epocas_plan, 3) if args.epocas_plan else 0
        args.ajuste = min(args.ajuste, 2)
    if not args.lote:
        args.lote = 512 if args.datos == "ligas" else 128
    if args.hilos:
        torch.set_num_threads(args.hilos)
    else:
        torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.dispositivo == "auto":
        args.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: este PyTorch no ve la gráfica. Usa ..\\venv-gpu\\Scripts\\python.exe")
    dev = torch.device(args.dispositivo)
    donde = (f"gráfica {torch.cuda.get_device_name(0)}" if dev.type == "cuda"
             else f"procesador, {torch.get_num_threads()} hilos")

    cfg = model_config(args.modelo)
    if args.rasgos:
        args.r_stats = ([0.0] * futbol_rasgos.N_RASGOS, [1.0] * futbol_rasgos.N_RASGOS)
    n_params = sum(p.numel() for p in crear(cfg, args).parameters())
    print(f"Paso 5 · Entrenar FutbolKronos ({args.modelo}, {n_params:,} parámetros, {donde})")
    print(f"   datos: {'22 divisiones europeas' if args.datos == 'ligas' else 'solo LaLiga'}"
          f"{f', con ajuste fino en LaLiga (hasta {args.ajuste} épocas)' if args.ajuste else ''}\n")

    s1, s2, st, R, meta, y, laliga, evaluable = load(args.datos, args.secuencias)
    if args.rasgos and R is None:
        sys.exit("ERROR: las secuencias no traen rasgos numéricos. Regenera con crear_secuencias_ligas.py")
    if not args.rasgos:
        R = None
    s1, s2, st = s1.to(dev), s2.to(dev), st.to(dev)
    R_cpu = R
    R = None if R is None else R.to(dev)
    data = (s1, s2, st, R)
    season = meta["temporada_inicio"].to_numpy()
    idx_tr = torch.from_numpy(np.where(season <= FIN_ENTRENAMIENTO)[0])
    en_val = (season > FIN_ENTRENAMIENTO) & (season <= FIN_VALIDACION)
    val_ok = meta["resultado"].notna().to_numpy() if (args.validacion == "todas" and args.datos == "ligas") else evaluable
    idx_va = torch.from_numpy(np.where(val_ok & en_val)[0])
    idx_va_es = torch.from_numpy(np.where(evaluable & en_val)[0])        # LaLiga: para la mezcla con el Elo
    idx_te = torch.from_numpy(np.where(evaluable & (season > FIN_VALIDACION))[0])
    es_laliga = torch.tensor(laliga)
    print(f"   entrenamiento: {len(idx_tr):,} partidos (1993/94–{FIN_ENTRENAMIENTO}/{(FIN_ENTRENAMIENTO + 1) % 100:02d})"
          f"{f', {int(es_laliga[idx_tr].sum()):,} de LaLiga' if args.datos == 'ligas' else ''}")
    donde_val = "todas las ligas" if len(idx_va) > len(idx_va_es) else "LaLiga"
    print(f"   validación:    {len(idx_va):,} partidos de {donde_val} ({FIN_ENTRENAMIENTO + 1}/{(FIN_ENTRENAMIENTO + 2) % 100:02d}–"
          f"{FIN_VALIDACION}/{(FIN_VALIDACION + 1) % 100:02d})")
    print(f"   prueba:        {len(idx_te):,} partidos de LaLiga ({FIN_VALIDACION + 1}/{(FIN_VALIDACION + 2) % 100:02d}–"
          f"{season.max()}/{(season.max() + 1) % 100:02d})\n")

    if args.produccion:
        if args.rasgos:
            args.r_stats = stats_rasgos(R_cpu, torch.arange(len(season)))
        return produccion(args, cfg, data, season, laliga)
    if args.rasgos:
        args.r_stats = stats_rasgos(R_cpu, idx_tr)

    # ---------------- Fase 1: elegir semivida y nº de épocas
    print("Fase 1 · Elegir semivida (entrenar hasta 2017/18, validar en 2018/19–2020/21)")
    y_va = y[idx_va.numpy()]
    y_va_es = y[idx_va_es.numpy()]
    elo_va = meta.loc[idx_va_es.numpy(), ["elo_prob_local", "elo_prob_empate", "elo_prob_visitante"]].to_numpy()
    print(f"   (referencia en validación LaLiga: Elo pérdida log {metrics(elo_va, y_va_es)['logloss']:.4f})\n")
    resultados = []                        # (semivida, pérdida validación, épocas preentreno, épocas ajuste)
    modelos_f1 = {}
    for h in parse_half_lives(args.semividas):
        w = recency_weights(season, FIN_ENTRENAMIENTO, h)
        w[idx_tr] = w[idx_tr] / w[idx_tr].mean()
        print(f" · Semivida: {fmt_hl(h)}")
        model, best, best_ep, _ = train(cfg, data, idx_tr, w, args, idx_va, y_va, semilla=args.semilla_base)
        print(f"   → mejor pérdida en validación {best:.4f} en la época {best_ep}")
        ep_aj = 0
        if args.ajuste:
            idx_aj = idx_tr[es_laliga[idx_tr]]
            model, best, ep_aj, _ = train(cfg, data, idx_aj, w, args, idx_va, y_va, epocas=args.ajuste,
                                          model=model, lr=args.lr_ajuste, best_inicial=best, etiqueta="ajuste · ")
            print(f"   → tras el ajuste fino: {best:.4f} ({ep_aj} épocas de ajuste)")
        resultados.append((h, best, best_ep, ep_aj))
        modelos_f1[h] = model
        print()
    h_best, val_best, ep_best, ep_aj_best = min(resultados, key=lambda r: r[1])
    print("   Resumen fase 1:")
    for h, best, ep, ea in resultados:
        print(f"     {fmt_hl(h):<16} {best:.4f}  (época {ep}{f' + {ea} de ajuste' if args.ajuste else ''})"
              f"{'   ← elegida' if h == h_best else ''}")

    # Calibración y combinación con el Elo, ajustadas en validación con el modelo de la fase 1
    m1 = modelos_f1[h_best]
    T = fit_temperature(m1, s1[idx_va], s2[idx_va], st[idx_va], y_va, R=sel(R, idx_va))
    val_T = metrics(predict_1x2(m1, s1[idx_va], s2[idx_va], st[idx_va], R=sel(R, idx_va), temperatura=T), y_va)["logloss"]
    p_va_T = predict_1x2(m1, s1[idx_va_es], s2[idx_va_es], st[idx_va_es], R=sel(R, idx_va_es), temperatura=T)
    mezcla = fit_blend(p_va_T, elo_va, y_va_es)
    val_mezcla = metrics(apply_blend(p_va_T, elo_va, mezcla), y_va_es)["logloss"]
    print(f"\n   Calibración: temperatura {T:.2f} → validación {val_T:.4f} (sin calibrar {val_best:.4f})")
    print(f"   Combinación con el Elo (validación LaLiga): pesos {mezcla[0]:.2f} · {mezcla[1]:.2f} → {val_mezcla:.4f}"
          f" (sin combinar {metrics(p_va_T, y_va_es)['logloss']:.4f})")
    if args.solo_validacion:
        print(f"\nVALIDACION {val_best:.4f} calibrado {val_T:.4f} combinado {val_mezcla:.4f}")
        return

    # ---------------- Fase 2: modelo final y prueba
    # Más datos que en la fase 1: se alargan las épocas en proporción
    idx_fin = torch.cat([idx_tr, torch.from_numpy(np.where((season > FIN_ENTRENAMIENTO) & (season <= FIN_VALIDACION))[0])])
    # Mismas épocas que las elegidas en la fase 1 (cada época ya es más larga con más datos)
    ep_final = max(1, ep_best)
    idx_fin_es = idx_fin[es_laliga[idx_fin]]
    ep_aj_final = ep_aj_best
    print(f"\nFase 2 · Modelo final: semivida {fmt_hl(h_best)}, {ep_final} épocas"
          f"{f' + {ep_aj_final} de ajuste en LaLiga' if args.ajuste else ''}, "
          f"entrenando hasta 2020/21, {args.semillas} semilla(s)")
    w = recency_weights(season, FIN_VALIDACION, h_best)
    w[idx_fin] = w[idx_fin] / w[idx_fin].mean()
    if args.rasgos:
        args.r_stats = stats_rasgos(R_cpu, idx_fin)
    nombre_modelo = f"futbol-{args.modelo}{'-ligas' if args.datos == 'ligas' else ''}-entrenado"
    out = os.path.join(BASE, "pesos", nombre_modelo)
    probs, probs_T = [], []
    # Champions League (si las secuencias incluyen Europa): se evalúa aparte, 2021/22 en adelante
    es_cl = (meta["liga"] == "CL").to_numpy()
    idx_cl = torch.from_numpy(np.where(es_cl & (season > FIN_VALIDACION))[0])
    probs_cl = []
    for k in range(args.semillas):
        model, _, _, _ = train(cfg, data, idx_fin, w, args, epocas=ep_final, semilla=100 + k,
                               etiqueta=f"semilla {k + 1} · ")
        if ep_aj_final:
            model, _, _, _ = train(cfg, data, idx_fin_es, w, args, epocas=ep_aj_final, semilla=100 + k,
                                   model=model, lr=args.lr_ajuste, etiqueta=f"semilla {k + 1} · ajuste · ")
        probs.append(predict_1x2(model, s1[idx_te], s2[idx_te], st[idx_te], R=sel(R, idx_te)))
        probs_T.append(predict_1x2(model, s1[idx_te], s2[idx_te], st[idx_te], R=sel(R, idx_te), temperatura=T))
        if len(idx_cl):
            probs_cl.append(predict_1x2(model, s1[idx_cl], s2[idx_cl], st[idx_cl], R=sel(R, idx_cl)))
        if k == 0 and not args.rapido:
            model.save_pretrained(out)
    p_te = np.mean(probs, axis=0)
    p_te_T = np.mean(probs_T, axis=0)

    te = meta.loc[idx_te.numpy()].reset_index(drop=True)
    y_te = y[idx_te.numpy()]
    elo_te = te[["elo_prob_local", "elo_prob_empate", "elo_prob_visitante"]].to_numpy()
    book_te = te[["prob_local", "prob_empate", "prob_visitante"]].to_numpy()
    freq = np.bincount(y[idx_fin_es.numpy()], minlength=3) / len(idx_fin_es)
    has_book = ~np.isnan(book_te).any(1)

    etiqueta_fk = f"FutbolKronos ({args.modelo}{', ligas' if args.datos == 'ligas' else ''})"
    filas = [("Siempre lo mismo (frecuencias históricas)", np.tile(freq, (len(y_te), 1))),
             ("Rating Elo", elo_te),
             (etiqueta_fk, p_te),
             (f"  … calibrado (temperatura {T:.2f})", p_te_T),
             ("  … calibrado y combinado con el Elo", apply_blend(p_te_T, elo_te, mezcla)),
             ("Casas de apuestas", book_te)]
    merc_te = None
    if "mercado_local" in te.columns:
        merc_te = te[["mercado_local", "mercado_empate", "mercado_visitante"]].to_numpy()
        has_book &= ~np.isnan(merc_te).any(1)
        filas.append(("Mercado (media de casas, sin margen)", merc_te))
    lines = [f"FutbolKronos · Resultados en prueba ({len(y_te):,} partidos de LaLiga, "
             f"{FIN_VALIDACION + 1}/{(FIN_VALIDACION + 2) % 100:02d}–{season.max()}/{(season.max() + 1) % 100:02d})",
             f"Modelo: {args.modelo}, {n_params:,} parámetros · datos: {args.datos} · semivida {fmt_hl(h_best)} · "
             f"{ep_final} épocas{f' + {ep_aj_final} de ajuste' if args.ajuste else ''} · "
             f"{args.semillas} semilla(s) promediadas",
             "",
             f"{'Modelo':<44}{'Pérdida log':>12}{'Brier':>9}{'Acierto':>10}"]
    for nombre, p in filas:
        m = metrics(p[has_book], y_te[has_book])
        lines.append(f"{nombre:<44}{m['logloss']:>12.4f}{m['brier']:>9.4f}{m['acierto'] * 100:>9.1f}%")
    # ¿La mejora sobre el Elo es real o suerte? Remuestreo de partidos
    d = (-np.log(np.clip(elo_te[np.arange(len(y_te)), y_te], 1e-12, 1))
         + np.log(np.clip(p_te[np.arange(len(y_te)), y_te], 1e-12, 1)))
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(f"\nMejora sobre el Elo: {d.mean():+.4f} (IC 95 % de {lo:+.4f} a {hi:+.4f}; "
                 f"mejor que el Elo en el {(boot > 0).mean() * 100:.0f} % de los remuestreos)")
    for nombre_ref, ref_p in [("las casas", book_te), ("el mercado", merc_te)]:
        if ref_p is None:
            continue
        ii = np.where(~np.isnan(ref_p).any(1))[0]
        for et, pp in [("FutbolKronos", p_te), ("FutbolKronos calibrado", p_te_T)]:
            dd_ = (-np.log(np.clip(ref_p[ii, y_te[ii]], 1e-12, 1)) + np.log(np.clip(pp[ii, y_te[ii]], 1e-12, 1)))
            bb = np.array([dd_[rng.integers(0, len(dd_), len(dd_))].mean() for _ in range(5000)])
            l2, h2 = np.percentile(bb, [2.5, 97.5])
            lines.append(f"{et} frente a {nombre_ref}: {dd_.mean():+.4f} (IC 95 % de {l2:+.4f} a {h2:+.4f}; "
                         f"mejor en el {(bb > 0).mean() * 100:.0f} % de los remuestreos)")
    if len(idx_cl) and "elo_local_antes" in meta.columns:
        icl = idx_cl.numpy()
        p_cl = np.mean(probs_cl, axis=0)
        y_cl = y[icl]
        dif = (meta["elo_local_antes"] - meta["elo_visitante_antes"]).to_numpy()
        tr_cl = np.where(es_cl & (season <= FIN_VALIDACION))[0]
        p_elo_cl = elo_logistico(dif[tr_cl], y[tr_cl], dif[icl])
        freq_cl = np.bincount(y[tr_cl], minlength=3) / len(tr_cl)
        lines.append("")
        lines.append(f"Champions League · prueba {FIN_VALIDACION + 1}/{(FIN_VALIDACION + 2) % 100:02d} en adelante "
                     f"({len(icl):,} partidos; no hay cuotas de casas para comparar)")
        for nombre, p in [("Siempre lo mismo", np.tile(freq_cl, (len(icl), 1))),
                          ("Elo europeo (desfase por país)", p_elo_cl), (etiqueta_fk, p_cl)]:
            m = metrics(p, y_cl)
            lines.append(f"{nombre:<44}{m['logloss']:>12.4f}{m['brier']:>9.4f}{m['acierto'] * 100:>9.1f}%")
    lines.append("")
    lines.append("Por temporada (pérdida log):  Elo · FutbolKronos · Casas")
    for t, g in te.groupby("temporada"):
        i = g.index.to_numpy()
        lines.append(f"   {t}   {metrics(elo_te[i], y_te[i])['logloss']:.4f} · "
                     f"{metrics(p_te[i], y_te[i])['logloss']:.4f} · {metrics(book_te[i], y_te[i])['logloss']:.4f}"
                     f"   ({len(i)} partidos)")
    lines.append("")
    lines.append("Fase 1 (validación 2018/19–2020/21), pérdida log por semivida:")
    for h, best, ep, ea in resultados:
        lines.append(f"   {fmt_hl(h):<16} {best:.4f}  (época {ep}{f' + {ea} de ajuste' if args.ajuste else ''})")
    texto = "\n".join(lines)
    print("\n" + texto)
    if args.rapido:
        print("\n(Prueba exprés: no se guarda nada.)")
        return

    sufijo = f"_ligas_{args.modelo}" if args.datos == "ligas" else ""
    with open(os.path.join(DATA, f"resultados_entrenamiento{sufijo}.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    pred = te[["fecha", "temporada", "local_id", "visitante_id", "goles_local", "goles_visitante", "resultado"]].copy()
    pred[["fk_local", "fk_empate", "fk_visitante"]] = p_te.round(4)
    pred[["elo_local", "elo_empate", "elo_visitante"]] = elo_te.round(4)
    pred[["casas_local", "casas_empate", "casas_visitante"]] = book_te.round(4)
    pred.to_csv(os.path.join(DATA, f"predicciones_prueba{sufijo}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out, "entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump({"modelo": args.modelo, "datos": args.datos, "parametros": n_params,
                   "semivida": None if math.isinf(h_best) else h_best,
                   "epocas": ep_final, "epocas_ajuste": ep_aj_final,
                   "entrenado_hasta": f"{FIN_VALIDACION}/{(FIN_VALIDACION + 1) % 100:02d}",
                   "temperatura_calibrada": T, "combinacion_elo": list(mezcla),
                   "secuencias": opciones_secuencias(args.secuencias),
                   "fase1": [{"semivida": None if math.isinf(h) else h, "val_logloss": b, "epoca": e, "epocas_ajuste": ea}
                             for h, b, e, ea in resultados],
                   "args": {k: v for k, v in vars(args).items() if k != "r_stats"}}, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: pesos/{nombre_modelo}, datos/resultados_entrenamiento{sufijo}.txt, "
          f"datos/predicciones_prueba{sufijo}.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: falta {e.filename}. Ejecuta antes crear_secuencias.py "
              f"(o descargar_ligas.py y crear_secuencias_ligas.py para --datos ligas).", file=sys.stderr)
        sys.exit(1)
