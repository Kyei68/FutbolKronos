"""
¿Cuánto pesa cada partido del historial en la predicción? (diagnóstico, no cambia nada)

Usa el modelo de evaluación (entrenado hasta 2020/21) sobre partidos de 2021/22 en adelante,
que no ha visto, con el historial completo (20 + 20 velas). Tres medidas:

1. Atención: desde la posición [PARTIDO] (la que predice) hacia cada vela del historial,
   por capa, media de cabezas y partidos. Posición 1 = el partido más reciente.
2. Sensibilidad: se tapa UNA vela (se convierte en [sin historial]) y se mide cuánto cambian
   las probabilidades 1X2 (distancia media absoluta). Mide la influencia real, no solo la atención.
3. Recortar el historial: se tapan los partidos más antiguos (quedarse con los últimos k) y se mide
   la pérdida logarítmica 1X2. Tapar los más antiguos es lo mismo que le pasa a un equipo recién
   llegado, así que el modelo ya lo conoce (no es una situación artificial).

Uso:  python3 analizar_atencion.py [--partidos 4000] [--liga SP1]
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
import futbol_tokenizer as tk
from crear_secuencias_ligas import construir, leer_resultados
from kronos_futbol import Kronos
from kronos_futbol.module import MultiHeadAttentionWithRoPE

GOLES = np.array([tk.score_from_id(i) for i in range(tk.N_SCORES)])
M1 = torch.tensor((GOLES[:, 0] > GOLES[:, 1]).astype(np.float32))
MX = torch.tensor((GOLES[:, 0] == GOLES[:, 1]).astype(np.float32))
M2 = torch.tensor((GOLES[:, 0] < GOLES[:, 1]).astype(np.float32))

CAPTURA = []          # atención de la última posición, una entrada por capa


def forward_con_captura(self, x, key_padding_mask=None):
    b, n, _ = x.shape
    q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    q, k = self.rotary(q, k)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
    s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    a = s.softmax(-1)
    CAPTURA.append(a[:, :, -1, :].mean(1))          # [lote, n] desde la última posición
    out = (a @ v).transpose(1, 2).contiguous().view(b, n, self.d_model)
    return self.resid_dropout(self.out_proj(out))


def p1x2(model, s1, s2, st, T, lote=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(s1), lote):
            lg, _ = model.decode_s1(s1[i:i + lote], s2[i:i + lote], st[i:i + lote])
            p = F.softmax(lg[:, -1, :tk.N_SCORES] / T, -1)
            out.append(torch.stack([p @ M1, p @ MX, p @ M2], 1))
    return torch.cat(out)


def tapar(s1, s2, st, posiciones):
    s1, s2, st = s1.clone(), s2.clone(), st.clone()
    s1[:, posiciones] = tk.S1_SIN_HISTORIAL
    s2[:, posiciones] = tk.S2_ESPECIAL
    st[:, posiciones] = 0
    return s1, s2, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidos", type=int, default=4000)
    ap.add_argument("--liga", default=None, help="p. ej. SP1; por defecto todas las ligas")
    args = ap.parse_args()
    torch.set_num_threads(4)

    ruta = os.path.join(BASE, "pesos", "futbol-reducido-ligas-entrenado")
    info = json.load(open(os.path.join(ruta, "entrenamiento.json"), encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        model = Kronos.from_pretrained(ruta)
    model.eval()
    T = info.get("temperatura_calibrada", 1.0)
    op = dict(info["secuencias"])
    N = op.pop("n_hist", 20)

    print("Construyendo secuencias (como en el entrenamiento)…", flush=True)
    df = leer_resultados(europa=op.get("europa", False))
    df = df.dropna(subset=["goles_local", "goles_visitante"]).reset_index(drop=True)
    S1, S2, ST, meta, _, nh = construir(df, verbose=False, n_hist=N, **op)
    sel = (meta["temporada_inicio"].to_numpy() >= 2021) & (nh[:, 0] == N) & (nh[:, 1] == N)
    if args.liga:
        sel &= (meta["liga"] == args.liga).to_numpy()
    idx = np.where(sel)[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(idx, min(args.partidos, len(idx)), replace=False))
    s1 = torch.from_numpy(S1[idx, :-1].astype(np.int64))
    s2 = torch.from_numpy(S2[idx, :-1].astype(np.int64))
    st = torch.from_numpy(ST[idx, :-1].astype(np.int64))
    gl, gv = meta["goles_local"].to_numpy()[idx], meta["goles_visitante"].to_numpy()[idx]
    y = torch.from_numpy(np.where(gl > gv, 0, np.where(gl == gv, 1, 2)))
    print(f"{len(idx)} partidos de 2021/22 en adelante{' · ' + args.liga if args.liga else ' · todas las ligas'},"
          f" historial completo\n")

    # posiciones del historial: local 1..N (N = más reciente), visitante N+2..2N+1
    loc = lambda r: N + 1 - r              # r = 1 (más reciente) … N (más antiguo)
    vis = lambda r: 2 * N + 2 - r

    # 1) atención
    orig = MultiHeadAttentionWithRoPE.forward
    MultiHeadAttentionWithRoPE.forward = forward_con_captura
    capas = None
    with torch.no_grad():
        for i in range(0, len(s1), 1024):
            CAPTURA.clear()
            model.decode_s1(s1[i:i + 1024], s2[i:i + 1024], st[i:i + 1024])
            c = torch.stack(CAPTURA[:len(model.transformer)]).sum(1)      # [capas, n]
            capas = c if capas is None else capas + c
    MultiHeadAttentionWithRoPE.forward = orig
    capas = (capas / len(s1)).numpy()
    print("1) ATENCIÓN desde [PARTIDO] (% del total, media de cabezas)")
    print("   capa  [INICIO] [SEP] [PARTIDO]  hist.local  hist.visit.")
    for c, a in enumerate(capas):
        print(f"   {c + 1:>4} {100 * a[0]:>8.1f} {100 * a[N + 1]:>5.1f} {100 * a[-1]:>9.1f}"
              f" {100 * a[1:N + 1].sum():>11.1f} {100 * a[N + 2:2 * N + 2].sum():>12.1f}")
    print("\n   Reparto dentro del historial (local+visitante, % de la atención al historial):")
    print("   partidos atrás  " + "".join(f"{f'{b}-{b + 4}':>9}" for b in range(1, N + 1, 5)))
    for c, a in enumerate(capas):
        h = np.array([a[loc(r)] + a[vis(r)] for r in range(1, N + 1)])
        h = 100 * h / h.sum()
        print(f"   capa {c + 1:<10}" + "".join(f"{h[b:b + 5].sum():>9.1f}" for b in range(0, N, 5)))

    # 2) sensibilidad a cada vela
    base = p1x2(model, s1, s2, st, T)
    sens = np.zeros(N)
    for r in range(1, N + 1):
        p = p1x2(model, *tapar(s1, s2, st, [loc(r), vis(r)]), T)
        sens[r - 1] = (p - base).abs().sum(1).mean().item() / 2      # cambio medio de probabilidad
    print("\n2) SENSIBILIDAD: cuánto cambian P(1X2) al tapar el partido r de cada equipo"
          " (puntos porcentuales)")
    print("   " + "  ".join(f"{r:>2}:{100 * sens[r - 1]:4.2f}" for r in range(1, 11)))
    print("   " + "  ".join(f"{r:>2}:{100 * sens[r - 1]:4.2f}" for r in range(11, N + 1)))
    print("   por bloques  " + "  ".join(f"{b}-{b + 4}: {100 * sens[b - 1:b + 4].mean():.2f}"
                                       for b in range(1, N + 1, 5)))

    # 3) recortar el historial (quedarse con los últimos k)
    llv = lambda p: -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-9))
    ll = lambda p: llv(p).mean().item()
    print("\n3) RECORTAR EL HISTORIAL: pérdida log 1X2 quedándose con los últimos k partidos de cada equipo")
    ref = ll(base)
    for k in [N, 15, 10, 5, 3, 1, 0]:
        tap = [loc(r) for r in range(k + 1, N + 1)] + [vis(r) for r in range(k + 1, N + 1)]
        p = base if k == N else p1x2(model, *tapar(s1, s2, st, tap), T)
        d = llv(p) - llv(base)
        print(f"   últimos {k:>2}: {ll(p):.4f}  ({d.mean().item():+.4f} ± {1.96 * d.std().item() / len(y) ** 0.5:.4f})")
    # ¿y si solo se quitan los recientes? (situación que el modelo no ha visto: solo orientativo)
    tap = [loc(r) for r in range(1, 6)] + [vis(r) for r in range(1, 6)]
    print(f"   sin los 5 más recientes (orientativo): {ll(p1x2(model, *tapar(s1, s2, st, tap), T)):.4f}")
    # error de muestreo aproximado de una diferencia emparejada
    print(f"\n   (referencia: {len(y)} partidos)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
