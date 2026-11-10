"""
Compara variantes de FutbolKronos SOLO en validación (2018/19–2020/21), con varias semillas.
La prueba (2021–2026) no se mira aquí: se reserva para la combinación ganadora.
Por defecto entrenar.py valida en todas las ligas (~22.000 partidos), no solo en LaLiga.

Cada variante = opciones de crear_secuencias_ligas.py + opciones de entrenar.py.
Resultados en datos/experimentos.txt (se añaden al final, con fecha).

Uso (desde esta carpeta):  python3 experimentos.py [--semillas 2] [--solo base,rojas]
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
GPU_PY = os.path.join(BASE, "..", "venv-gpu", "Scripts", "python.exe")
# Las variantes se escriben en datos/*_exp.*: las secuencias de producción nunca se tocan
# (antes se sobrescribían y, si el proceso se cortaba, quedaban con la última variante).
SUFIJO = "_exp"

# nombre: (opciones de secuencias, opciones de entrenamiento)
VARIANTES = {
    "base":            (["--europa"], []),
    "coseno":          (["--europa"], ["--lr-plan", "coseno"]),
    "poisson":         (["--europa"], ["--lr-plan", "coseno", "--cabeza", "poisson"]),
    "hibrido":         (["--europa"], ["--lr-plan", "coseno", "--rasgos"]),
    "poisson_hibrido": (["--europa"], ["--lr-plan", "coseno", "--cabeza", "poisson", "--rasgos"]),
    "peso3":           (["--europa"], ["--peso-marcador", "3"]),
    "peso10":          (["--europa"], ["--peso-marcador", "10"]),
    "rojas":           (["--europa", "--rojas"], []),
    "descanso":        (["--europa", "--descanso"], []),
    "rojas_descanso":  (["--europa", "--rojas", "--descanso"], []),
    "historial40":     (["--europa", "--historial", "40"], []),
    "historial10":     (["--europa", "--historial", "10"], []),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=2)
    ap.add_argument("--solo", help="variantes separadas por comas")
    ap.add_argument("--semividas", default="8")
    ap.add_argument("--extra", default="", help="opciones de entrenamiento comunes a todas")
    args = ap.parse_args()
    nombres = args.solo.split(",") if args.solo else list(VARIANTES)
    py = GPU_PY if os.path.exists(GPU_PY) else sys.executable
    filas, secuencias_actuales = [], None
    for nombre in nombres:
        seq, tr = VARIANTES[nombre]
        if seq != secuencias_actuales:
            subprocess.run([sys.executable, "crear_secuencias_ligas.py", *seq, "--sufijo", SUFIJO], cwd=BASE, check=True,
                           stdout=subprocess.DEVNULL)
            secuencias_actuales = seq
        vals = []
        for k in range(args.semillas):
            t0 = time.time()
            cmd = [py, "entrenar.py", "--datos", "ligas", "--solo-validacion", "--semividas", args.semividas,
                   "--semilla-base", str(k), "--hilos", "4", "--secuencias", SUFIJO, *tr, *args.extra.split()]
            if py == GPU_PY:
                cmd += ["--dispositivo", "cuda"]
            out = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, encoding="utf-8").stdout
            m = re.search(r"VALIDACION ([\d.]+) calibrado ([\d.]+) combinado ([\d.]+)", out)
            if not m:
                print(out[-2000:])
                raise SystemExit(f"falló la variante {nombre}")
            vals.append([float(x) for x in m.groups()])
            print(f"   {nombre:<16} semilla {k}: {vals[-1][0]:.4f} (calibrado {vals[-1][1]:.4f}, "
                  f"con Elo {vals[-1][2]:.4f})  {time.time() - t0:.0f} s", flush=True)
        v = np.array(vals)
        filas.append((nombre, v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 2].mean()))
        # se guarda al terminar cada variante: si la tanda se corta, lo hecho no se pierde
        with open(os.path.join(BASE, "datos", "experimentos_parcial.txt"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} · {nombre:<24} pérdida {filas[-1][1]:.4f} "
                    f"± {filas[-1][2]:.4f} · calibrado {filas[-1][3]:.4f} · con Elo {filas[-1][4]:.4f} · "
                    f"{' '.join(seq + tr)}\n")
    base = dict((f[0], f[1]) for f in filas).get("base")
    lineas = [f"\n=== {time.strftime('%Y-%m-%d %H:%M')} · validación 2018/19–2020/21 · {args.semillas} semillas · "
              f"semividas {args.semividas} {args.extra}".rstrip(),
              f"{'variante':<16}{'pérdida log':>12}{'± semillas':>12}{'calibrado':>11}{'con Elo':>10}{'vs base':>10}"]
    for n, m, sd, c, e in sorted(filas, key=lambda f: f[1]):
        dif = f"{m - base:+.4f}" if base is not None else ""
        lineas.append(f"{n:<16}{m:>12.4f}{sd:>12.4f}{c:>11.4f}{e:>10.4f}{dif:>10}")
    texto = "\n".join(lineas)
    print(texto)
    with open(os.path.join(BASE, "datos", "experimentos.txt"), "a", encoding="utf-8") as f:
        f.write(texto + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
