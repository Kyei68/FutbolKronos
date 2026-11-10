"""
Paso 1: crear un modelo con la arquitectura de Kronos-mini, VACIO (sin entrenar).

Construye la red con los mismos tamaños que Kronos-mini (configs/kronos_mini.json),
cuenta sus parámetros, le pasa una secuencia de prueba para comprobar que funciona
y guarda los pesos iniciales en la carpeta "pesos".

Uso (desde esta carpeta):  python3 crear_modelo.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

import torch
from kronos_futbol import Kronos


def main():
    # 1. Leer la configuración (los "planos" de Kronos-mini)
    with open(os.path.join(BASE, "configs", "kronos_mini.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    print("Configuración de Kronos-mini:")
    for k, v in cfg.items():
        print(f"   {k:<16} = {v}")

    # 2. Construir la red con pesos aleatorios
    torch.manual_seed(0)
    model = Kronos(**cfg)

    # 3. Contar parámetros por bloque
    print("\nParámetros por bloque:")
    total = 0
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        total += n
        if n:
            print(f"   {name:<12} {n:>10,}")
    print(f"   {'TOTAL':<12} {total:>10,}   (Kronos-mini oficial: ~4,1 millones)")

    # 4. Prueba: una "frase" inventada de 64 tokens, en 2 lotes
    vocab_s1 = 2 ** cfg["s1_bits"]
    vocab_s2 = 2 ** cfg["s2_bits"]
    B, T = 2, 64
    s1 = torch.randint(0, vocab_s1, (B, T))
    s2 = torch.randint(0, vocab_s2, (B, T))
    model.eval()
    with torch.no_grad():
        s1_logits, s2_logits = model(s1, s2, use_teacher_forcing=True, s1_targets=s1)
    print("\nPrueba de funcionamiento:")
    print(f"   entrada : {tuple(s1.shape)}  (lotes, tokens)")
    print(f"   salida 1: {tuple(s1_logits.shape)}  (probabilidad de cada token grueso siguiente)")
    print(f"   salida 2: {tuple(s2_logits.shape)}  (probabilidad de cada token fino siguiente)")

    # Sin entrenar, el modelo adivina al azar: la pérdida debe rondar ln(1024) = 6,93
    loss, _, _ = model.head.compute_loss(s1_logits[:, :-1], s2_logits[:, :-1], s1[:, 1:], s2[:, 1:])
    print(f"   pérdida sin entrenar: {loss.item():.2f}  (azar puro ≈ 6.93)")

    # 5. Guardar los pesos iniciales
    out = os.path.join(BASE, "pesos", "futbol-mini-sin-entrenar")
    try:
        model.save_pretrained(out)
        print(f"\nModelo vacío guardado en: {out}")
    except Exception as e:  # no es imprescindible para seguir
        torch.save(model.state_dict(), os.path.join(BASE, "pesos_sin_entrenar.pt"))
        print(f"\nModelo guardado como pesos_sin_entrenar.pt ({e})")

    print("\nTodo correcto. El siguiente paso es conseguir los datos y crear el vocabulario de fútbol.")


if __name__ == "__main__":
    main()
