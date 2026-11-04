"""
Prevision de precios con Kronos-base.

Descarga el historico de un activo de Yahoo Finance, le pide a Kronos que
genere las siguientes velas y guarda un grafico (PNG) y una tabla (CSV)
con el resultado en la carpeta "resultados".

Uso:  python predecir.py                 (te pregunta el activo)
      python predecir.py BTC-USD 1d 30   (simbolo, intervalo, velas a predecir)
"""
import os
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "Kronos-code"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from model import Kronos, KronosTokenizer, KronosPredictor

LOOKBACK = 400        # velas de historico que ve el modelo (maximo 512)
SAMPLES = 5           # trayectorias generadas y promediadas


def preguntar(texto, defecto):
    r = input(f"{texto} [{defecto}]: ").strip()
    return r or defecto


def cargar_modelo():
    """Usa los pesos locales si estan; si no, los baja de Hugging Face (solo la primera vez)."""
    local_tok = os.path.join(BASE, "models", "Kronos-Tokenizer-base")
    local_mod = os.path.join(BASE, "models", "Kronos-base")
    tiene_pesos = lambda p: os.path.exists(os.path.join(p, "model.safetensors"))
    tok_src = local_tok if tiene_pesos(local_tok) else "NeoQuasar/Kronos-Tokenizer-base"
    mod_src = local_mod if tiene_pesos(local_mod) else "NeoQuasar/Kronos-base"
    print(f"Cargando tokenizador desde: {tok_src}")
    tokenizer = KronosTokenizer.from_pretrained(tok_src)
    print(f"Cargando modelo desde:      {mod_src}")
    model = Kronos.from_pretrained(mod_src)
    return KronosPredictor(model, tokenizer, max_context=512)


def descargar_datos(simbolo, intervalo):
    periodo = "730d" if intervalo == "1h" else "max"
    print(f"Descargando datos de {simbolo} ({intervalo}) desde Yahoo Finance...")
    df = yf.download(simbolo, period=periodo, interval=intervalo,
                     auto_adjust=False, progress=False)
    if df.empty:
        raise SystemExit(f"No hay datos para '{simbolo}'. Revisa el simbolo (ej: BTC-USD, AAPL, SAN.MC).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df["amount"] = df["volume"] * df["close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "timestamps"
    return df.reset_index()


def fechas_futuras(ultima, n, intervalo, es_cripto):
    if intervalo == "1d":
        freq = "D" if es_cripto else "B"   # las acciones no cotizan en fin de semana
    else:
        freq = "h"
    return pd.Series(pd.date_range(ultima, periods=n + 1, freq=freq)[1:])


def main():
    args = sys.argv[1:]
    simbolo = args[0] if len(args) > 0 else preguntar("Simbolo de Yahoo Finance", "BTC-USD")
    intervalo = args[1] if len(args) > 1 else preguntar("Intervalo (1d o 1h)", "1d")
    pred_len = int(args[2] if len(args) > 2 else preguntar("Cuantas velas predecir", "30"))
    if intervalo not in ("1d", "1h"):
        raise SystemExit("El intervalo debe ser 1d o 1h.")

    df = descargar_datos(simbolo, intervalo)
    lookback = min(LOOKBACK, len(df))
    if lookback < 50:
        raise SystemExit(f"Solo hay {len(df)} velas; hacen falta al menos 50.")
    hist = df.tail(lookback).reset_index(drop=True)

    es_cripto = simbolo.upper().endswith(("-USD", "-EUR", "-USDT"))
    y_ts = fechas_futuras(hist["timestamps"].iloc[-1], pred_len, intervalo, es_cripto)

    predictor = cargar_modelo()
    print(f"Generando {pred_len} velas a partir de las ultimas {lookback} "
          f"({SAMPLES} trayectorias promediadas, dispositivo: {predictor.device})...")
    pred = predictor.predict(
        df=hist[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=hist["timestamps"],
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0, top_p=0.9, sample_count=SAMPLES,
    )

    out_dir = os.path.join(BASE, "resultados")
    os.makedirs(out_dir, exist_ok=True)
    nombre = f"{simbolo.replace('^', '')}_{intervalo}_{datetime.now():%Y%m%d_%H%M}"
    csv_path = os.path.join(out_dir, nombre + ".csv")
    png_path = os.path.join(out_dir, nombre + ".png")
    pred.to_csv(csv_path)

    ultimo = hist["close"].iloc[-1]
    final = pred["close"].iloc[-1]
    cambio = (final / ultimo - 1) * 100

    vista = hist.tail(150)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vista["timestamps"], vista["close"], label="Historico (cierre)", color="#1f77b4")
    ax.plot([vista["timestamps"].iloc[-1]] + list(y_ts),
            [ultimo] + list(pred["close"]), label="Prevision Kronos", color="#d62728")
    ax.axvline(vista["timestamps"].iloc[-1], color="gray", ls="--", lw=0.8)
    ax.set_title(f"{simbolo} ({intervalo}) - prevision de {pred_len} velas: {cambio:+.2f}%")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)

    print("\n================ RESULTADO ================")
    print(f"Ultimo cierre real:      {ultimo:,.2f}")
    print(f"Cierre previsto al final: {final:,.2f}  ({cambio:+.2f}%)")
    print(f"Grafico: {png_path}")
    print(f"Tabla:   {csv_path}")
    print("\nRecuerda: es una prevision estadistica, no una garantia ni un consejo de inversion.")
    os.startfile(png_path) if hasattr(os, "startfile") else None


if __name__ == "__main__":
    main()
