"""
Kronos - interfaz grafica de escritorio.

Abre con doble clic en "Abrir Kronos.bat", o desde PowerShell:
    python3 kronos_gui.py

Necesita las carpetas "Kronos-code" (codigo del proyecto) junto a este archivo.
"""
import os
import sys
import threading
import queue
import time
import traceback
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "Kronos-code"))
RESULTS_DIR = os.path.join(BASE, "resultados")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FuncFormatter, MaxNLocator

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
MODELS = {
    "Kronos-base  ·  102M  (más preciso)": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "Kronos-small  ·  25M  (equilibrado)": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "Kronos-mini  ·  4M  (rápido)": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
}

# Catalogo de activos: categoria -> [(simbolo de Yahoo, nombre)]
# Cualquier otro simbolo de Yahoo Finance se puede escribir a mano o buscar por nombre.
CATALOG = {
    "IBEX 35 (España)": [
        ("ACS.MC", "ACS"), ("ACX.MC", "Acerinox"), ("AENA.MC", "Aena"), ("AMS.MC", "Amadeus"),
        ("ANA.MC", "Acciona"), ("ANE.MC", "Acciona Energía"), ("BBVA.MC", "BBVA"), ("BKT.MC", "Bankinter"),
        ("CABK.MC", "CaixaBank"), ("CLNX.MC", "Cellnex"), ("COL.MC", "Inmobiliaria Colonial"),
        ("ELE.MC", "Endesa"), ("ENG.MC", "Enagás"), ("FDR.MC", "Fluidra"), ("FER.MC", "Ferrovial"),
        ("GRF.MC", "Grifols"), ("IAG.MC", "IAG (Iberia)"), ("IBE.MC", "Iberdrola"), ("IDR.MC", "Indra"),
        ("ITX.MC", "Inditex"), ("LOG.MC", "Logista"), ("MAP.MC", "Mapfre"), ("MRL.MC", "Merlin Properties"),
        ("MTS.MC", "ArcelorMittal"), ("NTGY.MC", "Naturgy"), ("PUIG.MC", "Puig"), ("RED.MC", "Redeia"),
        ("REP.MC", "Repsol"), ("ROVI.MC", "Laboratorios Rovi"), ("SAB.MC", "Banco Sabadell"),
        ("SAN.MC", "Banco Santander"), ("SCYR.MC", "Sacyr"), ("SLR.MC", "Solaria"),
        ("TEF.MC", "Telefónica"), ("UNI.MC", "Unicaja"),
    ],
    "Otras españolas": [
        ("MEL.MC", "Meliá Hotels"), ("VIS.MC", "Viscofan"), ("CIE.MC", "CIE Automotive"),
        ("EBRO.MC", "Ebro Foods"), ("ENC.MC", "Ence"), ("TRE.MC", "Técnicas Reunidas"),
        ("ALM.MC", "Almirall"), ("PHM.MC", "PharmaMar"), ("FAE.MC", "Faes Farma"), ("GEST.MC", "Gestamp"),
        ("CAF.MC", "CAF"), ("A3M.MC", "Atresmedia"), ("TL5.MC", "Mediaset España"), ("AIR.MC", "Airbus (Madrid)"),
    ],
    "EE. UU. · Tecnología": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"), ("GOOGL", "Alphabet (Google)"),
        ("AMZN", "Amazon"), ("META", "Meta (Facebook)"), ("TSLA", "Tesla"), ("AVGO", "Broadcom"),
        ("AMD", "AMD"), ("INTC", "Intel"), ("ORCL", "Oracle"), ("CRM", "Salesforce"), ("ADBE", "Adobe"),
        ("NFLX", "Netflix"), ("PLTR", "Palantir"), ("QCOM", "Qualcomm"), ("IBM", "IBM"), ("CSCO", "Cisco"),
        ("UBER", "Uber"), ("TSM", "TSMC (ADR)"), ("ASML", "ASML (ADR)"), ("COIN", "Coinbase"),
    ],
    "EE. UU. · Otros sectores": [
        ("BRK-B", "Berkshire Hathaway"), ("JPM", "JPMorgan"), ("BAC", "Bank of America"), ("GS", "Goldman Sachs"),
        ("V", "Visa"), ("MA", "Mastercard"), ("LLY", "Eli Lilly"), ("JNJ", "Johnson & Johnson"),
        ("PFE", "Pfizer"), ("UNH", "UnitedHealth"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
        ("MCD", "McDonald's"), ("WMT", "Walmart"), ("COST", "Costco"), ("DIS", "Disney"), ("NKE", "Nike"),
        ("XOM", "ExxonMobil"), ("CVX", "Chevron"), ("BA", "Boeing"), ("CAT", "Caterpillar"),
    ],
    "Europa": [
        ("ASML.AS", "ASML"), ("SAP.DE", "SAP"), ("SIE.DE", "Siemens"), ("ALV.DE", "Allianz"),
        ("DTE.DE", "Deutsche Telekom"), ("VOW3.DE", "Volkswagen"), ("BMW.DE", "BMW"),
        ("MBG.DE", "Mercedes-Benz"), ("ADS.DE", "Adidas"), ("MC.PA", "LVMH"), ("OR.PA", "L'Oréal"),
        ("RMS.PA", "Hermès"), ("TTE.PA", "TotalEnergies"), ("AIR.PA", "Airbus"), ("SAN.PA", "Sanofi"),
        ("BNP.PA", "BNP Paribas"), ("NESN.SW", "Nestlé"), ("NOVN.SW", "Novartis"), ("ROG.SW", "Roche"),
        ("NOVO-B.CO", "Novo Nordisk"), ("SHEL.L", "Shell"), ("AZN.L", "AstraZeneca"), ("HSBA.L", "HSBC"),
        ("ULVR.L", "Unilever"), ("BP.L", "BP"), ("ENI.MI", "Eni"), ("ISP.MI", "Intesa Sanpaolo"),
        ("UCG.MI", "UniCredit"), ("RACE.MI", "Ferrari"),
    ],
    "Asia": [
        ("7203.T", "Toyota"), ("6758.T", "Sony"), ("9984.T", "SoftBank"), ("7974.T", "Nintendo"),
        ("0700.HK", "Tencent"), ("9988.HK", "Alibaba (Hong Kong)"), ("BABA", "Alibaba (ADR)"),
        ("005930.KS", "Samsung Electronics"), ("2330.TW", "TSMC (Taiwán)"),
    ],
    "Índices": [
        ("^IBEX", "IBEX 35"), ("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "Nasdaq Composite"),
        ("^NDX", "Nasdaq 100"), ("^RUT", "Russell 2000"), ("^STOXX50E", "Euro Stoxx 50"),
        ("^GDAXI", "DAX (Alemania)"), ("^FCHI", "CAC 40 (Francia)"), ("^FTSE", "FTSE 100 (Reino Unido)"),
        ("FTSEMIB.MI", "FTSE MIB (Italia)"), ("^N225", "Nikkei 225 (Japón)"), ("^HSI", "Hang Seng"),
        ("^VIX", "VIX (volatilidad)"),
    ],
    "ETF": [
        ("SPY", "SPDR S&P 500"), ("VOO", "Vanguard S&P 500"), ("QQQ", "Invesco Nasdaq 100"),
        ("VTI", "Vanguard Total Market"), ("IWM", "iShares Russell 2000"), ("EEM", "iShares Mercados emergentes"),
        ("GLD", "SPDR Oro"), ("SLV", "iShares Plata"), ("TLT", "iShares Bonos EE. UU. 20+ años"),
        ("ARKK", "ARK Innovation"),
    ],
    "Criptomonedas": [
        ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"), ("XRP-USD", "XRP"),
        ("BNB-USD", "BNB"), ("ADA-USD", "Cardano"), ("DOGE-USD", "Dogecoin"), ("AVAX-USD", "Avalanche"),
        ("DOT-USD", "Polkadot"), ("LINK-USD", "Chainlink"), ("LTC-USD", "Litecoin"),
        ("BTC-EUR", "Bitcoin en euros"), ("ETH-EUR", "Ethereum en euros"),
    ],
    "Materias primas": [
        ("GC=F", "Oro"), ("SI=F", "Plata"), ("PL=F", "Platino"), ("HG=F", "Cobre"),
        ("CL=F", "Petróleo WTI"), ("BZ=F", "Petróleo Brent"), ("NG=F", "Gas natural"),
        ("ZW=F", "Trigo"), ("ZC=F", "Maíz"), ("KC=F", "Café"), ("CC=F", "Cacao"),
    ],
    "Divisas": [
        ("EURUSD=X", "Euro / Dólar"), ("GBPUSD=X", "Libra / Dólar"), ("USDJPY=X", "Dólar / Yen"),
        ("EURGBP=X", "Euro / Libra"), ("EURJPY=X", "Euro / Yen"), ("EURCHF=X", "Euro / Franco suizo"),
        ("USDCHF=X", "Dólar / Franco suizo"), ("AUDUSD=X", "Dólar australiano / Dólar"),
        ("USDCAD=X", "Dólar / Dólar canadiense"),
    ],
}
SEP = "  —  "


def entry_label(sym, name):
    return f"{sym}{SEP}{name}" if name else sym


# codigo -> (etiqueta, periodo de descarga en Yahoo, frecuencia pandas)
INTERVALS = {
    "1d": ("Diario", "max", "D"),
    "1wk": ("Semanal", "max", "7D"),
    "1h": ("1 hora", "730d", "h"),
    "30m": ("30 minutos", "60d", "30min"),
    "15m": ("15 minutos", "60d", "15min"),
}

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]
DISPLAY_CANDLES = 160          # velas de historico que se dibujan

# Paleta
BG = "#f5f6f8"
PANEL = "#ffffff"
TEXT = "#1f2328"
MUTED = "#667085"
BORDER = "#e3e6ea"
ACCENT = "#4f46e5"
UP = "#16a34a"
DOWN = "#dc2626"
PRED = "#7c3aed"
REAL = "#111827"


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def fetch_yahoo(symbol, interval):
    import yfinance as yf
    period = INTERVALS[interval][1]
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance no devolvió datos para «{symbol}» ({interval}).\n"
                         "Revisa el símbolo (ej.: BTC-USD, AAPL, SAN.MC, ^IBEX).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)            # conserva la hora local del mercado
    df.index = idx
    df.index.name = "timestamps"
    df["amount"] = df["volume"] * df["close"]
    return df.reset_index()


def load_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    for alias in ("timestamp", "date", "datetime", "time", "fecha"):
        if "timestamps" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "timestamps"})
    missing = [c for c in ["timestamps", "open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError("Al CSV le faltan columnas: " + ", ".join(missing) +
                         "\nNecesita: timestamps (o date), open, high, low, close "
                         "y opcionalmente volume y amount.")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df["close"]
    df = df[["timestamps"] + PRICE_COLS].dropna().sort_values("timestamps")
    return df.reset_index(drop=True)


def is_crypto(symbol):
    return symbol.upper().endswith(("-USD", "-EUR", "-USDT", "-BTC", "-ETH"))


def future_timestamps(ts, n, interval, symbol):
    """Fechas de las proximas n velas, imitando el calendario del historico."""
    ts = pd.Series(pd.to_datetime(ts)).reset_index(drop=True)
    last = ts.iloc[-1]
    if interval == "csv":
        step = ts.diff().dropna().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta("1D")
        freq = step
        trades_weekends = (ts.dt.dayofweek >= 5).any()
    else:
        freq = INTERVALS[interval][2]
        trades_weekends = is_crypto(symbol)
        if interval == "1d" and not trades_weekends:
            freq = "B"

    candidates = pd.date_range(last, periods=(n + 1) * 40, freq=freq)[1:]
    intraday = pd.Timedelta(candidates[1] - candidates[0]) < pd.Timedelta("1D")
    observed_times = set(ts.dt.time) if intraday and not trades_weekends else None

    out = []
    for t in candidates:
        if not trades_weekends and t.dayofweek >= 5:
            continue
        if observed_times is not None and t.time() not in observed_times:
            continue
        out.append(t)
        if len(out) == n:
            break
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Modelo (se carga una vez por tipo y se reutiliza)
# --------------------------------------------------------------------------- #
_PREDICTORS = {}


def get_predictor(model_key):
    if model_key not in _PREDICTORS:
        from model import Kronos, KronosTokenizer, KronosPredictor
        model_id, tok_id, ctx = MODELS[model_key]
        tokenizer = KronosTokenizer.from_pretrained(tok_id)
        model = Kronos.from_pretrained(model_id)
        _PREDICTORS[model_key] = KronosPredictor(model, tokenizer, max_context=ctx)
    return _PREDICTORS[model_key]


def run_forecast(p, q):
    """Se ejecuta en un hilo aparte. Comunica el progreso por la cola q."""
    try:
        q.put(("status", "Descargando datos…" if p["source"] == "yahoo" else "Leyendo CSV…"))
        if p["source"] == "yahoo":
            df = fetch_yahoo(p["symbol"], p["interval"])
            interval = p["interval"]
        else:
            df = load_csv(p["csv_path"])
            interval = "csv"

        pred_len = p["pred_len"]
        reserve = pred_len if p["test_mode"] else 0
        available = len(df) - reserve
        lookback = min(p["lookback"], available)
        if lookback < 50:
            raise ValueError(f"Hay muy pocos datos ({len(df)} velas). Hacen falta al menos "
                             f"{50 + reserve} con esta configuración.")

        if p["test_mode"]:
            hist = df.iloc[available - lookback:available].reset_index(drop=True)
            real = df.iloc[available:].reset_index(drop=True)
            y_ts = real["timestamps"]
        else:
            hist = df.iloc[-lookback:].reset_index(drop=True)
            real = None
            y_ts = future_timestamps(hist["timestamps"], pred_len, interval, p["symbol"])

        q.put(("status", "Cargando modelo… (la primera vez puede descargarlo)"))
        predictor = get_predictor(p["model"])

        sims = p["sims"]
        q.put(("status", f"Generando {sims} simulaciones de {pred_len} velas "
                         f"(dispositivo: {predictor.device})…"))
        x_df = hist[PRICE_COLS]
        x_ts = hist["timestamps"]
        paths = predictor.predict_batch(
            df_list=[x_df.copy() for _ in range(sims)],
            x_timestamp_list=[x_ts.copy() for _ in range(sims)],
            y_timestamp_list=[y_ts.copy() for _ in range(sims)],
            pred_len=pred_len, T=p["temperature"], top_k=0, top_p=p["top_p"],
            sample_count=1, verbose=True,
        )
        q.put(("done", {"params": p, "interval": interval, "hist": hist,
                        "real": real, "y_ts": y_ts, "paths": paths}))
    except Exception as e:
        q.put(("error", str(e), traceback.format_exc()))


def summarize(res):
    hist, paths, real = res["hist"], res["paths"], res["real"]
    closes = np.stack([pth["close"].to_numpy() for pth in paths])          # (sims, pred)
    stack = np.stack([pth[PRICE_COLS].to_numpy() for pth in paths])        # (sims, pred, 6)
    mean_df = pd.DataFrame(stack.mean(axis=0), columns=PRICE_COLS)
    mean_df.insert(0, "timestamps", pd.to_datetime(res["y_ts"]).to_numpy())
    mean_df["close_p10"] = np.percentile(closes, 10, axis=0)
    mean_df["close_p90"] = np.percentile(closes, 90, axis=0)
    last = float(hist["close"].iloc[-1])
    s = {
        "closes": closes, "mean": mean_df, "last": last,
        "final": float(mean_df["close"].iloc[-1]),
        "p10": float(mean_df["close_p10"].iloc[-1]),
        "p90": float(mean_df["close_p90"].iloc[-1]),
        "prob_up": float((closes[:, -1] > last).mean()),
    }
    s["change"] = s["final"] / last - 1
    if real is not None:
        r = real["close"].to_numpy()
        m = mean_df["close"].to_numpy()
        mean_df["real_close"] = r
        s["mape"] = float(np.mean(np.abs(m - r) / r))
        s["real_final"] = float(r[-1])
        s["dir_ok"] = bool(np.sign(m[-1] - last) == np.sign(r[-1] - last))
        s["coverage"] = float(np.mean((r >= mean_df["close_p10"]) & (r <= mean_df["close_p90"])))
    return s


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:,.2f} B"
    if a >= 1e6:
        return f"{v / 1e6:,.2f} M"
    if a >= 1000:
        return f"{v:,.2f}"
    if a >= 10:
        return f"{v:,.2f}"
    if a >= 2:
        return f"{v:,.3f}"
    return f"{v:,.4f}"


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
class KronosApp:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.worker = None
        self.t0 = None
        self.result = None
        self.summary = None
        self.plot_index = None
        self.csv_path = None

        root.title("Kronos · Previsión de mercados")
        root.geometry("1360x840")
        root.minsize(1100, 700)
        root.configure(bg=BG)
        self._styles()
        self._build()
        self._draw_empty()
        self.root.after(100, self._poll)

    # ---------- estilos ----------
    def _styles(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        base = ("Segoe UI", 10)
        self.root.option_add("*TCombobox*Listbox.font", base)
        st.configure(".", background=BG, foreground=TEXT, font=base)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        st.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        st.configure("Section.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9, "bold"))
        st.configure("Title.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 15, "bold"))
        st.configure("Big.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 20, "bold"))
        st.configure("Value.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10, "bold"))
        st.configure("KpiK.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        st.configure("KpiV.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 14, "bold"))
        st.configure("Info.TLabel", background=BG, foreground=MUTED, font=("Consolas", 9))
        st.configure("Panel.TCheckbutton", background=PANEL)
        st.configure("Panel.TRadiobutton", background=PANEL)
        st.map("Panel.TCheckbutton", background=[("active", PANEL)])
        st.map("Panel.TRadiobutton", background=[("active", PANEL)])
        st.configure("Accent.TButton", background=ACCENT, foreground="white",
                     font=("Segoe UI", 11, "bold"), padding=(10, 9), borderwidth=0)
        st.map("Accent.TButton", background=[("disabled", "#a5a6d8"), ("active", "#4338ca")],
               foreground=[("disabled", "#eef")])
        st.configure("TButton", padding=(8, 5))
        st.configure("Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT, thickness=6)
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(14, 6))
        st.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        st.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    # ---------- construccion ----------
    def _build(self):
        outer = ttk.Frame(self.root, style="TFrame", padding=12)
        outer.pack(fill="both", expand=True)

        # barra lateral con scroll
        side_wrap = tk.Frame(outer, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        side_wrap.pack(side="left", fill="y")
        canvas = tk.Canvas(side_wrap, bg=PANEL, width=330, highlightthickness=0)
        sb = ttk.Scrollbar(side_wrap, orient="vertical", command=canvas.yview)
        side = ttk.Frame(canvas, style="Panel.TFrame", padding=(16, 14))
        side.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=side, anchor="nw", width=330)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="y", expand=False)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
                        if self._mouse_over(canvas) else None)
        self._build_sidebar(side)

        # zona principal
        main = ttk.Frame(outer, padding=(12, 0, 0, 0))
        main.pack(side="left", fill="both", expand=True)
        self.kpi = tk.Frame(main, bg=BG)
        self.kpi.pack(fill="x", pady=(0, 10))
        self._kpi_placeholder()
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="both", expand=True)

        chart_tab = tk.Frame(self.nb, bg=PANEL)
        table_tab = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(chart_tab, text="  Gráfico  ")
        self.nb.add(table_tab, text="  Tabla de previsión  ")

        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor=PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_tab)
        tb_frame = tk.Frame(chart_tab, bg=PANEL)
        tb_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        self.info_var = tk.StringVar(value="Pasa el ratón sobre el gráfico para ver los valores de cada vela.")
        ttk.Label(main, textvariable=self.info_var, style="Info.TLabel").pack(fill="x", pady=(6, 0))

        cols = ("fecha", "open", "high", "low", "close", "p10", "p90", "volume", "real")
        heads = ("Fecha", "Apertura", "Máximo", "Mínimo", "Cierre (media)", "Cierre P10",
                 "Cierre P90", "Volumen", "Cierre real")
        self.tree = ttk.Treeview(table_tab, columns=cols, show="headings")
        for c, h in zip(cols, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=140 if c == "fecha" else 92, minwidth=70, anchor="e" if c != "fecha" else "w")
        ysb = ttk.Scrollbar(table_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        ysb.pack(side="right", fill="y", pady=8)

    def _mouse_over(self, widget):
        x, y = self.root.winfo_pointerxy()
        w = self.root.winfo_containing(x, y)
        while w is not None:
            if w == widget:
                return True
            w = w.master
        return False

    def _section(self, parent, text):
        ttk.Label(parent, text=text.upper(), style="Section.TLabel").pack(anchor="w", pady=(14, 4))

    def _build_sidebar(self, s):
        ttk.Label(s, text="Kronos", style="Title.TLabel").pack(anchor="w")
        ttk.Label(s, text="Modelo fundacional para velas financieras", style="Muted.TLabel").pack(anchor="w")

        # --- datos
        self._section(s, "Datos")
        self.source = tk.StringVar(value="yahoo")
        row = ttk.Frame(s, style="Panel.TFrame")
        row.pack(fill="x")
        self.source_row = row
        ttk.Radiobutton(row, text="Yahoo Finance", value="yahoo", variable=self.source,
                        style="Panel.TRadiobutton", command=self._toggle_source).pack(side="left")
        ttk.Radiobutton(row, text="Archivo CSV", value="csv", variable=self.source,
                        style="Panel.TRadiobutton", command=self._toggle_source).pack(side="left", padx=(12, 0))

        self.yahoo_box = ttk.Frame(s, style="Panel.TFrame")
        self.yahoo_box.pack(fill="x", pady=(6, 0))
        ttk.Label(self.yahoo_box, text="Categoría", style="Panel.TLabel").pack(anchor="w")
        self.category = tk.StringVar(value="Criptomonedas")
        cat_cb = ttk.Combobox(self.yahoo_box, textvariable=self.category, state="readonly", values=list(CATALOG))
        cat_cb.pack(fill="x", pady=(2, 6))
        cat_cb.bind("<<ComboboxSelected>>", self._category_changed)
        ttk.Label(self.yahoo_box, text="Activo (elige, o escribe para filtrar)", style="Panel.TLabel").pack(anchor="w")
        self.symbol = tk.StringVar(value=entry_label("BTC-USD", "Bitcoin"))
        self.symbol_cb = ttk.Combobox(self.yahoo_box, textvariable=self.symbol,
                                      values=[entry_label(*it) for it in CATALOG["Criptomonedas"]], height=16)
        self.symbol_cb.pack(fill="x", pady=(2, 4))
        self.symbol_cb.bind("<KeyRelease>", self._filter_symbols)
        ttk.Button(self.yahoo_box, text="🔍  Buscar cualquier empresa en Yahoo…",
                   command=self._open_search).pack(fill="x", pady=(0, 8))
        ttk.Label(self.yahoo_box, text="Intervalo de las velas", style="Panel.TLabel").pack(anchor="w")
        self.interval = tk.StringVar(value="1d  ·  Diario")
        ttk.Combobox(self.yahoo_box, textvariable=self.interval, state="readonly",
                     values=[f"{k}  ·  {v[0]}" for k, v in INTERVALS.items()]).pack(fill="x", pady=(2, 0))

        self.csv_box = ttk.Frame(s, style="Panel.TFrame")
        ttk.Button(self.csv_box, text="Elegir archivo CSV…", command=self._pick_csv).pack(fill="x")
        self.csv_label = ttk.Label(self.csv_box, text="Ningún archivo seleccionado",
                                   style="Muted.TLabel", wraplength=290)
        self.csv_label.pack(anchor="w", pady=(4, 0))
        ttk.Label(self.csv_box, text="Columnas: timestamps, open, high, low,\nclose (volume y amount opcionales)",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        # --- modelo
        self._section(s, "Modelo")
        self.model = tk.StringVar(value=list(MODELS)[0])
        cb = ttk.Combobox(s, textvariable=self.model, state="readonly", values=list(MODELS))
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_ctx())

        # --- prevision
        self._section(s, "Previsión")
        grid = ttk.Frame(s, style="Panel.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        self.lookback = tk.IntVar(value=400)
        self.pred_len = tk.IntVar(value=30)
        self.sims = tk.IntVar(value=8)
        self.lookback_lbl = ttk.Label(grid, text="Velas de histórico (máx. 512)", style="Panel.TLabel")
        self.lookback_lbl.grid(row=0, column=0, sticky="w", pady=3)
        self.lookback_spin = ttk.Spinbox(grid, from_=50, to=512, increment=10, textvariable=self.lookback, width=7)
        self.lookback_spin.grid(row=0, column=1, sticky="e")
        ttk.Label(grid, text="Velas a predecir", style="Panel.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Spinbox(grid, from_=1, to=240, increment=1, textvariable=self.pred_len, width=7).grid(row=1, column=1, sticky="e")
        ttk.Label(grid, text="Simulaciones", style="Panel.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(grid, from_=1, to=32, increment=1, textvariable=self.sims, width=7).grid(row=2, column=1, sticky="e")
        ttk.Label(s, text="Cada simulación es un futuro posible. Más simulaciones = banda de "
                          "incertidumbre más fiable, pero más lento.",
                  style="Muted.TLabel", wraplength=290).pack(anchor="w", pady=(2, 0))

        self.test_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(s, text="Modo prueba: predecir el pasado\nreciente y compararlo con la realidad",
                        variable=self.test_mode, style="Panel.TCheckbutton").pack(anchor="w", pady=(10, 0))

        # --- avanzado
        self._section(s, "Muestreo (avanzado)")
        self.temperature = tk.DoubleVar(value=1.0)
        self.top_p = tk.DoubleVar(value=0.9)
        self._slider(s, "Temperatura", self.temperature, 0.1, 1.5, "{:.2f}")
        self._slider(s, "Top-p", self.top_p, 0.5, 1.0, "{:.2f}")
        self.show_paths = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text="Mostrar cada simulación en el gráfico", variable=self.show_paths,
                        style="Panel.TCheckbutton", command=self._redraw).pack(anchor="w", pady=(6, 0))

        # --- accion
        self.run_btn = ttk.Button(s, text="▶  Predecir", style="Accent.TButton", command=self._start)
        self.run_btn.pack(fill="x", pady=(16, 6))
        self.progress = ttk.Progressbar(s, mode="determinate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value="Listo.")
        ttk.Label(s, textvariable=self.status, style="Muted.TLabel", wraplength=290).pack(anchor="w", pady=(4, 0))

        btns = ttk.Frame(s, style="Panel.TFrame")
        btns.pack(fill="x", pady=(12, 0))
        btns.columnconfigure((0, 1), weight=1)
        self.save_png_btn = ttk.Button(btns, text="Guardar gráfico", command=self._save_png, state="disabled")
        self.save_png_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.save_csv_btn = ttk.Button(btns, text="Exportar CSV", command=self._save_csv, state="disabled")
        self.save_csv_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ttk.Button(btns, text="Abrir carpeta de resultados", command=self._open_results).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Label(s, text="Previsión estadística. No es una garantía ni un consejo de inversión.",
                  style="Muted.TLabel", wraplength=290).pack(anchor="w", pady=(14, 0))

    def _slider(self, parent, label, var, lo, hi, f):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Panel.TLabel", width=12).pack(side="left")
        val = ttk.Label(row, text=f.format(var.get()), style="Value.TLabel", width=5)
        val.pack(side="right")
        ttk.Scale(row, from_=lo, to=hi, variable=var,
                  command=lambda v: val.configure(text=f.format(float(v)))).pack(side="left", fill="x", expand=True, padx=6)

    # ---------- eventos de la barra lateral ----------
    def _toggle_source(self):
        if self.source.get() == "yahoo":
            self.csv_box.pack_forget()
            self.yahoo_box.pack(fill="x", pady=(6, 0), after=self.source_row)
        else:
            self.yahoo_box.pack_forget()
            self.csv_box.pack(fill="x", pady=(6, 0), after=self.source_row)

    def _category_changed(self, _event=None):
        items = CATALOG[self.category.get()]
        self.symbol_cb.configure(values=[entry_label(*it) for it in items])
        self.symbol.set(entry_label(*items[0]))

    def _filter_symbols(self, event):
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab"):
            return
        text = self.symbol.get().strip().lower()
        if not text:
            self.symbol_cb.configure(values=[entry_label(*it) for it in CATALOG[self.category.get()]])
            return
        matches = [entry_label(sym, name) for items in CATALOG.values() for sym, name in items
                   if text in sym.lower() or text in name.lower()]
        self.symbol_cb.configure(values=list(dict.fromkeys(matches)) or ["(sin coincidencias: pulsa Buscar en Yahoo)"])

    def _open_search(self):
        SearchDialog(self.root, on_pick=lambda sym, name: self.symbol.set(entry_label(sym, name)))

    def _pick_csv(self):
        path = filedialog.askopenfilename(title="Elige un CSV con velas",
                                          filetypes=[("CSV", "*.csv"), ("Todos", "*.*")], initialdir=BASE)
        if path:
            self.csv_path = path
            self.csv_label.configure(text=os.path.basename(path))

    def _update_ctx(self):
        ctx = MODELS[self.model.get()][2]
        self.lookback_spin.configure(to=ctx)
        self.lookback_lbl.configure(text=f"Velas de histórico (máx. {ctx})")
        if self._int(self.lookback, 400) > ctx:
            self.lookback.set(ctx)

    @staticmethod
    def _int(var, default):
        try:
            return int(float(var.get()))
        except (tk.TclError, ValueError):
            return default

    # ---------- ejecucion ----------
    def _collect(self):
        ctx = MODELS[self.model.get()][2]
        p = {
            "source": self.source.get(),
            "symbol": self.symbol.get().split("—")[0].strip().upper() if self.source.get() == "yahoo"
                      else os.path.splitext(os.path.basename(self.csv_path or "csv"))[0],
            "interval": self.interval.get().split()[0],
            "csv_path": self.csv_path,
            "model": self.model.get(),
            "lookback": max(50, min(ctx, self._int(self.lookback, 400))),
            "pred_len": max(1, min(240, self._int(self.pred_len, 30))),
            "sims": max(1, min(32, self._int(self.sims, 8))),
            "temperature": round(float(self.temperature.get()), 2),
            "top_p": round(float(self.top_p.get()), 2),
            "test_mode": bool(self.test_mode.get()),
        }
        if p["source"] == "yahoo" and (not p["symbol"] or p["symbol"].startswith("(")):
            raise ValueError("Escribe un símbolo, por ejemplo BTC-USD.")
        if p["source"] == "csv" and not p["csv_path"]:
            raise ValueError("Elige primero un archivo CSV.")
        return p

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            p = self._collect()
        except ValueError as e:
            messagebox.showwarning("Kronos", str(e))
            return
        self.run_btn.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.t0 = time.time()
        self.worker = threading.Thread(target=run_forecast, args=(p, self.q), daemon=True)
        self.worker.start()
        self._tick()

    def _tick(self):
        if self.worker and self.worker.is_alive():
            self.root.title(f"Kronos · trabajando… {int(time.time() - self.t0)} s")
            self.root.after(500, self._tick)
        else:
            self.root.title("Kronos · Previsión de mercados")

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "status":
                    self.status.set(msg[1])
                elif msg[0] == "done":
                    self._finish()
                    self.result = msg[1]
                    self.summary = summarize(self.result)
                    secs = int(time.time() - self.t0)
                    self.status.set(f"Completado en {secs} s.")
                    self._show_summary()
                    self._fill_table()
                    self._redraw()
                elif msg[0] == "error":
                    self._finish()
                    self.status.set("Error: " + msg[1].splitlines()[0])
                    print(msg[2], file=sys.stderr)
                    messagebox.showerror("Kronos", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish(self):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.run_btn.configure(state="normal")

    # ---------- resultados ----------
    def _kpi_tile(self, title, value, sub="", color=TEXT):
        col = len(self.kpi.winfo_children())
        tile = tk.Frame(self.kpi, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        tile.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
        self.kpi.columnconfigure(col, weight=1, uniform="kpi")
        inner = ttk.Frame(tile, style="Panel.TFrame", padding=(12, 8))
        inner.pack(fill="both", expand=True)
        ttk.Label(inner, text=title, style="KpiK.TLabel").pack(anchor="w")
        ttk.Label(inner, text=value, style="KpiV.TLabel", foreground=color).pack(anchor="w")
        if sub:
            ttk.Label(inner, text=sub, style="KpiK.TLabel").pack(anchor="w")

    def _kpi_placeholder(self):
        for w in self.kpi.winfo_children():
            w.destroy()
        self._kpi_tile("Último cierre", "—")
        self._kpi_tile("Cierre previsto", "—")
        self._kpi_tile("Cambio previsto", "—")
        self._kpi_tile("Rango probable (P10–P90)", "—")
        self._kpi_tile("Simulaciones al alza", "—")

    def _show_summary(self):
        s, p = self.summary, self.result["params"]
        for w in self.kpi.winfo_children():
            w.destroy()
        color = UP if s["change"] >= 0 else DOWN
        tf = self._time_format()
        when = pd.Timestamp(s["mean"]["timestamps"].iloc[-1]).strftime(tf)
        last_when = pd.Timestamp(self.result["hist"]["timestamps"].iloc[-1]).strftime(tf)
        self._kpi_tile("Último cierre", fmt(s["last"]), last_when)
        if "mape" not in s:
            self._kpi_tile("Cierre previsto (media)", fmt(s["final"]), when)
            self._kpi_tile("Cambio previsto", f"{s['change'] * 100:+.2f} %", f"en {p['pred_len']} velas", color)
            self._kpi_tile("Rango probable", f"{fmt(s['p10'])} – {fmt(s['p90'])}", "P10 – P90")
            self._kpi_tile("Simulaciones al alza", f"{s['prob_up'] * 100:.0f} %", f"de {p['sims']}")
        else:
            real_chg = s["real_final"] / s["last"] - 1
            ok = UP if s["dir_ok"] else DOWN
            self._kpi_tile("Previsto (media)", fmt(s["final"]), f"{s['change'] * 100:+.2f} %", color)
            self._kpi_tile("Lo que ocurrió", fmt(s["real_final"]), f"{real_chg * 100:+.2f} %",
                           UP if real_chg >= 0 else DOWN)
            self._kpi_tile("Error medio (MAPE)", f"{s['mape'] * 100:.2f} %", "distancia media al real")
            self._kpi_tile("Dirección", "Acertada" if s["dir_ok"] else "Fallada",
                           f"real en banda: {s['coverage'] * 100:.0f} %", ok)
        self.save_png_btn.configure(state="normal")
        self.save_csv_btn.configure(state="normal")

    def _fill_table(self):
        self.tree.delete(*self.tree.get_children())
        m = self.summary["mean"]
        tfmt = self._time_format()
        for _, r in m.iterrows():
            self.tree.insert("", "end", values=(
                pd.Timestamp(r["timestamps"]).strftime(tfmt), fmt(r["open"]), fmt(r["high"]), fmt(r["low"]),
                fmt(r["close"]), fmt(r["close_p10"]), fmt(r["close_p90"]), fmt(r["volume"]),
                fmt(r["real_close"]) if "real_close" in m.columns else "—"))

    def _time_format(self):
        ts = pd.to_datetime(self.result["hist"]["timestamps"])
        step = ts.diff().dropna().median()
        return "%d/%m/%Y %H:%M" if step < pd.Timedelta("1D") else "%d/%m/%Y"

    # ---------- grafico ----------
    def _style_ax(self, ax):
        ax.set_facecolor(PANEL)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color="#eef0f3", linewidth=0.8)
        ax.set_axisbelow(True)

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_ax(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.5, 0.55, "Kronos", ha="center", va="center", fontsize=28, color=ACCENT,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.45, "Elige un activo a la izquierda y pulsa «Predecir»",
                ha="center", va="center", fontsize=11, color=MUTED, transform=ax.transAxes)
        self.canvas.draw_idle()

    @staticmethod
    def _candles(ax, x, o, h, l, c, alpha=1.0, width=0.62):
        up = c >= o
        colors = np.where(up, UP, DOWN)
        ax.vlines(x, l, h, colors=colors, linewidth=0.8, alpha=alpha, zorder=2)
        body = np.maximum(np.abs(c - o), (h - l).max() * 0.002 if len(h) else 0)
        ax.bar(x, body, bottom=np.minimum(o, c), width=width, color=colors,
               edgecolor=colors, linewidth=0.5, alpha=alpha, zorder=3)

    def _redraw(self):
        if self.result is None:
            return
        res, s, p = self.result, self.summary, self.result["params"]
        hist = res["hist"].tail(DISPLAY_CANDLES).reset_index(drop=True)
        n = len(hist)
        k = p["pred_len"]
        xh = np.arange(n)
        xf = np.arange(n, n + k)
        mean = s["mean"]

        self.fig.clear()
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05,
                                   left=0.06, right=0.985, top=0.92, bottom=0.08)
        ax = self.fig.add_subplot(gs[0])
        axv = self.fig.add_subplot(gs[1], sharex=ax)
        self._style_ax(ax)
        self._style_ax(axv)
        self.ax, self.axv = ax, axv

        # historico
        self._candles(ax, xh, *(hist[c].to_numpy() for c in ["open", "high", "low", "close"]))

        # real (modo prueba)
        real = res["real"]
        if real is not None:
            self._candles(ax, xf, *(real[c].to_numpy() for c in ["open", "high", "low", "close"]), alpha=0.28)
            ax.plot(np.r_[n - 1, xf], np.r_[s["last"], real["close"].to_numpy()], color=REAL,
                    lw=1.4, ls="--", label="Lo que ocurrió", zorder=5)

        # prevision
        xx = np.r_[n - 1, xf]
        ax.fill_between(xx, np.r_[s["last"], mean["close_p10"]], np.r_[s["last"], mean["close_p90"]],
                        color=PRED, alpha=0.14, lw=0, label="Rango probable (P10–P90)", zorder=1)
        if self.show_paths.get():
            for path in s["closes"]:
                ax.plot(xx, np.r_[s["last"], path], color=PRED, lw=0.6, alpha=0.25, zorder=4)
        ax.plot(xx, np.r_[s["last"], mean["close"]], color=PRED, lw=2.2, label="Previsión (media)", zorder=6)
        ax.scatter([xx[-1]], [s["final"]], color=PRED, s=28, zorder=7)
        ax.annotate(f" {fmt(s['final'])}", (xx[-1], s["final"]), color=PRED, fontsize=9,
                    fontweight="bold", va="center")
        ax.axvline(n - 0.5, color=MUTED, lw=0.8, ls=":")
        ax.text(n - 0.3, 0.98, "futuro" if real is None else "periodo de prueba", color=MUTED, fontsize=8,
                transform=ax.get_xaxis_transform(), va="top")

        # volumen
        vol_colors = np.where(hist["close"] >= hist["open"], UP, DOWN)
        axv.bar(xh, hist["volume"], color=vol_colors, alpha=0.45, width=0.62)
        axv.bar(xf, mean["volume"].clip(lower=0), color=PRED, alpha=0.45, width=0.62)
        axv.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v).replace(".00", "")))
        axv.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
        ax.tick_params(labelbottom=False)

        # eje x con fechas
        all_ts = list(pd.to_datetime(hist["timestamps"])) + list(pd.to_datetime(mean["timestamps"]))
        tfmt = "%d/%m %H:%M" if self._time_format().endswith("%M") else "%d/%m/%y"
        axv.xaxis.set_major_locator(MaxNLocator(9, integer=True))
        axv.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: all_ts[int(v)].strftime(tfmt) if 0 <= int(v) < len(all_ts) else ""))
        ax.set_xlim(-1, n + k + max(8, (n + k) * 0.07))

        name = p["model"].split()[0]
        label = INTERVALS[p["interval"]][0] if res["interval"] != "csv" else "CSV"
        self.fig.suptitle(f"{p['symbol']}  ·  {label}  ·  {name}", x=0.06, ha="left",
                          fontsize=13, fontweight="bold", color=TEXT)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

        self.plot_index = {"ts": all_ts, "hist": hist, "n": n, "real": real}
        self.canvas.draw_idle()

    def _on_hover(self, event):
        if self.plot_index is None or event.xdata is None or event.inaxes not in (getattr(self, "ax", None),
                                                                                  getattr(self, "axv", None)):
            return
        i = int(round(event.xdata))
        pi, s = self.plot_index, self.summary
        if not 0 <= i < len(pi["ts"]):
            return
        t = pi["ts"][i].strftime(self._time_format())
        if i < pi["n"]:
            r = pi["hist"].iloc[i]
            self.info_var.set(f"{t}   Apertura {fmt(r['open'])}   Máx {fmt(r['high'])}   Mín {fmt(r['low'])}"
                              f"   Cierre {fmt(r['close'])}   Vol {fmt(r['volume'])}")
        else:
            j = i - pi["n"]
            r = s["mean"].iloc[j]
            txt = (f"{t}   PREVISIÓN  cierre medio {fmt(r['close'])}   rango {fmt(r['close_p10'])} – "
                   f"{fmt(r['close_p90'])}")
            if pi["real"] is not None:
                txt += f"   real {fmt(pi['real']['close'].iloc[j])}"
            self.info_var.set(txt)

    # ---------- exportar ----------
    def _default_name(self):
        p = self.result["params"]
        sym = "".join(ch for ch in p["symbol"] if ch.isalnum() or ch in "-_")
        tag = p["interval"] if self.result["interval"] != "csv" else "csv"
        return f"{sym}_{tag}_{datetime.now():%Y%m%d_%H%M}"

    def _save_png(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(initialdir=RESULTS_DIR, initialfile=self._default_name() + ".png",
                                            defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=140, facecolor=PANEL)
            self.status.set(f"Gráfico guardado: {os.path.basename(path)}")

    def _save_csv(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(initialdir=RESULTS_DIR, initialfile=self._default_name() + ".csv",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.summary["mean"].to_csv(path, index=False)
            self.status.set(f"Tabla guardada: {os.path.basename(path)}")

    def _open_results(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(RESULTS_DIR)
        else:
            messagebox.showinfo("Kronos", RESULTS_DIR)


class SearchDialog:
    """Busca acciones, ETF, indices, etc. por nombre en Yahoo Finance."""

    def __init__(self, parent, on_pick):
        self.on_pick = on_pick
        self.q = queue.Queue()
        self.win = tk.Toplevel(parent)
        self.win.title("Buscar en Yahoo Finance")
        self.win.geometry("720x460")
        self.win.configure(bg=PANEL)
        self.win.transient(parent)
        frm = ttk.Frame(self.win, style="Panel.TFrame", padding=14)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Escribe el nombre de una empresa, ETF o índice (p. ej. «Ferrari», «Repsol», "
                            "«Coca Cola», «oro»):", style="Panel.TLabel", wraplength=680).pack(anchor="w")
        row = ttk.Frame(frm, style="Panel.TFrame")
        row.pack(fill="x", pady=(6, 8))
        self.query = tk.StringVar()
        ent = ttk.Entry(row, textvariable=self.query)
        ent.pack(side="left", fill="x", expand=True)
        ent.bind("<Return>", lambda e: self._search())
        ttk.Button(row, text="Buscar", command=self._search).pack(side="left", padx=(6, 0))
        cols = ("sym", "name", "exch", "type")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", selectmode="browse")
        for c, h, w in zip(cols, ("Símbolo", "Nombre", "Mercado", "Tipo"), (110, 330, 130, 100)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._use())
        bottom = ttk.Frame(frm, style="Panel.TFrame")
        bottom.pack(fill="x", pady=(8, 0))
        self.status = tk.StringVar(value="Doble clic en un resultado para usarlo.")
        ttk.Label(bottom, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        ttk.Button(bottom, text="Usar seleccionado", style="Accent.TButton", command=self._use).pack(side="right")
        ent.focus_set()
        self.win.after(100, self._poll)

    def _search(self):
        text = self.query.get().strip()
        if not text:
            return
        self.status.set("Buscando…")
        self.tree.delete(*self.tree.get_children())
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, text):
        try:
            import yfinance as yf
            if not hasattr(yf, "Search"):
                raise RuntimeError("Tu versión de yfinance no tiene buscador. Actualízala con:\n"
                                   "python3 -m pip install -U yfinance")
            quotes = yf.Search(text, max_results=25, news_count=0).quotes
            rows = [(qt.get("symbol", ""), qt.get("longname") or qt.get("shortname") or "",
                     qt.get("exchDisp", qt.get("exchange", "")), qt.get("typeDisp", qt.get("quoteType", "")))
                    for qt in quotes if qt.get("symbol")]
            self.q.put(("ok", rows))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll(self):
        if not self.win.winfo_exists():
            return
        try:
            kind, data = self.q.get_nowait()
            if kind == "ok":
                for r in data:
                    self.tree.insert("", "end", values=r)
                self.status.set(f"{len(data)} resultados. Doble clic para usar uno." if data
                                else "Sin resultados. Prueba con otro nombre (en inglés suele funcionar mejor).")
            else:
                self.status.set("Error en la búsqueda.")
                messagebox.showerror("Buscar", data, parent=self.win)
        except queue.Empty:
            pass
        self.win.after(100, self._poll)

    def _use(self):
        sel = self.tree.selection()
        if not sel:
            return
        sym, name = self.tree.item(sel[0], "values")[:2]
        self.on_pick(sym, name)
        self.win.destroy()


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    if not os.path.isdir(os.path.join(BASE, "Kronos-code", "model")):
        tk.Tk().withdraw()
        messagebox.showerror("Kronos", "No encuentro la carpeta «Kronos-code» junto a este archivo.\n"
                                       "Guarda kronos_gui.py dentro de la carpeta Kronos del escritorio.")
        return
    root = tk.Tk()
    KronosApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
