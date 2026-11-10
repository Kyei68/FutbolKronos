"""
FutbolKronos - simulador de partidos (interfaz gráfica de escritorio).

Abre con doble clic en "Abrir FutbolKronos.bat", o desde PowerShell:
    python3 futbolkronos_gui.py

Eliges liga, equipo local, visitante y fecha (o un partido de la próxima jornada de LaLiga),
y la aplicación simula el partido entre 10.000 y 100.000 veces con el modelo FutbolKronos.
Si el partido ya se jugó, lo simula con los datos de antes y muestra el resultado real.
"""
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "resultados")

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# ---------- paletas ----------
# Fondo neutro en dos capas; el color solo lo llevan los datos. Dos modos con los mismos papeles.
TEMAS = {
    "oscuro": dict(
        BG="#0F1115",         # fondo de la zona principal
        SIDE="#151920",       # barra lateral (segunda capa)
        SURF="#171B22",       # pestañas, tablas y gráficos
        FIELD="#1E232C",      # campos de formulario
        FIELD_HI="#262C37",   # campo con el ratón encima
        LINE="#262C36",       # filetes y bordes
        GRID="#222832",       # rejilla de los gráficos
        TEXT="#E7EAEF", MUTED="#9BA3B0", FAINT="#7D8695",
        LOCAL="#4C9BFF", EMPATE="#8B94A3", VISITA="#FF8A4C",
        UP="#43C07E", DOWN="#F26461", REAL="#F2C14E", SEL="#243650",
        HOVER_LINE="#38404C", CHECK_LINE="#3A424F", ON_ACCENT="#0B1220", ACTIVE_FG="#FFFFFF",
        DISABLED="#2E3440", PRIMARY_HI="#FFFFFF", PRIMARY_PRESS="#C9CFD8", TAB_HI="#13161C",
        SCROLL_HI="#353C48", ZEBRA="#1C212A", THUMB="#C9CFD8",
        HEAT=("#1E232C", "#2C4F7E", "#4C9BFF", "#BFD9FF"), HEAT_INK="#0B1220",
        SWITCH_OFF="#3A404A",
    ),
    "claro": dict(
        BG="#F3F4F6",
        SIDE="#FFFFFF",
        SURF="#FFFFFF",
        FIELD="#F1F2F5",
        FIELD_HI="#E8EAEE",
        LINE="#E1E4E9",
        GRID="#ECEEF1",
        TEXT="#15181D", MUTED="#5A6270", FAINT="#646C79",
        LOCAL="#1F6FE5", EMPATE="#9AA2AE", VISITA="#E2621B",
        UP="#16924F", DOWN="#D33A35", REAL="#C88A04", SEL="#DDE8FB",
        HOVER_LINE="#C9CED6", CHECK_LINE="#C3C8D0", ON_ACCENT="#FFFFFF", ACTIVE_FG="#000000",
        DISABLED="#E4E6EA", PRIMARY_HI="#2A2F37", PRIMARY_PRESS="#3A404A", TAB_HI="#E9EBEF",
        SCROLL_HI="#C9CED6", ZEBRA="#F7F8FA", THUMB="#5A6270",
        HEAT=("#F1F2F5", "#BFD5F7", "#1F6FE5", "#0F3F8C"), HEAT_INK="#FFFFFF",
        SWITCH_OFF="#E3E5E8",
    ),
}
SWITCH_ON = "#34C759"     # verde de los interruptores de iOS
PREFS = os.path.join(os.path.expanduser("~"), ".futbolkronos.json")


def aplicar_tema(nombre):
    """Cambia la paleta activa: todo lo que se dibuja lee estas constantes en el momento."""
    globals().update(TEMAS[nombre])


def leer_tema():
    try:
        import json
        with open(PREFS, encoding="utf-8") as f:
            t = json.load(f).get("tema")
        return t if t in TEMAS else "oscuro"
    except Exception:
        return "oscuro"


def guardar_tema(nombre):
    try:
        import json
        with open(PREFS, "w", encoding="utf-8") as f:
            json.dump({"tema": nombre}, f)
    except Exception:
        pass


aplicar_tema("oscuro")
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
EUROPA = ("CL", "CLQ", "EL", "ELQ", "ECL", "ECLQ")


def pct(x, dec=1):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def ease_out(t):
    """Curva de salida fuerte (≈ cubic-bezier(0.23, 1, 0.32, 1)): arranca rápido, frena suave."""
    return 1 - (1 - t) ** 5


def ease_in_out(t):
    """Para algo que ya está en pantalla y cambia de forma (≈ cubic-bezier(0.77, 0, 0.175, 1))."""
    return 8 * t ** 4 if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2


def _familia(*opciones):
    disponibles = set(tkfont.families())
    return next((f for f in opciones if f in disponibles), "Segoe UI")


def _barra_oscura(root, oscura=True):
    """Barra de título oscura o clara en Windows 10/11 (si no se puede, se queda la normal)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        valor = ctypes.c_int(1 if oscura else 0)
        for attr in (20, 19):           # DWMWA_USE_IMMERSIVE_DARK_MODE (nuevo y antiguo)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(valor),
                                                          ctypes.sizeof(valor)) == 0:
                break
    except Exception:
        pass


class FutbolKronosApp:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.worker = None
        self.sim = None
        self.res = None
        self.partido = None
        self.args = None
        self.nombres = {}          # liga -> {nombre mostrado: nombre interno}
        self._proximos = {}
        self.t0 = None
        self._p_vis = None         # reparto 1X2 que dibuja ahora mismo la barra (se anima hacia el real)
        self._anim = 0
        self._imgs = []
        self._fuentes = {}
        self._prev = {}            # estado del formulario al reconstruir por cambio de modo
        self.tema = leer_tema()
        aplicar_tema(self.tema)

        self.F = _familia("Segoe UI Variable Text", "Segoe UI")
        self.FD = _familia("Segoe UI Variable Display", "Segoe UI")
        self.FM = _familia("Cascadia Mono", "Consolas")

        root.title("FutbolKronos · Simulador de partidos")
        root.geometry("1400x880")
        root.minsize(1160, 740)
        root.configure(bg=BG)
        _barra_oscura(root, self.tema == "oscuro")
        self._styles()
        self._build()
        self._draw_empty()
        self.root.after(100, self._poll)
        self._run(self._cargar)

    # ---------- estilos ----------
    def _styles(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        F = self.F
        base = (F, 10)
        r = self.root
        r.option_add("*TCombobox*Listbox.font", base)
        r.option_add("*TCombobox*Listbox.background", FIELD)
        r.option_add("*TCombobox*Listbox.foreground", TEXT)
        r.option_add("*TCombobox*Listbox.selectBackground", SEL)
        r.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        r.option_add("*TCombobox*Listbox.borderWidth", 0)
        r.option_add("*TCombobox*Listbox.relief", "flat")

        st.configure(".", background=BG, foreground=TEXT, font=base, bordercolor=LINE,
                     lightcolor=LINE, darkcolor=LINE, troughcolor=LINE, focuscolor=LOCAL,
                     selectbackground=SEL, selectforeground=TEXT, insertcolor=TEXT)

        # textos
        st.configure("Side.TFrame", background=SIDE)
        st.configure("Side.TLabel", background=SIDE, foreground=TEXT)
        st.configure("Field.TLabel", background=SIDE, foreground=MUTED, font=(F, 9))
        st.configure("Hint.TLabel", background=SIDE, foreground=FAINT, font=(F, 9))
        st.configure("Group.TLabel", background=SIDE, foreground=TEXT, font=(F, 10, "bold"))
        st.configure("Brand.TLabel", background=SIDE, foreground=TEXT, font=(self.FD, 16, "bold"))
        st.configure("Value.TLabel", background=SIDE, foreground=TEXT, font=(F, 10, "bold"))
        st.configure("Info.TLabel", background=BG, foreground=FAINT, font=(F, 9))

        # campos
        campo = dict(fieldbackground=FIELD, background=FIELD, foreground=TEXT, bordercolor=LINE,
                     lightcolor=FIELD, darkcolor=FIELD, insertcolor=TEXT, padding=(8, 5),
                     arrowcolor=MUTED, selectbackground=FIELD, selectforeground=TEXT)
        st.configure("TEntry", **campo)
        st.configure("TCombobox", **campo, arrowsize=14)
        foco = [("focus", LOCAL), ("hover", HOVER_LINE)]
        st.map("TEntry", bordercolor=foco, fieldbackground=[("hover", FIELD_HI)])
        st.map("TCombobox", bordercolor=foco,
               fieldbackground=[("readonly", "hover", FIELD_HI), ("readonly", FIELD), ("disabled", SIDE)],
               background=[("readonly", "hover", FIELD_HI), ("readonly", FIELD)],
               foreground=[("disabled", FAINT), ("readonly", TEXT)],
               arrowcolor=[("disabled", LINE), ("hover", TEXT)],
               selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])

        # deslizadores, casillas y progreso
        st.configure("Side.Horizontal.TScale", background=MUTED, troughcolor=LINE, bordercolor=SIDE,
                     lightcolor=MUTED, darkcolor=MUTED, sliderlength=12, sliderthickness=12, gripcount=0)
        st.map("Side.Horizontal.TScale", background=[("pressed", TEXT), ("active", TEXT)],
               lightcolor=[("pressed", TEXT), ("active", TEXT)], darkcolor=[("pressed", TEXT), ("active", TEXT)])
        # un elemento de imagen por modo (ttk no deja redefinir uno que ya existe)
        indicador = f"Fk{self.tema}.Checkbutton.indicator"
        if indicador not in st.element_names():
            apagada, encima, marcada = (self._caja(FIELD, CHECK_LINE), self._caja(FIELD_HI, MUTED),
                                        self._caja(LOCAL, LOCAL, True))
            st.element_create(indicador, "image", apagada, ("selected", marcada),
                              ("active", encima), width=28, sticky="w")
        st.layout("Side.TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
            (indicador, {"side": "left", "sticky": ""}),
            ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                ("Checkbutton.label", {"sticky": "nswe"})]})]})])
        st.configure("Side.TCheckbutton", background=SIDE, foreground=TEXT, focuscolor=SIDE, padding=(0, 2))
        st.map("Side.TCheckbutton", background=[("active", SIDE)], foreground=[("active", ACTIVE_FG)])
        st.configure("Side.Horizontal.TProgressbar", troughcolor=SIDE, background=LOCAL, bordercolor=SIDE,
                     lightcolor=LOCAL, darkcolor=LOCAL, thickness=3)

        # botones: el principal va en el color del texto sobre el fondo, nunca en un color de los datos
        st.configure("Primary.TButton", background=TEXT, foreground=BG, font=(F, 11, "bold"),
                     padding=(12, 10), borderwidth=0, bordercolor=TEXT, lightcolor=TEXT, darkcolor=TEXT,
                     focuscolor=TEXT, focusthickness=0)
        st.map("Primary.TButton",
               background=[("disabled", DISABLED), ("pressed", PRIMARY_PRESS), ("active", PRIMARY_HI)],
               foreground=[("disabled", FAINT)],
               bordercolor=[("disabled", DISABLED), ("focus", LOCAL)],
               lightcolor=[("disabled", DISABLED)], darkcolor=[("disabled", DISABLED)])
        st.configure("Ghost.TButton", background=FIELD, foreground=TEXT, padding=(8, 6), borderwidth=1,
                     bordercolor=LINE, lightcolor=FIELD, darkcolor=FIELD, focuscolor=FIELD)
        st.map("Ghost.TButton", background=[("disabled", SIDE), ("pressed", LINE), ("active", FIELD_HI)],
               foreground=[("disabled", FAINT)], bordercolor=[("focus", LOCAL), ("disabled", LINE)],
               lightcolor=[("disabled", SIDE), ("active", FIELD_HI)], darkcolor=[("disabled", SIDE), ("active", FIELD_HI)])

        # pestañas
        st.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        st.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(16, 8), borderwidth=0,
                     bordercolor=BG, lightcolor=BG, darkcolor=BG, focuscolor=BG, font=(F, 10))
        st.map("TNotebook.Tab", background=[("selected", SURF), ("active", TAB_HI)],
               foreground=[("selected", TEXT), ("active", TEXT)],
               bordercolor=[("selected", SURF)], lightcolor=[("selected", SURF)], darkcolor=[("selected", SURF)],
               expand=[("selected", (0, 0, 0, 0)), ("!selected", (0, 0, 0, 0))],
               padding=[("selected", (16, 8)), ("!selected", (16, 8))])

        # tablas y barras de desplazamiento
        st.configure("Treeview", background=SURF, fieldbackground=SURF, foreground=TEXT, rowheight=28,
                     bordercolor=SURF, lightcolor=SURF, darkcolor=SURF, font=(F, 10), borderwidth=0)
        st.map("Treeview", background=[("selected", SEL)], foreground=[("selected", TEXT)])
        st.configure("Treeview.Heading", background=SURF, foreground=MUTED, font=(F, 9, "bold"),
                     relief="flat", borderwidth=0, bordercolor=LINE, lightcolor=SURF, darkcolor=LINE,
                     padding=(8, 8))
        st.map("Treeview.Heading", background=[("active", SURF)], foreground=[("active", TEXT)])
        st.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        st.configure("Vertical.TScrollbar", background=LINE, troughcolor=SURF, bordercolor=SURF,
                     lightcolor=LINE, darkcolor=LINE, arrowcolor=MUTED, gripcount=0, arrowsize=11)
        st.map("Vertical.TScrollbar", background=[("active", SCROLL_HI)])

        # gráficos: misma familia y colores que la ventana
        matplotlib.rcParams.update({
            "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
            "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": LINE,
            "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXT,
            "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlepad": 12,
            "axes.labelsize": 9, "axes.labelpad": 6, "legend.labelcolor": TEXT,
        })

    def _caja(self, fondo, borde, marca=False):
        """Casilla de 18 px con esquinas recortadas; la marcada lleva un check oscuro."""
        n = 18
        img = tk.PhotoImage(width=n, height=n)
        img.put(borde, to=(1, 0, n - 1, n))
        img.put(borde, to=(0, 1, n, n - 1))
        img.put(fondo, to=(1, 1, n - 1, n - 1))
        if marca:
            puntos = [(4 + i, 9 + i) for i in range(4)] + [(7 + i, 12 - i) for i in range(7)]
            for x, y in puntos:
                img.put(ON_ACCENT, to=(x, y, x + 2, y + 2))
        self._imgs.append(img)
        return img

    # ---------- construcción ----------
    def _build(self):
        side = tk.Frame(self.root, bg=SIDE, width=330)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Frame(self.root, bg=LINE, width=1).pack(side="left", fill="y")
        inner = ttk.Frame(side, style="Side.TFrame", padding=(22, 20, 22, 18))
        inner.pack(fill="both", expand=True)
        self._build_sidebar(inner)

        main = tk.Frame(self.root, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=(26, 22), pady=(18, 14))

        self.head = tk.Canvas(main, bg=BG, height=196, highlightthickness=0, bd=0)
        self.head.pack(fill="x", pady=(0, 12))
        self.head.bind("<Configure>", lambda e: self._draw_header())

        self.info = tk.StringVar(value="Son probabilidades, no certezas: las casas de apuestas siguen siendo más "
                                       "precisas. La banda indica la incertidumbre del propio modelo "
                                       "(percentiles 10–90).")
        pie = tk.Frame(main, bg=BG)
        pie.pack(side="bottom", fill="x", pady=(10, 0))
        self._interruptor(pie).pack(side="right")
        ttk.Label(pie, text="Modo oscuro", style="Info.TLabel").pack(side="right", padx=(0, 8))
        ttk.Label(pie, textvariable=self.info, style="Info.TLabel").pack(side="left", fill="x")
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="both", expand=True)
        tab_graf = tk.Frame(self.nb, bg=SURF)
        tab_marc = tk.Frame(self.nb, bg=SURF)
        tab_merc = tk.Frame(self.nb, bg=SURF)
        tab_ctx = tk.Frame(self.nb, bg=SURF)
        self.nb.add(tab_graf, text="Simulación")
        self.nb.add(tab_marc, text="Marcadores")
        self.nb.add(tab_merc, text="Goles y mercados")
        self.nb.add(tab_ctx, text="Contexto del partido")

        # exportar actúa sobre el resultado, así que vive junto a las pestañas y no en el formulario
        acciones = tk.Frame(main, bg=BG)
        acciones.place(in_=self.nb, relx=1.0, x=0, y=0, anchor="ne")
        self.png_btn = ttk.Button(acciones, text="Guardar gráfico PNG", style="Ghost.TButton",
                                  command=self._guardar_png, state="disabled")
        self.csv_btn = ttk.Button(acciones, text="Guardar tabla CSV", style="Ghost.TButton",
                                  command=self._guardar_csv, state="disabled")
        self.png_btn.pack(side="left", padx=(0, 6))
        self.csv_btn.pack(side="left")

        self.fig = Figure(figsize=(9, 6.0), dpi=100, facecolor=SURF)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_graf)
        self.canvas.get_tk_widget().configure(bg=SURF, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        self.tree_marc = self._tree(tab_marc, ("marcador", "resultado", "sims", "pct", "modelo"),
                                    ("Marcador", "Resultado", "Simulaciones",
                                     "% de simulaciones", "Probabilidad del modelo"), texto=2)
        self.tree_merc = self._tree(tab_merc, ("mercado", "si", "no"), ("Mercado", "Sí", "No"), first_width=360)

        self.ctx = tk.Text(tab_ctx, bg=SURF, fg=TEXT, font=(self.F, 10), relief="flat", bd=0,
                           padx=28, pady=22, wrap="word", insertbackground=TEXT, selectbackground=SEL,
                           highlightthickness=0, spacing1=2, spacing3=2)
        self.ctx.pack(fill="both", expand=True)
        self.ctx.tag_configure("h1", font=(self.FD, 15, "bold"), spacing3=2)
        self.ctx.tag_configure("h2", font=(self.F, 10, "bold"), spacing1=16, spacing3=6)
        self.ctx.tag_configure("muted", foreground=MUTED)
        self.ctx.tag_configure("faint", foreground=FAINT)
        self.ctx.tag_configure("mono", font=(self.FM, 10), lmargin1=2)
        self.ctx.tag_configure("V", foreground=UP, font=(self.FM, 10, "bold"))
        self.ctx.tag_configure("E", foreground=MUTED, font=(self.FM, 10, "bold"))
        self.ctx.tag_configure("D", foreground=DOWN, font=(self.FM, 10, "bold"))
        self.ctx.tag_configure("loc", foreground=LOCAL, font=(self.F, 10, "bold"))
        self.ctx.tag_configure("vis", foreground=VISITA, font=(self.F, 10, "bold"))
        self.ctx.tag_configure("aviso", foreground=REAL)
        self._ctx_write([("Simula un partido para ver la forma reciente de cada equipo y su Elo.",
                          "muted")])


        for var in (self.local, self.visitante, self.liga):
            var.trace_add("write", lambda *_: self.res is None and self._draw_header())

    # ---------- modo claro / oscuro ----------
    def _interruptor(self, parent):
        """Interruptor al estilo de iOS: pista en píldora y botón blanco con sombra suave.
        Al pulsar, el botón se estira como en iOS; al soltar, se desliza con una salida fuerte
        (200 ms) y, al llegar, la ventana cambia de modo. Espacio o Intro también lo cambian."""
        W, H, P = 46, 28, 2                       # ancho, alto y margen del botón dentro de la pista
        D = H - 2 * P                             # diámetro del botón
        cv = tk.Canvas(parent, width=W + 6, height=H + 6, bg=BG, highlightthickness=0, bd=0,
                       takefocus=1, cursor="hand2")
        st = {"pos": 1.0 if self.tema == "oscuro" else 0.0, "pulsado": False, "anim": 0}

        def mezcla(c1, c2, t):
            a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))

        def pildora(x0, y0, x1, y1, color, **kw):
            r = (y1 - y0) / 2
            cv.create_oval(x0, y0, x0 + 2 * r, y1, fill=color, outline="", **kw)
            cv.create_oval(x1 - 2 * r, y0, x1, y1, fill=color, outline="", **kw)
            if x1 - x0 > 2 * r:
                cv.create_rectangle(x0 + r, y0, x1 - r, y1, fill=color, outline="", **kw)

        def dibujar():
            cv.delete("all")
            ox, oy = 3, 3
            t = st["pos"]
            if cv.focus_get() == cv:
                pildora(ox - 2.5, oy - 2.5, ox + W + 2.5, oy + H + 2.5, LOCAL)
                pildora(ox - 1, oy - 1, ox + W + 1, oy + H + 1, BG)
            pildora(ox, oy, ox + W, oy + H, mezcla(SWITCH_OFF, SWITCH_ON, t))
            estira = 6 if st["pulsado"] else 0     # al pulsar el botón se alarga hacia el centro
            x0 = ox + P + t * (W - 2 * P - D) - (estira * t)
            x1 = x0 + D + estira
            sombra = mezcla(mezcla(SWITCH_OFF, SWITCH_ON, t), "#000000", 0.22)
            pildora(x0, oy + P + 1.2, x1, oy + P + D + 1.2, sombra)
            pildora(x0, oy + P, x1, oy + P + D, "#FFFFFF")

        def cambiar(_e=None):
            st["pulsado"] = False
            destino = 0.0 if st["pos"] > 0.5 else 1.0
            inicio, t0 = st["pos"], time.perf_counter()
            st["anim"] += 1
            mia = st["anim"]

            def paso():
                if mia != st["anim"] or not cv.winfo_exists():
                    return
                k = min(1.0, (time.perf_counter() - t0) / 0.2)
                st["pos"] = inicio + (destino - inicio) * ease_out(k)
                dibujar()
                if k < 1:
                    cv.after(12, paso)
                else:
                    cv.after(40, lambda: self._cambiar_tema("oscuro" if destino == 1.0 else "claro"))
            paso()

        def pulsa(_e):
            cv.focus_set()
            st["pulsado"] = True
            dibujar()

        cv.bind("<Configure>", lambda e: dibujar())
        cv.bind("<ButtonPress-1>", pulsa)
        cv.bind("<ButtonRelease-1>", cambiar)
        cv.bind("<Leave>", lambda e: st["pulsado"] and (st.update(pulsado=False), dibujar()))
        cv.bind("<FocusIn>", lambda e: dibujar())
        cv.bind("<FocusOut>", lambda e: dibujar())
        cv.bind("<space>", cambiar)
        cv.bind("<Return>", cambiar)
        return cv

    def _cambiar_tema(self, tema):
        """Reconstruye la ventana con la otra paleta sin perder el partido, los ajustes ni el resultado."""
        if tema == self.tema:
            return
        self._prev = dict(liga=self.liga.get(), local=self.local.get(), visitante=self.visitante.get(),
                          proximo=self.proximo.get(), fecha=self.fecha.get(), n_sims=self.n_sims.get(),
                          temp=self.temp.get(), dropout=self.dropout.get(), status=self.status.get(),
                          pestana=self.nb.index("current"))
        self.tema = tema
        guardar_tema(tema)
        aplicar_tema(tema)
        self._anim += 1                                   # corta una animación de la barra a medias
        if self.res is not None:
            self._p_vis = self.res["p"].astype(float)
        for w in self.root.winfo_children():
            w.destroy()
        self.root.configure(bg=BG)
        _barra_oscura(self.root, tema == "oscuro")
        self._styles()
        self._build()
        prev = self._prev
        if self.sim:
            self.liga_cb.configure(values=list(self.ligas), state="readonly")
            self.liga.set(prev["liga"])
            self._cambiar_liga()
            self.local.set(prev["local"])
            self.visitante.set(prev["visitante"])
            if self._proximos:
                self.proximo_cb.configure(values=list(self._proximos), state="readonly")
                self.proximo.set(prev["proximo"])
            self.fecha.set(prev["fecha"])
        trabajando = self.worker is not None and self.worker.is_alive()
        self.run_btn.configure(state="disabled" if trabajando or not self.sim else "normal",
                               text="Simulando…" if trabajando and self.sim else "Simular partido")
        if trabajando and self.sim:
            self.progress.configure(mode="indeterminate")
            self.progress.start(10)
        self.status.set(prev["status"])
        if self.res is not None:
            self._draw_header()
            self._dibujar()
            self._tablas()
            self._contexto()
            self.png_btn.configure(state="normal")
            self.csv_btn.configure(state="normal")
        else:
            self._draw_empty()
        self.nb.select(prev["pestana"])

    def _tree(self, parent, cols, heads, first_width=170, texto=1):
        """Tabla: las primeras `texto` columnas son texto (a la izquierda), el resto números (a la derecha)."""
        wrap = tk.Frame(parent, bg=SURF)
        wrap.pack(fill="both", expand=True, padx=14, pady=10)
        tree = ttk.Treeview(wrap, columns=cols, show="headings")
        for i, (c, h) in enumerate(zip(cols, heads)):
            lado = "w" if i < texto else "e"
            tree.heading(c, text=h, anchor=lado)
            tree.column(c, width=first_width if i == 0 else 150, anchor=lado)
        tree.tag_configure("par", background=ZEBRA)
        tree.tag_configure("top", foreground=TEXT, font=(self.F, 10, "bold"))
        sb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree

    def _group(self, parent, text, first=False):
        if not first:
            tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(18, 0))
        ttk.Label(parent, text=text, style="Group.TLabel").pack(anchor="w", pady=(14, 8))

    def _field(self, parent, text):
        ttk.Label(parent, text=text, style="Field.TLabel").pack(anchor="w", pady=(0, 3))

    def _combo(self, parent, var, pady=(0, 10)):
        cb = ttk.Combobox(parent, textvariable=var, state="disabled", width=40, font=(self.F, 10))
        cb.pack(fill="x", pady=pady)
        return cb

    def _build_sidebar(self, s):
        ttk.Label(s, text="FutbolKronos", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(s, text="Simulador de partidos", style="Hint.TLabel").pack(anchor="w", pady=(0, 2))

        self._group(s, "Próximos partidos")
        self.proximo = tk.StringVar()
        self.proximo_cb = self._combo(s, self.proximo, pady=(0, 4))
        self.proximo_cb.bind("<<ComboboxSelected>>", self._elegir_proximo)
        ttk.Label(s, text="O elige cualquier partido abajo.", style="Hint.TLabel").pack(anchor="w")

        self._group(s, "Partido")
        self._field(s, "Liga")
        self.liga = tk.StringVar()
        self.liga_cb = self._combo(s, self.liga)
        self.liga_cb.bind("<<ComboboxSelected>>", lambda e: self._cambiar_liga())
        self._field(s, "Local")
        self.local = tk.StringVar()
        self.local_cb = self._combo(s, self.local)
        self._field(s, "Visitante")
        self.visitante = tk.StringVar()
        self.visit_cb = self._combo(s, self.visitante)

        fila = ttk.Frame(s, style="Side.TFrame")
        fila.pack(fill="x")
        fila.columnconfigure(0, weight=1)
        ttk.Label(fila, text="Fecha", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 3))
        self.fecha = tk.StringVar(value=self._prev.get("fecha", datetime.today().strftime("%Y-%m-%d")))
        ttk.Entry(fila, textvariable=self.fecha, width=11, font=(self.F, 10)).grid(row=1, column=0, sticky="ew")
        ttk.Label(s, text="Fecha AAAA-MM-DD. Si ya se jugó, se simula con los datos de antes.",
                  style="Hint.TLabel", wraplength=284, justify="left").pack(anchor="w", pady=(6, 0))

        self._group(s, "Simulación")
        self.n_sims = tk.IntVar(value=self._prev.get("n_sims", 50_000))
        self._slider(s, "Simulaciones", self.n_sims, 10_000, 100_000, 10_000, miles)
        self.temp = tk.DoubleVar(value=self._prev.get("temp", 1.0))
        self._slider(s, "Temperatura · 1 = la del modelo", self.temp, 0.5, 1.5, 0.05,
                     lambda v: f"{float(v):.2f}".replace(".", ","))
        self.dropout = tk.BooleanVar(value=self._prev.get("dropout", True))
        ttk.Checkbutton(s, text="Incluir la incertidumbre del modelo", variable=self.dropout,
                        style="Side.TCheckbutton").pack(anchor="w", pady=(2, 0))

        self.run_btn = ttk.Button(s, text="Simular partido", style="Primary.TButton", command=self._simular,
                                  state="disabled", cursor="hand2")
        self.run_btn.pack(fill="x", pady=(18, 8))
        self.progress = ttk.Progressbar(s, mode="determinate", value=0, style="Side.Horizontal.TProgressbar")
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value="Cargando el modelo…")
        ttk.Label(s, textvariable=self.status, style="Hint.TLabel", wraplength=284,
                  justify="left").pack(anchor="w", pady=(6, 0))

    def _slider(self, parent, label, var, lo, hi, paso, f):
        row = ttk.Frame(parent, style="Side.TFrame")
        row.pack(fill="x", pady=(0, 2))
        ttk.Label(row, text=label, style="Field.TLabel").pack(side="left")
        val = ttk.Label(row, text=f(var.get()), style="Value.TLabel")
        val.pack(side="right")

        # deslizador dibujado a mano: carril fino, tramo recorrido y tirador redondo.
        # Sin animación: se arrastra decenas de veces y tiene que seguir al ratón al instante.
        cv = tk.Canvas(parent, bg=SIDE, height=22, highlightthickness=0, bd=0, takefocus=1, cursor="hand2")
        cv.pack(fill="x", pady=(2, 10))
        estado = {"hover": False, "drag": False}
        R = 7

        def dibujar():
            cv.delete("all")
            w = max(cv.winfo_width(), 40)
            x = R + (var.get() - lo) / (hi - lo) * (w - 2 * R)
            vivo = estado["hover"] or estado["drag"] or cv.focus_get() == cv
            cv.create_line(R, 11, w - R, 11, fill=LINE, width=3, capstyle="round")
            cv.create_line(R, 11, x, 11, fill=TEXT if vivo else MUTED, width=3, capstyle="round")
            if cv.focus_get() == cv:
                cv.create_oval(x - R - 3, 11 - R - 3, x + R + 3, 11 + R + 3, outline=LOCAL, width=1.5)
            cv.create_oval(x - R, 11 - R, x + R, 11 + R, fill=TEXT if vivo else THUMB, outline="")

        def poner(v):
            v = min(max(round(v / paso) * paso, lo), hi)
            var.set(int(round(v)) if isinstance(var, tk.IntVar) else round(v, 4))
            val.configure(text=f(var.get()))
            dibujar()

        def desde_x(e):
            w = max(cv.winfo_width(), 40)
            poner(lo + (e.x - R) / (w - 2 * R) * (hi - lo))

        def suelta(_e):
            estado["drag"] = False
            dibujar()

        cv.bind("<Configure>", lambda e: dibujar())
        cv.bind("<Button-1>", lambda e: (cv.focus_set(), estado.update(drag=True), desde_x(e)))
        cv.bind("<B1-Motion>", desde_x)
        cv.bind("<ButtonRelease-1>", suelta)
        cv.bind("<Enter>", lambda e: (estado.update(hover=True), dibujar()))
        cv.bind("<Leave>", lambda e: (estado.update(hover=False), dibujar()))
        cv.bind("<FocusIn>", lambda e: dibujar())
        cv.bind("<FocusOut>", lambda e: dibujar())
        cv.bind("<Left>", lambda e: poner(var.get() - paso))
        cv.bind("<Right>", lambda e: poner(var.get() + paso))

    # ---------- carga y listas ----------
    def _cargar(self):
        from simulador import Simulador
        sim = Simulador(log=lambda m: self.q.put(("status", m)))
        self.q.put(("cargado", sim))

    def _al_cargar(self, sim):
        self.sim = sim
        import descargar_ligas
        eq = pd.read_csv(os.path.join(BASE, "datos", "equivalencias_equipos.csv"), encoding="utf-8-sig")
        bonito = dict(zip(eq["nombre_football_data"], eq["nombre"]))
        self.ligas = {f"{d} · {nombre}": d for d, nombre in sim.ligas()}
        for etiqueta, d in self.ligas.items():
            internos = sim.equipos(d)
            vistos = {}
            for n in internos:
                v = bonito.get(n, n) if d in descargar_ligas.DIVISIONES and descargar_ligas.DIVISIONES[d][0] == "ESP" else n
                vistos[v if v not in vistos else n] = n
            self.nombres[d] = dict(sorted(vistos.items()))
        self.liga_cb.configure(values=list(self.ligas), state="readonly")
        self.liga.set(next(k for k, v in self.ligas.items() if v == "SP1"))
        self._cambiar_liga()
        if sim.proximos:
            self._proximos = {}
            elegidos = []
            for comp in sorted({p["liga"] for p in sim.proximos}):
                de_comp = [p for p in sim.proximos if p["liga"] == comp]
                jornadas = sorted({p["jornada"] for p in de_comp})[:2]       # las dos próximas jornadas
                elegidos += [p for p in de_comp if p["jornada"] in jornadas]
            for p in sorted(elegidos, key=lambda x: (x["fecha"], x["hora"])):
                d = pd.Timestamp(p["fecha"])
                inv = {v: k for k, v in self.nombres.get(p["liga"], {}).items()}
                pref = "Champions" if p["liga"] == "CL" else "LaLiga"
                txt = (f"{pref} J{p['jornada']} · {DIAS[d.weekday()]} {d:%d/%m} {p['hora']} · "
                       f"{inv.get(p['local'], p['local'])} – {inv.get(p['visitante'], p['visitante'])}")
                self._proximos[txt] = p
            self.proximo_cb.configure(values=list(self._proximos), state="readonly")
            self.proximo_cb.current(0)
            self._elegir_proximo()
        self.run_btn.configure(state="normal")
        aviso = f"  {sim.aviso_modelo}" if sim.aviso_modelo else ""
        self.status.set(f"Listo. Modelo: {len(sim.modelos)} semilla(s).{aviso}")
        self._draw_header()

    def _cambiar_liga(self):
        d = self.ligas[self.liga.get()]
        equipos = list(self.nombres[d])
        for cb, var in [(self.local_cb, self.local), (self.visit_cb, self.visitante)]:
            cb.configure(values=equipos, state="readonly")
            if var.get() not in equipos:
                var.set("")
        if equipos and not self.local.get():
            self.local.set(equipos[0])
            self.visitante.set(equipos[1] if len(equipos) > 1 else "")

    def _elegir_proximo(self, _e=None):
        p = self._proximos[self.proximo.get()]
        self.liga.set(next(k for k, v in self.ligas.items() if v == p["liga"]))
        self._cambiar_liga()
        inv = {v: k for k, v in self.nombres[p["liga"]].items()}
        self.local.set(inv.get(p["local"], p["local"]))
        self.visitante.set(inv.get(p["visitante"], p["visitante"]))
        self.fecha.set(pd.Timestamp(p["fecha"]).strftime("%Y-%m-%d"))

    # ---------- simulación ----------
    def _simular(self):
        if self.worker and self.worker.is_alive():
            return
        d = self.ligas.get(self.liga.get())
        loc, vis = self.nombres[d].get(self.local.get()), self.nombres[d].get(self.visitante.get())
        if not loc or not vis:
            messagebox.showwarning("FutbolKronos", "Elige el equipo local y el visitante.")
            return
        if loc == vis:
            messagebox.showwarning("FutbolKronos", "El local y el visitante tienen que ser distintos.")
            return
        try:
            fecha = pd.Timestamp(self.fecha.get())
        except ValueError:
            messagebox.showwarning("FutbolKronos", "La fecha tiene que tener el formato AAAA-MM-DD.")
            return
        args = dict(liga=d, local=loc, visitante=vis, fecha=fecha, n=int(self.n_sims.get()),
                    temperatura=float(self.temp.get()), dropout=bool(self.dropout.get()),
                    nombres=(self.local.get(), self.visitante.get()))
        self.run_btn.configure(state="disabled", text="Simulando…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        self.t0 = time.time()
        self._run(self._trabajo, args)

    def _trabajo(self, a):
        self.q.put(("status", "Preparando el historial de los dos equipos…"))
        p = self.sim.preparar(a["liga"], a["local"], a["visitante"], a["fecha"])
        self.q.put(("status", "Calculando probabilidades con el modelo…"))
        dist = self.sim.distribuciones(p, dropout=a["dropout"])
        self.q.put(("status", f"Simulando {miles(a['n'])} partidos…"))
        marcador, P = self.sim.simular(dist, a["n"], a["temperatura"])
        res = self.sim.resumir(marcador, P)
        res["modelo"] = P.mean(0)
        self.q.put(("hecho", (p, res, a)))

    def _run(self, fn, *args):
        def envoltura():
            try:
                fn(*args)
            except Exception as e:
                self.q.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))
        self.worker = threading.Thread(target=envoltura, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "status":
                    self.status.set(msg[1])
                elif msg[0] == "cargado":
                    self._al_cargar(msg[1])
                elif msg[0] == "hecho":
                    self._terminar()
                    self.partido, self.res, self.args = msg[1]
                    segundos = f"{time.time() - self.t0:.1f}".replace(".", ",")
                    self.status.set(f"{miles(self.res['n'])} simulaciones en {segundos} s.")
                    self._mostrar()
                elif msg[0] == "error":
                    self._terminar()
                    self.status.set("Error: " + msg[1].splitlines()[0])
                    print(msg[2], file=sys.stderr)
                    messagebox.showerror("FutbolKronos", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _terminar(self):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)    # en reposo no se ve nada
        self.run_btn.configure(state="normal" if self.sim else "disabled", text="Simular partido")

    # ---------- cabecera del partido ----------
    def _fuente(self, familia, tam, peso="normal"):
        clave = (familia, tam, peso)
        if clave not in self._fuentes:
            self._fuentes[clave] = tkfont.Font(family=familia, size=tam, weight=peso)
        return self._fuentes[clave]

    def _draw_header(self):
        """Cabecera del partido: equipos, barra 1·X·2 con la banda del modelo y una fila de datos.
        Todo se coloca midiendo las fuentes, así no se corta con otra escala de pantalla."""
        c = self.head
        c.delete("all")
        w = max(c.winfo_width(), 600)
        f_meta, f_eq = self._fuente(self.F, 9), self._fuente(self.FD, 22, "bold")
        f_big, f_pct = self._fuente(self.FD, 17, "bold"), self._fuente(self.FD, 15, "bold")
        f_val = self._fuente(self.FD, 13, "bold")
        ls = lambda f: f.metrics("linespace")
        hay = self.res is not None and self.args is not None
        if hay:
            r, p = self.res, self.partido
            nl, nv = self.args["nombres"]
            d = self.args["liga"]
            comp = next((k.split(" · ", 1)[-1] for k, v in getattr(self, "ligas", {}).items() if v == d), d)
            f = pd.Timestamp(p["fecha"])
            meta = f"{comp}  ·  {DIAS[f.weekday()]} {f:%d/%m/%Y}  ·  {miles(r['n'])} simulaciones"
        else:
            nl = self.local.get() or "Local"
            nv = self.visitante.get() or "Visitante"
            meta = "Sin simular  ·  elige un partido y pulsa «Simular partido»"

        y = 0
        c.create_text(0, y, text=meta, anchor="nw", fill=FAINT, font=f_meta)
        y += ls(f_meta) + 4
        c.create_text(0, y, text=nl, anchor="nw", fill=TEXT, font=f_eq)
        c.create_text(w, y, text=nv, anchor="ne", fill=TEXT, font=f_eq)
        y += ls(f_eq) + 16
        y0, alto, gap = y, 12, 3
        ytxt = y0 + alto + 10
        ysub = ytxt + ls(f_big) + 2
        yf = ysub + ls(f_meta) + 14
        cols = [LOCAL, EMPATE, VISITA]

        if hay:
            pr = r["p"]
            vis = self._p_vis if self._p_vis is not None else pr
            quieta = np.allclose(vis, pr)
            cortes = [0, vis[0] * w, (vis[0] + vis[1]) * w, w]
            for k in range(3):
                x1 = cortes[k] + (gap / 2 if k else 0)
                x2 = cortes[k + 1] - (gap / 2 if k < 2 else 0)
                if x2 > x1:
                    c.create_rectangle(x1, y0, x2, y0 + alto, fill=cols[k], outline="")
            # dónde podría caer cada frontera según la banda del modelo (percentiles 10–90);
            # aparece cuando la barra ya ha llegado a su sitio
            b = r["p_banda"]
            if self.args["dropout"] and quieta:
                ym = y0 - 7
                for lo, hi in [(b[0, 0] * w, b[1, 0] * w), ((1 - b[1, 2]) * w, (1 - b[0, 2]) * w)]:
                    c.create_line(lo, ym, hi, ym, fill=MUTED, width=1)
                    for x in (lo, hi):
                        c.create_line(x, ym - 3, x, ym + 4, fill=MUTED, width=1)
            favorito = int(np.argmax(pr))
            etiquetas = [f"Gana {nl}", "Empate", f"Gana {nv}"]
            centro = min(max((cortes[1] + cortes[2]) / 2, 170), w - 170)
            for k, (x, anc) in enumerate([(0, "nw"), (centro, "n"), (w, "ne")]):
                c.create_text(x, ytxt + (0 if k == favorito else 2), text=pct(pr[k]), anchor=anc,
                              fill=cols[k] if k != 1 else TEXT, font=f_big if k == favorito else f_pct)
                sub = etiquetas[k]
                if self.args["dropout"]:
                    sub += f"  ·  modelo {pct(b[0, k], 0).replace(' %', '')}–{pct(b[1, k], 0)}"
                c.create_text(x, ysub, text=sub, anchor=anc, fill=MUTED, font=f_meta)
        else:
            c.create_rectangle(0, y0, w, y0 + alto, fill=FIELD, outline="")
            for x, anc, t in [(0, "nw", "1 · local"), (w / 2, "n", "X · empate"), (w, "ne", "2 · visitante")]:
                c.create_text(x, ytxt, text="—", anchor=anc, fill=FAINT, font=f_pct)
                c.create_text(x, ysub, text=t, anchor=anc, fill=FAINT, font=f_meta)

        # fila de datos (en vacío deja el hueco con rayas para que nada salte al simular)
        c.create_line(0, yf, w, yf, fill=LINE)
        if hay:
            from simulador import etiqueta
            top = int(np.argmax(r["cuenta"]))
            datos = [("Marcador más probable", etiqueta(top), pct(r["cuenta"][top] / r["n"]), TEXT),
                     ("Goles esperados", f"{r['media_local']:.1f} – {r['media_visitante']:.1f}".replace(".", ","),
                      "aprox. (4+ cuenta como 4)", TEXT)]
            if p["real"]:
                gl, gv = p["real"]
                idx = 0 if gl > gv else 1 if gl == gv else 2
                ok = favorito == idx
                datos.append(("Resultado real", f"{gl}-{gv}",
                              f"{'acertado' if ok else 'no era el más probable'} · le daba {pct(pr[idx], 0)}",
                              UP if ok else DOWN))
        else:
            datos = [("Marcador más probable", "—", "", FAINT), ("Goles esperados", "—", "", FAINT)]
        col_w = w / len(datos)
        yt = yf + 12
        yv = yt + ls(f_meta) + 4
        for i, (titulo, valor, sub, color) in enumerate(datos):
            x = i * col_w + (0 if i == 0 else 20)
            if i:
                c.create_line(i * col_w, yt, i * col_w, yv + ls(f_val), fill=LINE)
            c.create_text(x, yt, text=titulo, anchor="nw", fill=FAINT, font=f_meta)
            c.create_text(x, yv, text=valor, anchor="nw", fill=color, font=f_val)
            if sub:
                c.create_text(x + f_val.measure(valor) + 10, yv + ls(f_val) - ls(f_meta) - 1, text=sub,
                              anchor="nw", fill=MUTED, font=f_meta)
        alto_total = yv + ls(f_val) + 4
        if int(c.cget("height")) != alto_total:
            c.configure(height=alto_total)

    def _animar_barra(self):
        """La barra 1·X·2 viaja del reparto anterior al nuevo (o desde tercios iguales la primera vez).
        Una sola animación con propósito: enseña cuánto ha cambiado la predicción. Interrumpible:
        si llega otra simulación a mitad, sale desde donde esté."""
        objetivo = self.res["p"].astype(float)
        primera = self._p_vis is None
        inicio = np.full(3, 1 / 3) if primera else self._p_vis.copy()
        curva, dur = (ease_out, 0.26) if primera else (ease_in_out, 0.28)
        self._anim += 1
        mia, t0 = self._anim, time.perf_counter()

        def paso():
            if mia != self._anim:
                return
            t = min(1.0, (time.perf_counter() - t0) / dur)
            self._p_vis = objetivo if t >= 1 else inicio + (objetivo - inicio) * curva(t)
            self._draw_header()
            if t < 1:
                self.root.after(12, paso)
        paso()

    def _mostrar(self):
        self._animar_barra()
        self._dibujar()
        self._tablas()
        self._contexto()
        self.png_btn.configure(state="normal")
        self.csv_btn.configure(state="normal")

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.54, "Aún no hay simulación", ha="center", va="center", color=TEXT, fontsize=12,
                fontweight="bold")
        ax.text(0.5, 0.46, "Elige un partido a la izquierda y pulsa «Simular partido».\n"
                           "Aquí verás el reparto 1·X·2, el mapa de marcadores, los goles de cada equipo\n"
                           "y cómo se estabilizan las probabilidades.",
                ha="center", va="top", color=MUTED, fontsize=9.5, linespacing=1.6)
        self.canvas.draw_idle()

    def _estilo_ejes(self, ax, rejilla="y"):
        ax.set_facecolor(SURF)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
        if rejilla:
            ax.grid(axis=rejilla, color=GRID, lw=0.8)
            ax.set_axisbelow(True)

    def _dibujar(self):
        r, nl, nv = self.res, *self.args["nombres"]
        f = self.fig
        f.clear()
        gs = f.add_gridspec(2, 2, hspace=0.62, wspace=0.26, left=0.075, right=0.94, top=0.86, bottom=0.1)
        f.text(0.075, 0.955, f"{nl} – {nv}", color=TEXT, fontsize=11, fontweight="bold", va="top")
        f.text(0.94, 0.955, f"{pd.Timestamp(self.partido['fecha']):%d/%m/%Y} · {miles(r['n'])} simulaciones",
               color=FAINT, fontsize=9, ha="right", va="top")
        etq = ["0", "1", "2", "3", "4+"]
        cols = [LOCAL, EMPATE, VISITA]

        # 1. Resultado 1X2 con la banda del modelo
        ax = f.add_subplot(gs[0, 0])
        self._estilo_ejes(ax, rejilla="x")
        y = np.arange(3)[::-1]
        ax.barh(y, r["p"] * 100, color=cols, height=0.42)
        techo = r["p"] * 100
        if self.args["dropout"]:
            b = r["p_banda"] * 100
            for k in range(3):
                ax.plot([b[0, k], b[1, k]], [y[k] + 0.34] * 2, color=MUTED, lw=1.2, solid_capstyle="butt")
                for x in b[:, k]:
                    ax.plot([x, x], [y[k] + 0.28, y[k] + 0.40], color=MUTED, lw=1.2)
            techo = np.maximum(techo, b[1])
        for k in range(3):
            ax.text(r["p"][k] * 100 + 1.2, y[k], pct(r["p"][k]), va="center", fontsize=9.5,
                    fontweight="bold", color=TEXT)
        ax.set_yticks(y, [nl, "Empate", nv])
        ax.set_xlim(0, max(techo) + 14)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %"))
        ax.spines["left"].set_visible(False)
        ax.set_title("Resultado · trazo fino = banda del modelo" if self.args["dropout"] else "Resultado",
                     loc="left")

        # 2. Mapa de marcadores
        ax = f.add_subplot(gs[0, 1])
        M = r["matriz"] * 100
        cmap = LinearSegmentedColormap.from_list("fk", list(HEAT))
        ax.imshow(M, cmap=cmap, origin="upper", vmin=0, vmax=max(M.max(), 1e-9), aspect="auto")
        for i in range(5):
            for j in range(5):
                claro = M[i, j] > M.max() * 0.62
                ax.text(j, i, f"{M[i, j]:.1f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                        color=HEAT_INK if claro else TEXT, fontweight="bold" if claro else "normal")
        for k in range(6):
            ax.axhline(k - 0.5, color=SURF, lw=2)
            ax.axvline(k - 0.5, color=SURF, lw=2)
        real = self.partido.get("real")
        if real:
            gi, gj = min(real[0], 4), min(real[1], 4)
            ax.add_patch(Rectangle((gj - 0.47, gi - 0.47), 0.94, 0.94, fill=False, ec=REAL, lw=2))
        ax.set_xticks(range(5), etq)
        ax.set_yticks(range(5), etq)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"Goles de {nv}")
        ax.set_ylabel(f"Goles de {nl}")
        ax.set_title("Marcadores · % de simulaciones" + (" · recuadro = real" if real else ""), loc="left")

        # 3. Goles de cada equipo
        ax = f.add_subplot(gs[1, 0])
        self._estilo_ejes(ax)
        x = np.arange(5)
        ax.bar(x - 0.19, r["goles_local"] * 100, width=0.36, color=LOCAL, label=nl)
        ax.bar(x + 0.19, r["goles_visitante"] * 100, width=0.36, color=VISITA, label=nv)
        ax.set_xticks(x, etq)
        ax.set_xlabel("Goles")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %"))
        ax.legend(frameon=False, fontsize=9, loc="upper right", handlelength=0.9, handleheight=0.9)
        ax.set_title("Goles de cada equipo", loc="left")

        # 4. Convergencia de las simulaciones
        ax = f.add_subplot(gs[1, 1])
        self._estilo_ejes(ax)
        c = r["convergencia"] * 100
        n = np.arange(1, len(c) + 1)
        paso = max(1, len(c) // 2000)
        desde = min(100, len(c) - 1)         # con menos simulaciones las proporciones solo dan saltos
        for k, col in enumerate(cols):
            ax.plot(n[desde::paso], c[desde::paso, k], color=col, lw=1.5)
        visibles = c[min(100, len(c) - 1):]
        bajo, alto = visibles.min(), visibles.max()
        margen = max(4.0, (alto - bajo) * 0.15)
        ax.set_ylim(max(0, bajo - margen), min(100, alto + margen))
        # etiquetas al final de cada línea, separadas para que no se pisen
        finales = c[-1]
        orden = np.argsort(finales)
        hueco = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.075
        pos = finales.astype(float).copy()
        for a, b2 in zip(orden[:-1], orden[1:]):
            pos[b2] = max(pos[b2], pos[a] + hueco)
        for k in range(3):
            ax.annotate(pct(finales[k] / 100), (n[-1], pos[k]), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=cols[k], fontweight="bold", annotation_clip=False)
        ax.set_xscale("log")
        ax.set_xlim(n[desde], n[-1])
        ax.set_xlabel("Simulaciones (escala logarítmica)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %"))
        ax.set_title("Cómo se estabilizan las probabilidades", loc="left")
        self.canvas.draw_idle()

    def _tablas(self):
        from simulador import GOLES, etiqueta
        r, nl, nv = self.res, *self.args["nombres"]
        self.tree_marc.delete(*self.tree_marc.get_children())
        for fila, i in enumerate(np.argsort(-r["cuenta"])):
            gl, gv = GOLES[i]
            res = f"Gana {nl}" if gl > gv else "Empate" if gl == gv else f"Gana {nv}"
            tags = (("par",) if fila % 2 else ()) + (("top",) if fila == 0 else ())
            self.tree_marc.insert("", "end", tags=tags,
                                  values=(etiqueta(i), res, miles(r["cuenta"][i]),
                                          pct(r["cuenta"][i] / r["n"], 2), pct(r["modelo"][i], 2)))
        self.tree_merc.delete(*self.tree_merc.get_children())
        filas = [(f"Más de {k:.1f} goles".replace(".", ","), v) for k, v in r["mas"].items()]
        filas += [("Marcan los dos equipos", r["ambos"]),
                  (f"{nl} deja la portería a cero", r["porteria_cero_local"]),
                  (f"{nv} deja la portería a cero", r["porteria_cero_visitante"]),
                  (f"Doble oportunidad: {nl} o empate (1X)", r["p"][0] + r["p"][1]),
                  (f"Doble oportunidad: {nv} o empate (X2)", r["p"][2] + r["p"][1]),
                  ("Doble oportunidad: no hay empate (12)", r["p"][0] + r["p"][2])]
        for fila, (nombre, v) in enumerate(filas):
            self.tree_merc.insert("", "end", tags=("par",) if fila % 2 else (), values=(nombre, pct(v), pct(1 - v)))

    def _ctx_write(self, trozos):
        self.ctx.configure(state="normal")
        self.ctx.delete("1.0", "end")
        for texto, tag in trozos:
            self.ctx.insert("end", texto, tag) if tag else self.ctx.insert("end", texto)
        self.ctx.configure(state="disabled")

    def _contexto(self):
        p, (nl, nv) = self.partido, self.args["nombres"]
        t = [(f"{nl} – {nv}\n", "h1"), (f"{pd.Timestamp(p['fecha']):%d/%m/%Y}\n", "muted")]
        if p["real"]:
            t += [(f"Ya se jugó: {p['real'][0]}-{p['real'][1]}. ", None),
                  ("La simulación solo usa datos anteriores.\n", "muted")]

        escala = "Elo europeo (Elo de liga + desfase del país)" if self.args["liga"] in EUROPA else "Elo (escala de su país)"
        t += [(f"{escala} antes del partido\n", "h2"),
              (f"{nl}  ", None), (f"{p['elo_local']:.0f}", "loc"), ("     ", None),
              (f"{nv}  ", None), (f"{p['elo_visitante']:.0f}", "vis"),
              (f"     diferencia {p['elo_local'] - p['elo_visitante']:+.0f}\n", "muted")]

        for nombre, forma, h in [(nl, p["forma_local"], p["historial"][0]),
                                 (nv, p["forma_visitante"], p["historial"][1])]:
            t += [(f"Últimos partidos de {nombre}", "h2"), (f"   el modelo ve {h} de {self.sim.n_hist}\n", "faint")]
            if not forma:
                t.append(("sin partidos previos en los datos\n", "muted"))
            for f, donde, rival, gf, gc, res in forma:
                t += [(f"{f:%d/%m/%Y}   ", "mono"), (res, res), (f"   {gf}-{gc}   {donde:<5}  ", "mono"),
                      (f"{rival}\n", None)]

        if min(p["historial"]) < self.sim.n_hist:
            t.append((f"\nAviso: algún equipo tiene menos de {self.sim.n_hist} partidos previos; "
                      f"la predicción es menos fiable.\n", "aviso"))
        t.append((f"\nModelo: {len(self.sim.modelos)} semilla(s)"
                  f"{' con incertidumbre (dropout, 100 versiones por semilla)' if self.args['dropout'] else ''} · "
                  f"temperatura {self.args['temperatura']:.2f}".replace(".", ",")
                  + (f" · calibración del modelo {self.sim.temperatura:.2f}".replace(".", ",")
                     if self.sim.temperatura != 1.0 else ""), "faint"))
        self._ctx_write(t)

    # ---------- guardar ----------
    def _nombre_archivo(self, ext):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        nl, nv = self.args["nombres"]
        limpio = lambda s: "".join(c for c in s if c.isalnum() or c in "-_")[:20]
        return os.path.join(RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{limpio(nl)}_{limpio(nv)}.{ext}")

    def _guardar_png(self):
        path = self._nombre_archivo("png")
        self.fig.savefig(path, dpi=150, facecolor=SURF)
        self.status.set(f"Guardado: {os.path.relpath(path, BASE)}")

    def _guardar_csv(self):
        from simulador import GOLES, etiqueta
        r = self.res
        df = pd.DataFrame({"marcador": [etiqueta(i) for i in range(len(r["cuenta"]))],
                           "goles_local": GOLES[:, 0], "goles_visitante": GOLES[:, 1],
                           "simulaciones": r["cuenta"], "pct_simulaciones": r["cuenta"] / r["n"],
                           "prob_modelo": r["modelo"]}).sort_values("simulaciones", ascending=False)
        path = self._nombre_archivo("csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        self.status.set(f"Guardado: {os.path.relpath(path, BASE)}")


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    FutbolKronosApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
