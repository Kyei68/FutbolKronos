"""
Paso 3a: unificar los nombres de los equipos y crear la tabla maestra de partidos.

- Crea datos/equivalencias_equipos.csv: el nombre de cada equipo en cada fuente
  y un identificador único (equipo_id). Si ya existe, lo respeta (puedes editarlo a mano).
- Comprueba que los partidos de las dos fuentes encajan (mismo día y mismo resultado).
- Crea datos/partidos_unificados.csv: un partido por fila desde 1993/94, uniendo
  resultados, estadísticas y cuotas (football-data) con entrenadores, sistemas,
  estadio y público (Transfermarkt, desde 2012/13).

Uso (desde esta carpeta):  python3 unificar_equipos.py
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "datos")
EQUIV = os.path.join(DATA, "equivalencias_equipos.csv")

# nombre en football-data -> (equipo_id, nombre oficial, club_id en Transfermarkt)
EQUIVALENCIAS = {
    "Alaves": ("alaves", "Deportivo Alavés", 1108),
    "Albacete": ("albacete", "Albacete Balompié", None),
    "Almeria": ("almeria", "UD Almería", 3302),
    "Ath Bilbao": ("athletic", "Athletic Club", 621),
    "Ath Madrid": ("atletico", "Atlético de Madrid", 13),
    "Barcelona": ("barcelona", "FC Barcelona", 131),
    "Betis": ("betis", "Real Betis", 150),
    "Cadiz": ("cadiz", "Cádiz CF", 2687),
    "Celta": ("celta", "RC Celta de Vigo", 940),
    "Compostela": ("compostela", "SD Compostela", None),
    "Cordoba": ("cordoba", "Córdoba CF", 993),
    "Eibar": ("eibar", "SD Eibar", 1533),
    "Elche": ("elche", "Elche CF", 1531),
    "Espanol": ("espanyol", "RCD Espanyol", 714),
    "Extremadura": ("extremadura", "CF Extremadura", None),
    "Getafe": ("getafe", "Getafe CF", 3709),
    "Gimnastic": ("gimnastic", "Gimnàstic de Tarragona", None),
    "Girona": ("girona", "Girona FC", 12321),
    "Granada": ("granada", "Granada CF", 16795),
    "Hercules": ("hercules", "Hércules CF", None),
    "Huesca": ("huesca", "SD Huesca", 5358),
    "La Coruna": ("deportivo", "RC Deportivo de La Coruña", 897),
    "Las Palmas": ("las-palmas", "UD Las Palmas", 472),
    "Leganes": ("leganes", "CD Leganés", 1244),
    "Lerida": ("lleida", "UE Lleida", None),
    "Levante": ("levante", "Levante UD", 3368),
    "Logrones": ("logrones", "CD Logroñés", None),
    "Malaga": ("malaga", "Málaga CF", 1084),
    "Mallorca": ("mallorca", "RCD Mallorca", 237),
    "Merida": ("merida", "CP Mérida", None),
    "Murcia": ("murcia", "Real Murcia", None),
    "Numancia": ("numancia", "CD Numancia", None),
    "Osasuna": ("osasuna", "CA Osasuna", 331),
    "Oviedo": ("oviedo", "Real Oviedo", 2497),
    "Real Madrid": ("real-madrid", "Real Madrid", 418),
    "Recreativo": ("recreativo", "Recreativo de Huelva", None),
    "Salamanca": ("salamanca", "UD Salamanca", None),
    "Santander": ("racing", "Racing de Santander", None),
    "Sevilla": ("sevilla", "Sevilla FC", 368),
    "Sociedad": ("real-sociedad", "Real Sociedad", 681),
    "Sp Gijon": ("sporting", "Real Sporting de Gijón", 2448),
    "Tenerife": ("tenerife", "CD Tenerife", None),
    "Valencia": ("valencia", "Valencia CF", 1049),
    "Valladolid": ("valladolid", "Real Valladolid", 366),
    "Vallecano": ("rayo", "Rayo Vallecano", 367),
    "Villarreal": ("villarreal", "Villarreal CF", 1050),
    "Villareal": ("villarreal", "Villarreal CF", 1050),   # errata en la temporada 1998/99
    "Xerez": ("xerez", "Xerez CD", None),
    "Zaragoza": ("zaragoza", "Real Zaragoza", 142),
}


def load_equivalencias():
    if os.path.exists(EQUIV):
        eq = pd.read_csv(EQUIV, encoding="utf-8-sig")
        print(f"   Usando {os.path.basename(EQUIV)} existente ({len(eq)} nombres).")
    else:
        eq = pd.DataFrame([(fd, *v) for fd, v in EQUIVALENCIAS.items()],
                          columns=["nombre_football_data", "equipo_id", "nombre", "tm_club_id"])
        eq["tm_club_id"] = eq["tm_club_id"].astype("Int64")
        eq.to_csv(EQUIV, index=False, encoding="utf-8-sig")
        print(f"   Creado {os.path.basename(EQUIV)} ({len(eq)} nombres, {eq['equipo_id'].nunique()} equipos).")
    return eq


def implied_probs(df):
    """Probabilidades implícitas en las cuotas (sin el margen de la casa).
    Usa la media del mercado; si no hay, Pinnacle; si no, Bet365."""
    out = pd.DataFrame(index=df.index, columns=["prob_local", "prob_empate", "prob_visitante"], dtype=float)
    src = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (h, d, a) in [("media", ("AvgH", "AvgD", "AvgA")), ("pinnacle", ("PSH", "PSD", "PSA")),
                            ("bet365", ("B365H", "B365D", "B365A"))]:
        if not all(c in df.columns for c in (h, d, a)):
            continue
        inv = 1 / df[[h, d, a]].apply(pd.to_numeric, errors="coerce")
        ok = inv.notna().all(axis=1) & out["prob_local"].isna()
        total = inv.sum(axis=1)
        out.loc[ok, "prob_local"] = (inv[h] / total)[ok]
        out.loc[ok, "prob_empate"] = (inv[d] / total)[ok]
        out.loc[ok, "prob_visitante"] = (inv[a] / total)[ok]
        src[ok] = name
    out = out.round(4)
    out["fuente_cuotas"] = src
    return out


def fill_coaches_from_api(out):
    """Rellena los entrenadores que falten (2026/27 en adelante) con los de api_football.py."""
    path = os.path.join(DATA, "api_football_laliga.csv")
    if not os.path.exists(path):
        return
    api = pd.read_csv(path, encoding="utf-8-sig")
    eq = pd.read_csv(os.path.join(DATA, "equivalencias_equipos.csv"), encoding="utf-8-sig")
    a_id = dict(zip(eq["nombre_football_data"], eq["equipo_id"]))
    api["fecha"] = pd.to_datetime(api["fecha"]).dt.date
    api["local_id"] = api["local"].map(a_id)
    api["visitante_id"] = api["visitante"].map(a_id)
    api = api.dropna(subset=["local_id", "visitante_id"]).drop_duplicates(["fecha", "local_id", "visitante_id"])
    m = out[["fecha", "local_id", "visitante_id"]].merge(
        api[["fecha", "local_id", "visitante_id", "entrenador_local", "entrenador_visitante"]],
        on=["fecha", "local_id", "visitante_id"], how="left")
    antes = out["entrenador_local"].notna().sum()
    for c in ["entrenador_local", "entrenador_visitante"]:
        out[c] = out[c].fillna(pd.Series(m[c].to_numpy(), index=out.index))
    print(f"   entrenadores añadidos desde API-Football: {out['entrenador_local'].notna().sum() - antes} partidos")


def fill_coaches_from_files(out):
    """Rellena los entrenadores que falten con datos/entrenadores_<temporada>.csv
    (entrenadores_actuales.py): para cada partido, el último entrenador con 'desde' <= fecha."""
    eq = pd.read_csv(os.path.join(DATA, "equivalencias_equipos.csv"), encoding="utf-8-sig")
    a_id = dict(zip(eq["nombre_football_data"], eq["equipo_id"]))
    antes = out["entrenador_local"].notna().sum() + out["entrenador_visitante"].notna().sum()
    for t in sorted(out["temporada_inicio"].unique()):
        path = os.path.join(DATA, f"entrenadores_{t}.csv")
        if not os.path.exists(path):
            continue
        ent = pd.read_csv(path, encoding="utf-8-sig")
        ent["equipo_id"] = ent["equipo"].map(a_id)
        ent["desde"] = pd.to_datetime(ent["desde"]).dt.date
        ent = ent.dropna(subset=["equipo_id"]).sort_values("desde")
        en_t = out["temporada_inicio"] == t
        for lado in ["local", "visitante"]:
            falta = en_t & out[f"entrenador_{lado}"].isna()
            for i in out.index[falta]:
                e = ent[(ent["equipo_id"] == out.at[i, f"{lado}_id"]) & (ent["desde"] <= out.at[i, "fecha"])]
                if len(e):
                    out.at[i, f"entrenador_{lado}"] = e["entrenador"].iloc[-1]
    despues = out["entrenador_local"].notna().sum() + out["entrenador_visitante"].notna().sum()
    if despues > antes:
        print(f"   entrenadores añadidos desde entrenadores_<temporada>.csv: {despues - antes}")


def main():
    print("Paso 3a · Unificar equipos\n")
    fd = pd.read_csv(os.path.join(DATA, "resultados_y_cuotas.csv"), encoding="utf-8-sig", low_memory=False)
    tm = pd.read_csv(os.path.join(DATA, "partidos.csv"), encoding="utf-8-sig", low_memory=False)
    eq = load_equivalencias()

    # 1. Nombres sin equivalencia
    names = set(fd["HomeTeam"]) | set(fd["AwayTeam"])
    missing = sorted(names - set(eq["nombre_football_data"]))
    if missing:
        print("\n   AVISO: estos nombres de football-data no tienen equivalencia y se descartan:")
        print("   " + ", ".join(missing))
        print(f"   Añádelos a {os.path.basename(EQUIV)} y vuelve a ejecutar.\n")

    to_id = dict(zip(eq["nombre_football_data"], eq["equipo_id"]))
    to_name = eq.drop_duplicates("equipo_id").set_index("equipo_id")["nombre"].to_dict()
    tm_to_id = eq.dropna(subset=["tm_club_id"]).drop_duplicates("tm_club_id")
    tm_to_id = dict(zip(tm_to_id["tm_club_id"].astype(int), tm_to_id["equipo_id"]))

    # 2. Tabla base: football-data
    fd["local_id"] = fd["HomeTeam"].map(to_id)
    fd["visitante_id"] = fd["AwayTeam"].map(to_id)
    fd = fd.dropna(subset=["local_id", "visitante_id"]).copy()
    fd["temporada_inicio"] = fd["temporada"].str[:4].astype(int)
    fd["fecha"] = pd.to_datetime(fd["fecha"], errors="coerce")

    # 3. Transfermarkt con los mismos identificadores
    tm["local_id"] = tm["home_club_id"].map(tm_to_id)
    tm["visitante_id"] = tm["away_club_id"].map(tm_to_id)
    unknown_tm = tm[tm["local_id"].isna() | tm["visitante_id"].isna()]
    if len(unknown_tm):
        print(f"   AVISO: {len(unknown_tm)} partidos de Transfermarkt con clubes sin equivalencia.")
    tm["temporada_inicio"] = tm["season"].astype(int)
    tm["fecha_tm"] = pd.to_datetime(tm["date"], errors="coerce")
    tm_cols = tm[["temporada_inicio", "local_id", "visitante_id", "game_id", "fecha_tm",
                  "home_club_goals", "away_club_goals", "round",
                  "home_club_manager_name", "away_club_manager_name",
                  "home_club_formation", "away_club_formation",
                  "home_club_position", "away_club_position", "stadium", "attendance", "referee"]]

    # Cada equipo recibe a cada rival una vez por temporada: (temporada, local, visitante) es único
    m = fd.merge(tm_cols, on=["temporada_inicio", "local_id", "visitante_id"], how="left", indicator=True)

    # 4. Comprobación cruzada
    both = m[m["_merge"] == "both"]
    tm_period = m[m["temporada_inicio"].between(tm["temporada_inicio"].min(), tm["temporada_inicio"].max())]
    same_score = (both["FTHG"] == both["home_club_goals"]) & (both["FTAG"] == both["away_club_goals"])
    day_diff = (both["fecha"] - both["fecha_tm"]).dt.days.abs()
    print("\n   Comprobación entre fuentes (temporadas con datos en ambas):")
    print(f"     partidos emparejados:     {len(both):,} de {len(tm_period):,} "
          f"({len(both) / max(len(tm_period), 1) * 100:.1f} %)")
    print(f"     mismo resultado:          {same_score.sum():,} ({same_score.mean() * 100:.1f} %)")
    print(f"     misma fecha (±1 día):     {(day_diff <= 1).sum():,} ({(day_diff <= 1).mean() * 100:.1f} %)")
    bad = both[~same_score | (day_diff > 1)]
    if len(bad):
        cols = ["temporada", "fecha", "fecha_tm", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                "home_club_goals", "away_club_goals"]
        bad[cols].to_csv(os.path.join(DATA, "revisar_discrepancias.csv"), index=False, encoding="utf-8-sig")
        print(f"     {len(bad)} discrepancias guardadas en revisar_discrepancias.csv")
    unmatched_tm = len(tm) - len(both)
    if unmatched_tm:
        print(f"     partidos de Transfermarkt sin pareja en football-data: {unmatched_tm}")

    # 5. Tabla maestra
    probs = implied_probs(m)
    out = pd.DataFrame({
        "temporada": m["temporada"],
        "temporada_inicio": m["temporada_inicio"],
        "fecha": m["fecha"].dt.date,
        "hora": m.get("Time"),
        "jornada": m["round"].astype(str).str.extract(r"(\d+)")[0].astype("Int64"),
        "local_id": m["local_id"],
        "local": m["local_id"].map(to_name),
        "visitante_id": m["visitante_id"],
        "visitante": m["visitante_id"].map(to_name),
        "goles_local": m["FTHG"],
        "goles_visitante": m["FTAG"],
        "resultado": m["FTR"].map({"H": "1", "D": "X", "A": "2"}),
        "goles_local_descanso": m.get("HTHG"),
        "goles_visitante_descanso": m.get("HTAG"),
    })
    stats = {"HS": "tiros_local", "AS": "tiros_visitante", "HST": "tiros_puerta_local",
             "AST": "tiros_puerta_visitante", "HC": "corners_local", "AC": "corners_visitante",
             "HF": "faltas_local", "AF": "faltas_visitante", "HY": "amarillas_local",
             "AY": "amarillas_visitante", "HR": "rojas_local", "AR": "rojas_visitante"}
    for src, dst in stats.items():
        if src in m.columns:
            out[dst] = m[src]
    out = pd.concat([out, probs], axis=1)
    out["entrenador_local"] = m["home_club_manager_name"]
    out["entrenador_visitante"] = m["away_club_manager_name"]
    fill_coaches_from_api(out)
    fill_coaches_from_files(out)
    out["sistema_local"] = m["home_club_formation"]
    out["sistema_visitante"] = m["away_club_formation"]
    out["posicion_local_antes"] = m["home_club_position"]
    out["posicion_visitante_antes"] = m["away_club_position"]
    out["estadio"] = m["stadium"]
    out["publico"] = m["attendance"]
    out["arbitro"] = m["referee"].fillna(m["Referee"]) if "Referee" in m.columns else m["referee"]
    out["tm_game_id"] = m["game_id"].astype("Int64")
    out = out.sort_values(["fecha", "local"]).reset_index(drop=True)

    path = os.path.join(DATA, "partidos_unificados.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n   partidos_unificados.csv: {len(out):,} partidos, {out['local_id'].nunique()} equipos, "
          f"{out['temporada'].nunique()} temporadas ({out['temporada'].iloc[0]} a {out['temporada'].iloc[-1]})")
    print(f"     con cuotas: {out['prob_local'].notna().mean() * 100:.0f} %   "
          f"con tiros: {out['tiros_local'].notna().mean() * 100:.0f} %   "
          f"con entrenadores: {out['entrenador_local'].notna().mean() * 100:.0f} %")
    print("\nListo.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: falta un archivo ({e.filename}). Ejecuta antes descargar_datos.py.", file=sys.stderr)
        sys.exit(1)
