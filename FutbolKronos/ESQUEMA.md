# FutbolKronos · Esquema copiado de Kronos-mini

Esta carpeta contiene una copia exacta de la arquitectura de **Kronos-mini**, el modelo
más pequeño de Kronos, como punto de partida para un modelo de fútbol.

| Archivo | Qué es |
|---|---|
| `model/` | Código original de Kronos (sin modificar): la red neuronal completa |
| `configs/kronos_mini.json` | Los tamaños de Kronos-mini (los "planos" del modelo) |
| `configs/kronos_tokenizer_original.json` | Configuración del tokenizador original de Kronos, como referencia |
| `crear_modelo.py` | Construye el modelo vacío, lo prueba y guarda los pesos iniciales |
| `LICENSE_Kronos.txt` | Licencia MIT de Kronos (obligatorio conservarla al copiar su código) |

Ejecuta `python3 crear_modelo.py` desde esta carpeta para comprobar que todo funciona.

---

## Cómo funciona Kronos, pieza a pieza

Kronos tiene dos redes que trabajan en cadena:

```
 Velas (números)  ──►  [1] TOKENIZADOR  ──►  tokens  ──►  [2] TRANSFORMER  ──►  siguiente token
 apertura, máx,         red neuronal que        (s1, s2)       predice qué viene        │
 mín, cierre, vol       convierte números                      después                  ▼
                        en "palabras"                                         [1] TOKENIZADOR
                                                                              vuelve a convertir
                                                                              el token en números
```

### [1] Tokenizador (`KronosTokenizer`)
Los precios son números continuos, y un transformer solo sabe trabajar con un
vocabulario finito. El tokenizador es una red (encoder + decoder) que comprime cada
vela en dos códigos binarios de 10 bits:

- **s1 (token grueso)**: 1.024 valores posibles. La forma general de la vela.
- **s2 (token fino)**: otros 1.024 valores. El detalle.

### [2] Transformer (`Kronos`), el que copiamos con los tamaños de "mini"

| Bloque | Qué hace | Tamaño en mini |
|---|---|---|
| `embedding` (HierarchicalEmbedding) | Convierte cada par (s1, s2) en un vector de 256 números | 2 tablas de 1.024 × 256 |
| `time_emb` (TemporalEmbedding) | Añade información de cuándo ocurre: minuto, hora, día de la semana, día y mes | 256 |
| `transformer` | 4 bloques de atención causal (cada token solo ve los anteriores) con RoPE | 4 capas, 4 cabezas, 256 → 512 → 256 |
| `dep_layer` (DependencyAwareLayer) | Primero decide s1 y, sabiendo s1, decide s2 | atención cruzada |
| `head` (DualHead) | Da la probabilidad de cada s1 y cada s2 siguientes | 2 × 1.024 salidas |

Total: unos **4 millones de parámetros**. Admite un contexto de 2.048 tokens.

Se entrena con una sola tarea: **predecir el siguiente token** (error de entropía cruzada
sobre s1 y s2). Para predecir, genera tokens uno a uno, muestreando con temperatura y
top-p, y repite varias veces para obtener varios futuros posibles.

---

## Qué cambia al pasar al fútbol

| Kronos (mercados) | FutbolKronos (fútbol) |
|---|---|
| Un token = una vela | Un token = un evento del partido |
| El tokenizador es una red neuronal, porque los precios son números continuos | **No hace falta red**: los eventos ya son categorías. Basta un diccionario (vocabulario) |
| s1 = forma general de la vela | s1 = **qué pasa**: equipo + tipo de evento (pase, tiro, gol, falta, tarjeta, córner, cambio…) |
| s2 = detalle de la vela | s2 = **dónde/contexto**: zona del campo, marcador, jugador… |
| `time_emb`: minuto, hora, día, mes | Minuto del partido, parte, día de la semana y mes (encaja casi directamente) |
| Contexto = velas anteriores | Contexto = eventos anteriores del partido (y de partidos previos) |
| Simular velas futuras | Simular el resto del partido → probabilidad de victoria, goles… |

La parte que hay que **reemplazar** es el tokenizador. El transformer (`model/kronos.py`,
clase `Kronos`) se puede usar tal cual: solo hay que ajustar `s1_bits` y `s2_bits` al
tamaño del vocabulario de fútbol cuando lo diseñemos.

## Principios de diseño

### 1. Los datos recientes pesan más (decidido el 01/11/2026)
Si un equipo lleva dos años mucho peor que antes, el modelo tiene que darle menos
probabilidad, aunque su historial desde 2012 sea bueno. Pero no todo caduca igual:

| Qué aprende el modelo | ¿Cambia con el tiempo? | Cómo se trata |
|---|---|---|
| **Cómo funciona el fútbol** (ventaja de local, reparto de goles, efecto de una expulsión, remontadas…) | Casi no | Todos los datos desde 2012, mismo peso |
| **Lo bueno que es cada equipo** | Sí, y rápido (fichajes, entrenadores, ventas, descensos) | Los datos recientes pesan mucho más |

Se aplica de tres formas combinadas:

1. **El modelo no memoriza a los equipos, los juzga por sus partidos recientes.**
   Nada de una ficha fija por club aprendida de todo su historial. Como contexto recibe
   los últimos partidos de cada equipo antes del que tiene que predecir (igual que Kronos
   mira las últimas velas, no el precio de hace 10 años).
2. **Indicadores que se actualizan solos**, como tokens de contexto:
   - rating tipo **Elo**, que sube y baja tras cada partido y olvida el pasado lejano;
   - **valor de mercado de la plantilla actual** y **cambios de entrenador**, que detectan un
     bajón (p. ej. venta de las estrellas) antes de que se note en los resultados.
3. **Peso por antigüedad al entrenar**, con una semivida:
   `peso = 0,5 ^ (antigüedad del partido / semivida)`.
   Con una semivida de 1 temporada: esta temporada = 1, hace 1 temporada = 0,5,
   hace 2 = 0,25… (como el modelo Dixon-Coles de las casas de apuestas).
   Además, como Kronos: **preentrenamiento con todo el histórico** y después
   **ajuste fino solo con las temporadas recientes**.

**La semivida no se fija a ojo**: se prueban varias (½, 1, 2 temporadas…) prediciendo
temporadas pasadas que el modelo no ha visto, y se elige la que mejor acierte.

Afecta a los pasos 3 a 5. La descarga (paso 2) baja todo el histórico igualmente.

### 2. Opción B: el equipo como serie temporal (decidido el 01/11/2026)
Cada partido es una **vela** del equipo, como las velas de Kronos. La trayectoria de un
equipo partido a partido es su "gráfico", y el modelo predice la siguiente vela.
(La opción A, el partido como frase de eventos para el modo en directo, queda para después.)

---

## Vocabulario (propuesta, pendiente de implementar)

Tokenizador **explícito**, no una red neuronal como el de Kronos: con pocos datos es más
fiable, se entiende a simple vista y da directamente las probabilidades de cada resultado.
Mantiene la estructura de dos niveles de Kronos:

**s1 · el marcador (lo que el modelo tiene que predecir)**
Goles a favor (0, 1, 2, 3, 4+) × goles en contra (0, 1, 2, 3, 4+) = 25 marcadores,
más 4 tokens especiales (inicio, separador, partido a predecir, relleno) → **5 bits**.
Solo el 1,9 % de las veces un equipo marca 5 o más, así que "4+" pierde muy poco.

| a favor \ en contra | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| **0** | 8,0 % | 9,2 % | 6,0 % | 3,1 % | 2,0 % |
| **1** | 9,2 % | 12,2 % | 7,7 % | 3,2 % | 2,1 % |
| **2** | 6,0 % | 7,7 % | 4,7 % | 2,1 % | 1,2 % |
| **3** | 3,1 % | 3,2 % | 2,1 % | 0,8 % | 0,5 % |
| **4+** | 2,0 % | 2,1 % | 1,2 % | 0,5 % | 0,1 % |

**s2 · el contexto y cómo jugó**
- Casa o fuera (2)
- Fuerza del rival según el Elo antes del partido (4 niveles: mucho más fuerte, más fuerte,
  más débil, mucho más débil)
- Dominio: diferencia de tiros a puerta (≤ −3, −2/−1, 0, +1/+2, ≥ +3, o "sin datos" antes de 2005)

2 × 4 × 6 = 48 combinaciones → **6 bits**. Vocabulario total: 32 × 64 = 2.048 tokens
(Kronos usa más de un millón).

**Secuencia para predecir Local–Visitante**
```
[INICIO] últimas N velas del local [SEP] últimas N velas del visitante [SEP] [PARTIDO] → marcador
```
El token [PARTIDO] lleva en s2 lo que se sabe antes de jugar (casa/fuera, fuerza del rival).
El modelo predice su s1 (el marcador) y de ahí salen P(1), P(X), P(2), más/menos goles y
marcador exacto, sumando las probabilidades de los marcadores correspondientes.

Información extra por vela (cambio de entrenador, nivel de Elo, valor de la plantilla) irá
en la capa de contexto que en Kronos codifica la fecha (`TemporalEmbedding`).

## Modelos de referencia (a batir)
Evaluados en 4.279 partidos de 2015/16 a 2026/27 que no se usaron para ajustar nada:

| Modelo | Pérdida log. | Acierto |
|---|---|---|
| Siempre lo mismo (frecuencias históricas) | 1,0665 | 45,7 % |
| **Rating Elo** (`crear_velas.py`) | **0,9866** | 52,2 % |
| **Casas de apuestas** | **0,9630** | 54,0 % |

Objetivo: bajar del Elo y acercarse a las casas de apuestas.

---

## Siguientes pasos
1. ✅ Copiar el esquema de Kronos-mini (esta carpeta).
2. ✅ Descargar los datos de LaLiga (`descargar_datos.py`).
3. ✅ Unificar equipos (`unificar_equipos.py`) y crear velas + Elo (`crear_velas.py`).
4. ✅ Implementar el vocabulario y convertir los partidos en secuencias de tokens
   (`futbol_tokenizer.py`, `crear_secuencias.py`).
5. ✅ Primera versión del entrenamiento (`entrenar.py`, resultados en `datos/resultados_entrenamiento.txt`).
   Modelo reducido (635 k parámetros), semivida elegida: sin olvido, 4 épocas, 3 semillas.
   En prueba (2021/22–2026/27): pérdida log 0,9815, frente a 0,9868 del Elo y 0,9650 de las casas.
   La mejora sobre el Elo aún no es estadísticamente segura (IC 95 % de −0,003 a +0,014),
   y el modelo memoriza a partir de la 4.ª época.
6. ✅ Preentrenar con 22 divisiones europeas (`descargar_ligas.py`, `crear_secuencias_ligas.py`,
   `entrenar.py --datos ligas`): 237.421 partidos, 10,4 M de tokens (19 veces más que solo LaLiga).
   Elo por país (Primera y Segunda comparten escala), Elo relativo a la división como contexto
   y la liga en el s2 de [INICIO]. Prueba (mismos 1.999 partidos de LaLiga):

   | Modelo | Pérdida log | Acierto |
   |---|---|---|
   | Rating Elo | 0,9868 | 52,5 % |
   | Reducido, solo LaLiga | 0,9815 | 52,6 % |
   | **Reducido, 22 ligas** | **0,9763** | **53,2 %** |
   | Mini, 22 ligas | 0,9778 | 52,3 % |
   | Casas de apuestas | 0,9650 | 54,3 % |

   El reducido con ligas bate al Elo con seguridad estadística (IC 95 % de +0,003 a +0,018)
   y recorre casi la mitad del camino entre el Elo y las casas. El ajuste fino en LaLiga no
   aportó (0 épocas elegidas) y el mini aún no mejora al reducido: sigue memorizando pronto.
7. ✅ Diferencia de Elo local − visitante (escalones de 20 puntos) en el contexto de [PARTIDO]:
   sin efecto apreciable (prueba 0,9767 frente a 0,9763; validación 1,0012 frente a 1,0023).
   El modelo ya deducía esa diferencia de las velas del historial. Se mantiene (no estorba);
   la versión anterior se regenera con `crear_secuencias_ligas.py --sin-elo-diferencia`.
8. ✅ Alineaciones y valor de mercado (`crear_alineaciones.py`, Transfermarkt, 11 primeras
   divisiones desde 2012/13, ~43.500 partidos). Indicadores, todos conocidos al publicarse el once:
   - ausencias de titulares habituales: conteo, y ponderada por titularidades recientes (quien
     juega siempre pesa más, sin depender de su edad ni de su valor);
   - valor del once: actual, y corregido por edad (máximo de cada titular en los últimos 3 años).
   Resultado: **no mejoran**. Las 5 variantes empatan en validación (1,0008–1,0013) y en prueba
   (0,9781–0,9785). Con una regresión logística sobre LaLiga, las ausencias tampoco aportan y el
   valor del once solo mejora al Elo en 0,0035, algo que FutbolKronos ya supera por sí solo.
   La ventaja de las casas no está en estos datos (probablemente en el propio mercado y en
   noticias de última hora). Se dejan como opción (`crear_secuencias_ligas.py --ausencias
   ponderada --valor max3`) y el modelo principal no las usa: así no depende de fuentes pausadas.
   `api_football.py` queda preparado para alineaciones y entrenadores partido a partido, pero
   su plan gratuito solo da las temporadas 2022–2024.
   Entrenadores de 2026/27: `entrenadores_actuales.py` (football-data.org, plan gratuito, clave
   en `football_data_clave.txt`) guarda el entrenador actual de cada equipo en
   `datos/entrenadores_2026.csv` (editable; revisar las fechas de los marcados "revisar"), y
   `unificar_equipos.py` lo usa para rellenar huecos. Modelo final reentrenado: 0,9768 en prueba.
9. ✅ Predecir jornadas (`predecir_jornada.py`, o doble clic en `Predecir jornada.bat`).
   - Modelo de producción: `entrenar.py --datos ligas --produccion` entrena con TODOS los partidos
     (la configuración elegida en la evaluación) y guarda 3 semillas en
     `pesos/futbol-reducido-ligas-produccion/`, que el predictor promedia.
   - Calendario y resultados recientes de football-data.org (si football-data.co.uk va con retraso,
     completa los últimos resultados con los goles de football-data.org).
   - Da P(1/X/2), más de 2,5 goles, ambos marcan y los 3 marcadores más probables.
   - Cada predicción se guarda en `datos/predicciones/` y en `datos/predicciones_historial.csv`;
     `predecir_jornada.py --evaluar` mide el acierto real cuando se juegan los partidos.

   **Rutina semanal:** `descargar_ligas.py` → (si cambia algún entrenador, `entrenadores_actuales.py`
   y la cadena `unificar_equipos.py` → `crear_velas.py` → `crear_secuencias.py`) →
   `crear_secuencias_ligas.py` → `predecir_jornada.py`. Reentrenar el modelo de producción de vez
   en cuando (p. ej. cada mes) con `entrenar.py --datos ligas --produccion`.
10. ✅ Simulador de partidos con interfaz gráfica (`futbolkronos_gui.py`, doble clic en
    `Abrir FutbolKronos.bat`; el motor está en `simulador.py`). Cualquier liga de las 22, equipos y
    fecha (o un partido de las dos próximas jornadas de LaLiga). Hace de 10.000 a 100.000
    simulaciones: cada una elige una versión del modelo (3 semillas × 100 pasadas con dropout,
    para recoger la incertidumbre del propio modelo) y sortea un marcador, con temperatura como
    Kronos. Muestra 1X2 con banda de incertidumbre, mapa de marcadores, goles de cada equipo,
    convergencia, mercados (más/menos, ambos marcan, portería a cero, doble oportunidad) y la
    forma reciente. Si el partido ya se jugó, simula con los datos anteriores y lo compara con el
    resultado real. Exporta PNG y CSV a `resultados/`.
11. ✅ Champions League y competiciones europeas (`crear_europa.py`, `crear_secuencias_ligas.py --europa`).
    - 10.534 partidos: Champions, Europa League, Conference y sus previas 2012/13–2025/26
      (Transfermarkt) + la Champions en curso (football-data.org, jugados y pendientes).
    - Cada club europeo tiene el mismo identificador que en su liga: su historial mezcla liga y Europa.
      Clubes de países sin liga en los datos (Shakhtar, Slavia, Bodø/Glimt…): identificador `EUR:` e
      historial solo europeo. Emparejamiento de nombres en `datos/europa_equipos_fdorg.csv`.
    - **Elo europeo**: en un partido europeo cada equipo usa su Elo de liga + el desfase de su país,
      ajustado con los cruces europeos de temporadas anteriores (sin mirar el futuro). Desfases
      actuales: Inglaterra +253, Italia +231, España +170, Alemania +162, Francia +80… Grecia −216.
    - Prueba (sin cuotas de casas para la Champions):

      | | LaLiga 2021–26 | Champions 2021–26 (806 partidos) |
      |---|---|---|
      | Elo (liga / europeo) | 0,9868 | 0,9164 |
      | FutbolKronos sin Europa | 0,9768 | — |
      | FutbolKronos con Europa | 0,9781 (calibrado 0,9776) | **0,9151** |

      LaLiga no cambia de forma apreciable; en Champions supera ligeramente al Elo europeo.
      La calibración por temperatura (0,95) ayuda un poco; combinar con el Elo empeora en prueba.
    - Producción: un único modelo con Europa y calibrado (`entrenar_produccion.bat`). El simulador
      ofrece la Champions y sus próximos partidos; `predecir_jornada.py --competicion champions`.
    - Los procesos largos se lanzan con `.bat` independientes (`entrenar_europa.bat`,
      `entrenar_produccion.bat`) para que no se corten si se cierra la sesión.
12. ⬜ Repetir los experimentos (`experimentos.py`: peso del marcador, rojas, descanso, historial 40),
    ahora con Europa incluida.
13. ⬜ Siguientes ideas: menos peso en las posiciones del historial, más regularización para el
   mini. (Las cuotas como entrada se probaron y se descartaron: ver «Sin cuotas», abajo.)

Para entrenar con la gráfica: `..\venv-gpu\Scripts\python.exe entrenar.py --dispositivo cuda`
(unas 5 veces más rápido que con el procesador).

---

## Revisión de noviembre de 2026 (08/11)

**Correcciones**
- `experimentos.py` ya no sobrescribe las secuencias de producción: sus variantes van a
  `datos/*_exp.*` (`crear_secuencias_ligas.py --sufijo _exp`, `entrenar.py --secuencias _exp`).
  Antes, al cortarse la tanda del 04/11, `secuencias_ligas.npz` se quedó con historial 10; se han
  regenerado con historial 20 y Europa, la configuración de producción.
- `entrenar.py` no reescala las épocas al pasar a más datos (fase 2 y producción). Cada época ya
  trae más pasos; el producto de ambos efectos hizo que producción se entrenara 4 épocas cuando la
  validación eligió 3.
- Validación por defecto en **todas las ligas** (`--validacion todas`, 24.229 partidos frente a
  1.140 de LaLiga): con LaLiga sola, una diferencia menor de ±0,005 en pérdida log no se distingue
  del ruido, y la mayoría de los experimentos estaban por debajo de eso. La prueba sigue en LaLiga.
  La mezcla con el Elo se ajusta en LaLiga, que es donde existe ese Elo de referencia.
- `--lr-plan coseno` (opcional) con plan de la misma longitud en todas las fases (`--epocas-coseno`).
- Simulador: con un modelo entrenado con Europa construye el historial con todos los partidos, como
  el entrenamiento y `predecir_jornada.py`. Antes, en el 60 % de los partidos de LaLiga el
  historial salía distinto y P(1X2) cambiaba de media 1 punto (hasta 6).
- Claves de API fuera de git (`.gitignore`, `git rm --cached`).

**Estructura**
- `model/` → `kronos_futbol/`, para que no choque con el paquete `model` de `Kronos-code` en el
  mismo entorno. Se quitan `KronosPredictor`, `auto_regressive_inference` y `calc_time_stamps`
  (OHLCV); el adaptador de fútbol es `futbol_predictor.py` (`FutbolKronosPredictor`:
  `predecir(historial, partidos)` y `probabilidades(S1, S2, ST)`), usado por `predecir_jornada.py`
  y `simulador.py`. Da las mismas probabilidades que antes (diferencia < 2·10⁻⁸).
- `kronos_futbol/module.py`: la entropía del cuantizador BSQ calculaba el índice con
  `(zq + 1) / 2` sobre `zq` ya escalado, así que todos los códigos caían en el 0. Además, la rama
  dura fallaba (`avg_prob` sin definir) y en float16/bfloat16 `0·log 0` daba NaN. Corregido:
  signo para los bits, float32, `xlogy` y backward vectorizado. FutbolKronos no usa este
  tokenizador; solo afecta a quien lo reutilice.
- `pyproject.toml` (PEP 517/518/621) y fuera los `sys.path.insert`. Desde esta carpeta todo
  funciona igual; para importar los módulos desde otro sitio:
  `python -m pip install -e . --no-deps` (en `venv-gpu` y/o en `python3`).

**Diagnóstico (prueba LaLiga 2021/22–2026/27, mismos 1.999 partidos)**

| Modelo | Pérdida log |
|---|---|
| Casas de apuestas | 0,9650 |
| FutbolKronos (3 semillas) | 0,9781 |
| Red pequeña (MLP) con Elo + forma + tiros + descanso | 0,9790 |
| Regresión logística con esos mismos rasgos | 0,9796 |
| Rating Elo | 0,9868 |

En las 22 ligas (40.955 partidos con cuotas), un Poisson-Dixon-Coles con esos rasgos supera a
FutbolKronos (una semilla) por 0,0017 ± 0,0012, y el mercado le saca 0,015. El transformer no
extrae del historial nada que no den unos promedios móviles. Ajustado sobre las cuotas,
FutbolKronos recibe peso ≈ 0: no añade información al mercado. El margen que queda está en datos
nuevos de fútbol (xG, alineaciones el día del partido), no en la arquitectura.

**Prototipos (solo validación 2018/19–2020/21, 24.229 partidos, 3 semillas; no se ha mirado la prueba)**

| Variante | LL por semilla | LL 3 semillas | vs base |
|---|---|---|---|
| A · base (LR constante) | 1,0193 | 1,0174 | — |
| B · LR coseno | 1,0185 | 1,0174 | 0,0000 ± 0,0006 |
| C · coseno + cabeza Poisson-Dixon-Coles | 1,0175 | 1,0163 | +0,0011 ± 0,0008 |
| D · coseno + rasgos numéricos (híbrido) | 1,0167 | 1,0162 | +0,0012 ± 0,0009 |

El coseno mejora cada semilla y su variación (la mejor época pasa a ser la última), pero no la
media de 3 semillas, así que no se cambia el valor por defecto. La cabeza Poisson y el híbrido
ganan ~0,001 en el límite del ruido, y también mejoran algo más/menos 2,5 goles. Sobre las cuotas:
mercado 1,0078; mercado + rasgos 1,0084 (−0,0005 ± 0,0007). Los rasgos de resultados no añaden
nada al mercado.

---
*La configuración de Kronos-mini procede de la copia publicada del modelo en Hugging Face
y cuadra con los 4,1 M de parámetros que indica el repositorio oficial.*

**Tanda de experimentos del 08/11** (validación en todas las ligas, 2 semillas; `datos/experimentos.txt`):
poisson_hibrido 1,0166 · hibrido 1,0171 · poisson 1,0175 · rojas_descanso 1,0179 · coseno 1,0180 ·
descanso 1,0182 · base 1,0185 · historial10 1,0185. Con cuotas como entrada: 1,0141, y 1,0124 si se
combinan con Poisson e híbrido.

**Sin cuotas (decisión del 08/11).** Las cuotas como entrada bajaban la pérdida, pero porque el modelo
pasaba a copiar al mercado: este ya da 1,0078 por sí solo y ningún dato de partido le añade nada. Se
quiere un modelo que prediga a partir del fútbol, así que la opción se ha quitado de todo el código y
de la app: `--cuotas`/`--sin-cuotas`, las casillas de cuotas en `[PARTIDO]`, `cuotas_proximas.py`
(movido a `_respaldo_cuotas/`) y las cuotas escritas a mano en el simulador. Las cuotas históricas se
siguen descargando solo como **vara de medir** (fila «Casas de apuestas» de los resultados).
Configuración de producción: `--lr-plan coseno --cabeza poisson --rasgos` (`entrenar_todo.bat`).

