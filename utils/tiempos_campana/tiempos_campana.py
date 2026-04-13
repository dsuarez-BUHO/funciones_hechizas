"""
tiempos_campana.py
------------------
Funciones específicas del pipeline de análisis de tiempos de picking por campaña.
Cubre tres capas:
    1. ETL        — transformación raw → df_clean → df_campanas
    2. Análisis   — eficiencia operacional, z-score contextual, categorización
    3. Resumen    — narrativa de métricas para exploración rápida
"""

import numpy as np
import pandas as pd


# ── Constantes de categorización ─────────────────────────────────────────────

ORDEN_CAT = ['critico', 'mejorable', 'bueno', 'excelente', 'sin_referencia']

COLORES_CAT = {
    'critico':        '#EA4335',
    'mejorable':      '#FBBC04',
    'bueno':          '#34A853',
    'excelente':      '#1E6B35',
    'sin_referencia': '#B0BEC5',
}


# ── 0. Normalización — API → esquema base ────────────────────────────────────

def categorizar_tipo_estacion(estacion):
    """
    Mapea el nombre de estación recibido desde la API al tipo semántico.

    Regla: estaciones agrupadas de 6 en 6.
        pos % 6 == 1 → Azul
        pos % 6 == 2 → Rojo
        pos % 6 == 3 → Verde
        pos % 6 == 4 → Amarillo
        pos % 6 == 5 → BG
        pos % 6 == 0 → Calidad
    Casos directos: "Calidad", "Pre-Calidad" y "BG" llegan como string,
    no como "Estacion N".

    Parámetros
    ----------
    estacion : str
        Valor de la columna estacion en los eventos crudos de la API.

    Retorna
    -------
    str — tipo semántico, o 'Otro' si el valor no coincide con ningún patrón.
    """
    if not isinstance(estacion, str):
        return 'Otro'

    est_lower = estacion.lower().strip()

    if est_lower == 'calidad':
        return 'Calidad'
    if est_lower in ('pre-calidad', 'pre calidad', 'precalidad'):
        return 'Pre-Calidad'
    if est_lower == 'bg':
        return 'BG'
    if est_lower in ('azul', 'rojo', 'verde', 'amarillo'):
        return estacion.strip().capitalize()

    if estacion.strip().startswith('Estacion '):
        try:
            n   = int(estacion.strip().split(' ', 1)[1])
            pos = n % 6
            if pos == 1: return 'Azul'
            if pos == 2: return 'Rojo'
            if pos == 3: return 'Verde'
            if pos == 4: return 'Amarillo'
            if pos == 5: return 'BG'
            if pos == 0: return 'Calidad'
        except ValueError:
            pass

    return 'Otro'


def transformer_api_eventos(df_raw):
    """
    Convierte eventos crudos de la API (box_open / box_close / scans)
    al esquema que espera transformar_base_campanas:
    una fila por operación (box_id × estacion).

    Parámetros
    ----------
    df_raw : pd.DataFrame
        Eventos crudos de la API con columnas:
        type, box_id, estacion, campana, id_campana, time, scan.
        Si incluye cliente_db (desde MySQL), se usa para asignar cliente;
        de lo contrario se extrae del prefijo del nombre de campaña.

    Retorna
    -------
    pd.DataFrame con columnas:
        box_id, campana, id_campana, tipo_estacion, init_scan, last_scan,
        picks, cliente.
    Solo incluye operaciones con par open+close completo.
    Las filas con tipo_estacion == 'Otro' deben filtrarse antes de subir a BQ.
    """
    df = df_raw.copy()
    df['time'] = pd.to_datetime(df['time'])

    KEY = ['box_id', 'estacion', 'campana', 'id_campana']

    opens = (
        df[df['type'] == 'box_open']
        .groupby(KEY)['time'].min()
        .rename('init_scan')
    )
    closes = (
        df[df['type'] == 'box_close']
        .groupby(KEY)['time'].max()
        .rename('last_scan')
    )
    scans = (
        df[df['scan'].notna()]
        .groupby(KEY)['scan'].count()
        .rename('picks')
    )

    ops = (
        opens.to_frame()
        .join(closes, how='inner')
        .join(scans, how='left')
        .fillna({'picks': 0})
        .reset_index()
    )

    ops['picks']         = ops['picks'].astype(int)
    ops['tipo_estacion'] = ops['estacion'].apply(categorizar_tipo_estacion)
    ops = ops.drop(columns='estacion')

    if 'cliente_db' in df_raw.columns:
        cliente_map = (
            df_raw[['campana', 'cliente_db']]
            .drop_duplicates()
            .set_index('campana')['cliente_db']
        )
        ops['cliente'] = ops['campana'].map(cliente_map)
    else:
        ops['cliente'] = ops['campana'].str.extract(r'^(\w+)')[0]

    return ops


# ── 1. ETL ────────────────────────────────────────────────────────────────────

def transformar_base_campanas(df):
    """
    Aplica transformaciones base al DataFrame raw de campañas.
    Castea IDs, parsea timestamps a UTC, extrae año y calcula tiempos totales.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame raw leído del Excel de campañas. Debe contener:
        box_id, init_scan, last_scan, campana.

    Retorna
    -------
    pd.DataFrame con columnas adicionales: año, tiempo_total_seg, tiempo_total_min.
    """
    df = df.copy()
    df['box_id']    = df['box_id'].astype(str)
    df['init_scan'] = pd.to_datetime(df['init_scan'], utc=True)
    df['last_scan'] = pd.to_datetime(df['last_scan'], utc=True)
    df['año'] = df['campana'].apply(
        lambda x: 2026 if '26' in x else (2025 if '25' in x else None)
    )
    df['tiempo_total_seg'] = (df['last_scan'] - df['init_scan']).dt.total_seconds()
    df['tiempo_total_min'] = df['tiempo_total_seg'] / 60
    return df


def limpiar_iqr_campanas(df_picking, col='tiempo_total_seg', group_col='tipo_estacion', k=1.5):
    """
    Elimina outliers por IQR calculado dentro de cada grupo (limpieza contextual).

    Parámetros
    ----------
    df_picking : pd.DataFrame
        Registros con picks > 0.
    col : str
        Columna sobre la que se aplica el IQR. Default: 'tiempo_total_seg'.
    group_col : str
        Columna de agrupación para calcular IQR por contexto. Default: 'tipo_estacion'.
    k : float
        Factor multiplicador del IQR. Default: 1.5 (estándar Tukey).

    Retorna
    -------
    pd.DataFrame con registros dentro de los límites IQR de su grupo.
    """
    from utils.ETL_EDA_functions import iqr_bounds
    mask = df_picking.groupby(group_col)[col].transform(
        lambda s: s.between(*iqr_bounds(s, k))
    )
    return df_picking[mask].copy()


def agregar_metricas_campanas(df_clean):
    """
    Deriva métricas de eficiencia sobre el DataFrame limpio de picking.
    Requiere que df_clean ya haya pasado por limpiar_iqr_campanas.

    Columnas añadidas
    -----------------
    tiempo_por_pick : float — segundos por pick (eficiencia por unidad)
    picks_por_min   : float — throughput operacional
    cliente         : str   — prefijo de la campaña (regex ^\w+)

    Parámetros
    ----------
    df_clean : pd.DataFrame
        DataFrame con picks > 0 y columnas tiempo_total_seg, tiempo_total_min, picks.

    Retorna
    -------
    pd.DataFrame con las tres columnas adicionales.
    """
    df = df_clean.copy()
    df['tiempo_por_pick'] = df['tiempo_total_seg'] / df['picks']
    df['picks_por_min']   = df['picks'] / df['tiempo_total_min']
    df['cliente']         = df['campana'].str.extract(r'^(\w+)')
    return df


def agregar_campanas(df):
    """
    Colapsa operaciones a nivel campaña y deriva dias_habiles.
    Punto de entrada para el pipeline ML — convierte df_clean en df_campanas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame limpio a nivel operación (df_clean). Debe contener:
        campana, init_scan, last_scan, picks, cliente, año, tipo_estacion,
        picks_por_min, tiempo_total_seg.

    Retorna
    -------
    pd.DataFrame con una fila por campaña y features derivados:
    picks_total, picks_por_op_mediana, picks_max_op, ops_total, cliente, año,
    mes_inicio, hora_inicio_mediana, pct_estacion_BG, pct_estacion_calidad,
    pct_estacion_pc, picks_por_min_mediana, picks_por_min_std,
    tiempo_total_horas, dias_habiles.
    """
    resumen = (
        df.groupby('campana')
        .agg(
            inicio                = ('init_scan',       'min'),
            fin                   = ('last_scan',        'max'),
            picks_total           = ('picks',            'sum'),
            picks_por_op_mediana  = ('picks',            'median'),
            picks_max_op          = ('picks',            'max'),
            ops_total             = ('picks',            'count'),
            cliente               = ('cliente',          'first'),
            año                   = ('año',              'first'),
            mes_inicio            = ('init_scan',        lambda x: x.min().month),
            hora_inicio_mediana   = ('init_scan',        lambda x: x.dt.hour.median()),
            pct_estacion_BG       = ('tipo_estacion',    lambda x: (x == 'BG').mean()),
            pct_estacion_calidad  = ('tipo_estacion',    lambda x: (x == 'Calidad').mean()),
            pct_estacion_pc       = ('tipo_estacion',    lambda x: (x == 'Pre-Calidad').mean()),
            picks_por_min_mediana = ('picks_por_min',    'median'),
            picks_por_min_std     = ('picks_por_min',    'std'),
            tiempo_total_horas    = ('tiempo_total_seg', lambda x: x.sum() / 3600),
        )
        .reset_index()
    )

    # Días hábiles (lun–vie) entre inicio y fin de campaña
    inicio_dates = resumen['inicio'].dt.tz_localize(None).values.astype('datetime64[D]')
    fin_dates    = resumen['fin'].dt.tz_localize(None).values.astype('datetime64[D]')
    resumen['dias_habiles'] = np.busday_count(inicio_dates, fin_dates)
    resumen['dias_habiles'] = resumen['dias_habiles'].clip(lower=1)

    return resumen


# ── 2. Análisis — z-score contextual ─────────────────────────────────────────

def agregar_camp_est(df, grupo_cols=None, col_metrica='tiempo_por_pick'):
    """
    Agrega df_clean a nivel campaña × tipo_estacion calculando la mediana
    de la métrica especificada.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame limpio a nivel operación con picks > 0.
    grupo_cols : list of str, optional
        Columnas de agrupación. Default: ['campana', 'tipo_estacion', 'cliente', 'año'].
    col_metrica : str
        Columna numérica a agregar. Default: 'tiempo_por_pick'.

    Retorna
    -------
    pd.DataFrame con columnas de grupo + mediana_tpp (mediana de col_metrica).
    """
    if grupo_cols is None:
        grupo_cols = ['campana', 'tipo_estacion', 'cliente', 'año']
    return (
        df.groupby(grupo_cols, observed=True, dropna=False)[col_metrica]
        .median()
        .reset_index()
        .rename(columns={col_metrica: 'mediana_tpp'})
    )


def categorizar_zscore(z):
    """
    Asigna categoría de eficiencia a partir del z-score contextual.
    z-score positivo = más lento = peor categoría.

    Parámetros
    ----------
    z : float | NaN

    Retorna
    -------
    str — una de: 'excelente', 'bueno', 'mejorable', 'critico', 'sin_referencia'

    Escala
    ------
    z > +1          → critico    (significativamente más lento que el grupo)
    0  ≤ z ≤ +1     → mejorable  (por encima de la mediana del grupo)
    -1 ≤ z <  0     → bueno      (por debajo de la mediana del grupo)
    z < -1          → excelente  (significativamente más rápido que el grupo)
    NaN             → sin_referencia (grupo con n < n_min)
    """
    if pd.isna(z):  return 'sin_referencia'
    if z >  1:      return 'critico'
    if z >= 0:      return 'mejorable'
    if z >= -1:     return 'bueno'
    return 'excelente'


def calcular_zscore_contextual(df, grupo_contexto=None, col='mediana_tpp', n_min=2):
    """
    Añade z_score y categoria al DataFrame resultado de agregar_camp_est(),
    comparando cada campaña contra la mediana de su grupo contextual.

    Parámetros
    ----------
    df : pd.DataFrame
        Resultado de agregar_camp_est(). Debe contener la columna `col`.
    grupo_contexto : list of str, optional
        Columnas que definen el contexto de comparación.
        Default: ['cliente', 'tipo_estacion', 'año'].
    col : str
        Columna sobre la que se calcula el z-score. Default: 'mediana_tpp'.
    n_min : int
        Mínimo de campañas en el grupo para calcular z.
        Grupos con menos filas reciben z = NaN → categoria = 'sin_referencia'.
        Default: 2 (mínimo estadístico para que exista std con corrección Bessel).

    Retorna
    -------
    pd.DataFrame con columnas adicionales:
        z_score  : float
        categoria: pd.Categorical (ordenada según ORDEN_CAT)
    """
    if grupo_contexto is None:
        grupo_contexto = ['cliente', 'tipo_estacion']

    df = df.copy()
    g = df.groupby(grupo_contexto, observed=True)[col]
    df['_n']   = g.transform('count')
    df['_mu']  = g.transform('median')
    df['_std'] = g.transform('std')

    df['z_score'] = np.where(
        df['_n'] >= n_min,
        (df[col] - df['_mu']) / df['_std'],
        float('nan')
    )
    df = df.drop(columns=['_n', '_mu', '_std'])
    df['categoria'] = pd.Categorical(
        df['z_score'].apply(categorizar_zscore),
        categories=ORDEN_CAT, ordered=True
    )
    return df


# ── 3. Resumen narrativo ──────────────────────────────────────────────────────

def resumen_eficiencia_picking(
    df,
    col_tiempo_por_pick='tiempo_por_pick',
    col_tiempo_total='tiempo_total_seg',
    col_picks='picks',
    col_picks_por_min='picks_por_min',
    contexto=None,
):
    """
    Imprime un resumen narrativo de eficiencia operacional.
    Usa mediana como tendencia central por robustez ante distribuciones asimétricas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame limpio con métricas de eficiencia ya calculadas
        (resultado de agregar_metricas_campanas).
    col_tiempo_por_pick : str
        Columna con segundos por pick. Default: 'tiempo_por_pick'.
    col_tiempo_total : str
        Columna con tiempo total por caja en segundos. Default: 'tiempo_total_seg'.
    col_picks : str
        Columna con cantidad de picks por caja. Default: 'picks'.
    col_picks_por_min : str
        Columna con throughput (picks/minuto). Default: 'picks_por_min'.
    contexto : str, optional
        Etiqueta libre para identificar el corte analizado
        (ej: 'FDA FEB26', 'Estación - 2026'). Se muestra en el encabezado.

    Notas
    -----
    El CV (coeficiente de variación) normaliza la dispersión respecto a la media,
    permitiendo comparar variabilidad entre grupos con distinta escala.
    Compatible con cualquier subconjunto del df (cliente, estación, campaña, etc.).
    """
    cols_req = [col_tiempo_por_pick, col_tiempo_total, col_picks, col_picks_por_min]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas no encontradas en el DataFrame: {faltantes}")
    if df.empty:
        print("El DataFrame está vacío — no hay estadísticas que mostrar.")
        return

    def stats(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        media  = s.mean()
        std    = s.std()
        cv     = (std / media * 100) if media != 0 else float('nan')
        return {
            'media': media, 'mediana': s.median(), 'std': std, 'cv': cv,
            'q1': q1, 'q3': q3, 'iqr': q3 - q1,
            'p90': s.quantile(0.90), 'p95': s.quantile(0.95), 'n': s.count(),
        }

    tpp = stats(df[col_tiempo_por_pick])
    tt  = stats(df[col_tiempo_total])
    pk  = stats(df[col_picks])
    ppm = stats(df[col_picks_por_min])

    sesgo_label = (
        "lo que sugiere que hay operaciones que se toman bastante más tiempo del habitual "
        "y jalan el promedio hacia arriba"
        if tpp['media'] > tpp['mediana'] * 1.15
        else "lo que indica que los tiempos son bastante simétricos y el promedio es representativo"
    )
    cv_label = "alta" if tpp['cv'] > 70 else "moderada" if tpp['cv'] > 40 else "baja"
    cv_interpretacion = (
        "Los tiempos son bastante dispares entre operaciones — hay variedad real "
        "en cómo se procesan las cajas, lo que puede reflejar diferencias en operadores, "
        "tipo de producto o complejidad de la caja."
        if tpp['cv'] > 70 else
        "Existe variabilidad notable pero controlada. La operación no es completamente "
        "uniforme, aunque la mayoría se mantiene dentro de un rango razonable."
        if tpp['cv'] > 40 else
        "Los tiempos son bastante consistentes. Las operaciones tienden a comportarse "
        "de forma homogénea, lo que es una buena señal de estandarización."
    )

    encabezado = f"  EFICIENCIA OPERACIONAL{f'  —  {contexto}' if contexto else ''}"
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{encabezado}
  n = {tpp['n']:,} operaciones
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TENDENCIA CENTRAL
  Una caja típica lleva {pk['mediana']:.0f} picks, se resuelve en
  {tt['mediana']:.0f} seg ({tt['mediana']/60:.1f} min) y tarda {tpp['mediana']:.0f} seg por pick,
  lo que equivale a un ritmo de {ppm['mediana']:.1f} picks/min.

  La media de tiempo por pick es {tpp['media']:.0f} seg versus una mediana de
  {tpp['mediana']:.0f} seg — {sesgo_label}.

DISTRIBUCIÓN CENTRAL  (el 50% de las operaciones)
  La mitad de las cajas procesa cada pick en entre {tpp['q1']:.0f} y {tpp['q3']:.0f} seg
  (IQR = {tpp['iqr']:.0f} seg). En tiempo total, ese mismo grupo opera
  entre {tt['q1']:.0f} y {tt['q3']:.0f} seg por caja.
  El throughput del grueso oscila entre {ppm['q1']:.1f} y {ppm['q3']:.1f} picks/min.

  El 90% de las operaciones resuelve cada pick en menos de {tpp['p90']:.0f} seg,
  y el 95% en menos de {tpp['p95']:.0f} seg — útil como referencia de techo operacional.

DISPERSIÓN  (variabilidad)
  Desviación estándar: {tpp['std']:.0f} seg/pick  |  CV: {tpp['cv']:.0f}%  →  variabilidad {cv_label}.
  {cv_interpretacion}

  En tiempo total por caja la dispersión es de ±{tt['std']:.0f} seg (CV = {tt['cv']:.0f}%).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
