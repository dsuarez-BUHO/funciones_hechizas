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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error
import category_encoders as ce
import lightgbm as lgb


# ── Constantes de categorización ─────────────────────────────────────────────

ORDEN_CAT = ['critico', 'mejorable', 'normal', 'bueno', 'excelente', 'sin_referencia']

COLORES_CAT = {
    'critico':        '#EA4335',
    'mejorable':      '#FBBC04',
    'normal':         '#9AA0A6',
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
        pos % 6 == 5 → Morado
        pos % 6 == 0 → Calidad
    Casos directos: "Calidad", "Pre-Calidad" y "bg" (→ Morado) llegan como string,
    no como "Estacion N". El input crudo sigue siendo "bg"; solo cambia la
    etiqueta semántica de salida ("BG" → "Morado").

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
    if est_lower in ('bg', 'morado'):
        return 'Morado'
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
            if pos == 5: return 'Morado'
            if pos == 0: return 'Calidad'
        except ValueError:
            pass

    return 'Otro'


# Estaciones extra (E1-E6) — confirmado con Talía (2026-06-19): desde esta fecha se
# tratan de forma homogénea como línea de tipo 'Naranja'. Antes del corte tenían otra
# naturaleza/varianza (quedan como 'Otro', excluidas — ver categorizar_tipo_estacion).
# Convención de color en dashboard: la estación 'Naranja' usa color naranja; Morado lleva su color homónimo.
# Detalle: vault contexto de proyectos/t_c/pipe_incluir_estaciones_extra.md.
# tz-naive a propósito: dentro de transformer_api_eventos, 'init_scan' todavía no tiene
# zona horaria (eso se asigna después, en transformar_base_campanas con utc=True).
FECHA_HOMOGENIZAR_EXTRAS = pd.Timestamp('2026-05-01')
ESTACIONES_EXTRAS = {
    'estacion e1', 'estacion e2', 'estacion e3',
    'estacion e4', 'estacion e5', 'estacion e6',
}


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

    # Estaciones extra (E1-E6) — homogéneas a 'Naranja' desde FECHA_HOMOGENIZAR_EXTRAS,
    # confirmado con Talía (2026-06-19). Antes del corte tenían otra naturaleza/varianza
    # y se quedan como 'Otro'. Detalle: vault pipe_incluir_estaciones_extra.md.
    mask_extra      = ops['estacion'].str.lower().str.strip().isin(ESTACIONES_EXTRAS)
    mask_extra_homog = mask_extra & (ops['init_scan'] >= FECHA_HOMOGENIZAR_EXTRAS)
    ops.loc[mask_extra_homog, 'tipo_estacion'] = 'Naranja'

    # AVISO — visibilidad de qué queda fuera del pipeline (terminal o notebook).
    # 'Otro' se filtra aguas abajo; sin este aviso una estación nueva o mal mapeada
    # se pierde en silencio.
    otro_mask = ops['tipo_estacion'] == 'Otro'
    if otro_mask.any():
        print(f"  AVISO transformer_api_eventos: {otro_mask.sum():,} operaciones con "
              f"tipo_estacion='Otro' (excluidas aguas abajo). Desglose por estación cruda:")
        print(ops.loc[otro_mask, 'estacion'].value_counts().to_string())

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
    df['año'] = df['init_scan'].dt.year
    df['tiempo_total_seg'] = (df['last_scan'] - df['init_scan']).dt.total_seconds()
    df['tiempo_total_min'] = df['tiempo_total_seg'] / 60
    return df


# v3-02 — campañas detectadas por regex sobre patrón canónico, revisadas y confirmadas
# por Talía 2026-06-19 (33 candidatas → 30 emergentes reales, 3 falsos positivos).
# 2026-06-25 — añadidas "FAB Mounjaro JUL26" y "FAB Wegovy Inglés" (total: 31).
EMERGENTES_DETECTADAS = {
    "Abril-Mayo Confeti": "DQ",
    "BUHO2025": "RAC",
    "Campaña Fantasma L3": "DQ",
    "DERMA FDA ENE24": "FDA",
    "DQ DÍA DEL CONO26": "DQ",
    "DQ Día del Cono25": "DQ",
    "FAB BR26": "FAB",
    "FAB Capacitación Calidad": "FAB",
    "FAB Mounjaro26": "FAB",
    "FAB Mounjaro JUL26": "FAB",
    "FAB Wegovy Inglés": "FAB",
    "FDA ABR26 Reenvío": "FDA",
    "FDA AGO25 DERMA": "FDA",
    "FDA Capacitación Calidad": "FDA",
    "FDA Estrategias Agresivas DIC24": "FDA",
    "FDA Estrategias Agresivas ENE25": "FDA",
    "FDA Estrategias Agresivas NOV24": "FDA",
    "FDA INVERNAL BUEN FIN": "FDA",
    "FDA MAR26 AG-INT": "FDA",
    "FDA MDP Naviderma": "FDA",
    "FDA Nuevos Mercados DIC24": "FDA",
    "FDA Nuevos Mercados ENE25": "FDA",
    "FDA Nuevos Mercados NOV24": "FDA",
    "FDA PROYECTO SUEÑO": "FDA",
    "FDA VINILES C1": "FDA",
    "FEBRERO MARZO RED VELVET NUEVO": "DQ",
    "POP MAR24 Emergente": "FDA",
    "Preciadores FDA ENE26": "FDA",
    "Preciadores FDA MAY26": "FDA",
    "Reenvío FDA MAY26": "FDA",
    "YZA ": "YZA",
}

# Confirmadas por Talía como NO emergentes (falsos positivos del regex) — no se usan
# para filtrar (ya están excluidas de EMERGENTES_DETECTADAS); quedan documentadas
# aquí para no repetir la revisión si el regex se vuelve a correr.
EXCEPCIONES_NO_RETIRAR = {
    "POP Marzo 2024",
    "POP NACIONAL MARZO 2024",
    "TIM HORTONS JUN26",
}


def retirar_campanas_emergentes(df_base):
    """
    Retira del DataFrame las operaciones de campañas emergentes (no representativas
    del flujo regular — capacitaciones, reenvíos, campañas fantasma, etc.), según
    la lista canónica EMERGENTES_DETECTADAS.

    Parámetros
    ----------
    df_base : pd.DataFrame
        Requiere columna campana.

    Retorna
    -------
    pd.DataFrame filtrado (copia), sin las campañas de EMERGENTES_DETECTADAS.
    """
    mask_retirar = df_base['campana'].isin(EMERGENTES_DETECTADAS.keys())
    return df_base[~mask_retirar].copy()


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
    cliente         : str   — prefijo del nombre de campaña (regex ^\\w+).
                              Solo se añade si la columna no existe; cuando el
                              DataFrame viene de transformer_api_eventos, cliente
                              ya llega desde MySQL y no se sobreescribe.

    Parámetros
    ----------
    df_clean : pd.DataFrame
        DataFrame con picks > 0 y columnas tiempo_total_seg, tiempo_total_min, picks.

    Retorna
    -------
    pd.DataFrame con las columnas adicionales.
    """
    df = df_clean.copy()
    df['tiempo_por_pick'] = df['tiempo_total_seg'] / df['picks']
    df['picks_por_min']   = df['picks'] / df['tiempo_total_min']
    if 'cliente' not in df.columns:
        df['cliente'] = df['campana'].str.extract(r'^(\w+)')
    return df


def agregar_tiempo_muerto(df_clean, pct_bajo=0.05, pct_alto=0.10):
    """
    Calcula el tiempo muerto entre operaciones consecutivas de la misma caja
    (box_id × campana): init_scan[n] - last_scan[n-1], ordenado por init_scan.

    Aplica a todos los tipo_estacion sin distinción (líneas, Naranja, Calidad/PC) —
    decisión explícita v3-06, no se filtra por TIPOS_LINEA.

    v3-06h (2026-06-22): dos capas de filtro — se retiró el tope absoluto fijo
    (max_gap_seg=3600) que tenía v3-06g. Bitácora completa en
    recursos_didacticos_t_c/tiempo_muerto_decisiones.html.

    1. Retroceso de flujo: si tipo_estacion_origen tiene un orden mayor a
       tipo_estacion destino en TIPO_ESTACION_ORDEN (ej. Amarillo→Azul), se
       excluye SIEMPRE, sin importar la duración — no es tránsito entre
       estaciones adyacentes, es la caja regresando a una estación anterior
       (recirculación de vuelta, o reutilización de box_id — causa raíz sin
       confirmar con piso/Talía). Esta capa es la que en la práctica reemplaza
       al tope absoluto: al sacar la contaminación estructural ANTES de
       calcular el percentil, el percentil ya no necesita un respaldo en
       segundos fijos para no inflarse (ver hallazgo Azul en v3-06e/f).
    2. Percentil por tipo_estacion destino, calculado SOLO sobre lo que pasó
       (1) — pct_bajo recorta la cola inferior, pct_alto la superior.
       Asimétrico por diseño: el tope superior es más permisivo (10% vs 5%)
       porque estaciones de cola/lote (Calidad/Pre-Calidad) tienen colas
       superiores legítimamente más largas — un tope absoluto en segundos
       (la versión anterior) les excluía >50% de sus transiciones reales.

    Columnas añadidas a df_clean
    ----------------------------
    tipo_estacion_origen : str — tipo_estacion de la operación anterior de la misma
                           caja. Se popula siempre que exista operación previa,
                           independientemente de si el gap es válido — es la pareja
                           origen→destino real, no depende de la confiabilidad del
                           tiempo. NaN solo para la primera operación de la caja.
    tiempo_muerto_seg    : float — gap respecto a la operación anterior. NaN para la
                           primera operación de la caja, o cuando el gap es inválido
                           (negativo, retroceso de flujo, o fuera del percentil de
                           su tipo_estacion destino). La transición en sí no se
                           pierde (ver tipo_estacion_origen); solo su duración.

    Con estas dos columnas se puede agregar tanto a nivel campaña (mediana de
    tiempo_muerto_seg por campana) como a nivel par de estaciones — origen→destino —
    vía groupby(['tipo_estacion_origen', 'tipo_estacion']), incluyendo el conteo total
    de transiciones (válidas + excluidas) por par.

    Parámetros
    ----------
    df_clean : pd.DataFrame
        Requiere columnas: box_id, campana, tipo_estacion, init_scan, last_scan.
    pct_bajo : float
        Fracción a recortar de la cola inferior, por tipo_estacion destino.
        Default 0.05.
    pct_alto : float
        Fracción a recortar de la cola superior, por tipo_estacion destino.
        Default 0.10 — más permisivo que pct_bajo, ver docstring.

    Retorna
    -------
    (df, df_outliers) — tupla:
      df          : copia de df_clean, ordenada por (campana, box_id, init_scan), con
                    las 2 columnas nuevas.
      df_outliers : 1 fila por transición con gap inválido — campana, box_id,
                    tipo_estacion_origen, tipo_estacion, gap_seg (valor real,
                    sin recortar), motivo ('negativo', 'retroceso_flujo',
                    'percentil_alto' o 'percentil_bajo'). Para trazabilidad: qué
                    se descartó del cálculo de tiempo muerto y por qué, sin
                    perderlo del todo. Pensado para exportarse a
                    assets/tablas/outliers_tiempos_muertos_salto_estacion.csv.
    """
    df = df_clean.sort_values(['campana', 'box_id', 'init_scan']).reset_index(drop=True)

    KEY = ['campana', 'box_id']
    prev_last_scan     = df.groupby(KEY)['last_scan'].shift(1)
    prev_tipo_estacion = df.groupby(KEY)['tipo_estacion'].shift(1)

    gap_seg    = (df['init_scan'] - prev_last_scan).dt.total_seconds()
    tiene_prev = prev_last_scan.notna()

    orden_origen  = prev_tipo_estacion.map(TIPO_ESTACION_ORDEN)
    orden_destino = df['tipo_estacion'].map(TIPO_ESTACION_ORDEN)
    es_retroceso  = tiene_prev & (orden_origen > orden_destino)

    base_valida = (gap_seg >= 0) & ~es_retroceso

    # Percentiles por tipo_estacion destino — calculados SOLO sobre lo que ya
    # pasó retroceso de flujo, sin tope absoluto en segundos (ver docstring).
    gap_base = gap_seg.where(base_valida)
    p_bajo = gap_base.groupby(df['tipo_estacion']).transform(
        lambda s: s.quantile(pct_bajo)
    )
    p_alto = gap_base.groupby(df['tipo_estacion']).transform(
        lambda s: s.quantile(1 - pct_alto)
    )
    gap_valido = base_valida & gap_seg.between(p_bajo, p_alto)

    df['tipo_estacion_origen'] = prev_tipo_estacion
    df['tiempo_muerto_seg']    = gap_seg.where(gap_valido)

    mask_outlier = tiene_prev & ~gap_valido
    df_outliers = df.loc[mask_outlier, ['campana', 'box_id', 'tipo_estacion']].copy()
    df_outliers['tipo_estacion_origen'] = prev_tipo_estacion[mask_outlier]
    df_outliers['gap_seg'] = gap_seg[mask_outlier]
    df_outliers['motivo']  = np.select(
        [
            gap_seg[mask_outlier].values < 0,
            es_retroceso[mask_outlier].values,
            gap_seg[mask_outlier].values > p_alto[mask_outlier].values,
        ],
        ['negativo', 'retroceso_flujo', 'percentil_alto'],
        default='percentil_bajo',
    )
    df_outliers = df_outliers[
        ['campana', 'box_id', 'tipo_estacion_origen', 'tipo_estacion', 'gap_seg', 'motivo']
    ].reset_index(drop=True)

    return df, df_outliers


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
    mes_inicio, hora_inicio_mediana, pct_estacion_morado, pct_estacion_calidad,
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
            pct_estacion_morado   = ('tipo_estacion',    lambda x: (x == 'Morado').mean()),
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


# v3-06b (2026-06-22) — agregación de tiempo muerto para dashboard. Grano
# campaña × par de estaciones: lo más fino que tiene sentido visualizar
# (por operación sería 1 fila por caja, demasiado para Looker) y desde ahí
# se puede subir a cliente, mes o global agrupando en Looker sin perder la
# vista por campaña puntual.
def agregar_tiempo_muerto_transiciones(df_clean, df_outliers_tiempo_muerto):
    """
    Agrega tiempo_muerto_seg (de agregar_tiempo_muerto) a nivel
    campaña × tipo_estacion_origen × tipo_estacion (destino).

    Parámetros
    ----------
    df_clean : pd.DataFrame
        Ya debe tener tipo_estacion_origen y tiempo_muerto_seg
        (salida de agregar_tiempo_muerto).
    df_outliers_tiempo_muerto : pd.DataFrame
        Segundo retorno de agregar_tiempo_muerto — transiciones con gap
        inválido, para sumarlas a n_transiciones_total por par.

    Retorna
    -------
    pd.DataFrame: campana, cliente, año, fecha_inicio, tipo_estacion_origen,
    tipo_estacion, mediana_tiempo_muerto_seg, promedio_tiempo_muerto_seg,
    n_transiciones_validas, n_transiciones_total. fecha_inicio = min(init_scan)
    de la campaña — sirve para ordenar cronológicamente (tendencia mensual,
    campaña anterior).
    """
    grupo = ['campana', 'cliente', 'año', 'tipo_estacion_origen', 'tipo_estacion']

    validas = (
        df_clean
        .dropna(subset=['tipo_estacion_origen'])
        .groupby(grupo, observed=True)
        .agg(
            mediana_tiempo_muerto_seg=('tiempo_muerto_seg', 'median'),
            promedio_tiempo_muerto_seg=('tiempo_muerto_seg', 'mean'),
            n_transiciones_validas=('tiempo_muerto_seg', 'count'),
        )
        .reset_index()
    )

    outliers_n = (
        df_outliers_tiempo_muerto
        .groupby(['campana', 'tipo_estacion_origen', 'tipo_estacion'])
        .size()
        .rename('n_outliers')
        .reset_index()
    )

    resultado = validas.merge(
        outliers_n, on=['campana', 'tipo_estacion_origen', 'tipo_estacion'], how='left'
    )
    resultado['n_outliers'] = resultado['n_outliers'].fillna(0).astype(int)
    resultado['n_transiciones_total'] = resultado['n_transiciones_validas'] + resultado['n_outliers']
    resultado = resultado.drop(columns='n_outliers')

    fecha_inicio = df_clean.groupby('campana')['init_scan'].min().rename('fecha_inicio')
    resultado = resultado.merge(fecha_inicio, on='campana', how='left')
    return resultado


def agregar_delta_tiempo_muerto_campana_anterior(df_tiempo_muerto, grupo_cols=None):
    """
    Para cada grupo (por defecto cliente × tipo_estacion_origen × tipo_estacion),
    ordena las campañas cronológicamente (fecha_inicio) y compara cada una contra
    la campaña inmediatamente anterior del mismo cliente en ese mismo grupo —
    "¿mejoró o empeoró el tiempo muerto vs la última vez que este cliente lo tuvo?".

    Primera campaña de cada grupo (sin anterior) queda con NaN — no es
    comparable, no se asume 0.

    Parámetros
    ----------
    df_tiempo_muerto : pd.DataFrame
        Requiere fecha_inicio, mediana_tiempo_muerto_seg y las columnas de
        grupo_cols.
    grupo_cols : list of str, optional
        Columnas que definen "la misma transición a través del tiempo".
        Default: ['cliente', 'tipo_estacion_origen', 'tipo_estacion'] (por par).
        Usar ['cliente', 'tipo_estacion'] para comparar por estación destino
        sola (ver agregar_tiempo_muerto_por_destino).

    Retorna
    -------
    Copia de df_tiempo_muerto + columnas:
      campana_anterior, mediana_tiempo_muerto_seg_anterior,
      delta_tiempo_muerto_seg, delta_tiempo_muerto_pct,
      promedio_tiempo_muerto_seg_anterior, n_transiciones_validas_anterior,
      y la versión _min (÷60) de las columnas en segundos — para no repetir
      el mismo campo calculado en cada widget de Looker.

      n_transiciones_validas_anterior existe específicamente para ponderar
      promedio_tiempo_muerto_seg_anterior al agregar en Looker — pesarlo con
      n_transiciones_validas (de la campaña ACTUAL) sería incorrecto, porque
      ese conteo no respalda el promedio de la campaña anterior.
    """
    if grupo_cols is None:
        grupo_cols = ['cliente', 'tipo_estacion_origen', 'tipo_estacion']
    df = df_tiempo_muerto.sort_values(grupo_cols + ['fecha_inicio']).reset_index(drop=True)

    g = df.groupby(grupo_cols)
    df['campana_anterior'] = g['campana'].shift(1)
    df['mediana_tiempo_muerto_seg_anterior'] = g['mediana_tiempo_muerto_seg'].shift(1)
    df['delta_tiempo_muerto_seg'] = (
        df['mediana_tiempo_muerto_seg'] - df['mediana_tiempo_muerto_seg_anterior']
    )
    df['delta_tiempo_muerto_pct'] = (
        df['delta_tiempo_muerto_seg'] / df['mediana_tiempo_muerto_seg_anterior'] * 100
    )
    df['promedio_tiempo_muerto_seg_anterior'] = g['promedio_tiempo_muerto_seg'].shift(1)
    df['n_transiciones_validas_anterior'] = g['n_transiciones_validas'].shift(1)

    for col_seg in [
        'mediana_tiempo_muerto_seg', 'mediana_tiempo_muerto_seg_anterior',
        'promedio_tiempo_muerto_seg', 'promedio_tiempo_muerto_seg_anterior',
        'delta_tiempo_muerto_seg',
    ]:
        col_min = col_seg.replace('_seg', '_min')
        df[col_min] = df[col_seg] / 60

    return df


def agregar_tiempo_muerto_por_destino(df_tiempo_muerto):
    """
    Colapsa campanas_tiempo_muerto_transiciones (grano campaña × par de
    estaciones) a campaña × tipo_estacion destino, sin importar el origen —
    "¿cuánto tiempo muerto hay en general antes de llegar a esta estación?".

    mediana_tiempo_muerto_seg se recalcula como promedio ponderado por
    n_transiciones_validas (mismo criterio que mediana_ponderada en el resto
    del proyecto — no es la mediana exacta recuperable sin los datos crudos,
    es la mejor aproximación disponible a partir de medianas ya agregadas).

    promedio_tiempo_muerto_seg, en cambio, sí se recupera exacto: el promedio
    ponderado de promedios por sus propios conteos es algebraicamente el
    promedio poblacional real (no una aproximación), porque la media es
    lineal en las observaciones — la mediana no.

    Parámetros
    ----------
    df_tiempo_muerto : pd.DataFrame
        Salida de agregar_tiempo_muerto_transiciones.

    Retorna
    -------
    pd.DataFrame: campana, cliente, año, fecha_inicio, tipo_estacion,
    mediana_tiempo_muerto_seg, promedio_tiempo_muerto_seg,
    n_transiciones_validas, n_transiciones_total.
    """
    df = df_tiempo_muerto.copy()
    df['_peso_mediana']   = df['mediana_tiempo_muerto_seg']  * df['n_transiciones_validas']
    df['_peso_promedio']  = df['promedio_tiempo_muerto_seg'] * df['n_transiciones_validas']

    grupo = ['campana', 'cliente', 'año', 'fecha_inicio', 'tipo_estacion']
    agg = (
        df.groupby(grupo, observed=True)
        .agg(
            _peso_mediana_sum=('_peso_mediana', 'sum'),
            _peso_promedio_sum=('_peso_promedio', 'sum'),
            n_transiciones_validas=('n_transiciones_validas', 'sum'),
            n_transiciones_total=('n_transiciones_total', 'sum'),
        )
        .reset_index()
    )
    agg['mediana_tiempo_muerto_seg'] = agg['_peso_mediana_sum'] / agg['n_transiciones_validas'].clip(lower=1)
    agg['promedio_tiempo_muerto_seg'] = agg['_peso_promedio_sum'] / agg['n_transiciones_validas'].clip(lower=1)
    return agg.drop(columns=['_peso_mediana_sum', '_peso_promedio_sum'])


# v3-04 (2026-06-19) — 5 categorías por percentiles 10/20/40/20/10 sobre normal
# estándar. Cortes validados exploratoriamente en exploraciones_rangos_eficiencia.ipynb
# antes de activarse aquí. 'normal' es categoría nueva, no reemplaza ninguna existente.
Z_EXCELENTE = -1.2816   # percentil 10
Z_BUENO     = -0.5244   # percentil 30
Z_MEJORABLE =  0.5244   # percentil 70
Z_CRITICO   =  1.2816   # percentil 90


def categorizar_zscore(z):
    """
    Asigna categoría de eficiencia a partir del z-score contextual.
    z-score positivo = más lento = peor categoría.

    Parámetros
    ----------
    z : float | NaN

    Retorna
    -------
    str — una de: 'excelente', 'bueno', 'normal', 'mejorable', 'critico', 'sin_referencia'

    Escala (percentiles 10/20/40/20/10 sobre normal estándar — v3-04)
    -------------------------------------------------------------------
    z > +1.2816                  → critico    (10% más lento del grupo)
    +0.5244 < z ≤ +1.2816        → mejorable  (siguiente 20%)
    -0.5244 ≤ z ≤ +0.5244        → normal     (40% central)
    -1.2816 ≤ z < -0.5244        → bueno      (siguiente 20%)
    z < -1.2816                  → excelente  (10% más rápido del grupo)
    NaN                          → sin_referencia (grupo con n < n_min)
    """
    if pd.isna(z):       return 'sin_referencia'
    if z >  Z_CRITICO:   return 'critico'
    if z >  Z_MEJORABLE: return 'mejorable'
    if z >= Z_BUENO:     return 'normal'
    if z >= Z_EXCELENTE: return 'bueno'
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


def mediana_ponderada(vals: np.ndarray, pesos: np.ndarray) -> float:
    """
    Mediana ponderada por numpy — reemplaza median() simple en groupby.apply.

    Parámetros
    ----------
    vals  : valores a mediar (ej. tiempo_por_pick de cada operación)
    pesos : pesos de cada valor (ej. picks)
    """
    orden     = np.argsort(vals)
    vals_ord  = vals[orden]
    pesos_ord = pesos[orden]
    acum      = np.cumsum(pesos_ord) / pesos_ord.sum()
    idx       = np.searchsorted(acum, 0.5)
    return float(vals_ord[min(idx, len(vals_ord) - 1)])


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


# ── 4. Dificultad de caja y complejidad de campaña ───────────────────────────

_COLS_DIFICULTAD = [
    'campana', 'box_id', 'n_skus_en_caja', 'n_categorias_en_caja',
    'cambios_de_row', 'n_filas_distintas',
]

_COLS_COMPLEJIDAD = [
    'campana', 'n_skus_distintos', 'n_materiales_distintos', 'n_categorias_distintas',
    'entropia_sku', 'hhi_sku', 'skus_por_caja_mediana',
]


def _cambios_de_row(group):
    """
    Cuenta cambios de fila entre picks consecutivos dentro de una caja.
    0 = todos los picks en la misma fila. Alto = zigzag entre filas (máxima fricción).
    """
    if 'row' not in group.columns:
        return float('nan')
    if len(group) <= 1:
        return 0
    if 'consecutivo' in group.columns:
        group = group.sort_values('consecutivo')
    cambios = (group['row'] != group['row'].shift()).sum() - 1
    return int(cambios)


def build_ops_dificultad(df_raw, tipos_estacion=None):
    """
    Construye features de dificultad por caja (box_id) a partir de eventos API crudos.
    Tabla paralela a ops_clean — 1 fila × box_id.

    Parámetros
    ----------
    df_raw : DataFrame de eventos crudos. Obligatorias: type, campana, box_id.
             Opcionales: sku, categoria, row, consecutivo.
    tipos_estacion : list of str, optional
        Si se provee, filtra a eventos de esas estaciones semánticas (ej. TIPOS_LINEA).
        Usa categorizar_tipo_estacion sobre la columna 'estacion'. Default: None (todos).

    Retorna
    -------
    DataFrame: campana, box_id, n_skus_en_caja, n_categorias_en_caja,
               cambios_de_row, n_filas_distintas.
    """
    # [2026-04-23] Añadido tipos_estacion para feature engineering por segmento.
    # Para revertir: eliminar este bloque y el parámetro. Eliminar también tipos_estacion
    # en build_campanas_complejidad. Contexto: Capa 1 Tarea 1 del roadmap t_c,
    # story_decision_segmentacion_metricas — modelos ML mixto vs líneas con features propios.
    if tipos_estacion is not None:
        mask = df_raw['estacion'].apply(categorizar_tipo_estacion).isin(tipos_estacion)
        df_raw = df_raw[mask].copy()

    if 'type' not in df_raw.columns or 'box_id' not in df_raw.columns:
        print("  AVISO build_ops_dificultad: faltan columnas 'type' o 'box_id'.")
        return pd.DataFrame(columns=_COLS_DIFICULTAD)

    _SCAN_TYPES = {'scan', 'pick_ok'}
    df_scans = df_raw[df_raw['type'].isin(_SCAN_TYPES & set(df_raw['type'].unique()))].copy()

    if df_scans.empty:
        print("  AVISO build_ops_dificultad: no hay eventos scan/pick_ok.")
        return pd.DataFrame(columns=_COLS_DIFICULTAD)

    for col in ('sku', 'categoria', 'row', 'consecutivo'):
        if col not in df_scans.columns:
            df_scans[col] = pd.NA

    agg = (
        df_scans
        .groupby(['campana', 'box_id'], observed=True)
        .agg(
            n_skus_en_caja       =('sku',       'nunique'),
            n_categorias_en_caja =('categoria', 'nunique'),
            n_filas_distintas    =('row',       'nunique'),
        )
        .reset_index()
    )

    cambios = (
        df_scans
        .groupby(['campana', 'box_id'], observed=True, group_keys=False)
        .apply(_cambios_de_row, include_groups=False)
        .rename('cambios_de_row')
        .reset_index()
    )
    cambios.columns = ['campana', 'box_id', 'cambios_de_row']

    result = agg.merge(cambios, on=['campana', 'box_id'], how='left')[_COLS_DIFICULTAD]
    result['box_id'] = result['box_id'].astype(str)
    return result


def _entropia_sku(series):
    """Entropía de Shannon (base 2) sobre distribución de picks por SKU."""
    counts = series.dropna().value_counts()
    n = len(counts)
    if n == 0:   return float('nan')
    if n == 1:   return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _hhi_sku(series):
    """Índice Herfindahl-Hirschman sobre distribución de picks por SKU. Alto → concentrado."""
    counts = series.dropna().value_counts(normalize=True)
    if counts.empty:
        return float('nan')
    return float((counts ** 2).sum())


def build_campanas_complejidad(df_raw, tipos_estacion=None):
    """
    Construye features de complejidad por campaña a partir de eventos API crudos.
    1 fila × campaña — qué se procesó (SKUs/materiales), no dónde ni cuánto tardó.

    Parámetros
    ----------
    df_raw : DataFrame de eventos crudos. Obligatorias: type, campana.
             Opcionales: sku, material, categoria, box_id.
    tipos_estacion : list of str, optional
        Si se provee, filtra a eventos de esas estaciones semánticas (ej. TIPOS_LINEA).
        Usa categorizar_tipo_estacion sobre la columna 'estacion'. Default: None (todos).

    Retorna
    -------
    DataFrame: campana, n_skus_distintos, n_materiales_distintos, n_categorias_distintas,
               entropia_sku, hhi_sku, skus_por_caja_mediana.
    """
    # [2026-04-23] Añadido tipos_estacion para feature engineering por segmento.
    # Para revertir: eliminar este bloque y el parámetro. Eliminar también tipos_estacion
    # en build_ops_dificultad. Contexto: Capa 1 Tarea 1 del roadmap t_c,
    # story_decision_segmentacion_metricas — modelos ML mixto vs líneas con features propios.
    if tipos_estacion is not None:
        mask = df_raw['estacion'].apply(categorizar_tipo_estacion).isin(tipos_estacion)
        df_raw = df_raw[mask].copy()

    if 'type' not in df_raw.columns or 'campana' not in df_raw.columns:
        print("  AVISO build_campanas_complejidad: faltan columnas 'type' o 'campana'.")
        return pd.DataFrame(columns=_COLS_COMPLEJIDAD)

    _SCAN_TYPES = {'scan', 'pick_ok'}
    df_scans = df_raw[df_raw['type'].isin(_SCAN_TYPES & set(df_raw['type'].unique()))].copy()

    if df_scans.empty:
        print("  AVISO build_campanas_complejidad: no hay eventos scan/pick_ok.")
        return pd.DataFrame(columns=_COLS_COMPLEJIDAD)

    for col in ('sku', 'material', 'categoria'):
        if col not in df_scans.columns:
            df_scans[col] = pd.NA

    agg = (
        df_scans
        .groupby('campana', observed=True)
        .agg(
            n_skus_distintos       =('sku',      'nunique'),
            n_materiales_distintos =('material', 'nunique'),
            n_categorias_distintas =('categoria','nunique'),
            entropia_sku           =('sku',      _entropia_sku),
            hhi_sku                =('sku',      _hhi_sku),
        )
        .reset_index()
    )

    if 'box_id' in df_scans.columns:
        skus_por_caja = (
            df_scans
            .groupby(['campana', 'box_id'], observed=True)['sku']
            .nunique()
            .reset_index(name='skus_en_caja')
            .groupby('campana')['skus_en_caja']
            .median()
            .rename('skus_por_caja_mediana')
        )
        agg = agg.merge(skus_por_caja, on='campana', how='left')
    else:
        agg['skus_por_caja_mediana'] = float('nan')

    return agg[_COLS_COMPLEJIDAD]


# ── 5. Z-score ajustado por dificultad ───────────────────────────────────────
#
# z_score_ajustado (campaña):
#   Target: mediana_seg_pick de campaña completa (todas las estaciones, incluyendo
#   Calidad y Pre-Calidad). Features de ajuste: complejidad de material y escala,
#   derivados principalmente de operaciones de línea. Limitación documentada: las
#   estaciones Calidad/Pre-Calidad afectan el target pero no están representadas en
#   los features de ajuste. Interpretación: "tiempo total de campaña ajustado por
#   complejidad de líneas". Una métrica equivalente usando solo picks de línea como
#   target queda pendiente como decisión de directivos (ver story_decision_segmentacion.md).
#
# z_score_ajustado_estn_cliente_lineas (estación):
#   Solo cubre TIPOS_LINEA. Calidad y Pre-Calidad quedan NaN — tienen drivers
#   operativos distintos que requieren análisis independiente (roadmap Capa 4a).

TIPOS_LINEA = {'Azul', 'Rojo', 'Amarillo', 'Verde', 'Morado', 'Naranja'}

# Migrado desde pipeline_campanas.py / pipeline_campanas_v2.py (2026-06-19) — estaban
# duplicados idénticos en ambos archivos; única fuente de verdad ahora.
# Azul/Rojo → 5 columnas × 3 filas (15 posiciones) | Amarillo/Verde → 9 × 3 (27 posiciones)
# Morado, Calidad, Pre-Calidad no son estaciones físicas de sorteo.
# Naranja (E1-E6): grid real sin confirmar con piso — placeholder explícito en vez de
# NaN porque configuracion_estacion es de tipo string. Ver vault
# pipe_incluir_estaciones_extra.md.
CONFIGURACION_ESTACION_MAP = {
    'Azul':        '5x3',
    'Rojo':        '5x3',
    'Amarillo':    '9x3',
    'Verde':       '9x3',
    'Morado':      '1x1',
    'Calidad':     '1x1',
    'Pre-Calidad': '1x1',
    'Naranja':     'Not defined',
}

# Mapeo tipo_estacion → columna pct en campanas_resumen
ESTACION_PCT_MAP = {
    'Azul':        'pct_azul',
    'Rojo':        'pct_rojo',
    'Verde':       'pct_verde',
    'Amarillo':    'pct_amarillo',
    'Morado':      'pct_morado',
    'Calidad':     'pct_calidad',
    'Pre-Calidad': 'pct_pc',
    'Naranja':     'pct_naranja',
}

# v3-06g (2026-06-22) — orden topológico del flujo físico de piso. Misma
# secuencia que el campo tipo_estacion_orden en Looker — única fuente de
# verdad, no duplicar el orden a mano en el dashboard. Usado por
# agregar_tiempo_muerto() para distinguir tránsito normal (origen → destino
# hacia adelante o igual) de retroceso de flujo (caja regresando a una
# estación anterior, ej. recirculación de vuelta — no es tiempo muerto de
# tránsito, es otro fenómeno).
TIPO_ESTACION_ORDEN = {
    'Naranja':     1,
    'Azul':        2,
    'Rojo':        3,
    'Verde':       4,
    'Amarillo':    5,
    'Morado':      6,
    'Pre-Calidad': 7,
    'Calidad':     8,
}

_FEATURES_MODELO_CAMP = ['entropia_sku', 'skus_por_caja_mediana', 'n_categorias_distintas']
_CATEG_COLS_CAMP      = ['cliente']
_TARGET_CAMP          = 'mediana_seg_pick'

_FEATURES_MODELO_EST  = ['n_categorias_mediana', 'n_filas_mediana']
_CATEG_COLS_EST       = ['cliente', 'tipo_estacion']
_TARGET_EST           = 'mediana_tpp'


def _make_preprocessor(num_cols: list, cat_cols: list):
    """Pipeline sklearn: imputer mediana + StandardScaler (num) + TargetEncoder (cat)."""
    return ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num_cols),
        ('cat', ce.TargetEncoder(cols=cat_cols, handle_missing='value', handle_unknown='value'), cat_cols),
    ], remainder='drop')


def _loco_evaluate(pipeline, X: pd.DataFrame, y: pd.Series):
    """Leave-One-Out CV. Retorna (preds_array, {'R²_LOO': float, 'MAE_LOO': float})."""
    preds  = cross_val_predict(pipeline, X, y, cv=LeaveOneOut())
    mae    = mean_absolute_error(y, preds)
    ss_res = np.sum((y.values - preds) ** 2)
    ss_tot = np.sum((y.values - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return preds, {'R²_LOO': round(r2, 3), 'MAE_LOO': round(mae, 3)}


def build_zscore_ajustado(
    df_complejidad: pd.DataFrame,
    campanas_resumen: pd.DataFrame,
    target_col: str = _TARGET_CAMP,
    features_extra: list = None,
) -> tuple:
    """
    Entrena modelos LOO-CV y devuelve z_score_ajustado por campaña.

    Parámetros
    ----------
    df_complejidad   : DataFrame con _FEATURES_MODELO_CAMP + 'campana'.
                       Para Fase 3 añadir features_extra antes de llamar.
    campanas_resumen : DataFrame con 'campana', 'cliente', target_col, 'z_score'.
    target_col       : columna de tiempo a predecir. Default: 'mediana_seg_pick'.
    features_extra   : features adicionales Fase 3, ej:
                       ['n_skus_en_caja_mediana_campana', 'log_ops_total'].

    Retorna
    -------
    df_resultado  : campana, cliente, target_col, residuo, z_score_ajustado, z_score, delta_z.
    tabla_modelos : R² y MAE LOO-CV. attrs['modelo_ganador'], attrs['features_usados'].
    """
    features_num = _FEATURES_MODELO_CAMP + (features_extra or [])

    df = df_complejidad.merge(
        campanas_resumen[
            ['campana', 'cliente', target_col]
            + (['z_score'] if 'z_score' in campanas_resumen.columns else [])
        ],
        on='campana', how='inner',
    )
    cols_keep = features_num + ['campana', 'cliente', target_col] + (['z_score'] if 'z_score' in df.columns else [])
    df = df[cols_keep].dropna().copy().reset_index(drop=True)

    X = df[features_num + _CATEG_COLS_CAMP]
    y = df[target_col].copy()

    modelos = {
        'Ridge':        Pipeline([('pre', _make_preprocessor(features_num, _CATEG_COLS_CAMP)),
                                   ('model', Ridge(alpha=1.0))]),
        'RandomForest': Pipeline([('pre', _make_preprocessor(features_num, _CATEG_COLS_CAMP)),
                                   ('model', RandomForestRegressor(n_estimators=100, max_depth=3,
                                                                    min_samples_leaf=5, random_state=42))]),
        'LightGBM':     Pipeline([('pre', _make_preprocessor(features_num, _CATEG_COLS_CAMP)),
                                   ('model', lgb.LGBMRegressor(num_leaves=8, min_child_samples=10,
                                                                n_estimators=100, random_state=42, verbose=-1))]),
    }

    resultados_cv = {}
    for nombre, pipe in modelos.items():
        preds, metricas = _loco_evaluate(pipe, X, y)
        resultados_cv[nombre] = {**metricas, 'y_pred_loo': preds}

    tabla_modelos = pd.DataFrame({
        k: {'R² LOO-CV': v['R²_LOO'], 'MAE LOO-CV': v['MAE_LOO']}
        for k, v in resultados_cv.items()
    }).T
    modelo_ganador = tabla_modelos['R² LOO-CV'].idxmax()

    df['y_pred_loo'] = resultados_cv[modelo_ganador]['y_pred_loo']
    df['residuo']    = df[target_col] - df['y_pred_loo']

    g = df.groupby('cliente')['residuo']
    df['_n']   = g.transform('count')
    df['_mu']  = g.transform('median')
    df['_std'] = g.transform('std')
    df['z_score_ajustado'] = np.where(df['_n'] >= 2, (df['residuo'] - df['_mu']) / df['_std'], np.nan)
    df = df.drop(columns=['_n', '_mu', '_std', 'y_pred_loo'])

    if 'z_score' in df.columns:
        df['delta_z'] = (df['z_score_ajustado'] - df['z_score']).round(3)

    tabla_modelos.attrs['modelo_ganador']  = modelo_ganador
    tabla_modelos.attrs['features_usados'] = features_num
    return df, tabla_modelos


def build_features_estacion(
    ops_dificultad: pd.DataFrame,
    estacion_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega features de dificultad de caja a nivel campaña × tipo_estacion.
    Filtrado a TIPOS_LINEA — Calidad y Pre-Calidad excluidos.

    Parámetros
    ----------
    ops_dificultad : resultado de build_ops_dificultad() — 1 fila × box_id.
    estacion_map   : DataFrame con ['box_id', 'tipo_estacion'].

    Retorna
    -------
    DataFrame: campana, tipo_estacion, n_categorias_mediana, n_filas_mediana.
    """
    df = ops_dificultad.merge(
        estacion_map[['box_id', 'tipo_estacion']].drop_duplicates(),
        on='box_id', how='left',
    )
    return (
        df[df['tipo_estacion'].isin(TIPOS_LINEA)]
        .groupby(['campana', 'tipo_estacion'], observed=True)
        .agg(
            n_categorias_mediana=('n_categorias_en_caja', 'median'),
            n_filas_mediana=     ('n_filas_distintas',    'median'),
        )
        .reset_index()
    )


def _zscore_por_grupo(
    df: pd.DataFrame,
    grupo: list,
    col_residuo: str = 'residuo_loo',
    n_min: int = 2,
) -> pd.Series:
    """Normaliza residuos por grupo (Bessel ddof=1). NaN si n < n_min."""
    g   = df.groupby(grupo, observed=True)[col_residuo]
    mu  = g.transform('median')
    std = g.transform('std')
    n   = g.transform('count')
    return np.where(n >= n_min, (df[col_residuo] - mu) / std, np.nan)


def build_zscore_ajustado_estacion(
    df_features_estacion: pd.DataFrame,
    campanas_estacion: pd.DataFrame,
    target_col: str = _TARGET_EST,
) -> tuple:
    """
    Entrena modelos LOO-CV y devuelve z_score ajustado a nivel campaña × tipo_estacion.
    Solo cubre estaciones de línea (TIPOS_LINEA). Calidad y Pre-Calidad quedan NaN.

    Parámetros
    ----------
    df_features_estacion : resultado de build_features_estacion().
    campanas_estacion    : tabla del pipeline con campana, cliente, tipo_estacion,
                           target_col y z_score.
    target_col           : columna de tiempo. Default: 'mediana_tpp'.

    Retorna
    -------
    df_resultado  : campana, cliente, tipo_estacion, <target_col>, residuo_loo,
                    z_score_ajustado_estn_cliente_lineas (Var A),
                    z_score_ajustado_dificultad (Var B — diferida [t_c-003]),
                    z_score, delta_z_estn_cliente_lineas, delta_z_dificultad.
    tabla_modelos : R² y MAE LOO-CV. attrs['modelo_ganador'].
    """
    cols_base = ['campana', 'cliente', 'tipo_estacion', target_col]
    if 'z_score' in campanas_estacion.columns:
        cols_base.append('z_score')

    df_ref = campanas_estacion[campanas_estacion['tipo_estacion'].isin(TIPOS_LINEA)][cols_base].copy()
    n_ref  = len(df_ref)

    df            = df_features_estacion.merge(df_ref, on=['campana', 'tipo_estacion'], how='inner')
    n_tras_join   = len(df)
    perdidas_join = n_ref - n_tras_join

    keep = _FEATURES_MODELO_EST + _CATEG_COLS_EST + ['campana', target_col]
    if 'z_score' in df.columns:
        keep.append('z_score')
    df_antes = df[keep].copy()
    df       = df_antes.dropna().reset_index(drop=True)

    print(f"\n── Cobertura build_zscore_ajustado_estacion ──")
    print(f"  Universo líneas en campanas_estacion : {n_ref:>4} filas")
    print(f"  Tras inner join con features         : {n_tras_join:>4} filas  ({perdidas_join} pérdidas)")
    print(f"  Tras dropna                          : {len(df):>4} filas  ({len(df_antes)-len(df)} pérdidas adicionales)")
    if perdidas_join > 0:
        perdidas_df = df_ref[
            ~df_ref.set_index(['campana', 'tipo_estacion']).index.isin(
                df_features_estacion.set_index(['campana', 'tipo_estacion']).index
            )
        ][['campana', 'cliente', 'tipo_estacion']].copy()
        print(f"\n  Detalle pérdidas join ({len(perdidas_df)} filas):")
        print(perdidas_df.to_string(index=False))
    print()

    X = df[_FEATURES_MODELO_EST + _CATEG_COLS_EST]
    y = df[target_col].copy()

    modelos = {
        'Ridge':        Pipeline([('pre', _make_preprocessor(_FEATURES_MODELO_EST, _CATEG_COLS_EST)),
                                   ('model', Ridge(alpha=1.0))]),
        'RandomForest': Pipeline([('pre', _make_preprocessor(_FEATURES_MODELO_EST, _CATEG_COLS_EST)),
                                   ('model', RandomForestRegressor(n_estimators=100, max_depth=3,
                                                                    min_samples_leaf=5, random_state=42))]),
        'LightGBM':     Pipeline([('pre', _make_preprocessor(_FEATURES_MODELO_EST, _CATEG_COLS_EST)),
                                   ('model', lgb.LGBMRegressor(num_leaves=8, min_child_samples=10,
                                                                n_estimators=100, random_state=42, verbose=-1))]),
    }

    resultados_cv = {}
    for nombre, pipe in modelos.items():
        preds, metricas = _loco_evaluate(pipe, X, y)
        resultados_cv[nombre] = {**metricas, 'y_pred_loo': preds}

    tabla_modelos = pd.DataFrame({
        k: {'R² LOO-CV': v['R²_LOO'], 'MAE LOO-CV': v['MAE_LOO']}
        for k, v in resultados_cv.items()
    }).T
    modelo_ganador = tabla_modelos['R² LOO-CV'].idxmax()

    df['y_pred_loo']  = resultados_cv[modelo_ganador]['y_pred_loo']
    df['residuo_loo'] = (df[target_col] - df['y_pred_loo']).round(4)

    # Variante A — normalizado por cliente × tipo_estacion
    df['z_score_ajustado_estn_cliente_lineas'] = _zscore_por_grupo(df, ['cliente', 'tipo_estacion'])

    # Variante B — normalizado por cuartil de dificultad × tipo_estacion ([t_c-003] diferida)
    df['_cuartil'] = pd.qcut(
        df['n_categorias_mediana'].rank(method='first'), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
    )
    df['z_score_ajustado_dificultad'] = _zscore_por_grupo(df, ['_cuartil', 'tipo_estacion'])
    df = df.drop(columns=['y_pred_loo', '_cuartil'])

    if 'z_score' in df.columns:
        df['delta_z_estn_cliente_lineas'] = (df['z_score_ajustado_estn_cliente_lineas'] - df['z_score']).round(3)
        df['delta_z_dificultad']          = (df['z_score_ajustado_dificultad']          - df['z_score']).round(3)

    tabla_modelos.attrs['modelo_ganador'] = modelo_ganador
    return df, tabla_modelos


# ─────────────────────────────────────────────────────────────
# §6  TPP GENUINO — Calidad / Pre-Calidad
# ─────────────────────────────────────────────────────────────
#
# Corrige el denominador inflado de Calidad/PC.
# tiempo_por_pick actual usa picks propios del checkpoint (mediana≈1),
# lo que produce medianas de ~77 seg/pick (Calidad) y ~57 seg/pick (PC).
# El denominador correcto es la cantidad de unidades del tramo de línea
# que ingresó al checkpoint.
#
# Supuesto Morado: 1 pick Morado = 1 unidad revisable por Calidad.
# Si las bolsas contienen múltiples ítems no registrados, el denominador
# quedaría subdimensionado — pendiente confirmación con Sam [R5].
#
# 7% de cajas Calidad sin denominador (flujo invertido: visita de línea
# ocurre temporalmente DESPUÉS del checkpoint). NaN es la respuesta
# honesta — no imputar.

_TIPOS_CALIDAD = ('Calidad', 'Pre-Calidad')
_EPOCH = pd.Timestamp('1970-01-01')


def build_tpp_genuino(
    df_base: pd.DataFrame,
    df_calidad_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula tpp_genuino para Calidad y Pre-Calidad.

    Parameters
    ----------
    df_base : DataFrame pre-IQR con columnas
        box_id, tipo_estacion, init_scan, picks, campana.
        Fuente del denominador — incluye todas las cajas, incluso outliers de tiempo.
    df_calidad_clean : DataFrame post-IQR filtrado a Calidad/Pre-Calidad con columnas
        box_id, tipo_estacion, tiempo_total_seg, campana.
        Fuente del numerador — excluye extremos de tiempo en el checkpoint.

    Returns
    -------
    DataFrame con columnas:
        campana, box_id, tipo_estacion, tpp_genuino, denom_tramo
    NaN en tpp_genuino donde denom_tramo es 0 o no calculable (flujo invertido).
    """
    # ── Normalizar timestamps a tz-naive ────────────────────────────────────
    base = df_base.copy()
    base['init_scan'] = (
        pd.to_datetime(base['init_scan'], utc=True).dt.tz_convert(None)
    )

    # ── Timestamps de primer checkpoint por box_id ───────────────────────────
    mask_linea = base['tipo_estacion'].isin(TIPOS_LINEA)
    mask_cal   = base['tipo_estacion'] == 'Calidad'
    mask_pc    = base['tipo_estacion'] == 'Pre-Calidad'

    t_cal = (
        base.loc[mask_cal]
        .groupby('box_id')['init_scan'].min()
        .rename('t_cal')
    )
    t_pc = (
        base.loc[mask_pc]
        .groupby('box_id')['init_scan'].min()
        .rename('t_pc')
    )

    # ── Base de líneas con timestamps de checkpoints ─────────────────────────
    lineas = (
        base.loc[mask_linea, ['box_id', 'campana', 'tipo_estacion', 'init_scan', 'picks']]
        .join(t_cal, on='box_id')
        .join(t_pc,  on='box_id')
    )

    # ── Denominador Pre-Calidad: picks de línea antes de PC ──────────────────
    pc_lineas = lineas[lineas['t_pc'].notna()].copy()
    pc_lineas['en_tramo'] = pc_lineas['init_scan'] < pc_lineas['t_pc']
    denom_pc = (
        pc_lineas[pc_lineas['en_tramo']]
        .groupby('box_id')['picks'].sum()
        .rename('denom_tramo')
        .reset_index()
        .assign(tipo_estacion='Pre-Calidad')
    )

    # ── Denominador Calidad: picks de línea entre PC y Cal ───────────────────
    # Si no hay PC, límite inferior = EPOCH (acepta todos los picks anteriores a Cal)
    cal_lineas = lineas[lineas['t_cal'].notna()].copy()
    cal_lineas['limite_inf'] = cal_lineas['t_pc'].fillna(_EPOCH)
    cal_lineas['en_tramo']   = (
        (cal_lineas['init_scan'] > cal_lineas['limite_inf']) &
        (cal_lineas['init_scan'] < cal_lineas['t_cal'])
    )
    denom_cal = (
        cal_lineas[cal_lineas['en_tramo']]
        .groupby('box_id')['picks'].sum()
        .rename('denom_tramo')
        .reset_index()
        .assign(tipo_estacion='Calidad')
    )

    denominadores = pd.concat([denom_pc, denom_cal], ignore_index=True)

    # ── Unir numerador (df_calidad_clean) con denominador ────────────────────
    cal_clean = df_calidad_clean[
        df_calidad_clean['tipo_estacion'].isin(_TIPOS_CALIDAD)
    ][['box_id', 'campana', 'tipo_estacion', 'tiempo_total_seg']].copy()

    resultado = cal_clean.merge(denominadores, on=['box_id', 'tipo_estacion'], how='left')

    # denom_tramo == 0 → NaN (no confundir con flujo invertido donde denom es NaN)
    resultado.loc[resultado['denom_tramo'] == 0, 'denom_tramo'] = float('nan')

    resultado['tpp_genuino'] = (
        resultado['tiempo_total_seg'] / resultado['denom_tramo']
    )

    return resultado[['campana', 'box_id', 'tipo_estacion', 'tpp_genuino', 'denom_tramo']]


# ─────────────────────────────────────────────────────────────
# §7  DIAGNÓSTICO — correlación features × tiempo_por_pick
# ─────────────────────────────────────────────────────────────
#
# Funciones de validación migradas desde exploraciones/legado/dificultad_caja.py
# No son funciones de producción — se usan para explorar señal de features
# antes de decidir si entran al modelo.
#
# Investigaciones pendientes (2026-04-30):
#   · n_skus_en_caja: es tailwind en tiempo_por_pick por efecto denominador,
#     pero puede describir complejidad. Verificar si añade R² al modelo
#     independientemente de n_categorias_en_caja.
#   · cambios_de_row: excluido del modelo mixto por artefacto Cal/PC.
#     Pendiente: evaluar si tiene señal real en segmento líneas (TIPOS_LINEA)
#     usando diagnostico_correlacion_dificultad con filtro previo a líneas.


def _computar_tpp_box(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula tiempo_por_pick a nivel box_id desde eventos crudos de apertura/cierre.

    Uso: validación de correlación — no sustituye al tpp del pipeline (que opera
    sobre ops_clean post-IQR a nivel box×estacion).

    Parámetros
    ----------
    df_raw : DataFrame de eventos crudos con columnas type, box_id, time, campana.

    Retorna
    -------
    DataFrame con: campana, box_id, tiempo_por_pick, seg_caja, n_picks
    Solo filas con par open+close válido y al menos 1 pick.
    """
    df_open = (
        df_raw[df_raw['type'] == 'box_open']
        .groupby('box_id', observed=True)['time'].min()
        .rename('t_open')
    )
    df_close = (
        df_raw[df_raw['type'] == 'box_close']
        .groupby('box_id', observed=True)['time'].max()
        .rename('t_close')
    )
    df_t = df_open.to_frame().join(df_close, how='inner')
    df_t['t_open']  = pd.to_datetime(df_t['t_open'],  utc=True, errors='coerce')
    df_t['t_close'] = pd.to_datetime(df_t['t_close'], utc=True, errors='coerce')
    df_t['seg_caja'] = (df_t['t_close'] - df_t['t_open']).dt.total_seconds()

    _SCAN_TYPES = {'scan', 'pick_ok'}
    n_picks = (
        df_raw[df_raw['type'].isin(_SCAN_TYPES & set(df_raw['type'].unique()))]
        .groupby('box_id', observed=True)
        .size()
        .rename('n_picks')
    )
    df_t = df_t.join(n_picks, how='left')
    df_t['n_picks'] = df_t['n_picks'].fillna(0)

    mask = (df_t['n_picks'] > 0) & (df_t['seg_caja'] > 0)
    df_t['tiempo_por_pick'] = np.where(mask, df_t['seg_caja'] / df_t['n_picks'], np.nan)

    campana_map = (
        df_raw[['box_id', 'campana']].dropna(subset=['campana'])
        .drop_duplicates('box_id').set_index('box_id')['campana']
    )
    return (
        df_t.join(campana_map, how='left')
        .reset_index()[['campana', 'box_id', 'tiempo_por_pick', 'seg_caja', 'n_picks']]
    )


def diagnostico_correlacion_dificultad(
    df_dificultad: pd.DataFrame,
    df_raw: pd.DataFrame,
    por_cliente: bool = True,
    tpp_max: float = 600.0,
) -> pd.DataFrame:
    """
    Correlación Spearman de features de dificultad × tiempo_por_pick a nivel caja.

    Retorna DataFrame con columnas: feature, scope, r, p_value, sig, n
    donde scope es 'global' o el nombre del cliente.

    Útil para:
    · Validar que n_categorias_en_caja sigue siendo headwind por cliente
    · Evaluar si n_skus_en_caja añade señal más allá del efecto denominador
    · Evaluar cambios_de_row en segmento líneas (pasar df_dificultad filtrado)

    Parámetros
    ----------
    df_dificultad : resultado de build_ops_dificultad (o filtrado a TIPOS_LINEA)
    df_raw        : eventos crudos — necesario para computar tpp_box
    por_cliente   : si True, incluye desglose por cliente_db
    tpp_max       : excluye outliers de tiempo (default 600 seg/pick)
    """
    from scipy.stats import spearmanr

    df_tpp = _computar_tpp_box(df_raw)
    df = df_dificultad.merge(df_tpp[['box_id', 'tiempo_por_pick']], on='box_id', how='inner')
    df = df[df['tiempo_por_pick'].notna() & df['tiempo_por_pick'].between(0, tpp_max)]

    features = ['n_skus_en_caja', 'n_categorias_en_caja', 'cambios_de_row', 'n_filas_distintas']
    features = [f for f in features if f in df.columns]

    rows = []

    def _corr_row(sub, scope, feat):
        vals = sub[['tiempo_por_pick', feat]].dropna()
        if len(vals) < 10:
            return None
        r, p = spearmanr(vals['tiempo_por_pick'], vals[feat])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        return {'feature': feat, 'scope': scope, 'r': round(r, 3),
                'p_value': round(p, 4), 'sig': sig, 'n': len(vals)}

    for feat in features:
        row = _corr_row(df, 'global', feat)
        if row:
            rows.append(row)

    if por_cliente and 'campana' in df.columns:
        cliente_map = (
            df_raw[['campana', 'cliente_db']].dropna()
            .drop_duplicates('campana').set_index('campana')['cliente_db']
        )
        df['cliente_db'] = df['campana'].map(cliente_map)
        for cliente in sorted(df['cliente_db'].dropna().unique()):
            sub = df[df['cliente_db'] == cliente]
            for feat in features:
                row = _corr_row(sub, cliente, feat)
                if row:
                    rows.append(row)

    return pd.DataFrame(rows)
