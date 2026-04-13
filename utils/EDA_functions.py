
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np
def parse_date(x,offset_hours=0):
    if x in ["", None, "NaT", "null"]:
        return None
    try:
        dt = pd.to_datetime(x, utc=True)
        if offset_hours != 0:
            dt = dt + pd.Timedelta(hours=offset_hours)
        return dt
    except:
        return None
    
def auto_parse_dates(df, parse_func, keyword="fecha", offset_hours=0):
    '''
    Busca llaves que contengan 'keyword' y aplica la función hechiza 'parse_date'.
    '''
    cols_a_modificar_a_fecha = [col for col in df.columns if keyword in col.lower()] 
    for col in cols_a_modificar_a_fecha: 
        df[col] = df[col].apply(lambda x: parse_func(x, offset_hours=offset_hours))
    return df

def calcular_antiguedad_bruta(df, col_inicio, col_fin=None, unidad='horas'):
    """
    Calcula la diferencia entre dos fechas en la unidad especificada.
    Si col_fin es None, usa la fecha actual.
    Fechas nulas es limitante.
    """
    fin = pd.to_datetime(col_fin) if col_fin else pd.Timestamp.now()
    inicio = pd.to_datetime(df[col_inicio])
    
    diff_segundos = (fin - inicio).dt.total_seconds()
    
    divisores = {'segundos': 1, 'minutos': 60, 'horas': 3600, 'dias': 86400}
    
    return diff_segundos / divisores.get(unidad, 3600) #horas totales de inicio a fin de fecha

import pandas as pd

def calcular_horas_laborales(df, col_inicio, col_fin=None):
    HORA_FIN = pd.Timedelta(hours=18, minutes=30)
    HORA_INICIO = pd.Timedelta(hours=9)

    now = pd.Timestamp.now()

    def horas_en_dia(fecha_inicio, fecha_fin):
        if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
            return 0

        if fecha_inicio.date() != fecha_fin.date():
            return 0
        
        start = max(fecha_inicio, fecha_inicio.normalize() + HORA_INICIO)
        end = min(fecha_fin, fecha_fin.normalize() + HORA_FIN)
        
        if start >= end:
            return 0
        
        return (end - start).total_seconds() / 3600

    def calcular(row):
        start = pd.to_datetime(row[col_inicio])
        end = pd.to_datetime(row[col_fin]) if col_fin else now

        if pd.isna(start) or pd.isna(end):
            return 0

        if start > end:
            start, end = end, start

        total_horas = 0
        dias = pd.date_range(start.normalize(), end.normalize(), freq='D')

        for dia in dias:
            if dia.weekday() >= 5:
                continue

            if dia.date() == start.date() and dia.date() == end.date():
                total_horas += horas_en_dia(start, end)

            elif dia.date() == start.date():
                cierre_dia = dia + HORA_FIN
                total_horas += horas_en_dia(start, cierre_dia)

            elif dia.date() == end.date():
                apertura_dia = dia + HORA_INICIO
                total_horas += horas_en_dia(apertura_dia, end)

            else:
                total_horas += 9.5

        return total_horas

    return df.apply(calcular, axis=1)


# ── Formato de tiempo (HH:MM) ─────────────────────────────────────────────────

def timedelta_a_hhmm(td):
    """
    Convierte un timedelta a string legible en formato "HH:MM".
    Soporta valores negativos (e.g. "-03:15").

    Args:
        td: pd.Timedelta | datetime.timedelta | pd.NaT

    Returns:
        str  →  "HH:MM" o "-HH:MM"
        None →  si td es NaT / None
    """
    if pd.isna(td):
        return None
    total_segundos = int(td.total_seconds())
    signo = "-" if total_segundos < 0 else ""
    total_segundos = abs(total_segundos)
    horas = total_segundos // 3600
    minutos = (total_segundos % 3600) // 60
    return f"{signo}{horas:02d}:{minutos:02d}"


def horas_a_hhmm(h):
    """
    Convierte un float de horas laborales a string "HH:MM".
    Soporta valores negativos (e.g. "-03:15").

    Args:
        h: float | None | NaN  — horas laborales

    Returns:
        str  →  "HH:MM" o "-HH:MM"
        None →  si h es None / NaN
    """
    if h is None or pd.isna(h):
        return None
    signo = "-" if h < 0 else ""
    h = abs(h)
    horas = int(h)
    minutos = round((h - horas) * 60)
    if minutos == 60:
        horas += 1
        minutos = 0
    return f"{signo}{horas:02d}:{minutos:02d}"


# ── Totales y porcentajes ─────────────────────────────────────────────────────

def calcular_totales_y_porcentajes(df, cols_invertido, cols_campana):
    """
    Agrega columnas de totales y porcentajes para dos grupos de columnas timedelta.
    Infraestructura conservada para análisis con timedelta a futuro.

    Columnas que crea:
        - total_tiempo_invertido  → suma de cols_invertido
        - total_tiempo_campaña    → suma de cols_campana
        - porcentaje_{col}        → % de cada col respecto a su total de grupo

    Args:
        df            : pd.DataFrame  — debe contener columnas timedelta
        cols_invertido: list[str]     — columnas del grupo "tiempo invertido"
        cols_campana  : list[str]     — columnas del grupo "tiempo de campaña"

    Returns:
        pd.DataFrame — el mismo df con las columnas nuevas añadidas
    """
    df["total_tiempo_invertido"] = df[cols_invertido].sum(axis=1)
    df["total_tiempo_campaña"]   = df[cols_campana].sum(axis=1)
    for col in cols_invertido:
        df[f"porcentaje_{col}"] = (
            df[col].dt.total_seconds() /
            df["total_tiempo_invertido"].dt.total_seconds()
        ) * 100
    for col in cols_campana:
        df[f"porcentaje_{col}"] = (
            df[col].dt.total_seconds() /
            df["total_tiempo_campaña"].dt.total_seconds()
        ) * 100
    cols_pct = [c for c in df.columns if c.startswith("porcentaje_")]
    df[cols_pct] = df[cols_pct].replace([float("inf"), float("-inf")], pd.NA)
    return df


def calcular_totales_y_porcentajes_bh(df, cols_invertido, cols_campana):
    """
    Agrega columnas de totales y porcentajes para dos grupos de columnas
    de horas laborales (float).

    Columnas que crea:
        - total_tiempo_invertido  → suma de cols_invertido
        - total_tiempo_campaña    → suma de cols_campana
        - porcentaje_{col}        → % de cada col respecto a su total de grupo

    Args:
        df            : pd.DataFrame  — debe contener columnas float (horas laborales)
        cols_invertido: list[str]     — columnas del grupo "tiempo invertido"
        cols_campana  : list[str]     — columnas del grupo "tiempo de campaña"

    Returns:
        pd.DataFrame — el mismo df con las columnas nuevas añadidas
    """
    df["total_tiempo_invertido"] = df[cols_invertido].sum(axis=1)
    df["total_tiempo_campaña"]   = df[cols_campana].sum(axis=1)
    for col in cols_invertido:
        df[f"porcentaje_{col}"] = (df[col] / df["total_tiempo_invertido"]) * 100
    for col in cols_campana:
        df[f"porcentaje_{col}"] = (df[col] / df["total_tiempo_campaña"]) * 100
    cols_pct = [c for c in df.columns if c.startswith("porcentaje_")]
    df[cols_pct] = df[cols_pct].replace([float("inf"), float("-inf")], pd.NA)
    return df


# ── IQR, agrupación y dispersión ─────────────────────────────────────────────

def iqr_bounds(series, k=1.5):
    """
    Calcula los límites inferior y superior IQR de una serie.

    Parámetros
    ----------
    series : pd.Series
    k      : float — factor multiplicador del IQR. Default 1.5 (estándar Tukey).

    Retorna
    -------
    (lower, upper) : tuple de floats
    """
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - k * IQR, Q3 + k * IQR


# ── picking_tiempos_campanas ───────────────────────────────────────────────────

def transformar_base_campanas(df):
    """
    Aplica transformaciones base al DataFrame raw de campañas:
    box_id → str, timestamps → datetime UTC, extrae año, calcula tiempos totales.

    Parámetros
    ----------
    df : pd.DataFrame — DataFrame raw leído del Excel de campañas.

    Retorna
    -------
    pd.DataFrame con columnas adicionales: año, tiempo_total_seg, tiempo_total_min.
    """
    df = df.copy()
    df['box_id']    = df['box_id'].astype(str)
    df['init_scan'] = pd.to_datetime(df['init_scan'], utc=True)
    df['last_scan'] = pd.to_datetime(df['last_scan'],  utc=True)
    df['año'] = df['campana'].apply(
        lambda x: 2026 if '26' in x else (2025 if '25' in x else None)
    )
    df['tiempo_total_seg'] = (df['last_scan'] - df['init_scan']).dt.total_seconds()
    df['tiempo_total_min'] = df['tiempo_total_seg'] / 60
    return df


def limpiar_iqr_campanas(df_picking, col='tiempo_total_seg', group_col='tipo_estacion', k=1.5):
    """
    Limpia outliers por IQR calculado dentro de cada grupo (contextual).

    Parámetros
    ----------
    df_picking : pd.DataFrame — registros con picks > 0.
    col        : str — columna sobre la que se aplica el IQR. Default: 'tiempo_total_seg'.
    group_col  : str — columna de agrupación. Default: 'tipo_estacion'.
    k          : float — factor IQR. Default: 1.5.

    Retorna
    -------
    pd.DataFrame limpio (registros dentro de límites IQR de su grupo).
    """
    mask = df_picking.groupby(group_col)[col].transform(
        lambda s: s.between(*iqr_bounds(s, k))
    )
    return df_picking[mask].copy()


def agregar_metricas_campanas(df_clean):
    """
    Agrega métricas de eficiencia derivadas al DataFrame limpio de picking.
    Columnas añadidas: tiempo_por_pick, picks_por_min, cliente.

    Parámetros
    ----------
    df_clean : pd.DataFrame — DataFrame con picks > 0, ya limpio por IQR.

    Retorna
    -------
    pd.DataFrame con columnas de eficiencia añadidas.
    """
    df = df_clean.copy()
    df['tiempo_por_pick'] = df['tiempo_total_seg'] / df['picks']
    df['picks_por_min']   = df['picks'] / df['tiempo_total_min']
    df['cliente']         = df['campana'].str.extract(r'^(\w+)')
    return df


# expresión dinámica y humana basada en datos
def resumen_eficiencia_picking(
    df,
    col_tiempo_por_pick = 'tiempo_por_pick',
    col_tiempo_total    = 'tiempo_total_seg',
    col_picks           = 'picks',
    col_picks_por_min   = 'picks_por_min',
    contexto            = None,
):
    """
    Genera un resumen narrativo de eficiencia operacional a partir de un DataFrame
    de tiempos de picking.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame limpio con métricas de eficiencia ya calculadas.
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
    - Usa mediana como tendencia central por robustez ante distribuciones asimétricas.
    - El coeficiente de variación (CV) normaliza la dispersión respecto a la media,
      permitiendo comparar variabilidad entre grupos con distinta escala.
    - Compatible con cualquier subconjunto del df (por cliente, estación, campaña, etc.).
    """
    # ── Validación de columnas ────────────────────────────────────────────────
    cols_req = [col_tiempo_por_pick, col_tiempo_total, col_picks, col_picks_por_min]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas no encontradas en el DataFrame: {faltantes}")
    if df.empty:
        print("⚠️  El DataFrame está vacío — no hay estadísticas que mostrar.")
        return

    # ── Cálculo de estadísticas ───────────────────────────────────────────────
    def stats(s):
        q1, q3   = s.quantile(0.25), s.quantile(0.75)
        media    = s.mean()
        std      = s.std()
        cv       = (std / media * 100) if media != 0 else float('nan')
        return {
            'media'  : media,
            'mediana': s.median(),
            'std'    : std,
            'cv'     : cv,
            'q1'     : q1,
            'q3'     : q3,
            'iqr'    : q3 - q1,
            'p90'    : s.quantile(0.90),
            'p95'    : s.quantile(0.95),
            'n'      : s.count(),
        }

    tpp = stats(df[col_tiempo_por_pick])
    tt  = stats(df[col_tiempo_total])
    pk  = stats(df[col_picks])
    ppm = stats(df[col_picks_por_min])

    # ── Interpretaciones dinámicas ────────────────────────────────────────────
    sesgo_label = (
        "lo que sugiere que hay operaciones que se toman bastante más tiempo del habitual "
        "y jalan el promedio hacia arriba"
        if tpp['media'] > tpp['mediana'] * 1.15
        else "lo que indica que los tiempos son bastante simétricos y el promedio es representativo"
    )

    cv_label = (
        "alta"    if tpp['cv'] > 70
        else "moderada" if tpp['cv'] > 40
        else "baja"
    )
    cv_interpretacion = (
        "Los tiempos son bastante dispares entre operaciones — hay variedad real "
        "en cómo se procesan las cajas, lo que puede reflejar diferencias en operadores, "
        "tipo de producto o complejidad de la caja."
        if tpp['cv'] > 70
        else
        "Existe variabilidad notable pero controlada. La operación no es completamente "
        "uniforme, aunque la mayoría se mantiene dentro de un rango razonable."
        if tpp['cv'] > 40
        else
        "Los tiempos son bastante consistentes. Las operaciones tienden a comportarse "
        "de forma homogénea, lo que es una buena señal de estandarización."
    )

    encabezado = f"  EFICIENCIA OPERACIONAL{f'  —  {contexto}' if contexto else ''}"

    # ── Impresión narrativa ───────────────────────────────────────────────────
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