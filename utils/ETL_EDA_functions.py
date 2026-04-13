
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np


# ── Casteo de tipos ───────────────────────────────────────────────────────────

def castear_ids(df):
    """
    Castea a str todas las columnas cuyo nombre sea 'id' o termine en '_id'.
    Evita que identificadores numéricos sean procesados como cantidades.
    Aplicar en ETL, justo después de la carga.

    Parámetros:
        df  DataFrame — tabla recién cargada

    Retorna:
        DataFrame con columnas ID casteadas a str
    """
    cols_id = [c for c in df.columns if c == 'id' or c.endswith('_id')]
    if cols_id:
        df = df.copy()
        df[cols_id] = df[cols_id].astype(str)
    return df


# ── Estadísticos descriptivos ─────────────────────────────────────────────────

def describir_numericas(df, nombre, cols=None):
    """
    Estadísticos de tendencia central y dispersión sobre columnas numéricas.
    Útil para perfilar qty_* de inventory_levels y cualquier tabla con numéricas.

    Parámetros:
        df      DataFrame   — tabla a describir
        nombre  str         — etiqueta para el encabezado del output
        cols    list[str] | None — columnas a incluir; None = todas las numéricas

    Retorna:
        DataFrame con métricas: count, mean, std, min, Q1, median, Q3, max, skew, zeros_%
    """
    cols_num = cols or df.select_dtypes(include='number').columns.tolist()
    if not cols_num:
        print(f"[{nombre}] sin columnas numéricas.")
        return pd.DataFrame()

    stats = df[cols_num].agg(['count', 'mean', 'std', 'min',
                               lambda x: x.quantile(0.25),
                               'median',
                               lambda x: x.quantile(0.75),
                               'max',
                               'skew']).T
    stats.columns = ['count', 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max', 'skew']
    stats['zeros_%'] = df[cols_num].apply(lambda c: (c == 0).mean() * 100).round(1)
    stats = stats.round(2)

    print(f"{'=' * 50}")
    print(f"{nombre}  —  estadísticos numéricos ({len(cols_num)} columnas)")
    print(f"{'=' * 50}")
    print(stats.to_string())
    print()
    return stats


# ── Exploración de estructuras anidadas ───────────────────────────────────────

def expandir_lista_col(df, col, nombre=None):
    """
    Explota una columna que contiene listas: genera una fila por elemento.
    Cada fila hereda las columnas del registro padre más los campos del elemento.

    Parámetros:
        df      DataFrame   — tabla con la columna de listas
        col     str         — nombre de la columna que contiene listas de dicts
        nombre  str | None  — etiqueta para el log (opcional)

    Retorna:
        DataFrame aplanado con una fila por elemento de la lista
    """
    etiq = nombre or col
    df_exp = df.copy()

    df_exp[col] = df_exp[col].apply(
        lambda x: x if isinstance(x, list) else []
    )

    df_exp = df_exp.explode(col).reset_index(drop=True)

    tiene_dicts = df_exp[col].dropna().apply(lambda x: isinstance(x, dict)).any()
    if tiene_dicts:
        expandido = pd.json_normalize(
            df_exp[col].dropna().apply(lambda x: x if isinstance(x, dict) else {}).tolist()
        )
        expandido.index = df_exp[col].dropna().index
        expandido.columns = [f"{col}.{c}" for c in expandido.columns]
        df_exp = df_exp.drop(columns=[col]).join(expandido)

    n_elem = len(df_exp)
    print(f"[{etiq}] {n_elem} filas después de expandir '{col}'")
    if tiene_dicts:
        nuevas = [c for c in df_exp.columns if c.startswith(f"{col}.")]
        print(f"  Subcampos: {nuevas}")
    return df_exp


# ── Parseo de fechas ──────────────────────────────────────────────────────────

def parse_date(x, offset_hours=0):
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


# ── Antigüedad y tiempos ──────────────────────────────────────────────────────

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

    return diff_segundos / divisores.get(unidad, 3600)


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


