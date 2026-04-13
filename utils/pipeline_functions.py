import pandas as pd
import requests
import sys
import os
from pathlib import Path

# ── Versión anterior de api_to_df (respaldo temporal) ────────────────────────
# def api_to_df(url, params=None, headers=None, records_key=None):
#     try:
#         response = requests.get(url, params=params, headers=headers)
#         response.raise_for_status()
#         data = response.json()
#         if records_key:
#             data = data[records_key]
#         return pd.DataFrame(data)
#     except Exception as e:
#         print(f"Error al obtener datos de la API: {e}")
#         return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────

def api_to_df(url, params=None, headers=None, records_key=None, limit=None, fetch_all=False):
    """
    Convierte una respuesta de API directamente a un DataFrame.

    :param url:         URL base del endpoint (sin parámetros de paginación).
    :param params:      Diccionario con parámetros adicionales (filtros, tokens).
    :param headers:     Headers HTTP (ej. Authorization).
    :param records_key: Llave del JSON donde están los registros (ej: 'collection', 'results').
    :param limit:       Registros por página. Si se indica, se agrega a params automáticamente.
    :param fetch_all:   False (default) → solo primera página, avisa si hay más datos.
                        True → loopea con offset hasta obtener todos los registros.
    """
    try:
        all_records = []
        offset = 0
        label = url.rstrip('/').split('/')[-1]

        while True:
            # Construir parámetros de la petición
            call_params = dict(params or {})
            if limit is not None:
                call_params['limit'] = limit
                call_params['offset'] = offset

            response = requests.get(url, params=call_params, headers=headers)
            response.raise_for_status()
            data = response.json()

            records = data[records_key] if records_key else data
            if not isinstance(records, list):
                records = [records]
            all_records.extend(records)

            if not fetch_all:
                if limit is not None and len(records) >= limit:
                    print(f"[{label}] AVISO: se alcanzó el límite de {limit} registros. "
                          f"Puede haber más datos. Usa fetch_all=True para obtenerlos todos.")
                else:
                    print(f"[{label}] {len(records)} registros cargados.")
                break

            # fetch_all=True: continúa hasta página final
            print(f"[{label}] offset={offset} → {len(records)} registros")
            if limit is None or len(records) < limit:
                print(f"[{label}] Total: {len(all_records)} registros cargados.")
                break
            offset += limit

        return pd.DataFrame(all_records)

    except Exception as e:
        print(f"Error al obtener datos de la API: {e}")
        return pd.DataFrame()  # Retorna un DF vacío para que el pipeline no truene
    

# ── Preparación para exportación ─────────────────────────────────────────────

def preparar_para_export(df, cols_timedelta, cols_porcentaje, cols_bh=None):
    """
    Prepara una copia de un DataFrame para exportar a Excel:
        - Convierte columnas timedelta a string "HH:MM" (infraestructura futura)
        - Convierte columnas de horas laborales (float) a string "HH:MM"
        - Convierte porcentajes de escala 0-100 a decimal 0-1 (formato Excel)

    Args:
        df              : pd.DataFrame — DataFrame fuente (no se modifica el original)
        cols_timedelta  : list[str]    — columnas con pd.Timedelta a formatear
        cols_porcentaje : list[str]    — columnas con porcentajes 0-100 a dividir entre 100
        cols_bh         : list[str]    — columnas con horas laborales (float) a formatear

    Returns:
        pd.DataFrame — copia lista para openpyxl / to_excel
    """
    import utils.ETL_EDA_functions as EDA
    df = df.copy()
    for col in cols_timedelta:
        if col in df.columns:
            df[col] = df[col].apply(EDA.timedelta_a_hhmm)
    for col in (cols_bh or []):
        if col in df.columns:
            df[col] = df[col].apply(EDA.horas_a_hhmm)
    for col in cols_porcentaje:
        if col in df.columns:
            df[col] = df[col] / 100
    return df


# ── Detección de relaciones entre tablas ─────────────────────────────────────

def mapear_fks(tablas):
    """
    Extrae y muestra los IDs de columnas anidadas (claves foráneas implícitas).
    Sirve para identificar los JOINs posibles entre tablas de la API.

    Parámetros:
        tablas  dict[str, DataFrame] — tablas crudas (con columnas anidadas)

    Imprime un mapa de FK detectadas y retorna dict con los valores únicos por FK.
    """
    resultado = {}
    print(f"{'=' * 50}")
    print("Mapa de claves foráneas (FK) detectadas")
    print(f"{'=' * 50}")

    for nombre, df in tablas.items():
        fks = {}
        for col in df.columns:
            tiene_dicts = df[col].apply(lambda x: isinstance(x, dict)).any()
            if tiene_dicts:
                sample = df[col].apply(lambda x: x if isinstance(x, dict) else {})
                expandido = pd.json_normalize(sample.tolist())
                id_cols = [c for c in expandido.columns if 'id' in c.lower()]
                for ic in id_cols:
                    fks[f"{col}.{ic}"] = expandido[ic].dropna().unique().tolist()

        if fks:
            print(f"\n  {nombre}:")
            for fk, vals in fks.items():
                muestra = vals[:3]
                print(f"    {fk:<35} ejemplo: {muestra}")
            resultado[nombre] = fks

    print(f"\n{'=' * 50}")
    return resultado


# ── Entorno y conexión ────────────────────────────────────────────────────────

def conectar_utils(nombre_carpeta="utils"):
    """
    Agrega la carpeta de utilidades al sys.path detectando si es script o notebook.
    """
    try:
        # Detecta si existe __file__ (scripts .py)
        base_path = Path(__file__).resolve().parent
    except NameError:
        # Si no (Notebooks), usa el directorio actual
        base_path = Path.cwd()

    # Buscamos la carpeta subiendo un nivel
    ruta = str(base_path.parent / nombre_carpeta)

    if ruta not in sys.path:
        sys.path.append(ruta)
    
    return ruta