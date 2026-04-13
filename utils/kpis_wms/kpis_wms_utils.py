"""
kpis_wms_utils.py
-----------------
Funciones específicas del proyecto KPIs WMS.
Estructuras o lógica atípica que no aplica a utils generales.

Funciones disponibles:
    revisar_calidad(tablas)     — nulos y duplicados por tabla (columnas planas)
    aplanar_objetos(df, nombre) — expande dicts anidados en subcampos
"""

import pandas as pd


def aplanar_objetos(df, nombre):
    """
    Expande columnas con dicts anidados en subcampos (col.subcampo).
    Convierte listas a string para poder comparar en duplicados.
    Uso: exploración de estructuras anidadas y extracción de subcampos para KPIs.

    Parámetros:
        df      DataFrame — tabla con columnas anidadas
        nombre  str       — etiqueta para el log de columnas expandidas

    Retorna:
        DataFrame con dicts expandidos en subcampos y listas como str
    """
    df_flat = df.copy()
    cols_log = []

    for col in df_flat.select_dtypes(include='object').columns:
        tiene_dicts  = df_flat[col].apply(lambda x: isinstance(x, dict)).any()
        tiene_listas = df_flat[col].apply(lambda x: isinstance(x, list)).any()

        if tiene_dicts:
            serie_limpia = df_flat[col].apply(lambda x: x if isinstance(x, dict) else {})
            expandido = pd.json_normalize(serie_limpia.tolist())
            expandido.columns = [f"{col}.{c}" for c in expandido.columns]
            df_flat = df_flat.drop(columns=[col])
            df_flat = pd.concat([df_flat.reset_index(drop=True), expandido.reset_index(drop=True)], axis=1)
            cols_log.append(f"{col} → {list(expandido.columns)}")

        elif tiene_listas:
            df_flat[col] = df_flat[col].apply(lambda x: str(x) if isinstance(x, list) else x)
            cols_log.append(f"{col} (lista → str)")

    print(f"[{nombre}]")
    for entry in cols_log:
        print(f"  {entry}")
    if not cols_log:
        print("  sin columnas anidadas")

    return df_flat


def revisar_calidad(tablas):
    """
    Revisa nulos y duplicados sobre columnas planas.
    Columnas anidadas (dict/list) se catalogan aparte — no se abren aquí.

    Parámetros:
        tablas  dict[str, DataFrame] — tablas crudas (sin aplanar)

    Retorna:
        dict[str, Series] con % de nulos por columna para cada tabla
    """
    resultado = {}
    print('=' * 50)
    for nombre, df in tablas.items():
        print(f"\n{nombre}  ({len(df)} registros, {df.shape[1]} columnas)")

        cols_anidadas = [
            col for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any()
        ]
        cols_planas = [col for col in df.columns if col not in cols_anidadas]

        nulos = df[cols_planas].isnull().mean().mul(100).round(1)
        nulos_relevantes = nulos[nulos > 0].sort_values(ascending=False)

        df_str = df.astype(str)
        dupes    = df_str.duplicated().sum()
        dupes_id = df_str.duplicated(subset=['id']).sum() if 'id' in df.columns else None

        sin_problemas = (
            nulos_relevantes.empty
            and dupes == 0
            and (dupes_id == 0 if dupes_id is not None else True)
        )

        if sin_problemas:
            print("  Campos planos: sin nulos ni duplicados.")
        else:
            if not nulos_relevantes.empty:
                print("  Nulos (%):")
                for col, pct in nulos_relevantes.items():
                    print(f"    {col:<45} {pct}%")
            print(f"  Duplicados (fila completa) : {dupes}")
            if dupes_id is not None:
                print(f"  Duplicados (por id)        : {dupes_id}")

        if cols_anidadas:
            print(f"  Anidadas (explorar aparte) : {cols_anidadas}")

        resultado[nombre] = nulos_relevantes

    print(f'\n{"=" * 50}')
    return resultado
