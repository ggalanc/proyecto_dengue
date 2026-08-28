"""
Ingeniería de features segura (sin fuga) para DengAI.

Implementa las mitigaciones identificadas en la Fase 1 (Paso 7):
- Las lag features SIEMPRE usan shift(+k) (trae el pasado), nunca shift(-k).
- El shift se hace agrupado por 'city' para no filtrar información entre ciudades
  (la última fila de SJ no debe "traer" la primera fila de IQ).
"""
import numpy as np
import pandas as pd

# Variables climáticas con mayor señal según el EDA (Fase 1, Paso 5); NDVI se deja
# fuera de los lags porque el EDA mostró correlación ~0 con total_cases en todos los
# rezagos y es la variable con más nulos.
FEATURES_CLIMATICAS = [
    'reanalysis_specific_humidity_g_per_kg',
    'reanalysis_dew_point_temp_k',
    'station_avg_temp_c',
    'precipitation_amt_mm',
    'reanalysis_precip_amt_kg_per_m2',
]

LAGS_SEMANAS = [1, 2, 3, 4, 8, 12]

# Resto de las variables climáticas crudas que el modelo usa SIN rezago (contemporáneas).
OTRAS_FEATURES_CONTEMPORANEAS = [
    'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_max_air_temp_k',
    'reanalysis_min_air_temp_k', 'reanalysis_relative_humidity_percent',
    'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'station_max_temp_c',
    'station_min_temp_c', 'station_precip_mm', 'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
]

# Las 20 columnas crudas tal como vienen en dengue_features_*.csv (sin las llaves
# city/year/weekofyear/week_start_date). El servicio de inferencia (Fase 3) recibe y
# almacena estas 20 en su histórico -- aunque el modelo actual solo usa 19 de ellas
# (no incluye 'reanalysis_sat_precip_amt_mm', quedó fuera en la Fase 2; se documenta
# como limitación conocida en vez de reentrenar sobre la marcha).
RAW_FEATURE_COLS = sorted(set(
    FEATURES_CLIMATICAS + OTRAS_FEATURES_CONTEMPORANEAS + ['reanalysis_sat_precip_amt_mm']
))


def construir_lag_features(df, columnas=FEATURES_CLIMATICAS, lags=LAGS_SEMANAS):
    """Agrega columnas col_lag{k} = valor de 'col' hace k semanas, dentro de cada ciudad."""
    df = df.sort_values(['city', 'week_start_date']).copy()
    nuevas_cols = []
    for col in columnas:
        for k in lags:
            nombre = f'{col}_lag{k}'
            df[nombre] = df.groupby('city')[col].shift(k)  # shift(+k) -> SOLO pasado
            nuevas_cols.append(nombre)
    return df, nuevas_cols


def verificar_sin_fuga_temporal(df, columnas=FEATURES_CLIMATICAS, lags=LAGS_SEMANAS):
    """
    Test de auditoría: para una muestra de filas, confirma que cada lag feature
    coincide exactamente con el valor original k semanas atrás (misma ciudad) y
    NO con ningún valor de una fecha posterior. Lanza AssertionError si falla.
    """
    df = df.sort_values(['city', 'week_start_date']).reset_index(drop=True)
    for city, sub in df.groupby('city'):
        sub = sub.reset_index(drop=True)
        for col in columnas:
            for k in lags:
                lag_col = f'{col}_lag{k}'
                if lag_col not in sub.columns:
                    continue
                # Para cada fila i con i >= k, el valor de lag_col debe ser exactamente
                # el valor de 'col' en la fila (i - k) DENTRO de esa misma ciudad.
                esperado = sub[col].shift(k)
                obtenido = sub[lag_col]
                comparables = esperado.notna() & obtenido.notna()
                assert np.allclose(esperado[comparables], obtenido[comparables]), \
                    f"Fuga detectada en {city}/{lag_col}: no coincide con el valor real k semanas atrás"
    return True


def preparar_dataset_modelado(df):
    """
    Aplica feature engineering completo y devuelve (X, y, columnas_numericas, columnas_categoricas)
    listos para el ColumnTransformer. Elimina las filas de "calentamiento" (warm-up) que quedan
    con NaN en los lags más largos porque no hay suficiente historia previa en esa ciudad.
    """
    df_feat, lag_cols = construir_lag_features(df)
    verificar_sin_fuga_temporal(df_feat)

    columnas_numericas = FEATURES_CLIMATICAS + lag_cols + OTRAS_FEATURES_CONTEMPORANEAS
    columnas_categoricas = ['city']

    max_lag = max(LAGS_SEMANAS)
    n_antes = len(df_feat)
    # nos quedamos solo con filas donde el lag mas largo ya tiene valor (fin del calentamiento)
    df_feat = df_feat.dropna(subset=[f'{FEATURES_CLIMATICAS[0]}_lag{max_lag}'])
    n_despues = len(df_feat)

    X = df_feat[columnas_numericas + columnas_categoricas].reset_index(drop=True)
    y = df_feat['total_cases'].reset_index(drop=True)
    fechas = df_feat['week_start_date'].reset_index(drop=True)
    ciudades = df_feat['city'].reset_index(drop=True)

    info = {'n_antes_dropna': n_antes, 'n_despues_dropna': n_despues,
            'filas_descartadas_calentamiento': n_antes - n_despues}

    return X, y, fechas, ciudades, columnas_numericas, columnas_categoricas, info


def columnas_modelo():
    """Nombres de columnas numéricas + categóricas que el modelo de la Fase 2 espera, en el
    mismo orden que produce preparar_dataset_modelado (el ColumnTransformer selecciona por
    nombre, así que el orden exacto no es obligatorio, pero mantenerlo evita sorpresas)."""
    lag_cols = [f'{col}_lag{k}' for col in FEATURES_CLIMATICAS for k in LAGS_SEMANAS]
    columnas_numericas = FEATURES_CLIMATICAS + lag_cols + OTRAS_FEATURES_CONTEMPORANEAS
    return columnas_numericas, ['city']


def construir_vector_prediccion(df_ventana):
    """
    Usado por el servicio de inferencia (Fase 3): dado un pequeño histórico reciente de UNA
    ciudad (df_ventana, ordenado por fecha, con la observación a predecir en la ÚLTIMA fila),
    calcula lag features con la MISMA función que en el entrenamiento (construir_lag_features)
    y devuelve el vector de features de esa última fila, listo para pipeline.predict().

    Devuelve (vector_dict, cobertura_suficiente): cobertura_suficiente es False si el
    histórico disponible no alcanza para llenar todos los lags (ej. muy al inicio de la
    serie de una ciudad nueva) -- el pipeline igual puede predecir (el imputer entrena con
    la mediana de referencia), pero se marca para dejarlo trazable.
    """
    df_feat, lag_cols = construir_lag_features(df_ventana)
    ultima = df_feat.sort_values('week_start_date').iloc[-1]

    columnas_numericas, columnas_categoricas = columnas_modelo()
    vector = {c: (None if pd.isna(ultima.get(c)) else float(ultima[c])) for c in columnas_numericas}
    vector['city'] = ultima['city']

    cobertura_suficiente = not any(vector[c] is None for c in lag_cols)
    return vector, cobertura_suficiente
