"""
Lógica de inferencia compartida entre el servicio FastAPI (Fase 3, vía HTTP) y el
notebook de simulación/monitoreo de drift (Fase 4, llamado directamente en proceso,
sin necesidad de levantar el servidor para poder ejecutar el notebook de punta a punta
con un solo comando). Mantener esto en un solo lugar evita que la API y el notebook
calculen la predicción de formas ligeramente distintas.
"""
import json

import joblib
import pandas as pd

from src.features import construir_vector_prediccion
from service import store


def cargar_modelo_y_metadata(model_path, metadata_path):
    modelo = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return modelo, metadata


def procesar_observacion(obs_dict, modelo, metadata, db_path, n_semanas_ventana=20):
    """
    Dado el dict crudo de UNA observación semanal (city, year, weekofyear, week_start_date,
    + variables climáticas), lo guarda en el histórico, calcula sus lag features, predice
    y registra la predicción. Devuelve el registro completo (incluye la predicción).

    Es exactamente lo que hace POST /ingest del servicio (Fase 3); el notebook de Fase 4
    llama esta misma función para simular el streaming sin pasar por HTTP.
    """
    obs_dict = dict(obs_dict)
    week_start_date = pd.Timestamp(obs_dict['week_start_date']).strftime('%Y-%m-%d')
    obs_dict['week_start_date'] = week_start_date

    store.ingest_observacion(obs_dict, db_path)

    ventana = store.obtener_ventana_historial(obs_dict['city'], week_start_date, db_path,
                                               n_semanas=n_semanas_ventana)
    vector, cobertura_suficiente = construir_vector_prediccion(ventana)

    X_nueva = pd.DataFrame([vector])
    pred = max(0.0, float(modelo.predict(X_nueva)[0]))

    registro = {
        'city': obs_dict['city'], 'year': int(obs_dict['year']), 'weekofyear': int(obs_dict['weekofyear']),
        'week_start_date': week_start_date,
        'total_cases_predicho': round(pred, 2),
        'cobertura_historica_suficiente': cobertura_suficiente,
        'modelo_entrenado_hasta': metadata['fecha_entrenamiento_referencia'],
        'features_usadas': vector,
    }
    store.registrar_prediccion(registro, db_path)
    return registro
