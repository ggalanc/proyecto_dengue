"""
Servicio de inferencia local — Fase 3.

Arranca con:  cd service && uvicorn app:app --reload --port 8000

Endpoints:
- POST /ingest        -> recibe UNA observación semanal cruda de una ciudad, calcula sus
                          lag features a partir del histórico acumulado (misma lógica que
                          el entrenamiento), predice total_cases y registra la predicción.
- GET  /predicciones   -> devuelve el log de predicciones (insumo de la Fase 4).
- GET  /historial/{city} -> histórico crudo acumulado de una ciudad (debug).
- GET  /health         -> estado del servicio + metadata del modelo cargado.
- POST /reiniciar      -> (solo demo/testing) limpia predicciones e historial, y resiembra
                          desde data/processed/referencia.csv.

Nota de diseño (ligado a la Fase 1, riesgo de fuga #4): el modelo se entrenó con variables
de reanálisis climático (NOAA) que en la práctica se publican con retraso respecto a la
fecha que describen. Este servicio asume que el llamador ya tiene esos valores disponibles
al momento de invocar /ingest (igual que el propio dataset de entrenamiento los presenta
alineados por semana) -- es una simplificación explícita para el alcance de este práctico,
documentada aquí y en el README para no pasarla por alto.
"""
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inferencia import cargar_modelo_y_metadata, procesar_observacion
from service import store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_dengue.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "..", "models", "metadata.json")
REFERENCIA_CSV = os.path.join(BASE_DIR, "..", "data", "processed", "referencia.csv")
DB_PATH = os.path.join(BASE_DIR, "..", "logs", "dengue_service.db")

app = FastAPI(title="Servicio de Inferencia — DengAI MLOps", version="1.0")

_modelo = None
_metadata = None


class ObservacionSemanal(BaseModel):
    city: str
    year: int
    weekofyear: int
    week_start_date: str  # 'YYYY-MM-DD'
    ndvi_ne: Optional[float] = None
    ndvi_nw: Optional[float] = None
    ndvi_se: Optional[float] = None
    ndvi_sw: Optional[float] = None
    precipitation_amt_mm: Optional[float] = None
    reanalysis_air_temp_k: Optional[float] = None
    reanalysis_avg_temp_k: Optional[float] = None
    reanalysis_dew_point_temp_k: Optional[float] = None
    reanalysis_max_air_temp_k: Optional[float] = None
    reanalysis_min_air_temp_k: Optional[float] = None
    reanalysis_precip_amt_kg_per_m2: Optional[float] = None
    reanalysis_relative_humidity_percent: Optional[float] = None
    reanalysis_sat_precip_amt_mm: Optional[float] = None
    reanalysis_specific_humidity_g_per_kg: Optional[float] = None
    reanalysis_tdtr_k: Optional[float] = None
    station_avg_temp_c: Optional[float] = None
    station_diur_temp_rng_c: Optional[float] = None
    station_max_temp_c: Optional[float] = None
    station_min_temp_c: Optional[float] = None
    station_precip_mm: Optional[float] = None


@app.on_event("startup")
def startup():
    global _modelo, _metadata
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    store.init_db(DB_PATH)
    if store.historial_vacio(DB_PATH):
        store.sembrar_historial_desde_referencia(REFERENCIA_CSV, DB_PATH, n_semanas=16)

    _modelo, _metadata = cargar_modelo_y_metadata(MODEL_PATH, METADATA_PATH)
    print(f"Modelo cargado (entrenado hasta {_metadata['fecha_entrenamiento_referencia']}), "
          f"MAE validación={_metadata['mae_validacion_timeseriessplit']:.2f}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_entrenado_hasta": _metadata["fecha_entrenamiento_referencia"],
        "mae_validacion_timeseriessplit": _metadata["mae_validacion_timeseriessplit"],
        "n_predicciones_registradas": len(store.obtener_predicciones(DB_PATH)),
    }


@app.post("/ingest")
def ingest(obs: ObservacionSemanal):
    if obs.city not in ("sj", "iq"):
        raise HTTPException(400, f"Ciudad desconocida: {obs.city} (se esperaba 'sj' o 'iq')")

    registro = procesar_observacion(obs.model_dump(), _modelo, _metadata, DB_PATH)

    if not registro["cobertura_historica_suficiente"]:
        registro["advertencia"] = (
            "Histórico insuficiente para completar todos los lags; el pipeline imputó "
            "los faltantes con la mediana de referencia (comportamiento esperado del "
            "SimpleImputer, no un error, pero se marca para trazabilidad)."
        )
    return registro


@app.get("/predicciones")
def predicciones(city: Optional[str] = None):
    df = store.obtener_predicciones(DB_PATH, city=city)
    return json.loads(df.to_json(orient="records"))


@app.get("/historial/{city}")
def historial(city: str):
    df = store.obtener_ventana_historial(city, "2100-01-01", DB_PATH, n_semanas=10_000)
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.post("/reiniciar")
def reiniciar():
    """SOLO para demo/testing: borra el histórico y las predicciones, y resiembra desde referencia.csv."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    store.init_db(DB_PATH)
    store.sembrar_historial_desde_referencia(REFERENCIA_CSV, DB_PATH, n_semanas=16)
    return {"status": "reiniciado"}
