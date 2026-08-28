"""
Gatillo de reentrenamiento — Fase 5.

Semi-automático a propósito (ver enunciado): NO se dispara solo cuando llega una alerta;
el equipo lo ejecuta a mano al recibir una alerta `REENTRENAR` del dashboard/Fase 4. Esto
es una decisión de diseño, no una limitación de tiempo: un reentrenamiento 100% automático
sin revisión humana es riesgoso quando el propio umbral de alerta puede ser ruidoso con
ventanas chicas (ver Fase 4, Paso 3) -- mejor que una persona confirme antes de promover un
modelo nuevo a producción.

Uso:
    cd service
    python reentrenar.py --city sj --ventanas V2 V3        # incorpora esas ventanas al set de referencia
    python reentrenar.py --rollback                        # revierte al modelo anterior

Qué hace (cuando NO es --rollback):
1. Carga referencia.csv + produccion_simulada.csv (con etiquetas verdaderas).
2. Agrega las ventanas indicadas (que ya tienen etiqueta real, porque produccion_simulada
   viene del propio dataset histórico, no de una predicción) al set de referencia.
3. Reentrena el pipeline completo (src.modelo, misma arquitectura que la Fase 2) sobre el
   set de referencia ampliado, con la misma validación temporal (TimeSeriesSplit).
4. Guardrail: solo PROMUEVE el modelo nuevo si su MAE de validación no empeora más de un
   10% respecto al modelo vigente. Si empeora más que eso, lo deja en `models/candidatos/`
   para revisión manual y NO reemplaza el modelo en producción.
5. Si promueve, hace backup versionado del modelo anterior en `models/archivo/` (con eso
   `--rollback` puede revertir) antes de sobrescribir `models/modelo_dengue.joblib`.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd

from src.drift import asignar_ventanas
from src.features import preparar_dataset_modelado
from src.modelo import entrenar_y_evaluar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
ARCHIVO_DIR = os.path.join(MODELS_DIR, "archivo")
CANDIDATOS_DIR = os.path.join(MODELS_DIR, "candidatos")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_dengue.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
HISTORIAL_PATH = os.path.join(MODELS_DIR, "historial_versiones.json")
REFERENCIA_CSV = os.path.join(BASE_DIR, "..", "data", "processed", "referencia.csv")
PRODUCCION_CSV = os.path.join(BASE_DIR, "..", "data", "processed", "produccion_simulada.csv")

UMBRAL_EMPEORAMIENTO_ACEPTABLE = 1.10  # el modelo nuevo puede ser hasta 10% peor y aun así promoverse


def _leer_historial():
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH) as f:
            return json.load(f)
    return []


def _guardar_historial(historial):
    with open(HISTORIAL_PATH, "w") as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)


def reentrenar(city, ventanas):
    os.makedirs(ARCHIVO_DIR, exist_ok=True)
    os.makedirs(CANDIDATOS_DIR, exist_ok=True)

    with open(METADATA_PATH) as f:
        metadata_actual = json.load(f)
    mae_actual = metadata_actual["mae_validacion_timeseriessplit"]

    df_ref = pd.read_csv(REFERENCIA_CSV, parse_dates=["week_start_date"])
    df_prod = pd.read_csv(PRODUCCION_CSV, parse_dates=["week_start_date"])

    df_prod_city = asignar_ventanas(df_prod[df_prod.city == city], n_ventanas=4)
    incorporar = df_prod_city[df_prod_city["ventana"].isin(ventanas)].drop(columns=["ventana"])
    if incorporar.empty:
        raise ValueError(f"No hay filas para city={city}, ventanas={ventanas}")

    print(f"Incorporando {len(incorporar)} semanas de {city.upper()} (ventanas {ventanas}) al set de referencia "
          f"({incorporar['week_start_date'].min().date()} .. {incorporar['week_start_date'].max().date()})")

    df_ref_ampliado = pd.concat([df_ref, incorporar]).sort_values(["city", "week_start_date"]).reset_index(drop=True)

    X, y, fechas, ciudades, num_cols, cat_cols, info = preparar_dataset_modelado(df_ref_ampliado)
    modelo_nuevo, mae_nuevo = entrenar_y_evaluar(X, y, num_cols, cat_cols)

    print(f"MAE modelo vigente (referencia original): {mae_actual:.2f}")
    print(f"MAE modelo candidato (referencia + {ventanas}): {mae_nuevo:.2f}")

    promovido = mae_nuevo <= mae_actual * UMBRAL_EMPEORAMIENTO_ACEPTABLE
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    metadata_nuevo = dict(metadata_actual)
    metadata_nuevo.update({
        "fecha_entrenamiento_referencia": str(df_ref_ampliado["week_start_date"].max().date()),
        "columnas_numericas": num_cols,
        "columnas_categoricas": cat_cols,
        "mae_validacion_timeseriessplit": mae_nuevo,
        "reentrenado_en": timestamp,
        "ventanas_incorporadas": {"city": city, "ventanas": ventanas},
        "version": metadata_actual.get("version", 1) + 1,
    })

    if promovido:
        # backup versionado del modelo vigente (permite --rollback)
        version_anterior = metadata_actual.get("version", 1)
        shutil.copy(MODEL_PATH, os.path.join(ARCHIVO_DIR, f"modelo_dengue_v{version_anterior}.joblib"))
        with open(os.path.join(ARCHIVO_DIR, f"metadata_v{version_anterior}.json"), "w") as f:
            json.dump(metadata_actual, f, indent=2, ensure_ascii=False)

        joblib.dump(modelo_nuevo, MODEL_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata_nuevo, f, indent=2, ensure_ascii=False)

        historial = _leer_historial()
        historial.append({"version": metadata_nuevo["version"], "timestamp": timestamp,
                           "mae": mae_nuevo, "promovido": True, "motivo": f"reentrenado con {city}/{ventanas}"})
        _guardar_historial(historial)
        print(f"\nPROMOVIDO: nuevo modelo (v{metadata_nuevo['version']}) reemplaza a models/modelo_dengue.joblib. "
              f"Versión anterior respaldada en models/archivo/modelo_dengue_v{version_anterior}.joblib")
    else:
        candidato_path = os.path.join(CANDIDATOS_DIR, f"modelo_dengue_candidato_{timestamp}.joblib")
        joblib.dump(modelo_nuevo, candidato_path)
        with open(os.path.join(CANDIDATOS_DIR, f"metadata_candidato_{timestamp}.json"), "w") as f:
            json.dump(metadata_nuevo, f, indent=2, ensure_ascii=False)
        print(f"\nNO PROMOVIDO: el modelo candidato empeora el MAE más de {(UMBRAL_EMPEORAMIENTO_ACEPTABLE-1)*100:.0f}%. "
              f"Se guardó en {candidato_path} para revisión manual del equipo. El modelo en producción NO cambió.")

    return promovido, mae_actual, mae_nuevo


def rollback():
    versiones = sorted(
        [f for f in os.listdir(ARCHIVO_DIR) if f.startswith("modelo_dengue_v") and f.endswith(".joblib")],
        key=lambda f: int(f.split("_v")[-1].replace(".joblib", "")),
    ) if os.path.exists(ARCHIVO_DIR) else []

    if not versiones:
        print("No hay versiones respaldadas en models/archivo/ -- nada que revertir.")
        return

    ultima = versiones[-1]
    version_num = ultima.split("_v")[-1].replace(".joblib", "")
    print(f"Revirtiendo a {ultima} ...")

    shutil.copy(os.path.join(ARCHIVO_DIR, ultima), MODEL_PATH)
    shutil.copy(os.path.join(ARCHIVO_DIR, f"metadata_v{version_num}.json"), METADATA_PATH)

    historial = _leer_historial()
    historial.append({"version": f"rollback_a_v{version_num}",
                       "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                       "promovido": True, "motivo": "rollback manual"})
    _guardar_historial(historial)
    print(f"Listo. models/modelo_dengue.joblib ahora es la versión {version_num}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=["sj", "iq"])
    parser.add_argument("--ventanas", nargs="+", choices=["V1", "V2", "V3", "V4"])
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        if not args.city or not args.ventanas:
            parser.error("Se requiere --city y --ventanas (o usar --rollback)")
        reentrenar(args.city, args.ventanas)
