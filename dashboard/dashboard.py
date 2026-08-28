"""
Dashboard de monitoreo — Fase 4.

Arranca con:  cd dashboard && streamlit run dashboard.py

Lee directamente los artefactos que dejaron los notebooks de las Fases 2-4 (no necesita
que el servicio FastAPI esté corriendo):
- models/metadata.json           -> info del modelo entrenado
- reports/drift_detalle.csv      -> PSI/KS por feature x ventana x ciudad
- reports/mae_por_ventana.csv    -> desempeño por ventana x ciudad
- reports/alertas_por_ventana.csv-> resumen de alertas (Paso 5 de la Fase 4)
- logs/dengue_service.db         -> log de predicciones individuales (tabla `predicciones`)
- data/processed/produccion_simulada.csv -> etiquetas verdaderas, para el gráfico predicho vs real
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from src.drift import UMBRAL_PSI_SIGNIFICATIVO, UMBRAL_PSI_MODERADO

MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "reports")
DB_PATH = os.path.join(BASE_DIR, "..", "logs", "dengue_service.db")
PROD_CSV = os.path.join(BASE_DIR, "..", "data", "processed", "produccion_simulada.csv")

st.set_page_config(page_title="Monitoreo de Drift — DengAI", layout="wide")


@st.cache_data
def cargar_datos():
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    faltantes = []
    for nombre in ["drift_detalle.csv", "mae_por_ventana.csv", "alertas_por_ventana.csv"]:
        if not os.path.exists(os.path.join(REPORTS_DIR, nombre)):
            faltantes.append(nombre)
    if faltantes:
        return metadata, None, None, None, None, faltantes

    drift = pd.read_csv(os.path.join(REPORTS_DIR, "drift_detalle.csv"))
    mae = pd.read_csv(os.path.join(REPORTS_DIR, "mae_por_ventana.csv"))
    alertas = pd.read_csv(os.path.join(REPORTS_DIR, "alertas_por_ventana.csv"))

    preds = pd.DataFrame()
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        preds = pd.read_sql("SELECT * FROM predicciones ORDER BY week_start_date", conn)
        conn.close()
        if len(preds):
            preds["week_start_date"] = pd.to_datetime(preds["week_start_date"])

    prod_real = pd.DataFrame()
    if os.path.exists(PROD_CSV):
        prod_real = pd.read_csv(PROD_CSV, parse_dates=["week_start_date"])[["city", "week_start_date", "total_cases"]]

    return metadata, drift, mae, alertas, (preds, prod_real), []


metadata, drift, mae, alertas, extra, faltantes = cargar_datos()

st.title("🦟 Monitoreo de Drift — DengAI (San Juan / Iquitos)")

if faltantes:
    st.error(
        "Faltan archivos que genera el notebook de la Fase 4 (`notebooks/practico_06_proyecto_final_"
        f"fase4_drift_Gerardo_Galan_Mabel_Herrera.ipynb`): {', '.join(faltantes)}. "
        "Corre ese notebook completo primero — este dashboard solo LEE sus resultados, no los recalcula."
    )
    st.stop()

preds, prod_real = extra

col1, col2, col3 = st.columns(3)
col1.metric("Modelo entrenado hasta", metadata["fecha_entrenamiento_referencia"])
col2.metric("MAE de validación (referencia)", f"{metadata['mae_validacion_timeseriessplit']:.2f}")
col3.metric("Predicciones registradas", len(preds) if len(preds) else 0)

st.sidebar.header("Filtros")
ciudad = st.sidebar.selectbox("Ciudad", ["sj", "iq"], format_func=lambda c: "San Juan" if c == "sj" else "Iquitos")

st.markdown("---")

# ── Alertas ──
st.subheader("🚨 Alertas por ventana")
alertas_c = alertas[alertas.city == ciudad].sort_values("ventana")


def color_accion(val):
    color = {"OK": "#2ecc71", "REVISAR": "#f39c12", "REENTRENAR": "#e74c3c"}.get(val, "white")
    return f"background-color: {color}; color: white; font-weight: bold"


try:
    estilo = alertas_c.style.map(color_accion, subset=["accion_sugerida"])  # pandas >= 2.1
except AttributeError:
    estilo = alertas_c.style.applymap(color_accion, subset=["accion_sugerida"])  # pandas < 2.1

st.dataframe(estilo, width='stretch', hide_index=True)

st.markdown("---")

# ── Desempeño ──
st.subheader("📈 Desempeño del modelo (MAE) por ventana")
mae_c = mae[mae.city == ciudad].sort_values("ventana")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mae_c["ventana"], mae_c["MAE"], marker="o", linewidth=2, color="#e74c3c", label="MAE en producción")
ax.axhline(metadata["mae_validacion_timeseriessplit"], color="gray", linestyle="--",
           label="MAE de validación (referencia)")
ax.set_ylabel("MAE")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# ── Drift por feature ──
st.subheader("🌡️ Drift por variable (PSI)")
st.caption(
    f"PSI < {UMBRAL_PSI_MODERADO} = sin drift relevante · "
    f"{UMBRAL_PSI_MODERADO}-{UMBRAL_PSI_SIGNIFICATIVO} = drift moderado (monitorear) · "
    f"> {UMBRAL_PSI_SIGNIFICATIVO} = drift significativo"
)

drift_c = drift[drift.city == ciudad]
piv = drift_c.pivot(index="feature", columns="ventana", values="psi").reindex(columns=["V1", "V2", "V3", "V4"])

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn_r", center=UMBRAL_PSI_SIGNIFICATIVO,
            vmin=0, vmax=max(1.0, float(piv.values.max())), ax=ax2, cbar_kws={"label": "PSI"})
st.pyplot(fig2)

st.markdown("---")

# ── Predicho vs. real ──
st.subheader("🔍 Predicho vs. real")
if len(preds):
    p_c = preds[preds.city == ciudad].sort_values("week_start_date")
    m = p_c.merge(prod_real[prod_real.city == ciudad], on=["city", "week_start_date"], how="left")

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(m["week_start_date"], m["total_cases"], label="Real", color="#2c3e50", linewidth=1.5)
    ax3.plot(m["week_start_date"], m["total_cases_predicho"], label="Predicho", color="#e74c3c",
              linewidth=1.5, linestyle="--")
    ax3.set_ylabel("Casos por semana")
    ax3.legend()
    st.pyplot(fig3)

    with st.expander("Ver registro de predicciones (auditoría)"):
        st.dataframe(m[["week_start_date", "total_cases", "total_cases_predicho",
                         "cobertura_historica_suficiente"]], width='stretch', hide_index=True)
else:
    st.info(
        "No hay predicciones registradas todavía. Corre el notebook de la Fase 4 "
        "(que alimenta el servicio con la producción simulada) o `service/simular_produccion.py` "
        "contra el servicio en vivo."
    )
