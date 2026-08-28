# MLOps Local con Monitoreo de Data Drift — DengAI

**Práctico Final — Magíster en Ciencia de Datos, Tópicos en Data Science II**
**Integrantes:** Gerardo Galán, Mabel Herrera

Sistema local (sin Docker, sin cloud) que predice casos semanales de dengue en San Juan
(Puerto Rico) e Iquitos (Perú), los sirve vía una API de inferencia, y **monitorea cuándo el
modelo empieza a fallar por drift climático/epidemiológico** — con un plan de acción y un
gatillo de reentrenamiento implementados, no solo descritos.

## Estructura del repositorio

```
dengue_mlops_drift/
├── data/
│   ├── raw/                 4 CSV originales de DengAI (Kaggle: arashnic/epidemy)
│   └── processed/           referencia.csv (train) y produccion_simulada.csv (holdout con etiquetas)
├── notebooks/                un notebook ejecutado por fase (evidencia completa: código + outputs + gráficos)
│   ├── practico_06_proyecto_final_f0_f1_eda_Gerardo_Galan_Mabel_Herrera.ipynb
│   ├── practico_06_proyecto_final_fase2_modelado_Gerardo_Galan_Mabel_Herrera.ipynb
│   └── practico_06_proyecto_final_fase4_drift_Gerardo_Galan_Mabel_Herrera.ipynb
├── src/                       código compartido (evita reimplementar lógica en cada fase)
│   ├── data.py                 carga + split temporal referencia/producción
│   ├── features.py              lag features sin fuga + auditoría automática
│   ├── modelo.py                 construcción/entrenamiento del pipeline
│   ├── drift.py                   PSI, KS-test, ventanas temporales
│   └── inferencia.py               lógica de predicción compartida por la API y el notebook de Fase 4
├── service/
│   ├── app.py                  servicio de inferencia FastAPI (Fase 3)
│   ├── store.py                  histórico + registro de predicciones (SQLite)
│   ├── simular_produccion.py       alimenta el servicio vía HTTP con produccion_simulada.csv
│   └── reentrenar.py               gatillo de reentrenamiento + rollback (Fase 5)
├── dashboard/
│   └── dashboard.py            dashboard Streamlit de monitoreo (Fase 4)
├── models/
│   ├── modelo_dengue.joblib      pipeline entrenado (preprocesamiento + modelo)
│   └── metadata.json               metadata del modelo vigente
├── reports/                    salidas del notebook de Fase 4 (insumo del dashboard)
├── logs/                       dengue_service.db (SQLite: histórico + predicciones)
├── PLAN_DE_ACCION.md           Fase 5: qué hace el sistema ante cada nivel de alerta
├── INFORME_EJECUTIVO_template.md  plantilla para el informe ejecutivo (a completar por el equipo)
└── requirements.txt
```

## Instalación y ejecución (< 10 minutos)

```bash
# 1) Entorno
python3 -m venv venv && source venv/bin/activate   # opcional pero recomendado
pip install -r requirements.txt

# 2) (Ya viene entrenado en el repo) Si quieren reentrenar desde cero, correr en orden:
#    jupyter nbconvert --to notebook --execute notebooks/practico_06_proyecto_final_f0_f1_eda_Gerardo_Galan_Mabel_Herrera.ipynb
#    jupyter nbconvert --to notebook --execute notebooks/practico_06_proyecto_final_fase2_modelado_Gerardo_Galan_Mabel_Herrera.ipynb
#    jupyter nbconvert --to notebook --execute notebooks/practico_06_proyecto_final_fase4_drift_Gerardo_Galan_Mabel_Herrera.ipynb
#    (el paso de Fase 4 es el que genera reports/, necesario para el dashboard)

# 3) Levantar el servicio de inferencia
cd service
uvicorn app:app --port 8000
# probar:  curl http://127.0.0.1:8000/health

# 4) (en otra terminal) Alimentar el servicio con la producción simulada
cd service
python simular_produccion.py

# 5) (en otra terminal) Levantar el dashboard de monitoreo
cd dashboard
streamlit run dashboard.py
# abre http://localhost:8501
```

Los pasos 3-5 son la demo en vivo para la defensa oral: el servicio prediciendo, el
dashboard mostrando drift y alertas en tiempo real a medida que se alimenta.

### Reentrenar / revertir (Fase 5)

```bash
cd service
python reentrenar.py --city sj --ventanas V2 V3    # incorpora esas ventanas y reentrena
python reentrenar.py --rollback                     # revierte a la versión anterior
```

## Arquitectura (resumen)

```
Kaggle DengAI (arashnic/epidemy)
        │
        ▼
Fase 1: EDA + detección de fuga de datos potencial  ──►  notebooks/..._f0_f1_eda...ipynb
        │
        ▼
Fase 2: split referencia/producción + features de rezago + modelo (RandomForest + log1p)
        │  sin fuga: TimeSeriesSplit, imputación aislada por fold, lag shift(+k) auditado
        ▼
   models/modelo_dengue.joblib  ◄────────────┐
        │                                     │ reentrenar.py (Fase 5, semi-automático)
        ▼                                     │
Fase 3: service/app.py (FastAPI)              │
   POST /ingest → predice + registra ─────────┘
        │
        ▼
   logs/dengue_service.db (histórico + predicciones)
        │
        ▼
Fase 4: notebook de drift → PSI/KS por ventana + MAE por ventana → reports/*.csv
        │
        ▼
   dashboard/dashboard.py (Streamlit)  +  PLAN_DE_ACCION.md (Fase 5)
```

## Limitaciones conocidas (documentadas, no escondidas)

- El modelo no usa `reanalysis_sat_precip_amt_mm` (quedó fuera en la Fase 2 por un criterio
  de selección de features que no la incluyó explícitamente; no afecta la validez del
  pipeline, es una variable menos de las 20 disponibles).
- El servicio asume que las variables de reanálisis climático (NOAA) están disponibles al
  momento de invocar `/ingest`, aunque en la práctica esos productos se publican con
  retraso — ver Fase 1, riesgo de fuga #4, y la nota de diseño en `service/app.py`.
- Los umbrales de alerta (PSI > 0.2, ≥2 features con drift, MAE 30% sobre referencia) están
  justificados con el análisis de la Fase 4, pero no fueron calibrados con un experimento de
  A/A testing (comparar dos ventanas sin drift real para estimar la tasa de falsos positivos)
  por el alcance de tiempo del práctico — se documenta como mejora futura.

## Ejecución en Google Colab (versión alternativa)

Además de la ejecución local descrita arriba, el proyecto completo también se puede correr en Google Colab (a pedido del profesor).

Los notebooks de las Fases 0-1, 2 y 4 ya traen una celda de **"Bootstrap para Google Colab"** justo al inicio, que trae el repositorio al entorno (por URL de GitHub o subiendo un `.zip`) y ajusta el directorio de trabajo. El resto de cada notebook no cambió nada -- las mismas celdas que corren localmente corren igual en Colab.

Para la Fase 3 (servicio de inferencia), la Fase 5 (reentrenamiento y rollback) y el dashboard, usar:

```
notebooks/practico_06_proyecto_final_fase3_fase5_colab_Gerardo_Galan_Mabel_Herrera.ipynb
```

Ese notebook levanta `service/app.py` en un hilo de fondo (en vez de una terminal aparte), lo alimenta con `service/simular_produccion.py`, prueba `service/reentrenar.py` (reentrenamiento + rollback), y muestra `dashboard/dashboard.py` usando el proxy de puertos integrado de Colab (`google.colab.output`) -- sin cuentas ni tokens externos. Como alternativa comentada, si se necesita un link público real (para alguien que no tiene esa sesión de Colab abierta), el notebook deja lista la opción de `localtunnel`, que tampoco requiere token.

**Diferencias respecto a la ejecución local:**
- Cada sesión de Colab es efímera: lo que se reentrena o registra durante la sesión no persiste al cerrar el entorno de ejecución (decisión deliberada, para mantenerlo simple -- ver la nota dentro del notebook).
- El dashboard se ve a través del proxy de puertos integrado de Colab, por lo que solo es visible para quien tiene esa sesión de Colab abierta (no genera un link público salvo que se use la alternativa de `localtunnel`).
- Para traer el repositorio a Colab hace falta una URL de GitHub (variable `REPO_URL` en la celda de bootstrap) o subir un `.zip` del proyecto cuando la celda lo pida.

## Dataset

DengAI — predicción de casos de dengue, San Juan (Puerto Rico) e Iquitos (Perú).
Fuente: [Kaggle `arashnic/epidemy`](https://www.kaggle.com/datasets/arashnic/epidemy)
(originalmente de la competencia DrivenData "DengAI: Predicting Disease Spread").
