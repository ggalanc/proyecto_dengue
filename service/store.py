"""
Almacenamiento local (SQLite) del servicio de inferencia — Fase 3.

Dos tablas:
- `historial`: observaciones climáticas crudas recibidas semana a semana (una fila por
  ciudad-semana). Es la fuente para calcular las lag features en el momento de predecir,
  usando EXACTAMENTE la misma función que en el entrenamiento (src.features), para que
  entrenamiento e inferencia nunca calculen features de forma distinta.
- `predicciones`: registro de cada predicción hecha por el servicio, junto con los
  inputs (features) efectivamente usados — trazabilidad para auditoría y para la Fase 4
  (el módulo de drift lee esta tabla).
"""
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src.features import RAW_FEATURE_COLS

DB_PATH_DEFAULT = "../logs/dengue_service.db"


def get_conn(db_path=DB_PATH_DEFAULT):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH_DEFAULT):
    conn = get_conn(db_path)
    cols_sql = ", ".join(f'"{c}" REAL' for c in RAW_FEATURE_COLS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS historial (
            city TEXT NOT NULL,
            year INTEGER NOT NULL,
            weekofyear INTEGER NOT NULL,
            week_start_date TEXT NOT NULL,
            {cols_sql},
            PRIMARY KEY (city, week_start_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            city TEXT NOT NULL,
            year INTEGER NOT NULL,
            weekofyear INTEGER NOT NULL,
            week_start_date TEXT NOT NULL,
            total_cases_predicho REAL NOT NULL,
            cobertura_historica_suficiente INTEGER NOT NULL,
            modelo_entrenado_hasta TEXT,
            features_usadas TEXT
        )
    """)
    conn.commit()
    conn.close()


def historial_vacio(db_path=DB_PATH_DEFAULT):
    conn = get_conn(db_path)
    n = conn.execute("SELECT COUNT(*) AS n FROM historial").fetchone()["n"]
    conn.close()
    return n == 0


def sembrar_historial_desde_referencia(referencia_csv, db_path=DB_PATH_DEFAULT, n_semanas=16):
    """
    Carga en `historial` las últimas n_semanas (por ciudad) del set de REFERENCIA
    (el mismo con el que se entrenó el modelo en la Fase 2). Así el servicio puede
    calcular lag features (hasta 12 semanas) desde la primera semana de "producción"
    que reciba, sin un vacío al arrancar.
    """
    df = pd.read_csv(referencia_csv, parse_dates=["week_start_date"])
    conn = get_conn(db_path)
    for city, sub in df.groupby("city"):
        sub = sub.sort_values("week_start_date").tail(n_semanas)
        _insertar_filas_historial(conn, sub)
    conn.commit()
    conn.close()


def _insertar_filas_historial(conn, df_filas):
    cols = ["city", "year", "weekofyear", "week_start_date"] + RAW_FEATURE_COLS
    placeholders = ", ".join(["?"] * len(cols))
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    for _, row in df_filas.iterrows():
        valores = [row["city"], int(row["year"]), int(row["weekofyear"]),
                   pd.Timestamp(row["week_start_date"]).strftime("%Y-%m-%d")]
        valores += [None if pd.isna(row[c]) else float(row[c]) for c in RAW_FEATURE_COLS]
        conn.execute(f"INSERT OR REPLACE INTO historial ({cols_sql}) VALUES ({placeholders})", valores)


def ingest_observacion(obs_dict, db_path=DB_PATH_DEFAULT):
    """Inserta (o reemplaza) una observación semanal cruda en `historial`."""
    conn = get_conn(db_path)
    fila = pd.DataFrame([obs_dict])
    fila["week_start_date"] = pd.to_datetime(fila["week_start_date"])
    _insertar_filas_historial(conn, fila)
    conn.commit()
    conn.close()


def obtener_ventana_historial(city, hasta_fecha, db_path=DB_PATH_DEFAULT, n_semanas=20):
    """Últimas n_semanas de historial de una ciudad, hasta (e incluyendo) hasta_fecha."""
    conn = get_conn(db_path)
    q = """
        SELECT * FROM historial
        WHERE city = ? AND week_start_date <= ?
        ORDER BY week_start_date DESC
        LIMIT ?
    """
    filas = conn.execute(q, (city, hasta_fecha, n_semanas)).fetchall()
    conn.close()
    df = pd.DataFrame([dict(f) for f in filas])
    if len(df):
        df["week_start_date"] = pd.to_datetime(df["week_start_date"])
        df = df.sort_values("week_start_date").reset_index(drop=True)
    return df


def registrar_prediccion(registro, db_path=DB_PATH_DEFAULT):
    conn = get_conn(db_path)
    conn.execute("""
        INSERT INTO predicciones
            (logged_at, city, year, weekofyear, week_start_date, total_cases_predicho,
             cobertura_historica_suficiente, modelo_entrenado_hasta, features_usadas)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        registro["city"], registro["year"], registro["weekofyear"], registro["week_start_date"],
        registro["total_cases_predicho"], int(registro["cobertura_historica_suficiente"]),
        registro.get("modelo_entrenado_hasta"), json.dumps(registro.get("features_usadas", {})),
    ))
    conn.commit()
    conn.close()


def obtener_predicciones(db_path=DB_PATH_DEFAULT, city=None):
    conn = get_conn(db_path)
    if city:
        filas = conn.execute("SELECT * FROM predicciones WHERE city = ? ORDER BY week_start_date",
                              (city,)).fetchall()
    else:
        filas = conn.execute("SELECT * FROM predicciones ORDER BY week_start_date").fetchall()
    conn.close()
    return pd.DataFrame([dict(f) for f in filas])
