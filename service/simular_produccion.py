"""
Simula la llegada de datos "en producción": lee data/processed/produccion_simulada.csv
(el 20% más reciente por ciudad, reservado en la Fase 2 y NUNCA visto en entrenamiento) y
envía cada semana, en orden cronológico, al endpoint POST /ingest del servicio de la Fase 3.

Uso:
    # 1) en una terminal: levantar el servicio
    cd service && uvicorn app:app --port 8000

    # 2) en otra terminal: alimentarlo con la producción simulada
    cd service && python simular_produccion.py

Este script es el puente entre la Fase 3 (servicio) y la Fase 4 (monitoreo de drift): el
log de predicciones que genera es exactamente el insumo que la Fase 4 usa para calcular
métricas de drift y de desempeño por ventana temporal.
"""
import argparse
import sys
import time

import httpx
import pandas as pd

RAW_COLS = [
    'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
    'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c',
    'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm',
]


def main(csv_path, base_url, pausa_seg):
    df = pd.read_csv(csv_path, parse_dates=['week_start_date']).sort_values('week_start_date')
    print(f"Alimentando {len(df)} semanas de producción simulada a {base_url}/ingest ...")

    ok, fallidas = 0, 0
    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{base_url}/health")
        r.raise_for_status()
        print("Servicio activo:", r.json())

        for _, row in df.iterrows():
            payload = {
                'city': row['city'], 'year': int(row['year']), 'weekofyear': int(row['weekofyear']),
                'week_start_date': row['week_start_date'].strftime('%Y-%m-%d'),
            }
            for c in RAW_COLS:
                v = row.get(c)
                payload[c] = None if pd.isna(v) else float(v)

            resp = client.post(f"{base_url}/ingest", json=payload)
            if resp.status_code == 200:
                ok += 1
                r = resp.json()
                print(f"  {r['city']} {r['week_start_date']}: predicho={r['total_cases_predicho']:.1f} "
                      f"(real={row['total_cases']:.0f})")
            else:
                fallidas += 1
                print(f"  ERROR en {row['city']} {row['week_start_date']}: {resp.status_code} {resp.text[:200]}")
            if pausa_seg:
                time.sleep(pausa_seg)

    print(f"\nListo. {ok} predicciones registradas, {fallidas} fallidas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../data/processed/produccion_simulada.csv")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--pausa", type=float, default=0.0, help="segundos de pausa entre requests (streaming lento)")
    args = parser.parse_args()
    main(args.csv, args.url, args.pausa)
