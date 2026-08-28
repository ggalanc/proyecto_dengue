"""
Métricas de drift propias (sin librerías externas de MLOps) — Fase 4.

- PSI (Population Stability Index): mide cuánto cambió la distribución de una variable
  numérica entre una ventana de referencia y una ventana "actual", usando deciles de la
  referencia como bins comunes. Regla de bolsillo estándar en la industria (documentada
  en el enunciado del práctico): PSI < 0.1 sin drift relevante, 0.1-0.2 drift moderado
  (monitorear), > 0.2 drift significativo (justifica alerta / revisión del modelo).
- KS-test (Kolmogorov-Smirnov de 2 muestras): complementa el PSI con un test de hipótesis
  formal (H0: misma distribución); p-valor < 0.05 se interpreta como evidencia de drift.

Se usan AMBAS métricas a propósito: el PSI es más interpretable en magnitud (un número
comparable entre features y ventanas) pero es sensible al binning elegido; el KS-test no
depende de bins pero su p-valor es sensible al tamaño de muestra (con ventanas grandes,
diferencias mínimas pueden dar p-valor significativo aunque el drift práctico sea chico).
Cruzarlas da una lectura más robusta que confiar en una sola.
"""
import numpy as np
import pandas as pd
from scipy import stats

UMBRAL_PSI_MODERADO = 0.1
UMBRAL_PSI_SIGNIFICATIVO = 0.2
UMBRAL_KS_PVALUE = 0.05


def calcular_psi(referencia, actual, bins=10):
    """PSI de una variable numérica entre dos muestras, usando los deciles de `referencia`
    como límites de bin (bins comunes para ambas distribuciones)."""
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()
    if len(referencia) < bins or len(actual) == 0:
        return np.nan

    cuantiles = np.linspace(0, 1, bins + 1)
    limites = np.unique(referencia.quantile(cuantiles).values)
    if len(limites) < 3:  # variable casi constante en la referencia -> PSI no es informativo
        return np.nan
    limites[0], limites[-1] = -np.inf, np.inf

    ref_bins = pd.cut(referencia, bins=limites)
    act_bins = pd.cut(actual, bins=limites)

    ref_pct = ref_bins.value_counts(normalize=True, sort=False)
    act_pct = act_bins.value_counts(normalize=True, sort=False)

    epsilon = 1e-4  # evita log(0) / división por 0 en bins vacíos
    ref_pct = ref_pct.reindex(ref_pct.index, fill_value=0) + epsilon
    act_pct = act_pct.reindex(ref_pct.index, fill_value=0) + epsilon

    psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return float(psi)


def calcular_ks(referencia, actual):
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan, np.nan
    resultado = stats.ks_2samp(referencia, actual)
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_drift_features(df_referencia, df_ventana, columnas, bins=10):
    """Calcula PSI y KS para cada columna en `columnas`, comparando df_referencia vs df_ventana.
    Devuelve un DataFrame con una fila por feature."""
    filas = []
    for col in columnas:
        psi = calcular_psi(df_referencia[col], df_ventana[col], bins=bins)
        ks_stat, ks_p = calcular_ks(df_referencia[col], df_ventana[col])
        filas.append({
            'feature': col,
            'psi': psi,
            'drift_psi': (psi is not None) and (not np.isnan(psi)) and psi > UMBRAL_PSI_SIGNIFICATIVO,
            'psi_moderado': (psi is not None) and (not np.isnan(psi)) and UMBRAL_PSI_MODERADO < psi <= UMBRAL_PSI_SIGNIFICATIVO,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'drift_ks': (ks_p is not None) and (not np.isnan(ks_p)) and ks_p < UMBRAL_KS_PVALUE,
        })
    return pd.DataFrame(filas)


def asignar_ventanas(df, n_ventanas=4, col_fecha='week_start_date'):
    """Divide un dataframe en n_ventanas secuenciales de ancho de tiempo IGUAL (no de igual
    cantidad de filas) según el rango de fechas de TODO el df -- así una ventana con pocos
    datos (ej. semanas con más nulos) sigue representando el mismo tramo de tiempo."""
    df = df.copy()
    fechas = pd.to_datetime(df[col_fecha])
    inicio, fin = fechas.min(), fechas.max()
    bordes = pd.date_range(inicio, fin, periods=n_ventanas + 1)
    etiquetas = [f"V{i+1}" for i in range(n_ventanas)]
    df['ventana'] = pd.cut(fechas, bins=bordes, labels=etiquetas, include_lowest=True)
    return df
