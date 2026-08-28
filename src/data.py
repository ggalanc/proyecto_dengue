"""
Carga y partición temporal del dataset DengAI.

Reutilizado por: notebook de Fase 2 (modelado), servicio de inferencia (Fase 3)
y módulo de monitoreo de drift (Fase 4). Centralizar esto en un solo lugar evita
que cada fase reimplemente su propia lógica de carga/split y que se desincronicen.
"""
import os
import pandas as pd


def cargar_datos_dengue(data_dir):
    """Carga y mergea features + labels de entrenamiento. Devuelve también feat_test/sub_format."""
    archivos = ['dengue_features_train.csv', 'dengue_labels_train.csv',
                'dengue_features_test.csv', 'submission_format.csv']
    if not all(os.path.exists(os.path.join(data_dir, a)) for a in archivos):
        import kagglehub
        data_dir = kagglehub.dataset_download("arashnic/epidemy")

    feat_train = pd.read_csv(os.path.join(data_dir, 'dengue_features_train.csv'), parse_dates=['week_start_date'])
    lab_train = pd.read_csv(os.path.join(data_dir, 'dengue_labels_train.csv'))
    feat_test = pd.read_csv(os.path.join(data_dir, 'dengue_features_test.csv'), parse_dates=['week_start_date'])
    sub_format = pd.read_csv(os.path.join(data_dir, 'submission_format.csv'))

    df = feat_train.merge(lab_train, on=['city', 'year', 'weekofyear'], how='inner')
    df = df.sort_values(['city', 'week_start_date']).reset_index(drop=True)
    return df, feat_test, sub_format


def split_referencia_produccion(df, frac_referencia=0.8):
    """
    Split temporal por ciudad: el frac_referencia (ej. 80%) más antiguo de cada
    ciudad es 'referencia' (para entrenar/validar el modelo en Fase 2); el resto
    (más reciente) es 'producción simulada' — CON etiquetas verdaderas, pero que
    el modelo NUNCA ve en entrenamiento. Se usa en Fase 4 para medir drift y
    degradación de desempeño reales, sin necesidad de simular drift artificialmente.
    """
    refs, prods = [], []
    for city, sub in df.groupby('city'):
        sub = sub.sort_values('week_start_date')
        cutoff = sub['week_start_date'].quantile(frac_referencia)
        refs.append(sub[sub['week_start_date'] < cutoff])
        prods.append(sub[sub['week_start_date'] >= cutoff])
    df_ref = pd.concat(refs).sort_values(['city', 'week_start_date']).reset_index(drop=True)
    df_prod = pd.concat(prods).sort_values(['city', 'week_start_date']).reset_index(drop=True)
    return df_ref, df_prod
