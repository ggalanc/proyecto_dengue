"""
Construcción y entrenamiento del pipeline de modelado — compartido entre el notebook de
la Fase 2 (entrenamiento inicial) y `service/reentrenar.py` (Fase 5, gatillo de
reentrenamiento), para que ambos entrenen exactamente de la misma forma.
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

RANDOM_STATE = 42


def construir_pipeline(num_c, cat_c, n_estimators=300, log_target=True):
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), num_c),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_c),
    ])
    modelo = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([('pre', pre), ('modelo', modelo)])
    if log_target:
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def entrenar_y_evaluar(X, y, num_c, cat_c, n_splits=5):
    """Entrena el pipeline final sobre TODO (X, y) y devuelve (pipeline_entrenado, mae_validacion_cv)."""
    pipe_cv = construir_pipeline(num_c, cat_c)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_cv = -cross_val_score(pipe_cv, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

    pipe_final = construir_pipeline(num_c, cat_c)
    pipe_final.fit(X, y)
    return pipe_final, float(mae_cv)
