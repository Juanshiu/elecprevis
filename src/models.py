"""
Entrenamiento y comparación de los cuatro modelos (Fase 4):
regresión lineal, random forest, XGBoost y LSTM.

Todos se evalúan con el mismo protocolo: split temporal (no aleatorio) y
las métricas MAE, RMSE, R² y sMAPE definidas en el Entregable 1.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_EXCLUDE = {"y"}  # todo lo demás en el df supervisado es feature


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return float(100 * np.mean(2 * np.abs(y_true - y_pred) / denom))


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def split_xy(df: pd.DataFrame, target_col: str = "y"):
    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    return df[feature_cols], df[target_col]


def fit_linear_regression(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    model = LinearRegression().fit(scaler.transform(X_train), y_train)
    return model, scaler


def fit_random_forest(X_train, y_train, **kwargs):
    model = RandomForestRegressor(
        n_estimators=kwargs.pop("n_estimators", 200),
        random_state=42,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train, y_train, **kwargs):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=kwargs.pop("n_estimators", 300),
        max_depth=kwargs.pop("max_depth", 6),
        learning_rate=kwargs.pop("learning_rate", 0.05),
        random_state=42,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def build_lstm_sequences(X: np.ndarray, y: np.ndarray, n_steps: int = 48):
    """Convierte el dataset tabular en ventanas (samples, n_steps, features) para LSTM."""
    Xs, ys = [], []
    for i in range(n_steps, len(X)):
        Xs.append(X[i - n_steps : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def fit_lstm(X_train_seq, y_train_seq, n_features: int, epochs: int = 20, batch_size: int = 64):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train_seq.shape[1], n_features)),
            keras.layers.LSTM(64),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def compare_models(results: dict) -> pd.DataFrame:
    """results: {"Regresión lineal": {...métricas...}, "Random Forest": {...}, ...}"""
    return pd.DataFrame(results).T.sort_values("RMSE")
