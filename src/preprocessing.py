"""
Limpieza y preparación del dataset UCI (Fase 1 de la metodología ElecPrevis).

Pasos que sigue este módulo, en orden:
    1. Carga del .txt separado por ';', con '?' como marcador de nulo.
    2. Combinación de Date + Time en un índice datetime.
    3. Optimización de memoria (downcast numérico).
    4. Imputación de nulos por interpolación temporal (no por mediana global:
       es una serie de tiempo, no un dataset tabular independiente).
    5. Remuestreo de 1 minuto a 30 minutos.
    6. Reporte de outliers vía IQR (solo diagnóstico, no se recortan valores
       reales de consumo sin justificar por qué).

Nunca se usa información futura en estos pasos: cada operación mira hacia
atrás (interpolación, resample), no hacia adelante.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga el .txt crudo y arma el índice datetime."""
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast de columnas numéricas para reducir el consumo de RAM."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def report_missing(df: pd.DataFrame) -> pd.Series:
    """Porcentaje de nulos por columna, para documentar antes de imputar."""
    return (df.isna().mean() * 100).round(3)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolación temporal: usa los puntos vecinos en el tiempo para rellenar
    huecos. Preferible a una mediana global porque respeta la continuidad
    de la serie (un hueco a las 3am no debería rellenarse con el promedio
    del día completo).
    """
    df[NUMERIC_COLS] = df[NUMERIC_COLS].interpolate(method="time", limit_direction="both")
    return df


def report_outliers_iqr(df: pd.DataFrame, cols=NUMERIC_COLS) -> pd.DataFrame:
    """
    Reporta cuántos valores caen fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    por columna. Es diagnóstico: en consumo eléctrico un pico puede ser
    un evento real (encender el horno), no ruido a eliminar sin criterio.
    """
    rows = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append(
            {"columna": col, "q1": q1, "q3": q3, "limite_inf": lower,
             "limite_sup": upper, "n_outliers": n_out,
             "pct_outliers": round(n_out / len(df) * 100, 3)}
        )
    return pd.DataFrame(rows)


def resample_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Remuestreo de 1 minuto a intervalos de 30 minutos (promedio)."""
    return df[NUMERIC_COLS].resample("30min").mean()


def run_pipeline(save: bool = True) -> pd.DataFrame:
    print("Cargando datos crudos...")
    df = load_raw()
    print(f"  {len(df):,} filas cargadas (resolución 1 min).")

    df = optimize_memory(df)

    missing = report_missing(df)
    print("Nulos por columna (%):")
    print(missing[missing > 0])

    df = impute_missing(df)

    outliers = report_outliers_iqr(df)
    print("\nOutliers detectados (IQR, solo diagnóstico):")
    print(outliers.to_string(index=False))

    df_30min = resample_30min(df)
    print(f"\nRemuestreo a 30 min: {len(df_30min):,} intervalos.")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "power_30min.csv"
        df_30min.to_csv(out_path)
        print(f"Guardado en {out_path}")

    return df_30min


if __name__ == "__main__":
    run_pipeline()
