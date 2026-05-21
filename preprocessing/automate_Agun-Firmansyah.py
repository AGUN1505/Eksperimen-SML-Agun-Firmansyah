from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari file CSV."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def handle_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Handle missing values: median untuk numerik, modus untuk kategorikal."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_feat = [c for c in num_cols if c != target_col]
    cat_feat = [c for c in cat_cols if c != target_col]

    if num_feat:
        imputer = SimpleImputer(strategy="median")
        df[num_feat] = imputer.fit_transform(df[num_feat])

    if cat_feat:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_feat] = cat_imputer.fit_transform(df[cat_feat])

    print(f"[handle_missing_values] Missing values tersisa: {df.isnull().sum().sum()}")
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[handle_duplicates] Dihapus {before - len(df)} baris duplikat")
    return df


def encode_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Encoding fitur kategorikal dan target (jika string)."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()

    for col in cat_cols:
        if col != target_col:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"[encode_features] Encoded fitur kategorikal: {col}")

    if df[target_col].dtype == "object":
        target_le = LabelEncoder()
        df[target_col] = target_le.fit_transform(df[target_col])
        mapping = dict(zip(target_le.classes_, range(len(target_le.classes_))))
        print(f"[encode_features] Encoded target '{target_col}' -> {mapping}")

    return df


def split_and_scale(
    df: pd.DataFrame, target_col: str, scaler_path: str | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train-test split + StandardScaler.

    Jika `scaler_path` diberikan, scaler ter-fit akan disimpan ke path tsb
    dengan pickle agar bisa dipakai kembali di tahap inference.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump({"scaler": scaler, "feature_names": list(X_train.columns)}, f)
        print(f"[split_and_scale] Scaler disimpan ke '{scaler_path}'")

    print(f"[split_and_scale] Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")
    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def save_preprocessed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """Simpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"[save_preprocessed] File disimpan di '{output_dir}/'")


def run_preprocessing(
    input_path: str = "../iris_raw/iris.csv",
    output_dir: str = "iris_preprocessing",
    target_col: str = "species",
) -> None:
    """Pipeline preprocessing end-to-end."""
    print("=" * 60)
    print("Memulai preprocessing otomatis dataset Iris...")
    print("=" * 60)

    df = load_data(input_path)
    df = handle_missing_values(df, target_col)
    df = handle_duplicates(df)
    df = encode_features(df, target_col)

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    X_train, X_test, y_train, y_test = split_and_scale(df, target_col, scaler_path=scaler_path)
    save_preprocessed(X_train, X_test, y_train, y_test, output_dir)

    print("=" * 60)
    print("Preprocessing selesai! Data siap dilatih.")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing(
        input_path="../iris_raw/iris.csv",
        output_dir="iris_preprocessing",
        target_col="species",
    )
