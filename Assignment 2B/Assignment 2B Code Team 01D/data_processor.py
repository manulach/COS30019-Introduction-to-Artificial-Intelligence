"""
COS30019 Assignment 2B: Traffic-Based Route Guidance System
Data Processing Module

Extracts and cleans the SCATS Boroondara traffic dataset, reshapes it into
time-series sequences suitable for LSTM, GRU, and CNN-LSTM training.

Team ID: 01D
Team Members: Saniru, Haresh, Manula

Usage:
    python data_processor.py
    (Run once before training — outputs .npy files to data/)
"""

import os
import math
import numpy as np
import pandas as pd
import xlrd
from sklearn.preprocessing import MinMaxScaler
import pickle


# ───────────
# CONSTANTS
# ───────────

DATA_FILE   = os.path.join(os.path.dirname(__file__), "data", "Scats Data October 2006.xls")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
SEQUENCE_LEN = 12          # 12 × 15 min = 3-hour look-back window
TRAIN_RATIO  = 0.8         # 80 % train / 20 % test split
FLOW_COLS    = [f"V{i:02d}" for i in range(96)]   # V00 … V95


# ───────────────────────
# STEP 1 — READ RAW XLS
# ───────────────────────

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Read the 'Data' sheet from the SCATS October 2006 XLS file.

    Returns a tidy DataFrame with columns:
        scats_id, location, latitude, longitude, date_serial,
        V00 … V95  (15-minute vehicle counts)
    """
    wb = xlrd.open_workbook(filepath)
    sh = wb.sheet_by_name("Data")

    # Row 1 (index 1) is the header; rows 2+ are data
    header = sh.row_values(1)
    rows   = [sh.row_values(i) for i in range(2, sh.nrows)]

    df = pd.DataFrame(rows, columns=header)

    # Rename metadata columns for clarity
    df = df.rename(columns={
        "SCATS Number":  "scats_id",
        "Location":      "location",
        "NB_LATITUDE":   "latitude",
        "NB_LONGITUDE":  "longitude",
        "Date":          "date_serial",
    })

    # Keep only the columns we need
    keep = ["scats_id", "location", "latitude", "longitude", "date_serial"] + FLOW_COLS
    df = df[keep].copy()

    # Coerce flow columns to numeric (xlrd returns floats already, but be safe)
    df[FLOW_COLS] = df[FLOW_COLS].apply(pd.to_numeric, errors="coerce")

    # Convert SCATS id to string with zero-padding
    df["scats_id"] = df["scats_id"].astype(str).str.strip().str.zfill(4)

    return df


# ────────────────
# STEP 2 — CLEAN
# ────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      - Remove rows where all 96 flow values are NaN (missing day).
      - Clip negative values to 0 (sensor glitches).
      - Forward-fill isolated NaN readings within a row, then fill
        remaining NaNs with the column median.
      - Remove sites with zero/invalid coordinates (e.g., site 4266).
    """
    # Drop rows where every flow reading is NaN
    before = len(df)
    df = df.dropna(subset=FLOW_COLS, how="all").reset_index(drop=True)
    print(f"  Dropped {before - len(df)} fully-empty rows.")

    # Clip negatives to 0
    df[FLOW_COLS] = df[FLOW_COLS].clip(lower=0)

    # Forward-fill then backward-fill within each row for isolated gaps
    flow_data = df[FLOW_COLS].copy()
    flow_data = flow_data.T.ffill().bfill().T
    # Any remaining NaNs → column median across all rows
    col_medians = flow_data.median()
    flow_data   = flow_data.fillna(col_medians)
    df[FLOW_COLS] = flow_data

    # Remove sites with missing/zero coordinates
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].reset_index(drop=True)

    print(f"  Clean dataset: {len(df)} rows, {df['scats_id'].nunique()} unique sites.")
    return df


# ────────────────────────────────────
# STEP 3 — BUILD TIME SERIES PER SITE
# ────────────────────────────────────

def build_time_series(df: pd.DataFrame) -> dict:
    """
    For each SCATS site, concatenate all daily readings (V00–V95) into a
    single continuous time series (one value per 15-minute interval).

    Returns:
        dict  {scats_id: np.ndarray of shape (T,)}
    """
    site_series = {}
    for site_id, group in df.groupby("scats_id"):
        # Sort by date to ensure chronological order
        group = group.sort_values("date_serial")
        # Stack the 96 columns row-by-row → 1-D time series
        series = group[FLOW_COLS].values.flatten()
        site_series[site_id] = series.astype(np.float32)
    return site_series


# ───────────────────────────────────
# STEP 4 — BUILD SITE METADATA TABLE
# ───────────────────────────────────

def build_site_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per SCATS site with id, location name, lat, lon.
    """
    info = (
        df.groupby("scats_id")
        .agg(location=("location", "first"),
             latitude=("latitude",  "first"),
             longitude=("longitude", "first"))
        .reset_index()
    )
    return info


# ──────────────────────────────────
# STEP 5 — SLIDING-WINDOW SEQUENCES
# ──────────────────────────────────

def make_sequences(series: np.ndarray, seq_len: int):
    """
    Convert a 1-D time series into overlapping (X, y) pairs.

    X shape: (N, seq_len)
    y shape: (N,)
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ──────────────────────────────────────
# STEP 6 — NORMALISE + TRAIN/TEST SPLIT
# ──────────────────────────────────────

def prepare_ml_data(site_series: dict, seq_len: int = SEQUENCE_LEN,
                    train_ratio: float = TRAIN_RATIO):
    """
    For every site:
      1. Fit a MinMaxScaler on the training portion only (no leakage).
      2. Create sliding-window sequences.
      3. Split chronologically into train / test.

    Returns:
        X_train, y_train, X_test, y_test  (pooled across all sites)
        scaler                            (fitted on combined training data)
        per_site_data                     (dict for site-level evaluation)
    """
    all_train_X, all_train_y = [], []
    all_test_X,  all_test_y  = [], []
    per_site_data = {}

    # Fit a single global scaler on the training portion of ALL sites
    # (consistent scaling for the integrated system)
    all_train_raw = []
    for series in site_series.values():
        split = int(len(series) * train_ratio)
        all_train_raw.append(series[:split])

    global_scaler = MinMaxScaler(feature_range=(0, 1))
    global_scaler.fit(np.concatenate(all_train_raw).reshape(-1, 1))

    for site_id, series in site_series.items():
        split = int(len(series) * train_ratio)
        train_raw = series[:split]
        test_raw  = series[split:]

        train_norm = global_scaler.transform(train_raw.reshape(-1, 1)).flatten()
        test_norm  = global_scaler.transform(test_raw.reshape(-1, 1)).flatten()

        X_tr, y_tr = make_sequences(train_norm, seq_len)
        X_te, y_te = make_sequences(test_norm,  seq_len)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue   # skip sites with too little data

        all_train_X.append(X_tr);  all_train_y.append(y_tr)
        all_test_X.append(X_te);   all_test_y.append(y_te)

        per_site_data[site_id] = {
            "X_train": X_tr, "y_train": y_tr,
            "X_test":  X_te, "y_test":  y_te,
        }

    X_train = np.concatenate(all_train_X)
    y_train = np.concatenate(all_train_y)
    X_test  = np.concatenate(all_test_X)
    y_test  = np.concatenate(all_test_y)

    return X_train, y_train, X_test, y_test, global_scaler, per_site_data


# ──────────────────────────────
# STEP 7 — SAVE PROCESSED DATA
# ──────────────────────────────

def save_processed(output_dir, X_train, y_train, X_test, y_test,
                   scaler, site_info, per_site_data):
    """Save all processed artefacts to disk."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    site_info.to_csv(os.path.join(output_dir, "site_info.csv"), index=False)

    with open(os.path.join(output_dir, "per_site_data.pkl"), "wb") as f:
        pickle.dump(per_site_data, f)

    print(f"\n  Saved to: {output_dir}")
    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}    y_test:  {y_test.shape}")


# ────────
# MAIN
# ────────

def run_pipeline(data_file: str = DATA_FILE,
                 output_dir: str = OUTPUT_DIR,
                 seq_len: int    = SEQUENCE_LEN,
                 train_ratio: float = TRAIN_RATIO) -> dict:
    """
    End-to-end data pipeline.
    Returns a summary dict for use by train_models.py / gui.py.
    """
    print("=" * 55)
    print("  TBRGS Data Processing Pipeline")
    print("=" * 55)

    print("\n[1] Loading raw data …")
    df_raw = load_raw_data(data_file)
    print(f"  Raw rows: {len(df_raw)} | Sites: {df_raw['scats_id'].nunique()}")

    print("\n[2] Cleaning …")
    df_clean = clean_data(df_raw)

    print("\n[3] Building time series per site …")
    site_series = build_time_series(df_clean)
    site_info   = build_site_info(df_clean)
    total_points = sum(len(v) for v in site_series.values())
    print(f"  Sites: {len(site_series)} | Total readings: {total_points}")

    print("\n[4] Generating ML sequences (window = {seq_len} steps × 15 min) …".format(seq_len=seq_len))
    X_train, y_train, X_test, y_test, scaler, per_site_data = \
        prepare_ml_data(site_series, seq_len=seq_len, train_ratio=train_ratio)

    print("\n[5] Saving artefacts …")
    save_processed(output_dir, X_train, y_train, X_test, y_test,
                   scaler, site_info, per_site_data)

    print("\n✓ Pipeline complete.\n")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "site_info": site_info,
        "per_site_data": per_site_data,
    }


if __name__ == "__main__":
    run_pipeline()
