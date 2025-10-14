"""
Preprocess raw traffic data:
- Load .h5 dataset (auto-detects correct key)
- Handle zeros (treat long consecutive zeros as missing)
- Interpolate missing values
- Normalize per sensor
"""

import pandas as pd
import numpy as np
import h5py

def load_and_clean_data(file_path, zero_threshold=12):
    """
    Loads either METR-LA or PEMS-BAY dataset automatically.
    Replaces long consecutive zeros with NaN, interpolates missing values.
    """
    # --- Detect key automatically ---
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        print(f"📂 Keys found in {file_path}: {keys}")

    possible_keys = ["df", "speed", "data"]
    df = None
    for key in possible_keys:
        try:
            df = pd.read_hdf(file_path, key=key)
            print(f"✅ Loaded using key='{key}' → shape={df.shape}")
            break
        except Exception:
            continue

    if df is None:
        raise KeyError(f"❌ No valid dataset key found in {file_path}. Available keys: {keys}")

    # --- Replace long consecutive zeros with NaN ---
    def replace_long_zeros(series, threshold):
        values = series.values.copy()
        zero_indices = np.where(values == 0)[0]
        if len(zero_indices) == 0:
            return series

        # Split consecutive zero runs
        zero_runs = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
        for run in zero_runs:
            if len(run) >= threshold:
                values[run] = np.nan
        return pd.Series(values, index=series.index)

    df_clean = df.apply(lambda col: replace_long_zeros(col, threshold=zero_threshold))

    # --- Interpolate missing values ---
    df_clean = df_clean.interpolate(method="linear", axis=0).ffill().bfill()
    print(f"✅ After cleaning: {df_clean.shape}")

    return df_clean


def normalize_data(df):
    """Z-score normalization per sensor."""
    means = df.mean()
    stds = df.std()
    df_norm = (df - means) / stds
    return df_norm, means, stds
