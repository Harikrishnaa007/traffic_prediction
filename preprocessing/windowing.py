"""
Create sliding windows for supervised learning.
Automatically adapts to number of sensors (columns) in the dataset.
"""

import numpy as np
import torch

def create_windowed_dataset(df, input_len=12, output_len=6):
    """
    Create sliding windows for supervised learning.
    Ensures that only sensor columns are used for prediction targets (Y).
    """

    # --- 1️⃣ Identify sensor columns ---
    # Assume time features are added at the END of the dataframe by add_time_features()
    # So, we can detect sensors as all numeric columns before those time features.
    df_numeric = df.select_dtypes(include=[np.number])
    all_cols = list(df_numeric.columns)

    # Infer number of sensors based on std/mean alignment (done later in dashboard)
    # For now, assume sensors come first, time features last
    # If you have a fixed known count (like 207 for METR-LA, 325 for PEMS-BAY),
    # you can explicitly set it here.
    if "hour_sin" in all_cols or "hour_cos" in all_cols:
        # assume time features appended last
        time_feature_start = all_cols.index("hour_sin")
        sensor_cols = all_cols[:time_feature_start]
    else:
        sensor_cols = all_cols

    sensor_values = df_numeric[sensor_cols].values
    all_values = df_numeric.values

    # --- 2️⃣ Window creation ---
    num_samples = all_values.shape[0] - input_len - output_len + 1
    X, Y = [], []

    for i in range(num_samples):
        # Input window includes all numeric data (sensors + time features)
        X.append(all_values[i:i + input_len])
        # Output window includes ONLY sensors
        Y.append(sensor_values[i + input_len:i + input_len + output_len])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    # --- 3️⃣ Chronological split ---
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
    X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
