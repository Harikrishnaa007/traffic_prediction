"""
Create sliding windows for supervised learning.
"""

import numpy as np
import torch

def create_windowed_dataset(df, input_len=12, output_len=6):
    df_numeric = df.select_dtypes(include=[np.number])  # drop timestamp

    values = df_numeric.values
    num_samples = values.shape[0] - input_len - output_len + 1

    X, Y = [], []
    for i in range(num_samples):
        X.append(values[i:i+input_len])
        Y.append(values[i+input_len:i+input_len+output_len, :207])  # only sensors in Y

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    # Chronological split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size

    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
    X_test, Y_test = X[train_size+val_size:], Y[train_size+val_size:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
