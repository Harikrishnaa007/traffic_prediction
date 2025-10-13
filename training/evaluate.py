"""
Model evaluation for Hybrid LSTM + TransformerXL
Computes MAE, RMSE, and optionally logs predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from math import sqrt


def evaluate_model(model, test_loader, device="cuda"):
    """Evaluates the model on test data and returns (MAE, RMSE)."""
    model.eval()
    model.to(device)

    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    total_mse, total_mae, count = 0.0, 0.0, 0

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)

            mse = criterion(preds, Y).item()
            mae = mae_fn(preds, Y).item()

            total_mse += mse
            total_mae += mae
            count += 1

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    rmse = sqrt(avg_mse)

    print(f"Test MAE: {avg_mae:.4f}, RMSE: {rmse:.4f}")
    return avg_mae, rmse
