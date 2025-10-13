"""
Model evaluation for Hybrid LSTM + TransformerXL.
Computes MAE, RMSE, and returns predictions for visualization.
"""

import torch
import torch.nn as nn
import numpy as np
from math import sqrt

def evaluate_model(model, test_loader, device="cuda"):
    """Evaluates the model on test data and returns (MAE, RMSE, Y_pred, Y_true)."""
    model.eval()
    model.to(device)

    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    total_mse, total_mae, count = 0.0, 0.0, 0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)

            mse = criterion(preds, Y).item()
            mae = mae_fn(preds, Y).item()

            total_mse += mse
            total_mae += mae
            count += 1

            all_preds.append(preds.cpu())
            all_trues.append(Y.cpu())

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    rmse = sqrt(avg_mse)

    # Combine predictions and truths
    Y_pred = torch.cat(all_preds)
    Y_true = torch.cat(all_trues)

    print(f"Test MAE: {avg_mae:.4f}, RMSE: {rmse:.4f}")
    return avg_mae, rmse, Y_pred, Y_true
