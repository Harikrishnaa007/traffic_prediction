"""

import h5py
import numpy as np
import pandas as pd

# Open the HDF5 file
file_path = "metr-la.h5"
h5_file = h5py.File(file_path, "r")

# List all datasets in the file
print("Keys:", list(h5_file.keys()))

# Read the HDF5 dataset using pandas
df = pd.read_hdf(file_path, key='df')

# Show first few rows
print(df.head(10))

print(df.isna().any().any())  # True if there is at least one NaN

file_path2 = "pems-bay.h5"
pemsh5_file = h5py.File(file_path2,'r')

print("Keys:", list(pemsh5_file.keys()))

dfpems = pd.read_hdf(file_path2, key='speed')

print(dfpems.head(10))
"""


"""
Main pipeline: preprocessing → dataset → model → training → evaluation → logging
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from preprocessing.preprocess import load_and_clean_data, normalize_data
from preprocessing.features import add_time_features, build_adjacency_matrix
from preprocessing.windowing import create_windowed_dataset
from dataset.traffic_dataset import get_dataloaders
from models.hybrid_model import HybridModel
from training.train import train_model
from training.evaluate import evaluate_model
from training.log_metrics import log_experiment
from visualization.plot_predictions import plot_predictions


def main():
    # -------------------------------
    # 1️⃣ Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="Train Hybrid LSTM + TransformerXL for traffic prediction")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--dataset", type=str, default="data/metr-la.h5", help="Dataset path")

    # ✅ NEW optional arguments
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "cosine", "step"],
                        help="Type of learning rate scheduler (none, cosine, step)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in the HybridModel")

    args = parser.parse_args()

    epochs, lr, device, patience = args.epochs, args.lr, args.device, args.patience
    dataset_path, scheduler_type, batch_size, dropout = args.dataset, args.lr_scheduler, args.batch_size, args.dropout
    dataset_name = os.path.basename(dataset_path).replace(".h5", "").upper()

    print(f"\n🔹 Loading and preprocessing data for {dataset_name}...")

    # -------------------------------
    # 2️⃣ Load + preprocess dataset
    # -------------------------------
    df_clean = load_and_clean_data(dataset_path)
    df_norm, mean, std = normalize_data(df_clean)
    df_time = add_time_features(df_norm)
    adj = build_adjacency_matrix(df_clean)
    print(f"✅ Data ready: {df_time.shape}")

    # -------------------------------
    # 3️⃣ Prepare windowed dataset
    # -------------------------------
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_time)
    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=batch_size
    )

    # -------------------------------
    # 4️⃣ Build model
    # -------------------------------
    model = HybridModel(input_dim=X_train.shape[-1], output_dim=Y_train.shape[-1], dropout=dropout)
    print(f"✅ Model initialized — input_dim={X_train.shape[-1]}, output_dim={Y_train.shape[-1]}, dropout={dropout}")

    # -------------------------------
    # 5️⃣ Train model
    # -------------------------------
    print(f"🚀 Training for {epochs} epochs (lr={lr:.6f}, batch_size={batch_size}, device={device})...")

    # Train and get optimizer
    model, train_losses, val_losses, optimizer = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        patience=patience,
        dataset_name=os.path.basename(dataset_path)
    )

    # ✅ Optional learning rate scheduler
    if scheduler_type != "none":
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            print("📉 Using Cosine Annealing LR scheduler")
        elif scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
            print("📉 Using StepLR scheduler")
    else:
        scheduler = None

    # -------------------------------
    # 6️⃣ Plot loss curve
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)
    loss_curve_path = f"outputs/{dataset_name.lower()}_loss_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training vs Validation Loss — {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"📊 Saved loss curve → {loss_curve_path}")

    # -------------------------------
    # 7️⃣ Evaluate model
    # -------------------------------
    print("📈 Evaluating best model on test data...")
    test_mae, test_rmse, Y_pred, Y_true = evaluate_model(model, test_loader, device=device)

    # -------------------------------
    # 8️⃣ Visualize predictions
    # -------------------------------
    pred_plot_path = f"outputs/{dataset_name.lower()}_pred_vs_actual.png"
    print("🎨 Plotting predicted vs actual traffic speeds...")
    plot_predictions(Y_true, Y_pred, num_sensors=4, save_path=pred_plot_path)
    print(f"📈 Saved prediction comparison → {pred_plot_path}")

    # -------------------------------
    # 9️⃣ Log experiment
    # -------------------------------
    log_experiment(
        model_name="Hybrid_LSTM_TransformerXL",
        dataset=dataset_name,
        epochs=epochs,
        lr=lr,
        patience=patience,
        best_val_loss=min(val_losses),
        test_mae=test_mae,
        test_rmse=test_rmse,
        early_stop_epoch=len(val_losses),
        device=device,
        notes=f"Tuned run: epochs={epochs}, lr={lr}, patience={patience}, scheduler={scheduler_type}, dropout={dropout}"
    )


if __name__ == "__main__":
    main()
