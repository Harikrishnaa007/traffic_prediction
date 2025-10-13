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
Main pipeline: preprocessing → dataset → model → training → evaluation
"""

"""
Main pipeline for Hybrid LSTM + Transformer-XL:
Data preprocessing → Dataset creation → Model training → Evaluation
"""

import torch
import matplotlib.pyplot as plt
import argparse
import os
from preprocessing.preprocess import load_and_clean_data, normalize_data
from preprocessing.features import add_time_features, build_adjacency_matrix
from preprocessing.windowing import create_windowed_dataset
from dataset.traffic_dataset import get_dataloaders
from models.hybrid_model import HybridModel
from training.train import train_model
from training.evaluate import evaluate_model


def main(epochs=30, lr=5e-4, device="cuda"):
    # 1️⃣ Data Loading and Preprocessing
    print("🔹 Loading and preprocessing data...")
    df_clean = load_and_clean_data("data/metr-la.h5")
    df_norm, mean, std = normalize_data(df_clean)
    df_time = add_time_features(df_norm)
    adj = build_adjacency_matrix(df_clean)
    print(f"✅ Data ready: {df_time.shape}")

    # 2️⃣ Create Train/Val/Test Datasets
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_time)
    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    )

    # 3️⃣ Initialize Model
    model = HybridModel(input_dim=X_train.shape[-1], output_dim=Y_train.shape[-1])
    print(f"✅ Model initialized — input_dim={X_train.shape[-1]}, output_dim={Y_train.shape[-1]}")

    # 4️⃣ Train Model
    print(f"🚀 Training for {epochs} epochs (lr={lr}, device={device})...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
    )

    # 5️⃣ Plot & Save Loss Curve
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/loss_curve.png")
    plt.show()
    print("📊 Saved loss curve → outputs/loss_curve.png")

    # 6️⃣ Evaluate Model
    print("📈 Evaluating best model on test data...")
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu or cuda")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    main(epochs=args.epochs, lr=args.lr, device=args.device)

