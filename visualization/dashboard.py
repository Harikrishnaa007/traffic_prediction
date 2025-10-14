"""
Streamlit Dashboard for Real-Time Traffic Prediction
----------------------------------------------------
Visualizes predicted vs actual traffic speeds for selected sensors.
Automatically detects dataset (METR-LA or PEMS-BAY) and loads matching model.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from models.hybrid_model import HybridModel
from preprocessing.preprocess import load_and_clean_data, normalize_data
from preprocessing.features import add_time_features
from preprocessing.windowing import create_windowed_dataset


# ============== CACHED HELPERS ==============
@st.cache_resource
def load_model(model_path, input_dim, output_dim, device="cpu"):
    """Load pre-trained model."""
    model = HybridModel(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_dataset(dataset_path):
    """Load and preprocess dataset (with time features)."""
    df = load_and_clean_data(dataset_path)
    df_norm, mean, std = normalize_data(df)
    df_time = add_time_features(df_norm)
    return df_time, mean, std


# ============== PREDICTION HELPER ==============
def predict_for_sensor(model, df_time, sensor_id, mean, std, device="cpu"):
    """Predicts future timesteps for a selected sensor."""
    df_sensor = df_time[[sensor_id]]
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_sensor)
    X_latest = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_latest).cpu().numpy()[0, :, 0]

    # De-normalize
    preds = preds * std[sensor_id] + mean[sensor_id]
    actual = (Y_test[-1, :, 0] * std[sensor_id]) + mean[sensor_id]
    return preds, actual


# ============== STREAMLIT DASHBOARD ==============
def main():
    st.set_page_config(page_title="Traffic Prediction Dashboard", layout="wide")
    st.title("🚦 Real-Time Traffic Speed Prediction Dashboard")
    st.markdown("### Hybrid LSTM + TransformerXL Model — Interactive Visualization")
    st.markdown("---")

    # ========== Sidebar Controls ==========
    st.sidebar.header("⚙️ Configuration")

    # Google Drive-based dataset paths
    dataset_map = {
        "METR-LA": "/content/drive/MyDrive/traffic_prediction/data/metr-la.h5",
        "PEMS-BAY": "/content/drive/MyDrive/traffic_prediction/data/pems-bay.h5"
    }

    model_map = {
        "METR-LA": "/content/drive/MyDrive/traffic_prediction/best_model_metrla.pth",
        "PEMS-BAY": "/content/drive/MyDrive/traffic_prediction/best_model_pemsbay.pth"
    }

    dataset_choice = st.sidebar.selectbox("Select Dataset", list(dataset_map.keys()))
    dataset_path = dataset_map[dataset_choice]
    model_path = model_map[dataset_choice]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.sidebar.markdown(f"📂 **Dataset:** `{dataset_path}`")
    st.sidebar.markdown(f"💾 **Model:** `{os.path.basename(model_path)}`")

    # ========== Load Model + Data ==========
    with st.spinner("Loading dataset and model..."):
        df_time, mean, std = load_dataset(dataset_path)
        output_dim = 1  # predicting 1 sensor at a time
        model = load_model(model_path, input_dim=df_time.shape[1], output_dim=output_dim, device=device)

    st.success(f"✅ Loaded model and dataset: {dataset_choice}")
    st.write(f"Dataset shape: {df_time.shape}")

    # ========== User Input ==========
    sensor_id = st.selectbox("Select Sensor ID", df_time.columns)
    update_interval = st.slider("⏱️ Update Interval (seconds)", 1, 10, 3)

    st.markdown("### 📊 Live Traffic Prediction")
    plot_placeholder = st.empty()

    # ========== Real-Time Loop ==========
    while True:
        preds, actual = predict_for_sensor(model, df_time, sensor_id, mean, std, device)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(actual, label="Actual", color="blue")
        ax.plot(preds, label="Predicted", color="red", linestyle="--")
        ax.set_title(f"Sensor {sensor_id} — Real-Time Traffic Speed")
        ax.set_xlabel("Future Timesteps (5-min intervals)")
        ax.set_ylabel("Speed (mph)")
        ax.legend()
        ax.grid(True)

        plot_placeholder.pyplot(fig)
        time.sleep(update_interval)


# ============== ENTRY POINT ==============
if __name__ == "__main__":
    main()
