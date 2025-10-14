"""
Streamlit Dashboard for Real-Time Traffic Prediction
----------------------------------------------------
Displays predicted vs actual traffic speeds for selected sensors.
Works with METR-LA or PEMS-BAY datasets.
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


@st.cache_resource
def load_model(model_path, input_dim, output_dim, device="cpu"):
    model = HybridModel(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_dataset(dataset_path):
    df = load_and_clean_data(dataset_path)
    df_norm, mean, std = normalize_data(df)
    df_time = add_time_features(df_norm)
    return df_time, mean, std


def predict_for_sensor(model, df_time, sensor_id, mean, std, device="cpu"):
    """
    Predicts the next few timesteps for a single sensor.
    """
    df_sensor = df_time[[sensor_id]]
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_sensor)
    X_latest = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_latest).cpu().numpy()[0, :, 0]

    # De-normalize
    preds = preds * std[sensor_id] + mean[sensor_id]
    actual = (Y_test[-1, :, 0] * std[sensor_id]) + mean[sensor_id]
    return preds, actual


def main():
    st.title("🚦 Real-Time Traffic Speed Prediction Dashboard")
    st.markdown("### Hybrid LSTM + TransformerXL Model (Live Demo)")
    st.markdown("---")

    # Sidebar controls
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset", ["METR-LA", "PEMS-BAY"], index=0
    )
    model_path = st.sidebar.text_input("Model Path", "best_model.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_map = {
        "METR-LA": "data/metr-la.h5",
        "PEMS-BAY": "data/pems-bay.h5"
    }
    dataset_path = dataset_map[dataset_choice]

    st.sidebar.write(f"📂 Using dataset: `{dataset_path}`")

    # Load dataset and model
    with st.spinner("Loading dataset and model..."):
        df_time, mean, std = load_dataset(dataset_path)
        model = load_model(model_path, input_dim=df_time.shape[1], output_dim=1, device=device)

    st.success(f"✅ Loaded model and dataset: {dataset_choice}")
    st.write(f"Dataset shape: {df_time.shape}")

    # Sensor selection
    sensor_id = st.selectbox("Select Sensor ID", df_time.columns)
    update_interval = st.slider("Update Interval (seconds)", 1, 10, 3)

    # Real-time prediction loop
    st.markdown("### 📊 Live Traffic Prediction")
    plot_placeholder = st.empty()

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


if __name__ == "__main__":
    main()
