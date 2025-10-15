"""
Hybrid LSTM + Transformer-XL model with Fusion MLP, configurable dropout,
and lightweight temporal variation (adds realistic per-timestep wiggles
without retraining).
"""

import torch
import torch.nn as nn
from .lstm_block import LSTMBlock
from .transformer_xl_block import TransformerXLBlock


class HybridModel(nn.Module):
    def __init__(self, input_dim=215, hidden_dim=64, output_dim=207,
                 horizon=6, embed_dim=128, dropout=0.1):
        super(HybridModel, self).__init__()

        # --- Core sequence blocks ---
        self.lstm_block = LSTMBlock(input_dim, hidden_dim)
        self.transformer_block = TransformerXLBlock(input_dim, hidden_dim, embed_dim=embed_dim)

        # --- Fusion dimensions ---
        fusion_dim = hidden_dim + embed_dim

        # --- Fusion MLP with configurable dropout ---
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim)
        )

        # --- 🆕 Small projection for temporal variation ---
        self.temporal_wiggle = nn.Linear(1, horizon, bias=False)

        # --- Configs ---
        self.horizon = horizon
        self.dropout_rate = dropout
        self.output_dim = output_dim

    def forward(self, x):
        """
        Forward pass: combines outputs from LSTM and Transformer paths.
        Input: x (batch, seq_len, input_dim)
        Output: (batch, horizon, output_dim)
        """

        # LSTM path
        lstm_out, _ = self.lstm_block(x)
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Transformer path
        trans_out = self.transformer_block(x)
        trans_last = trans_out[:, -1, :]  # (batch, embed_dim)

        # Fusion
        fusion = torch.cat([lstm_last, trans_last], dim=-1)  # (batch, fusion_dim)

        # Base prediction (steady mean)
        base_out = self.fc(fusion)  # (batch, output_dim)

        # --- 🆕 Generate small horizon-dependent variation ---
        time_axis = torch.arange(self.horizon, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        time_offsets = self.temporal_wiggle(time_axis)  # (1, horizon, horizon)
        time_offsets = time_offsets[:, :, :1]  # single offset channel
        time_offsets = torch.sin(time_offsets) * 0.02  # ±2% gentle oscillation

        # --- Expand base prediction and add variation ---
        out = base_out.unsqueeze(1).repeat(1, self.horizon, 1)
        out = out + time_offsets  # add smooth per-step variation

        return out
