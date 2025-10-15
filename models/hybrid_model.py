"""
Hybrid LSTM + Transformer-XL model with Fusion MLP and configurable dropout.
"""

import torch
import torch.nn as nn
from .lstm_block import LSTMBlock
from .transformer_xl_block import TransformerXLBlock


class HybridModel(nn.Module):
    def __init__(self, input_dim=215, hidden_dim=64, output_dim=207, horizon=6, embed_dim=128, dropout=0.1):
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

        # --- Configs ---
        self.horizon = horizon
        self.dropout_rate = dropout

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
        fusion = torch.cat([lstm_last, trans_last], dim=-1)  # (batch, hidden_dim+embed_dim)

        # Multi-horizon prediction via MLP + dropout
        out = self.fc(fusion)  # (batch, output_dim)
        out = out.unsqueeze(1).repeat(1, self.horizon, 1)  # (batch, horizon, output_dim)

        return out
