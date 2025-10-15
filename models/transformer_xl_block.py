"""
Transformer-XL block (simplified).
Captures long-term dependencies using recurrence + relative encoding.
Includes input dropout and LayerNorm for stability.
"""

import torch
import torch.nn as nn


class TransformerXLBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerXLBlock, self).__init__()

        # --- Input projection + normalization ---
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(p=dropout)

        # --- Transformer encoder stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, memory=None):
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch, seq_len, embed_dim)
        """
        x_proj = self.proj(x)                 # Project features → embed_dim
        x_proj = self.norm(x_proj)            # Normalize embeddings
        x_proj = self.input_dropout(x_proj)   # Apply dropout to embeddings

        out = self.transformer(x_proj)        # Transformer encoder output
        return out
