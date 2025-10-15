"""
Streaming LSTM block.
Captures short-term temporal patterns.
"""

import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, x, hidden=None):
        """
        Forward pass for LSTM block.
        Args:
            x: (batch, seq_len, input_dim)
            hidden: optional hidden state tuple (h_0, c_0)
        Returns:
            out: (batch, seq_len, hidden_dim)
            hidden: (h_n, c_n)
        """
        out, hidden = self.lstm(x, hidden)
        return out, hidden
