# Temporal transformer module


import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=256, transformer_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, transformer_dim)

    def forward(self, x):
        """
        x: (T, D) → temporal sequence of spatial features
        Returns:
            prob: (T,), accident probability per frame
        """
        x = self.fc(x)              # (T, transformer_dim)
        x = self.transformer(x.unsqueeze(0)).squeeze(0)  # (T, transformer_dim)
        return x
