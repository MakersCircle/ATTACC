# Temporal attention module - GRU
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, num_layers=1, bidirectional=False):
        super(TemporalAttention, self).__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        gru_out, _ = self.gru(features)
        output = self.fc(gru_out).squeeze(-1)
        return output
