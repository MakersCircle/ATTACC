# Final model integrating all modules

import torch.nn as nn
from models.architecture.spatial_attention import CombineFeatures
from models.architecture.temporal_attention import TemporalAttention


class AccidentPredictor(nn.Module):
    def __init__(self, dimension, depth):
        super(AccidentPredictor, self).__init__()

        self.cbam_attention = CombineFeatures(dimension, depth)  # Spatial attention module
        self.gru_out = TemporalAttention(dimension, depth)

    def forward(self, f_obj, f_frame, f_depth):
        att_out = self.cbam_attention(f_obj, f_frame, f_depth)
        features = att_out.mean(dim=2)

        accident_scores = self.gru_out(features)
        return accident_scores


# Example usage
# if __name__ == "__main__":
    # Example dimensions
    # batch_size, T, N, D, d = 2, 50, 19, 4096, 16
    # F_obj = torch.rand(batch_size, T, N, D)
    # F_frame = torch.rand(batch_size, T, 1, D)
    # F_depth = torch.rand(batch_size, T, N, d)

    # Initialize model
    # model = AccidentPredictor(D, d)

    # Forward pass
    # accident_score = model(F_obj, F_frame, F_depth)

    # print("Accident Scores Shape:", accident_score.shape)
    # print("Accident Scores:", accident_score)
