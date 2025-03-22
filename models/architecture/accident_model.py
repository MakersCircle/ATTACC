# Final model integrating all modules

import torch
import torch.nn as nn
from models.architecture.spatial_attention import SpatialEncoder
from models.architecture.temporal_attention import TemporalEncoder

class GraphTransformerAccidentModel(nn.Module):
    def __init__(self):
        super(GraphTransformerAccidentModel, self).__init__()
        self.spatial = SpatialEncoder(in_channels=4097)  # 4096 + 1 (depth)
        self.temporal = TemporalEncoder(input_dim=256)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, 1)

    def forward(self, object_features, object_depths):
        """
        object_features: (T, N, 4096)
        object_depths: (T, N, 1)
        Returns:
            probs: (T,)
            uncertainty: (T,)
        """
        print(object_features.shape)
        T, N, _ = object_features.shape
        spatial_features = []

        for t in range(T):
            obj_feat = object_features[t]  # (N, 4096)
            obj_depth = object_depths[t]  # (N, 1)
            x = torch.cat([obj_feat, obj_depth], dim=-1)  # (N, 4097)

            # Construct edge_index: fully connected or based on heuristics
            edge_index = torch.combinations(torch.arange(N), r=2).T  # shape (2, N*(N-1)/2)
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # Make it bidirectional

            # Construct batch: all nodes belong to graph 0
            batch = torch.zeros(N, dtype=torch.long, device=x.device)

            spatial_feat = self.spatial(x, edge_index, batch)  # (1, D)
            spatial_features.append(spatial_feat)

        spatial_features = torch.cat(spatial_features, dim=0)  # (T, D)

        temporal_out = self.temporal(spatial_features)  # (T, D)

        out_drop = self.dropout(temporal_out)
        logits = self.classifier(out_drop).squeeze(-1)
        probs = torch.sigmoid(logits)

        with torch.no_grad():
            logits2 = self.classifier(self.dropout(temporal_out)).squeeze(-1)
            probs2 = torch.sigmoid(logits2)
            uncertainty = (probs - probs2).abs()

        return probs, uncertainty



if __name__ == "__main__":
    model = GraphTransformerAccidentModel()
    obj_det = torch.randn(50, 19, 6)
    obj_depth = torch.randn(50, 19, 1)
    obj_feat = torch.randn(50, 19, 4096)
    frm_feat = torch.randn(50, 1, 4096)


    probs, uncert = model(obj_feat, obj_depth)
    print("Prob shape:", probs.shape)       # [50]
    print("Uncertainty shape:", uncert.shape)
