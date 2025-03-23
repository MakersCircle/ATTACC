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

        B, T, N, _ = object_features.shape
        spatial_features = []

        for b in range(B):
            per_video_spatial = []
            for t in range(T):
                obj_feat = object_features[b, t]  # (N, 4096)
                obj_depth = object_depths[b, t]  # (N, 1)
                x = torch.cat([obj_feat, obj_depth], dim=-1)  # (N, 4097)

                # Construct fully-connected edge index
                edge_index = torch.combinations(torch.arange(N, device=x.device), r=2).T
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

                batch = torch.zeros(N, dtype=torch.long, device=x.device)
                spatial_feat = self.spatial(x, edge_index, batch)  # (1, D)
                per_video_spatial.append(spatial_feat)

            per_video_spatial = torch.cat(per_video_spatial, dim=0)  # (T, D)
            spatial_features.append(per_video_spatial)

        spatial_features = torch.stack(spatial_features, dim=0)  # (B, T, D)
        temporal_out = self.temporal(spatial_features)  # (B, T, D)

        # Classification + MC Dropout
        out_drop = self.dropout(temporal_out)
        logits = self.classifier(out_drop).squeeze(-1)  # (B, T)
        probs = torch.sigmoid(logits)

        # Uncertainty via MC Dropout
        with torch.no_grad():
            logits2 = self.classifier(self.dropout(temporal_out)).squeeze(-1)
            probs2 = torch.sigmoid(logits2)
            uncertainty = (probs - probs2).abs()

        return probs, uncertainty



if __name__ == "__main__":
    model = GraphTransformerAccidentModel()
    obj_det = torch.randn(4, 50, 19, 6)
    obj_depth = torch.randn(4, 50, 19, 1)
    obj_feat = torch.randn(4, 50, 19, 4096)
    frm_feat = torch.randn(4, 50, 1, 4096)


    probs, uncert = model(obj_feat, obj_depth)
    print("Prob shape:", probs.shape)       # [50]
    print("Uncertainty shape:", uncert.shape)
