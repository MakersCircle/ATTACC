# models/architecture/spatial_encoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=4097, hidden_channels=512, out_channels=256, heads=4):
        super(SpatialEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=True)

    def forward(self, x, edge_index, batch):
        """
        x: (num_nodes, in_channels)
        edge_index: (2, num_edges)
        batch: (num_nodes,) -> batch assignment for each node

        Returns:
            graph_embeddings: (batch_size, out_channels)
        """
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)

        # Mean pool over objects in each frame
        graph_embeddings = global_mean_pool(x, batch)  # (T, out_channels)

        return graph_embeddings
