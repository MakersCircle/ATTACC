# Spatial attention module
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)

        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        out = x * self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_out, max_out], dim=1)
        out = x * self.sigmoid(self.conv(concat))
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CombineFeatures(nn.Module):
    def __init__(self, feature_dim, depth_dim):
        super(CombineFeatures, self).__init__()
        self.feature_dim = feature_dim
        self.depth_dim = depth_dim

        self.fc = nn.Linear(feature_dim * 2 + depth_dim, feature_dim)
        self.cbam = CBAM(in_channels=feature_dim)

    def forward(self, f_obj, f_frame, f_depth):
        N = f_obj.shape[2]
        frame_expanded = f_frame.expand(-1, -1, N, -1)

        combined_features = torch.cat([f_obj, frame_expanded, f_depth], dim=-1)
        combined_features = self.fc(combined_features)

        cbam_input = combined_features.permute(0, 3, 1, 2)
        out = self.cbam(cbam_input)
        att_out = out.permute(0, 2, 3, 1)

        return att_out
