import torch
import torch.nn as nn
import torch.nn.functional as F

class TUGLLoss(nn.Module):
    """
    Temporal Uncertainty-Guided Loss (TUGL)
    - Adjusts learning dynamically based on uncertainty and time-to-accident signals.
    """

    def __init__(self, alpha=0.1, beta=0, gamma=0):
        super(TUGLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, accident_scores, uncertainty, labels, time_to_accident, cls):
        """
        Args:
            accident_scores (Tensor): (B, T) accident probabilities
            uncertainty (Tensor): (B, T) uncertainty estimates
            labels (Tensor): (B, T) binary accident labels
            time_to_accident (Tensor): (B, T) time-to-accident
            cls (Tensor): (B,) binary class label per video

        Returns:
            loss (Tensor): scalar loss averaged across batch
        """
        B, T = accident_scores.shape
        losses = []

        for b in range(B):
            # 1. Uncertainty Weight
            U_t = 1 - uncertainty[b] ** 2

            # 2. Exponential Weighting for Positives
            L_exp_weight = torch.exp(-self.alpha * time_to_accident[b])
            BCE = F.binary_cross_entropy(accident_scores[b], labels[b], reduction='none')

            # L_exp: only apply exponential if positive video
            L_exp = BCE * (cls[b] * L_exp_weight + (1 - cls[b]))

            # 3. Contrastive Loss (skip if only 0 or 1 class exists)
            positive_mask = labels[b] == 1
            negative_mask = labels[b] == 0
            if positive_mask.any() and negative_mask.any():
                pos_scores = accident_scores[b][positive_mask]
                neg_scores = accident_scores[b][negative_mask]
                L_cont = torch.mean((pos_scores.mean() - neg_scores.mean()) ** 2)
            else:
                L_cont = torch.tensor(0.0, device=accident_scores.device)

            # 4. RL-style weighted BCE
            rewards = torch.where(labels[b] == 1,
                                  torch.tensor(1.0, device=accident_scores.device),
                                  torch.tensor(-1.0, device=accident_scores.device))
            L_RL = -rewards * torch.log(accident_scores[b] + 1e-6) - (1 - rewards) * torch.log(1 - accident_scores[b] + 1e-6)

            # Final
            loss = torch.mean(U_t * (L_exp + self.beta * L_cont) + self.gamma * L_RL)
            losses.append(loss)

        return torch.mean(torch.stack(losses))  # returns scalar for backprop
