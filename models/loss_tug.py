import torch
import torch.nn as nn

class TUGLLoss(nn.Module):
    """
    Temporal Uncertainty-Guided Loss (TUGL)
    - Adjusts learning dynamically based on uncertainty and time-to-accident signals.
    """

    def __init__(self, alpha=0.1, beta=1.0, gamma=0.5):
        super(TUGLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, accident_scores, uncertainty, labels, time_to_accident):
        """
        Args:
            accident_scores (Tensor): (T,) accident probabilities.
            uncertainty (Tensor): (T,) uncertainty estimates.
            labels (Tensor): (T,) ground-truth accident labels.
            time_to_accident (Tensor): (T,) time-to-accident per frame.

        Returns:
            loss (Tensor): Final loss value.
        """
        # 1. Uncertainty-Based Dynamic Weighting
        U_t = 1 - uncertainty**2  # Higher weight for uncertain predictions

        # 2. Exponential Loss for Early Detection
        L_exp = torch.exp(-self.alpha * (time_to_accident - torch.arange(len(time_to_accident), device=accident_scores.device)))

        # 3. Contrastive Loss for Separating Normal vs Accident Frames
        L_cont = torch.mean((accident_scores[labels == 1] - accident_scores[labels == 0])**2)

        # 4. Reinforcement Learning-Adjusted Risk Penalty
        rewards = torch.where(labels == 1, torch.tensor(1.0, device=accident_scores.device),
                              torch.tensor(-1.0, device=accident_scores.device))
        L_RL = -rewards * torch.log(accident_scores) - (1 - rewards) * torch.log(1 - accident_scores)

        # Final Loss
        loss = torch.mean(U_t * (L_exp + self.beta * L_cont) + self.gamma * L_RL)
        return loss
