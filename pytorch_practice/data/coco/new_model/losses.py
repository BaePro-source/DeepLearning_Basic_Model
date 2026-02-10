"""
Loss Functions for Semantic Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be None, scalar, or Tensor[C]
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B,C,H,W]
            target: [B,H,W] (int64)
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Get probability and log-probability for target class
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        # Apply class weights (alpha)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(target.device)[target]
                loss = alpha_t * loss
            else:
                loss = self.alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation"""
    
    def __init__(self, num_classes: int, smooth: float = 1.0, exclude_bg: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.exclude_bg = exclude_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B,C,H,W]
            target: [B,H,W]
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        # Convert target to one-hot
        target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        # Exclude background if specified
        start_c = 1 if self.exclude_bg else 0
        probs = probs[:, start_c:]
        target_onehot = target_onehot[:, start_c:]

        # Compute Dice coefficient per class
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        
        return loss