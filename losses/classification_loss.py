# losses/classification_loss.py
# Classification loss functions for the Whisper framework.
# Supports cross-entropy, label-smoothed cross-entropy, and focal loss.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Flexible classification loss supporting:
      - Standard cross-entropy
      - Label-smoothed cross-entropy
      - Focal loss

    Args:
        loss_type:       One of 'cross_entropy', 'label_smooth', 'focal'.
        label_smoothing: Smoothing factor (used when loss_type='label_smooth').
        focal_gamma:     Focusing parameter for focal loss.
        weight:          Class weight tensor for imbalanced datasets.
    """

    def __init__(
        self,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw class scores.
            labels: (B,) ground-truth class indices.

        Returns:
            Scalar loss value.
        """
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(logits, labels, weight=self.weight)

        elif self.loss_type == "label_smooth":
            return F.cross_entropy(
                logits,
                labels,
                weight=self.weight,
                label_smoothing=self.label_smoothing,
            )

        elif self.loss_type == "focal":
            return self._focal_loss(logits, labels)

        else:
            raise ValueError(f"Unknown loss_type: '{self.loss_type}'")

    def _focal_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss: down-weights easy examples to focus training on hard ones.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()
