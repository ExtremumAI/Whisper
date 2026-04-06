# models/fixed_text_bridge.py
# Ablation variant: semantic bridge with frozen (non-attended) text embeddings.
# Instead of using cross-attention, the text embedding is directly projected
# and added to the CLS token. This tests the value of the attention mechanism
# vs simply injecting text information.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedTextBridge(nn.Module):
    """
    Ablation: semantic bridge that adds a projected text embedding directly
    to the CLS token without any cross-attention operation.

    Args:
        text_dim:     Dimensionality of text embeddings.
        patch_dim:    Dimensionality of ViT patch embeddings.
        dropout:      Dropout probability.
        temperature:  Temperature for the consistency loss.
    """

    def __init__(
        self,
        text_dim: int = 768,
        patch_dim: int = 768,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Linear(patch_dim * 2, patch_dim)
        self.norm = nn.LayerNorm(patch_dim)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            patch_embeddings: (B, N, D) ViT patch token sequence.
            text_embeddings:  (C, text_dim) one text embedding per class.
            labels:           (B,) ground-truth class indices (optional).

        Returns:
            dict with keys:
                'attended':  (B, N, D) patch features with fused text.
                'loss':      scalar consistency loss.
        """
        B, N, D = patch_embeddings.shape
        cls_token = patch_embeddings[:, 0, :]  # (B, D)

        if labels is not None:
            text_selected = text_embeddings[labels]   # (B, text_dim)
        else:
            text_selected = text_embeddings.mean(0, keepdim=True).expand(B, -1)

        text_proj = self.text_proj(text_selected)  # (B, D)

        # Fuse CLS token with text embedding (no attention)
        fused_cls = self.fusion(torch.cat([cls_token, text_proj], dim=-1))  # (B, D)
        fused_cls = self.norm(fused_cls + cls_token)                         # residual

        # Replace CLS token in the sequence
        attended = patch_embeddings.clone()
        attended[:, 0, :] = fused_cls

        # Consistency loss
        cls_original = F.normalize(cls_token, dim=-1)
        cls_fused = F.normalize(fused_cls, dim=-1)

        sim = torch.mm(cls_fused, cls_original.t()) / self.temperature
        targets = torch.arange(B, device=patch_embeddings.device)
        loss = F.cross_entropy(sim, targets)

        return {"attended": attended, "loss": loss}
