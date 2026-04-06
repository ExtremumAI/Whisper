# models/random_attention_bridge.py
# Ablation variant: semantic bridge with random (untrained) attention weights.
# Used to verify that the semantic benefit comes from meaningful text guidance
# rather than from the additional attention operation itself.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cross_attention import CrossAttention


class RandomAttentionBridge(nn.Module):
    """
    Ablation: semantic bridge where the cross-attention key/values are
    replaced with randomly initialized, frozen text-like vectors.
    This isolates the effect of the attention structure from the
    semantic content of the text embeddings.

    Args:
        text_dim:     Dimensionality of the (unused) text input.
        patch_dim:    Dimensionality of ViT patch embeddings.
        num_heads:    Number of cross-attention heads.
        dropout:      Dropout probability.
        temperature:  Temperature for the consistency loss.
    """

    def __init__(
        self,
        text_dim: int = 768,
        patch_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.patch_dim = patch_dim

        # Fixed random key embedding — not updated during training
        self.register_buffer(
            "random_key",
            torch.randn(1, 1, patch_dim),
        )

        self.cross_attn = CrossAttention(
            query_dim=patch_dim,
            key_dim=patch_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            patch_embeddings: (B, N, D) ViT patch token sequence.
            text_embeddings:  Ignored in this ablation variant.
            labels:           Ignored in this ablation variant.

        Returns:
            dict with keys:
                'attended':  (B, N, D) attended patch features.
                'loss':      scalar consistency loss.
        """
        B, N, D = patch_embeddings.shape

        # Expand random key to batch size
        random_key = self.random_key.expand(B, -1, -1)  # (B, 1, D)

        # Cross-attention with fixed random keys
        attended = self.cross_attn(patch_embeddings, random_key)  # (B, N, D)

        # Consistency loss between attended and original CLS tokens
        cls_original = F.normalize(patch_embeddings[:, 0, :], dim=-1)
        cls_attended = F.normalize(attended[:, 0, :], dim=-1)

        sim = torch.mm(cls_attended, cls_original.t()) / self.temperature
        targets = torch.arange(B, device=patch_embeddings.device)
        loss = F.cross_entropy(sim, targets)

        return {"attended": attended, "loss": loss}
