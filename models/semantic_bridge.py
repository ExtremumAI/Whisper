# models/semantic_bridge.py
# Semantic bridge module for the Whisper framework.
# Bridges medical text embeddings with visual patch features via cross-attention,
# producing a consistency loss that encourages patch features to align with
# their corresponding class text embeddings.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cross_attention import CrossAttention


class SemanticBridge(nn.Module):
    """
    Semantic bridge: text-guided cross-attention over visual patch tokens.

    For each image, the text embedding corresponding to the predicted class
    is used as the key/value in a cross-attention operation with the patch
    tokens as queries. A contrastive consistency loss is computed between
    the attended features and the original CLS token.

    Args:
        text_dim:     Dimensionality of text embeddings (e.g., 768 for BioBERT).
        patch_dim:    Dimensionality of ViT patch embeddings.
        num_heads:    Number of cross-attention heads.
        dropout:      Dropout probability.
        temperature:  Temperature for contrastive loss.
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

        # Project text embeddings to patch dimension if needed
        if text_dim != patch_dim:
            self.text_proj = nn.Linear(text_dim, patch_dim)
        else:
            self.text_proj = nn.Identity()

        self.cross_attn = CrossAttention(
            query_dim=patch_dim,
            key_dim=patch_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(patch_dim, patch_dim)
        self.norm = nn.LayerNorm(patch_dim)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            patch_embeddings: (B, N, D) ViT patch token sequence (incl. CLS).
            text_embeddings:  (C, text_dim) one text embedding per class.
            labels:           (B,) ground-truth class indices (optional).

        Returns:
            dict with keys:
                'attended':  (B, N, D) text-attended patch features.
                'loss':      scalar consistency loss.
        """
        B, N, D = patch_embeddings.shape
        C = text_embeddings.shape[0]

        # Project text to patch dimension
        text_proj = self.text_proj(text_embeddings)  # (C, D)

        if labels is not None:
            # Use ground-truth text embeddings as keys/values
            text_keys = text_proj[labels]  # (B, D)
        else:
            # Use mean text embedding as a fallback
            text_keys = text_proj.mean(dim=0, keepdim=True).expand(B, -1)  # (B, D)

        text_keys = text_keys.unsqueeze(1)  # (B, 1, D) — single key per sample

        # Cross-attention: patch tokens attend to the class text embedding
        attended = self.cross_attn(patch_embeddings, text_keys)  # (B, N, D)

        # Consistency loss: CLS token of attended features vs original CLS token
        cls_original = patch_embeddings[:, 0, :]   # (B, D)
        cls_attended = attended[:, 0, :]            # (B, D)

        cls_original_norm = F.normalize(cls_original, dim=-1)
        cls_attended_norm = F.normalize(cls_attended, dim=-1)

        # InfoNCE-style contrastive loss
        sim = torch.mm(cls_attended_norm, cls_original_norm.t()) / self.temperature
        targets = torch.arange(B, device=patch_embeddings.device)
        loss = F.cross_entropy(sim, targets)

        return {"attended": attended, "loss": loss}
