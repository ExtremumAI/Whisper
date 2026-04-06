# models/vit.py
# Vision Transformer (ViT) backbone with optional MaskedAttention.
# Wraps a HuggingFace ViT model for use in the Whisper framework.

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class MaskedAttention(nn.Module):
    """
    Multi-head self-attention with an optional binary mask.
    The mask can suppress attention to certain patch positions (e.g., background).
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, N, D) patch token sequence.
            mask: (B, N) boolean tensor; True positions are suppressed.

        Returns:
            (B, N, D) attended patch token sequence.
        """
        residual = x
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.norm(residual + attn_out)


class ViTClassifier(nn.Module):
    """
    ViT-based classifier that exposes patch embeddings for downstream modules.

    Args:
        pretrained_model_name: HuggingFace model name or local path.
        num_labels:            Number of output classes.
        use_masked_attention:  Whether to add a MaskedAttention layer on top.
        dropout:               Dropout probability for the classification head.
    """

    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 6,
        use_masked_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.vit.config.hidden_size

        self.use_masked_attention = use_masked_attention
        if use_masked_attention:
            self.masked_attention = MaskedAttention(
                embed_dim=self.hidden_size,
                num_heads=self.vit.config.num_attention_heads,
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        patch_mask: torch.Tensor = None,
        return_patch_embeddings: bool = False,
    ):
        """
        Args:
            pixel_values:            (B, C, H, W) preprocessed images.
            attention_mask:          Optional HuggingFace-style attention mask.
            patch_mask:              (B, N) boolean mask for MaskedAttention.
            return_patch_embeddings: If True, also return patch token embeddings.

        Returns:
            logits:      (B, num_labels) classification logits.
            patch_emb:   (B, N+1, D) all token embeddings (only if
                         return_patch_embeddings=True).
        """
        outputs = self.vit(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state: (B, N+1, D)  [CLS + patch tokens]
        patch_emb = outputs.last_hidden_state

        if self.use_masked_attention and patch_mask is not None:
            patch_emb = self.masked_attention(patch_emb, mask=patch_mask)

        cls_token = patch_emb[:, 0, :]
        logits = self.classifier(cls_token)

        if return_patch_embeddings:
            return logits, patch_emb
        return logits
