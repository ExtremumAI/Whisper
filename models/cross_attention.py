# models/cross_attention.py
# Cross-attention module for attending from one feature space to another.
# Used by the semantic bridge to let patch features attend to text embeddings.

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: queries from one modality attend to
    keys/values from another modality.

    Args:
        query_dim:  Dimensionality of the query input.
        key_dim:    Dimensionality of the key/value input.
        num_heads:  Number of attention heads.
        dropout:    Dropout probability.
        bias:       Whether to use bias in projection layers.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, (
            f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.k_proj = nn.Linear(key_dim, query_dim, bias=bias)
        self.v_proj = nn.Linear(key_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query:            (B, Nq, query_dim) query features.
            key:              (B, Nk, key_dim) key features.
            value:            (B, Nk, key_dim) value features (defaults to key).
            key_padding_mask: (B, Nk) boolean mask; True positions are ignored.

        Returns:
            out: (B, Nq, query_dim) attended query features.
        """
        if value is None:
            value = key

        B, Nq, _ = query.shape
        Nk = key.shape[1]

        # Project queries, keys, values
        q = self.q_proj(query)   # (B, Nq, D)
        k = self.k_proj(key)     # (B, Nk, D)
        v = self.v_proj(value)   # (B, Nk, D)

        # Reshape for multi-head attention: (B, heads, N, head_dim)
        def reshape(x, seq_len):
            return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q, Nq)
        k = reshape(k, Nk)
        v = reshape(v, Nk)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, Nq, Nk)

        if key_padding_mask is not None:
            # Expand mask: (B, 1, 1, Nk)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).bool()
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)  # (B, H, Nq, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, Nq, -1)  # (B, Nq, D)

        out = self.out_proj(attended)
        out = self.norm(out + query)  # residual connection
        return out
