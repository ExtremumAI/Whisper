# models/dec.py
# Deep Embedded Clustering (DEC) module.
# Computes soft cluster assignments from embeddings using
# a Student's t-distribution kernel.

import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class DEC(nn.Module):
    """
    DEC clustering head.

    Args:
        n_clusters:    Number of cluster centroids.
        embedding_dim: Dimensionality of input embeddings.
        alpha:         Degrees of freedom for the Student's t-distribution.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        embedding_dim: int = 768,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self._exponent = -(alpha + 1.0) / 2.0  # precompute constant exponent
        self.cluster_centers = nn.Parameter(
            torch.zeros(n_clusters, embedding_dim),
            requires_grad=True,
        )

    @torch.no_grad()
    def init_cluster_centers(self, embeddings: torch.Tensor):
        """
        Initialize cluster centers using k-means.

        Args:
            embeddings: (N, D) tensor of embeddings.
        """
        emb_np = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        kmeans.fit(emb_np)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.cluster_centers.data.copy_(centers.to(self.cluster_centers.device))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments Q.

        Args:
            z: (B, D) embedding vectors.

        Returns:
            q: (B, K) soft assignment probabilities.
        """
        # Squared Euclidean distance between z and each cluster center
        # z: (B, D), centers: (K, D)
        diff = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)  # (B, K, D)
        dist_sq = (diff ** 2).sum(dim=-1)                          # (B, K)

        # Student's t-distribution kernel
        numerator = (1.0 + dist_sq / self.alpha) ** self._exponent
        q = numerator / numerator.sum(dim=1, keepdim=True)
        return q
