# utils/utils.py
# General utilities for the Whisper framework.
# Includes GPU memory management, loss functions, training helpers,
# and the EarlyStopping callback.

import os
import json
import gc
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GPU / memory utilities
# ---------------------------------------------------------------------------

def clear_cuda_cache():
    """Release unused GPU memory held by the PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_memory(prefix: str = ""):
    """Print current GPU memory usage (allocated / reserved)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        tag = f"[{prefix}] " if prefix else ""
        print(f"{tag}GPU memory — allocated: {allocated:.1f} MB | reserved: {reserved:.1f} MB")


# ---------------------------------------------------------------------------
# DEC loss utilities
# ---------------------------------------------------------------------------

def dec_supervised_loss(
    q: torch.Tensor,
    p: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """
    Semi-supervised DEC loss that combines:
      1. KL divergence between soft assignments Q and target distribution P.
      2. Cross-entropy between cluster assignments and class labels
         (assumes cluster index == class index).

    Args:
        q:          (B, K) soft cluster assignments.
        p:          (B, K) target distribution (from target_distribution()).
        labels:     (B,) ground-truth class labels.
        n_clusters: Number of clusters K.

    Returns:
        Scalar combined loss.
    """
    # KL divergence: sum_ij p_ij * log(p_ij / q_ij)
    kl_loss = F.kl_div(q.log(), p, reduction="batchmean")

    # Supervised cross-entropy over cluster assignments
    if labels.max() < n_clusters:
        sup_loss = F.cross_entropy(q, labels)
    else:
        sup_loss = torch.tensor(0.0, device=q.device)

    return kl_loss + 0.5 * sup_loss


def compute_joint_loss_fixed(
    task_losses: List[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute a weighted sum of per-task losses.
    If no weights are provided, all tasks are weighted equally.

    Args:
        task_losses: List of scalar loss tensors.
        weights:     Optional (n_tasks,) weight tensor.

    Returns:
        Scalar total loss.
    """
    if not task_losses:
        return torch.tensor(0.0)

    if weights is None:
        weights = torch.ones(len(task_losses), device=task_losses[0].device)

    weight_sum = weights.sum()
    if weight_sum == 0:
        return torch.tensor(0.0, device=task_losses[0].device)
    weights = weights / weight_sum  # normalize
    return sum(w * l for w, l in zip(weights, task_losses))


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a validation metric and stop training when it stops improving.
    Saves the model checkpoint at the best observed metric value.

    Args:
        patience:  Number of epochs with no improvement before stopping.
        delta:     Minimum change to qualify as an improvement.
        save_path: File path to save the best model weights.
        mode:      'max' (higher is better) or 'min' (lower is better).
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        save_path: str = "best_model.pth",
        mode: str = "max",
    ):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.mode = mode

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module):
        """
        Update state based on the latest validation score.

        Args:
            score: Current epoch's validation metric.
            model: Model whose weights to save when a new best is found.
        """
        if self.best_score is None:
            self.best_score = score
            self._save(model)
            return

        improved = (
            score >= self.best_score + self.delta
            if self.mode == "max"
            else score <= self.best_score - self.delta
        )

        if improved:
            self.best_score = score
            self._save(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model: nn.Module):
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), self.save_path)


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def save_training_config(config: dict, save_path: str):
    """
    Serialize a training configuration dictionary to a JSON file.

    Args:
        config:    Dictionary of configuration parameters.
        save_path: Destination file path (.json).
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to: {save_path}")


# ---------------------------------------------------------------------------
# GradNorm parameter extraction
# ---------------------------------------------------------------------------

def extract_gradnorm_params_vit(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract the last shared layer parameters from a ViT model for
    GradNorm loss weighting.  GradNorm anchors gradients at the last
    shared representation layer (i.e., the final encoder layer).

    Args:
        model: A ViTClassifier instance (or any model with a .vit attribute).

    Returns:
        List of parameter tensors from the last encoder layer.
    """
    # Navigate to the last transformer encoder layer
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        encoder_layers = model.vit.encoder.layer
        if len(encoder_layers) > 0:
            last_layer = encoder_layers[-1]
            return list(last_layer.parameters())

    # Fallback: return all parameters
    return list(model.parameters())
