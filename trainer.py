# trainer.py
# Joint multi-task trainer for the Whisper framework.
# Handles classification, semantic bridge, and DEC clustering tasks
# with GradNorm loss balancing.

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vit import ViTClassifier
from models.dec import DEC
from models.semantic_bridge import SemanticBridge
from models.random_attention_bridge import RandomAttentionBridge
from models.fixed_text_bridge import FixedTextBridge
from losses.classification_loss import ClassificationLoss
from utils.customdatasetloader import build_dataloaders
from utils.dec_utils import target_distribution, cluster_accuracy
from utils.utils import (
    clear_cuda_cache,
    print_gpu_memory,
    dec_supervised_loss,
    compute_joint_loss_fixed,
    EarlyStopping,
    save_training_config,
    extract_gradnorm_params_vit,
)


class JointTrainer:
    """
    Joint multi-task trainer that coordinates:
      - ViT-based image classification
      - Semantic bridge (cross-attention between text and visual patch embeddings)
      - DEC unsupervised/semi-supervised clustering
    Loss balancing is performed with GradNorm.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        text_embedding_path: str = None,
        pretrained_vit_path: str = "google/vit-base-patch16-224",
        vit_num_labels: int = 6,
        dec_n_clusters: int = 6,
        tasks: list = None,
        ablation_mode: str = "full",
        batch_size: int = 32,
        num_workers: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        gradnorm_alpha: float = 1.5,
        gradnorm_lr: float = 1e-3,
        dec_update_interval: int = 5,
        early_stopping_patience: int = 10,
        save_dir: str = "./saved_models/whisper",
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.text_embedding_path = text_embedding_path
        self.pretrained_vit_path = pretrained_vit_path
        self.vit_num_labels = vit_num_labels
        self.dec_n_clusters = dec_n_clusters
        self.tasks = tasks if tasks is not None else ["classification", "semantic", "clustering"]
        self.ablation_mode = ablation_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr
        self.dec_update_interval = dec_update_interval
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dec_centers_path = os.path.join(self.save_dir, "dec_stable_centers.pt")

        os.makedirs(self.save_dir, exist_ok=True)

        self._build_dataloaders()
        self._build_models()
        self._build_optimizers()
        self._build_losses()

        self.early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            save_path=os.path.join(self.save_dir, "best_model.pth"),
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_dataloaders(self):
        self.train_loader, self.val_loader = build_dataloaders(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            processor_name=self.pretrained_vit_path,
        )
        print(f"Train batches: {len(self.train_loader)}", flush=True)
        if self.val_loader is not None:
            print(f"Val batches:   {len(self.val_loader)}", flush=True)

    def _build_models(self):
        # ViT backbone
        self.vit = ViTClassifier(
            pretrained_model_name=self.pretrained_vit_path,
            num_labels=self.vit_num_labels,
        ).to(self.device)

        # Semantic bridge
        if "semantic" in self.tasks:
            text_dim = self._get_text_embedding_dim()
            patch_dim = self.vit.hidden_size

            if self.ablation_mode == "random_attention":
                self.bridge = RandomAttentionBridge(
                    text_dim=text_dim, patch_dim=patch_dim
                ).to(self.device)
            elif self.ablation_mode == "fixed_text":
                self.bridge = FixedTextBridge(
                    text_dim=text_dim, patch_dim=patch_dim
                ).to(self.device)
            else:
                self.bridge = SemanticBridge(
                    text_dim=text_dim, patch_dim=patch_dim
                ).to(self.device)
        else:
            self.bridge = None

        # DEC clustering head
        if "clustering" in self.tasks:
            self.dec = DEC(
                n_clusters=self.dec_n_clusters,
                embedding_dim=self.vit.hidden_size,
            ).to(self.device)
        else:
            self.dec = None

        # Load text embeddings
        if self.text_embedding_path is not None and "semantic" in self.tasks:
            text_emb = np.load(self.text_embedding_path)
            self.text_embeddings = torch.tensor(
                text_emb, dtype=torch.float32
            ).to(self.device)
            print(f"Text embeddings loaded: {self.text_embeddings.shape}", flush=True)
        else:
            self.text_embeddings = None

    def _get_text_embedding_dim(self):
        if self.text_embedding_path is not None:
            text_emb = np.load(self.text_embedding_path)
            return text_emb.shape[-1]
        return 768  # default BioBERT dimension

    def _build_optimizers(self):
        params = list(self.vit.parameters())
        if self.bridge is not None:
            params += list(self.bridge.parameters())
        if self.dec is not None:
            params += list(self.dec.parameters())

        self.optimizer = optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )

        # GradNorm loss weights
        n_tasks = len(self.tasks)
        self.loss_weights = nn.Parameter(
            torch.ones(n_tasks, device=self.device), requires_grad=True
        )
        self.gradnorm_optimizer = optim.Adam([self.loss_weights], lr=self.gradnorm_lr)

    def _build_losses(self):
        self.cls_loss_fn = ClassificationLoss().to(self.device)

    # ------------------------------------------------------------------
    # DEC initialization
    # ------------------------------------------------------------------

    def _init_dec_centers(self):
        """Initialize DEC cluster centers from ViT patch embeddings."""
        print("Initializing DEC cluster centers...", flush=True)
        self.vit.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Extracting embeddings"):
                images = batch[0].to(self.device)
                _, patch_emb = self.vit(images, return_patch_embeddings=True)
                # Use CLS token or mean of patch tokens
                cls_emb = patch_emb[:, 0, :]
                all_embeddings.append(cls_emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.dec.init_cluster_centers(all_embeddings.to(self.device))
        print("DEC cluster centers initialized.", flush=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full training loop."""
        print(f"Training on device: {self.device}", flush=True)
        print(f"Active tasks: {self.tasks}", flush=True)

        # Initialize DEC centers before training if clustering is active
        if "clustering" in self.tasks:
            self._init_dec_centers()

        initial_losses = None

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self._train_epoch(epoch, initial_losses)

            # Update DEC target distribution periodically
            if "clustering" in self.tasks and epoch % self.dec_update_interval == 0:
                self._update_dec_targets()

            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)

            # Scheduler step
            self.scheduler.step()

            # Store initial loss magnitudes for GradNorm
            if initial_losses is None and "total_loss" in train_metrics:
                initial_losses = {
                    k: v for k, v in train_metrics.items() if k != "total_loss"
                }

            # Early stopping on validation accuracy (or training loss)
            monitor_val = val_metrics.get(
                "val_acc", -train_metrics.get("total_loss", 0.0)
            )
            self.early_stopping(monitor_val, self.vit)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

        # Save final model and DEC centers
        self._save_final(epoch)
        print("Training complete.", flush=True)

    def _train_epoch(self, epoch, initial_losses):
        self.vit.train()
        if self.bridge is not None:
            self.bridge.train()
        if self.dec is not None:
            self.dec.train()

        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        sem_loss_sum = 0.0
        dec_loss_sum = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        for batch in pbar:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            self.gradnorm_optimizer.zero_grad()

            # Forward pass
            logits, patch_emb = self.vit(images, return_patch_embeddings=True)

            task_losses = []

            # Classification loss
            if "classification" in self.tasks:
                cls_loss = self.cls_loss_fn(logits, labels)
                task_losses.append(cls_loss)
                cls_loss_sum += cls_loss.item()

            # Semantic bridge loss
            if "semantic" in self.tasks and self.bridge is not None:
                sem_out = self.bridge(patch_emb, self.text_embeddings)
                sem_loss = sem_out["loss"] if isinstance(sem_out, dict) else sem_out
                task_losses.append(sem_loss)
                sem_loss_sum += sem_loss.item()

            # DEC clustering loss
            if "clustering" in self.tasks and self.dec is not None:
                cls_token = patch_emb[:, 0, :]
                q = self.dec(cls_token)
                p = target_distribution(q).detach()
                dec_loss = dec_supervised_loss(q, p, labels, self.dec_n_clusters)
                task_losses.append(dec_loss)
                dec_loss_sum += dec_loss.item()

            if not task_losses:
                continue

            # GradNorm-weighted total loss (guard against length mismatch)
            n_active = min(len(task_losses), self.loss_weights.shape[0])
            weights = torch.softmax(self.loss_weights[:n_active], dim=0)
            total_loss = sum(w * l for w, l in zip(weights, task_losses[:n_active]))

            total_loss.backward()
            # Clip gradients for all trained parameters
            all_params = list(self.vit.parameters())
            if self.bridge is not None:
                all_params += list(self.bridge.parameters())
            if self.dec is not None:
                all_params += list(self.dec.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            self.optimizer.step()

            total_loss_sum += total_loss.item()

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        n = len(self.train_loader)
        acc = correct / max(total, 1)
        metrics = {
            "total_loss": total_loss_sum / n,
            "cls_loss": cls_loss_sum / n,
            "sem_loss": sem_loss_sum / n,
            "dec_loss": dec_loss_sum / n,
            "train_acc": acc,
        }
        print(
            f"Epoch {epoch} | loss={metrics['total_loss']:.4f} "
            f"| cls={metrics['cls_loss']:.4f} "
            f"| sem={metrics['sem_loss']:.4f} "
            f"| dec={metrics['dec_loss']:.4f} "
            f"| acc={acc:.4f}",
            flush=True,
        )
        return metrics

    @torch.no_grad()
    def _validate_epoch(self, epoch):
        self.vit.eval()
        if self.bridge is not None:
            self.bridge.eval()

        correct = 0
        total = 0
        val_loss_sum = 0.0

        for batch in self.val_loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits, _ = self.vit(images, return_patch_embeddings=True)
            loss = self.cls_loss_fn(logits, labels)
            val_loss_sum += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        val_loss = val_loss_sum / len(self.val_loader)
        print(
            f"Epoch {epoch} | Val loss={val_loss:.4f} | Val acc={acc:.4f}",
            flush=True,
        )
        return {"val_loss": val_loss, "val_acc": acc}

    @torch.no_grad()
    def _update_dec_targets(self):
        """Periodically log clustering statistics over the full training set."""
        self.vit.eval()
        self.dec.eval()
        all_q = []
        for batch in self.train_loader:
            images = batch[0].to(self.device)
            _, patch_emb = self.vit(images, return_patch_embeddings=True)
            cls_token = patch_emb[:, 0, :]
            q = self.dec(cls_token)
            all_q.append(q.cpu())
        all_q = torch.cat(all_q, dim=0)
        cluster_assignments = all_q.argmax(dim=1)
        counts = torch.bincount(cluster_assignments, minlength=self.dec_n_clusters)
        print(f"DEC cluster distribution: {counts.tolist()}", flush=True)

    def _save_final(self, epoch):
        """Save final model weights and DEC centers."""
        final_path = os.path.join(self.save_dir, "final_model.pth")
        torch.save(self.vit.state_dict(), final_path)
        print(f"Final model saved to: {final_path}", flush=True)

        if self.dec is not None and hasattr(self.dec, "cluster_centers"):
            torch.save(self.dec.cluster_centers, self.dec_centers_path)
            print(f"DEC centers saved to: {self.dec_centers_path}", flush=True)
