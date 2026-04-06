# main.py
# Training entry point for the Whisper framework.

import os
import json
import argparse
import torch
from trainer import JointTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Whisper: Semantic-Guided Multi-Granularity Medical Image Classification"
    )

    # Data paths
    parser.add_argument(
        "--train_dir", type=str, required=True,
        help="Path to the training data directory (ImageFolder format)."
    )
    parser.add_argument(
        "--val_dir", type=str, default=None,
        help="Path to the validation data directory (ImageFolder format)."
    )
    parser.add_argument(
        "--text_embedding_path", type=str, default=None,
        help="Path to the pre-extracted text embedding file (.npy)."
    )
    parser.add_argument(
        "--pretrained_vit_path", type=str, default="google/vit-base-patch16-224",
        help="Path or HuggingFace model name for the pretrained ViT."
    )

    # Model configuration
    parser.add_argument(
        "--vit_num_labels", type=int, default=6,
        help="Number of output classes for the ViT classifier."
    )
    parser.add_argument(
        "--dec_n_clusters", type=int, default=6,
        help="Number of clusters for DEC."
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["classification", "semantic", "clustering"],
        choices=["classification", "semantic", "clustering"],
        help="Active training tasks."
    )
    parser.add_argument(
        "--ablation_mode", type=str, default="full",
        choices=["full", "random_attention", "fixed_text"],
        help="Ablation variant to use for the semantic bridge."
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the main optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--gradnorm_alpha", type=float, default=1.5,
                        help="Alpha value for GradNorm loss balancing.")
    parser.add_argument("--gradnorm_lr", type=float, default=1e-3,
                        help="Learning rate for GradNorm loss weights.")
    parser.add_argument("--dec_update_interval", type=int, default=5,
                        help="Epoch interval for updating DEC target distribution.")
    parser.add_argument("--dec_pretrain_epochs", type=int, default=10,
                        help="Number of epochs to pretrain ViT before DEC init.")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Patience (epochs) for early stopping.")

    # Output
    parser.add_argument(
        "--save_dir", type=str, default="./saved_models/whisper",
        help="Directory for saving checkpoints and logs."
    )

    return parser.parse_args()


def train_vit_only(args):
    """Pre-train the ViT backbone with classification loss only."""
    print("Pre-training ViT backbone (classification only)...")
    trainer = JointTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        text_embedding_path=args.text_embedding_path,
        pretrained_vit_path=args.pretrained_vit_path,
        vit_num_labels=args.vit_num_labels,
        dec_n_clusters=args.dec_n_clusters,
        tasks=["classification"],
        ablation_mode=args.ablation_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.dec_pretrain_epochs,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr,
        dec_update_interval=args.dec_update_interval,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir,
    )
    trainer.train()
    print("ViT pre-training complete.")


def save_experiment_config(args, save_dir):
    """Save the experiment configuration to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "experiment_config.json")
    config = vars(args)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config saved to: {config_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Whisper: Semantic-Guided Multi-Granularity Learning")
    print("=" * 60)
    print(f"  Tasks:          {args.tasks}")
    print(f"  Ablation mode:  {args.ablation_mode}")
    print(f"  Num classes:    {args.vit_num_labels}")
    print(f"  Num clusters:   {args.dec_n_clusters}")
    print(f"  Save dir:       {args.save_dir}")
    print("=" * 60)

    # Save experiment configuration
    save_experiment_config(args, args.save_dir)

    # Build and run trainer
    trainer = JointTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        text_embedding_path=args.text_embedding_path,
        pretrained_vit_path=args.pretrained_vit_path,
        vit_num_labels=args.vit_num_labels,
        dec_n_clusters=args.dec_n_clusters,
        tasks=args.tasks,
        ablation_mode=args.ablation_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr,
        dec_update_interval=args.dec_update_interval,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir,
    )
    trainer.train()

    print("Training complete.")


if __name__ == "__main__":
    main()
