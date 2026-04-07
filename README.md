<img width="144" height="144" alt="Whisper Logo" src="https://github.com/user-attachments/assets/c76a10d1-3e2d-4324-bbd0-f7256656b61e" />
# Whisper: Simple Semantic Guidance for Multi-Granularity Medical Image Understanding

Whisper is a multi-task learning framework for medical image classification that jointly optimizes image-level classification and patch-level subtype clustering. It bridges medical text embeddings with visual patch features through cross-modal attention, requiring only a single medical term per class—no pretraining or image-text paired data needed.

## Installation

```bash
git clone https://github.com/ExtremumAI/Whisper.git
cd Whisper
pip install -r requirements.txt
```

## Usage

```bash
# Full Whisper (classification + semantic bridge + clustering)
python main.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --text_embedding_path /path/to/text_embeddings.npy \
    --tasks classification semantic clustering \
    --ablation_mode full \
    --vit_num_labels 6 \
    --save_dir ./saved_models/whisper_full

# Baseline (ViT + DEC, no semantic bridge)
python main.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --tasks classification clustering \
    --vit_num_labels 6 \
    --save_dir ./saved_models/baseline

# Ablation: random attention bridge
python main.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --text_embedding_path /path/to/text_embeddings.npy \
    --tasks classification semantic clustering \
    --ablation_mode random_attention \
    --vit_num_labels 6 \
    --save_dir ./saved_models/ablation_random

# Extract text embeddings (BioBERT)
python features/extract_text_embeddings.py \
    --terms "chorditis" "vocal fold polyp" "leukoplakia" \
    --output_path text_embeddings.npy
```

## Project Structure

```
Whisper/
├── main.py                          # Training entry point
├── trainer.py                       # JointTrainer: multi-task training loop
├── models/
│   ├── vit.py                       # ViT backbone with MaskedAttention
│   ├── dec.py                       # Deep Embedded Clustering (DEC) module
│   ├── cross_attention.py           # Cross-attention module
│   ├── semantic_bridge.py           # Semantic bridge (full Whisper)
│   ├── random_attention_bridge.py   # Ablation: random attention bridge
│   └── fixed_text_bridge.py         # Ablation: fixed text embedding bridge
├── losses/
│   └── classification_loss.py       # Classification loss functions
├── utils/
│   ├── customdatasetloader.py       # Dataset loader (ImageFolder + ViT processor)
│   ├── dec_utils.py                 # DEC utilities (cluster accuracy, target dist.)
│   └── utils.py                     # General utilities (EarlyStopping, GradNorm, etc.)
└── features/
    └── extract_text_embeddings.py   # BioBERT-based text embedding extraction
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{
  title   = {Whisper: Semantic-Guided Multi-Granularity Learning for Medical Image Classification},
  author  = {Zhen Jiang, Yan Zhang, Shuxin Yin, Zhaoyan Yu, Liang Sun},
  journal = {},
  year    = {2026}
}
```

## License

[Apache License 2.0](LICENSE)

## Disclaimer

This software is intended for research purposes only and has not been approved for clinical use. The authors make no warranties regarding the accuracy or reliability of the model outputs for any medical application.
