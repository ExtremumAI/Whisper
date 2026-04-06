<img width="2048" height="2048" alt="IEEE稿件撰写指南" src="https://github.com/user-attachments/assets/c78708cd-1b6e-417d-b62b-bce5cffdcaf5" />
<img width="2048" height="2048" alt="IEEE稿件撰写指南" src="https://github.com/user-attachments/assets/f2497597-5b54-475a-ba82-059654d8f6d0" />
# Whisper
**Semantic-Guided Multi-Granularity Learning for Medical Image Classification**

Whisper jointly learns image-level classification and patch-level subtype discovery by bridging medical text embeddings with visual patch features through cross-modal attention. It requires only a single medical term per class, with no pretraining or image-text paired data.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/whisper-framework.git
cd whisper-framework
pip install -r requirements.txt
```

## Usage

```bash
# Full Whisper
python main.py \
    --tasks classification semantic clustering \
    --ablation_mode full \
    --vit_num_labels 6 \
    --save_dir ./saved_models/whisper_full

# Baseline (ViT + DEC)
python main.py \
    --tasks classification clustering \
    --vit_num_labels 6 \
    --save_dir ./saved_models/baseline

# Ablation: random attention / fixed text
python main.py \
    --tasks classification semantic clustering \
    --ablation_mode random_attention \
    --vit_num_labels 6 \
    --save_dir ./saved_models/ablation_random
```



## License

[Apache License 2.0](LICENSE)

## Disclaimer

For research purposes only. Not approved for clinical use.
