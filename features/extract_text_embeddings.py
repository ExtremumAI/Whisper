# features/extract_text_embeddings.py
# Extract BioBERT text embeddings for a list of medical class terms.
# Saves a NumPy array of shape (n_classes, embedding_dim) to disk.
#
# Usage:
#   python features/extract_text_embeddings.py \
#       --terms "chorditis" "vocal fold polyp" "leukoplakia" \
#       --output_path text_embeddings.npy
#
#   python features/extract_text_embeddings.py \
#       --terms_file class_terms.txt \
#       --output_path text_embeddings.npy \
#       --model_name dmis-lab/biobert-base-cased-v1.2

import os
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def load_model(model_name: str):
    """Load a HuggingFace tokenizer and model."""
    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_terms(
    terms: list,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
    pooling: str = "cls",
) -> np.ndarray:
    """
    Encode a list of text terms into embeddings.

    Args:
        terms:      List of strings to encode.
        tokenizer:  HuggingFace tokenizer.
        model:      HuggingFace transformer model.
        device:     Torch device.
        batch_size: Number of terms to process at once.
        pooling:    'cls' to use the [CLS] token, 'mean' to average all tokens.

    Returns:
        (N, D) NumPy array of embeddings.
    """
    model.to(device)
    all_embeddings = []

    for i in range(0, len(terms), batch_size):
        batch_terms = terms[i: i + batch_size]
        encoded = tokenizer(
            batch_terms,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        # outputs.last_hidden_state: (B, L, D)
        hidden_states = outputs.last_hidden_state

        if pooling == "cls":
            embeddings = hidden_states[:, 0, :]   # [CLS] token
        elif pooling == "mean":
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: '{pooling}'")

        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract BioBERT embeddings for medical class terms."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--terms",
        nargs="+",
        metavar="TERM",
        help="One or more medical class terms to embed (space-separated).",
    )
    group.add_argument(
        "--terms_file",
        type=str,
        metavar="FILE",
        help="Path to a plain-text file with one term per line.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the .npy embedding file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.2",
        help="HuggingFace model name or local path for the text encoder.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling strategy for extracting sentence embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect terms
    if args.terms is not None:
        terms = args.terms
    else:
        with open(args.terms_file, "r") as f:
            terms = [line.strip() for line in f if line.strip()]

    print(f"Encoding {len(terms)} term(s):", flush=True)
    for i, t in enumerate(terms):
        print(f"  [{i}] {t}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_name)

    embeddings = encode_terms(
        terms=terms,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )

    print(f"Embedding shape: {embeddings.shape}", flush=True)

    # Save output
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output_path, embeddings)
    print(f"Embeddings saved to: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
