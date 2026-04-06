# utils/customdatasetloader.py
# Dataset loader for the Whisper framework.
# Builds train/val DataLoaders from ImageFolder-structured directories
# using the HuggingFace ViT feature extractor for preprocessing.

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

try:
    from transformers import ViTImageProcessor
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ImageNet normalization constants (used as default)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViTPreprocessTransform:
    """
    Wraps the HuggingFace ViTImageProcessor so it can be used as a
    torchvision transform.  Falls back to standard resize + normalize if
    the transformers library is not installed.

    Args:
        processor_name: HuggingFace model name or local path.
        image_size:     Target image size (height == width).
    """

    def __init__(
        self,
        processor_name: str = "google/vit-base-patch16-224",
        image_size: int = 224,
    ):
        if _HF_AVAILABLE:
            self.processor = ViTImageProcessor.from_pretrained(processor_name)
            self._use_hf = True
        else:
            self._use_hf = False
            self._fallback = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self._use_hf:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)  # (C, H, W)
        return self._fallback(image)


class ImageFolderWithTransform(Dataset):
    """
    ImageFolder dataset that applies a callable transform to each image.

    Args:
        root:      Root directory containing class subdirectories.
        transform: Callable applied to each PIL image.
    """

    def __init__(self, root: str, transform=None):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        path, label = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    processor_name: str = "google/vit-base-patch16-224",
    image_size: int = 224,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and optional validation DataLoaders.

    Args:
        train_dir:      Path to the training data (ImageFolder format).
        val_dir:        Path to the validation data (ImageFolder format).
                        If None, only the train loader is returned.
        batch_size:     Number of samples per batch.
        num_workers:    Number of DataLoader worker processes.
        processor_name: HuggingFace ViT processor name or local path.
        image_size:     Target image resolution.
        pin_memory:     Pin memory for faster GPU transfers.
        drop_last:      Drop the last incomplete training batch (default: False).

    Returns:
        (train_loader, val_loader) — val_loader is None if val_dir is None.
    """
    preprocess = ViTPreprocessTransform(
        processor_name=processor_name,
        image_size=image_size,
    )

    train_dataset = ImageFolderWithTransform(root=train_dir, transform=preprocess)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = None
    if val_dir is not None and os.path.isdir(val_dir):
        val_dataset = ImageFolderWithTransform(root=val_dir, transform=preprocess)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader
