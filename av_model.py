"""Model definition and image preprocessing for AuthenticVision GenImage Trainer."""
from __future__ import annotations

from typing import Callable
import io
import random

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

MODEL_NAME = "convnext_tiny"
CLASS_NAMES = ["real", "ai"]  # Positive class is AI-generated, index 1.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(pretrained: bool = True) -> nn.Module:
    """Build a ConvNeXt-Tiny image classifier with two output classes.

    If pretrained=True, the backbone starts from torchvision ImageNet-1K
    weights. Training then fine-tunes this model for real-vs-AI classification.
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


class RandomJPEGCompression:
    """Simulate common social-media JPEG compression during training."""
    def __init__(self, probability: float = 0.35, quality_range: tuple[int, int] = (55, 95)):
        self.probability = float(probability)
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        quality = random.randint(*self.quality_range)
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()


def build_transforms(image_size: int = 224) -> tuple[Callable, Callable]:
    """Return robust augmented training transforms and deterministic evaluation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.85, 1.0),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06, hue=0.015)
        ], p=0.25),
        RandomJPEGCompression(probability=0.35, quality_range=(55, 95)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Load one of this project's checkpoints for inference/evaluation."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location=device)
    model = build_model(pretrained=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError(
            "Checkpoint has no 'model_state_dict'. Please use best_model.pt "
            "created by this project."
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint
