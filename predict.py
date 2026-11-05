"""Run one trained AuthenticVision checkpoint on a local image."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from av_model import build_transforms, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--image", required=True, help="Image to analyse")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override trained threshold; normally leave unset")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(str(checkpoint_path), device=device)
    _, transform = build_transforms(int(ckpt.get("image_size", 224)))
    threshold = float(args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5))
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probability_ai = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
    verdict = "LIKELY AI-GENERATED" if probability_ai >= threshold else "LIKELY REAL"
    print(f"Image: {image_path}")
    print(f"AI probability: {probability_ai:.1%}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Verdict: {verdict}")
    print("Warning: this is a model estimate, not proof of the image's origin.")


if __name__ == "__main__":
    main()
