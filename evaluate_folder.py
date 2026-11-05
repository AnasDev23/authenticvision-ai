"""Evaluate a trained checkpoint on a separate image folder.

Expected layout:
  evaluation_data/real/*image files*
  evaluation_data/ai/*image files*
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from av_model import build_transforms, load_checkpoint
from metrics_utils import calculate_metrics

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="external_evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    root = Path(args.data_dir)
    samples = []
    for folder, label in [(root / "real", 0), (root / "ai", 1)]:
        if not folder.exists():
            raise SystemExit(f"Required folder not found: {folder}")
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        samples.extend((p, label) for p in files)
    if not samples:
        raise SystemExit("No image files found in real/ and ai/ folders.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    _, transform = build_transforms(int(ckpt.get("image_size", 224)))
    threshold = float(args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5))
    y_true, p_ai, rows = [], [], []
    model.eval()
    for path, label in tqdm(samples, desc="Evaluating images"):
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(tensor)
                prob = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
            y_true.append(label)
            p_ai.append(prob)
            pred = int(prob >= threshold)
            rows.append([str(path), label, prob, pred, int(label == pred)])
        except Exception as exc:
            print(f"Skipping unreadable file {path}: {exc}")
    if not y_true:
        raise SystemExit("All image files failed to load.")
    report = calculate_metrics(y_true, p_ai, threshold)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import json
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "probability_ai", "predicted_label", "correct"])
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
