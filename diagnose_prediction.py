"""Print full-precision prediction diagnostics for a local image."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageOps
from av_model import build_transforms, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/best_model.pt')
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    checkpoint_path, image_path = Path(args.checkpoint), Path(args.image)
    if not checkpoint_path.is_file():
        raise SystemExit(f'Checkpoint not found: {checkpoint_path.resolve()}')
    if not image_path.is_file():
        raise SystemExit(f'Image not found: {image_path.resolve()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_checkpoint(str(checkpoint_path), device)
    _, transform = build_transforms(int(ckpt.get('image_size', 224)))
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor).float().cpu()[0]
        probabilities = torch.softmax(logits, dim=0)
    p_real, p_ai = float(probabilities[0]), float(probabilities[1])
    threshold = float(ckpt.get('threshold', 0.5))
    print(f'Image: {image_path.resolve()}')
    print(f'Checkpoint: {checkpoint_path.resolve()}')
    print(f'Checkpoint epoch: {ckpt.get("epoch", "unknown")}')
    print(f'Class names in checkpoint: {ckpt.get("class_names", "not saved")}')
    print(f'Stored trained threshold: {threshold:.10f}')
    print(f'Logit class 0 (real): {float(logits[0]):.10f}')
    print(f'Logit class 1 (AI):   {float(logits[1]):.10f}')
    print(f'P(real): {p_real:.10f} ({p_real*100:.6f}%)')
    print(f'P(AI):   {p_ai:.10f} ({p_ai*100:.6f}%)')
    print(f'Verdict at stored threshold: {"LIKELY AI-GENERATED" if p_ai >= threshold else "LIKELY REAL"}')
    print(f'Verdict at 0.5 threshold:    {"LIKELY AI-GENERATED" if p_ai >= 0.5 else "LIKELY REAL"}')
    if isinstance(ckpt.get('holdout_metrics'), dict):
        print('Checkpoint holdout metrics:', ckpt['holdout_metrics'])
    print('Note: a result outside the training data distribution can be wrong.')

if __name__ == '__main__':
    main()
