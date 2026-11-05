"""Train AuthenticVision from Tiny-GenImage on a cloud GPU or local machine.

This project trains a new real-vs-AI classifier by fine-tuning an ImageNet-
pretrained ConvNeXt-Tiny model. One AI generator is held out from training and
used for a separate generalization test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from av_model import MODEL_NAME, build_model, build_transforms
from hf_tiny_data import DATASET_ID, build_splits
from metrics_utils import calculate_metrics, calibrate_threshold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_hf_dataset(dataset_id: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is missing. Install it with: pip install datasets"
        ) from exc
    print(f"Loading dataset: {dataset_id}")
    print("First download can be several GB; the dataset is cached for later runs.")
    dataset = load_dataset(dataset_id)
    print("Available splits:", {name: len(split) for name, split in dataset.items()})
    return dataset


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, device,
                balanced: bool = False) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if balanced:
        labels = [int(y) for y in dataset.labels]
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Training split must contain real and AI images; counts={counts.tolist()}")
        item_weights = [1.0 / float(counts[y]) for y in labels]
        # Approximately equal real/AI class exposure per epoch, with replacement.
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(item_weights, dtype=torch.double),
            num_samples=int(2 * max(counts)),
            replacement=True,
        )
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(**kwargs)


def predict_loader(model, loader, device, criterion=None, description="Evaluating"):
    model.eval()
    all_y, all_p = [], []
    loss_sum, count = 0.0, 0
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels) if criterion is not None else None
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            all_y.extend(labels.cpu().numpy().astype(int).tolist())
            all_p.extend(probs.cpu().numpy().astype(float).tolist())
            if loss is not None:
                loss_sum += float(loss.item()) * int(labels.shape[0])
                count += int(labels.shape[0])
    avg_loss = loss_sum / count if count else None
    return np.asarray(all_y, dtype=np.int64), np.asarray(all_p, dtype=np.float64), avg_loss


def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, epoch=None, threshold=0.5,
                    metadata=None, best_val_auc=None):
    checkpoint = {
        "model_name": MODEL_NAME,
        "class_names": ["real", "ai"],
        "image_size": int((metadata or {}).get("image_size", 224)),
        "threshold": float(threshold),
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch or 0),
        "best_val_auc": None if best_val_auc is None else float(best_val_auc),
        "metadata": json_safe(metadata or {}),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def write_predictions_csv(path: Path, y_true, prob_ai, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = (np.asarray(prob_ai) >= threshold).astype(int)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_number", "true_label", "true_name", "probability_ai",
                         "predicted_label", "predicted_name", "correct"])
        for i, (y, p, pred) in enumerate(zip(y_true, prob_ai, preds), start=1):
            writer.writerow([i, int(y), "ai" if y else "real", f"{float(p):.8f}",
                             int(pred), "ai" if pred else "real", bool(int(y) == int(pred))])


def save_plots(output_dir: Path, y_true, prob_ai, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_ai, dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted real", "Predicted AI"])
    ax.set_yticks([0, 1], labels=["Actual real", "Actual AI"])
    ax.set_title("Test split confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=140)
    plt.close(fig)

    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label="Test split")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
        ax.set_xlabel("False-positive rate on real images")
        ax.set_ylabel("True-positive rate on AI images")
        ax.set_title("Test split ROC curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "test_roc_curve.png", dpi=140)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", default=DATASET_ID, help="Hugging Face dataset ID")
    p.add_argument("--output-dir", default="outputs", help="Where to save checkpoints and reports")
    p.add_argument("--holdout-generator", default="all", help="Use all generators for deployment training, or specify one generator to run a leave-one-generator-out research test")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the classifier head")
    p.add_argument("--backbone-learning-rate", type=float, default=1e-5, help="Lower learning rate for pretrained visual backbone")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-train-per-generator", type=int, default=2000,
                   help="Maximum AI images selected per generator")
    p.add_argument("--max-val-per-class", type=int, default=1000)
    p.add_argument("--max-test-per-class", type=int, default=1000)
    p.add_argument("--target-fpr", type=float, default=0.05,
                   help="Target validation false-positive rate for real images")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="", help="Path to last_checkpoint.pt to resume training")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train ConvNeXt-Tiny from random initialization (not recommended)")
    p.add_argument("--cpu", action="store_true", help="Force CPU (cloud GPU training is recommended)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: CPU training will be much slower. A cloud GPU is recommended.")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    dataset_dict = load_hf_dataset(args.dataset_id)
    train_tf, eval_tf = build_transforms(args.image_size)
    splits = build_splits(
        dataset_dict,
        train_transform=train_tf,
        eval_transform=eval_tf,
        holdout_generator=args.holdout_generator,
        max_train_per_generator=args.max_train_per_generator,
        max_val_per_class=args.max_val_per_class,
        max_test_per_class=args.max_test_per_class,
        seed=args.seed,
    )
    print("Generator labels:", ", ".join(splits.generator_names))
    if splits.holdout_generator.startswith("ALL"):
        print("Training mode: all generator families present in the training split are included.")
    else:
        print("Held-out generator (not used for training):", splits.holdout_generator)
    print("Evaluation protocol:", splits.counts.get("evaluation_protocol"))
    print("Split sizes:", json.dumps(json_safe(splits.counts), indent=2))
    (output_dir / "split_counts.json").write_text(
        json.dumps(json_safe(splits.counts), indent=2), encoding="utf-8"
    )

    train_loader = make_loader(splits.train, args.batch_size, True, args.num_workers, device, balanced=True)
    val_loader = make_loader(splits.validation, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(splits.holdout_test, args.batch_size, False, args.num_workers, device)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.backbone_learning_rate},
            {"params": model.classifier.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, threshold=0.002,
        min_lr=[1e-6, 1e-5],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    start_epoch = 1
    best_val_auc = -1.0
    best_val_loss = float("inf")
    history = []
    history_path = output_dir / "training_history.csv"
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_model.pt"

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as exc:
                raise RuntimeError(
                    "This checkpoint was made by an older training version with a different optimizer layout. "
                    "Start a fresh run in a NEW output folder instead of resuming that checkpoint."
                ) from exc
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        saved_best = ckpt.get("best_val_auc")
        best_val_auc = float(saved_best) if saved_best is not None else -1.0
        if history_path.exists():
            with history_path.open("r", newline="", encoding="utf-8") as history_file:
                history = list(csv.DictReader(history_file))

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item()) * int(labels.shape[0])
            seen += int(labels.shape[0])
            progress.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")

        train_loss = running_loss / max(seen, 1)
        val_y, val_p, val_loss = predict_loader(model, val_loader, device, criterion, "Validation")
        threshold = calibrate_threshold(val_y, val_p, args.target_fpr)
        val_metrics = calculate_metrics(val_y, val_p, threshold)
        val_auc = val_metrics.get("roc_auc")
        # Use validation AUC for ranking quality; monitor loss separately and
        # lower both learning rates when validation loss plateaus.
        scheduler.step(float(val_loss if val_loss is not None else math.inf))
        score_for_selection = float(val_auc) if val_auc is not None else -float(val_loss or math.inf)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={float(val_loss or 0):.4f}, "
            f"val_auc={val_auc}, val_accuracy={val_metrics.get('accuracy'):.4f}, "
            f"calibrated_threshold={threshold:.8g}, time={time.time() - epoch_start:.1f}s"
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_auc": val_auc,
            "validation_accuracy_at_calibrated_threshold": val_metrics.get("accuracy"),
            "threshold": threshold,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(history_row)
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        metadata = {
            "dataset_id": args.dataset_id,
            "generator_training_mode": "all_generators" if splits.holdout_generator.startswith("ALL") else f"leave_one_generator_out:{splits.holdout_generator}",
            "evaluation_protocol": splits.counts.get("evaluation_protocol"),
            "train_generators": list(splits.counts.get("per_generator_train_ai", {}).keys()),
            "image_size": args.image_size,
            "pretrained_imagenet_initialization": not args.no_pretrained,
            "backbone_learning_rate": args.backbone_learning_rate,
            "head_learning_rate": args.learning_rate,
            "label_smoothing": 0.05,
            "train_counts": splits.counts,
            "validation_metrics": json_safe(val_metrics),
            "target_validation_fpr": args.target_fpr,
            "torch_version": str(torch.__version__),
        }
        improved = score_for_selection > best_val_auc
        if improved:
            best_val_auc = score_for_selection
            best_val_loss = float(val_loss or math.inf)
            save_checkpoint(
                best_path, model, None, None, epoch, threshold, metadata, best_val_auc
            )
            print(f"Saved best model: {best_path}")
        # Save the current epoch and the updated best score after model selection.
        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, threshold, metadata, best_val_auc
        )

    if not best_path.exists():
        raise RuntimeError("Training did not produce best_model.pt; check logs above.")

    # Evaluate on disjoint images from the dataset's validation split.
    # In all-generators mode, test generators have also appeared in training;
    # this is a deployment benchmark, not a generator-held-out benchmark.
    best_model, best_ckpt = load_model_from_checkpoint(best_path, device)
    val_y, val_p, _ = predict_loader(best_model, val_loader, device, None, "Final validation")
    final_threshold = calibrate_threshold(val_y, val_p, args.target_fpr)
    test_y, test_p, _ = predict_loader(best_model, test_loader, device, None, "Test split evaluation")
    holdout_metrics = calculate_metrics(test_y, test_p, final_threshold)
    val_metrics = calculate_metrics(val_y, val_p, final_threshold)

    # Record calibrated threshold in the final deployable checkpoint.
    best_ckpt["threshold"] = float(final_threshold)
    best_ckpt["validation_metrics"] = json_safe(val_metrics)
    best_ckpt["test_metrics"] = json_safe(holdout_metrics)
    best_ckpt["evaluation_protocol"] = splits.counts.get("evaluation_protocol")
    if not splits.holdout_generator.startswith("ALL"):
        best_ckpt["holdout_metrics"] = json_safe(holdout_metrics)
    temp_path = best_path.with_suffix(".pt.tmp")
    torch.save(best_ckpt, temp_path)
    temp_path.replace(best_path)

    write_predictions_csv(output_dir / "test_predictions.csv", test_y, test_p, final_threshold)
    if not splits.holdout_generator.startswith("ALL"):
        write_predictions_csv(output_dir / "holdout_predictions.csv", test_y, test_p, final_threshold)
    save_plots(output_dir, test_y, test_p, final_threshold)
    report = {
        "model": MODEL_NAME,
        "dataset_id": args.dataset_id,
        "generator_training_mode": "all_generators" if splits.holdout_generator.startswith("ALL") else f"leave_one_generator_out:{splits.holdout_generator}",
        "evaluation_protocol": splits.counts.get("evaluation_protocol"),
        "training_generators": list(splits.counts.get("per_generator_train_ai", {}).keys()),
        "training_epochs_requested": args.epochs,
        "best_checkpoint_epoch": int(best_ckpt.get("epoch", 0)),
        "device": str(device),
        "threshold": float(final_threshold),
        "target_validation_fpr": args.target_fpr,
        "validation_metrics": json_safe(val_metrics),
        "test_metrics": json_safe(holdout_metrics),
        "split_counts": json_safe(splits.counts),
        "notes": [
            splits.counts.get("evaluation_protocol", "Evaluation protocol was not recorded."),
            "Threshold was calibrated on validation real images, not the test set.",
            "This result applies to this dataset/split; it is not a guarantee for all generators or real-world images.",
            "Tiny-GenImage is licensed CC-BY-NC-SA-4.0; check its terms before redistribution or commercial use.",
        ],
    }
    if not splits.holdout_generator.startswith("ALL"):
        report["held_out_generator"] = splits.holdout_generator
        report["held_out_generator_metrics"] = json_safe(holdout_metrics)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTest evaluation:")
    print("Protocol:", splits.counts.get("evaluation_protocol"))
    print(json.dumps(json_safe(holdout_metrics), indent=2))
    print(f"\nSaved model: {best_path}")
    print(f"Saved reports: {output_dir}")
    print("Resume after interruption with --resume \"" + str(last_path) + "\"")


def load_model_from_checkpoint(path: Path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    main()
