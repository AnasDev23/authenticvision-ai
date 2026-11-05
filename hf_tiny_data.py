"""Data loading and generator-held-out split builder for Tiny-GenImage."""
from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset, ConcatDataset

DATASET_ID = "TheKernel01/Tiny-GenImage"


@dataclass
class DataSplits:
    train: Dataset
    validation: Dataset
    holdout_test: Dataset
    generator_names: list[str]
    holdout_generator: str
    counts: dict


class HFImageDataset(Dataset):
    """A view into a Hugging Face image dataset with PyTorch transforms."""
    def __init__(self, base_dataset, indices: list[int], transform, labels=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        # Pass scalar labels from metadata columns to avoid decoding every image at setup.
        self.labels = list(labels) if labels is not None else [int(base_dataset["label"][i]) for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row = self.base_dataset[self.indices[item]]
        image = row["image"]
        if image is None:
            raise RuntimeError(f"Missing image at dataset index {self.indices[item]}")
        image = image.convert("RGB")
        return self.transform(image), int(row["label"])  # 0=real, 1=fake.


def _subsample(indices: list[int], limit: int | None, seed: int) -> list[int]:
    if limit is None or limit <= 0 or len(indices) <= limit:
        return list(indices)
    return random.Random(seed).sample(list(indices), limit)


def _generator_label_names(dataset) -> list[str]:
    names = getattr(dataset.features["generator"], "names", None)
    if not names:
        raise ValueError("The dataset's 'generator' field has no ClassLabel names.")
    return list(names)


def _resolve_holdout(name: str, generator_names: list[str]) -> tuple[int, str]:
    wanted = name.strip().casefold()
    exact = [i for i, label in enumerate(generator_names) if label.casefold() == wanted]
    if exact:
        return exact[0], generator_names[exact[0]]
    partial = [i for i, label in enumerate(generator_names) if wanted in label.casefold()]
    if len(partial) == 1:
        return partial[0], generator_names[partial[0]]
    raise ValueError(
        f"Could not uniquely match holdout generator {name!r}. Available labels: "
        + ", ".join(generator_names)
    )


def build_splits(
    dataset_dict,
    train_transform,
    eval_transform,
    holdout_generator: str = "Midjourney",
    max_train_per_generator: int = 2500,
    max_val_per_class: int = 1000,
    max_test_per_class: int = 1000,
    validation_real_fraction: float = 0.20,
    seed: int = 42,
) -> DataSplits:
    """Build disjoint image splits for either deployment or generator-holdout mode.

    If holdout_generator names a generator, it is excluded from AI training and
    used in the final test. If holdout_generator='all', every generator present
    in the train split is used for training and validation/test are disjoint image
    subsets from the official validation split. In all-generator mode the test is
    not a test on unseen generator families.
    """
    if "train" not in dataset_dict or "validation" not in dataset_dict:
        raise ValueError("Dataset must contain 'train' and 'validation' splits.")
    train_base = dataset_dict["train"]
    val_base = dataset_dict["validation"]
    generator_names = _generator_label_names(train_base)
    all_generators_mode = holdout_generator.strip().casefold() in {
        "all", "none", "all-generators", "all_generators", "off", "disable"
    }
    if all_generators_mode:
        holdout_id = None
        resolved_name = "ALL generators included in training; test is image-disjoint, not generator-held-out"
    else:
        holdout_id, resolved_name = _resolve_holdout(holdout_generator, generator_names)

    # Reading scalar metadata columns does not decode all image files.
    train_labels = list(train_base["label"])
    train_generators = list(train_base["generator"])
    val_labels = list(val_base["label"])
    val_generators = list(val_base["generator"])

    train_real_all = [i for i, y in enumerate(train_labels) if int(y) == 0]
    train_ai_by_generator: dict[int, list[int]] = {}
    for i, (y, g) in enumerate(zip(train_labels, train_generators)):
        if int(y) == 1 and (all_generators_mode or int(g) != holdout_id):
            train_ai_by_generator.setdefault(int(g), []).append(i)
    if not train_real_all:
        raise ValueError("No real images (label 0) found in the train split.")
    if not train_ai_by_generator:
        raise ValueError("No AI images remain after excluding the held-out generator.")

    rng = random.Random(seed)
    rng.shuffle(train_real_all)
    if all_generators_mode:
        # Validation/test come from the official validation split in this mode,
        # so use every real image in the training split for training.
        val_real_indices = []
        train_real_indices = train_real_all
    else:
        val_real_count = max(1, int(len(train_real_all) * validation_real_fraction))
        val_real_indices = train_real_all[:val_real_count]
        train_real_indices = train_real_all[val_real_count:]

    train_ai_indices: list[int] = []
    per_generator_counts: dict[str, int] = {}
    for gen_id in sorted(train_ai_by_generator):
        gen_name = generator_names[gen_id]
        chosen = _subsample(
            train_ai_by_generator[gen_id], max_train_per_generator, seed + gen_id * 101
        )
        train_ai_indices.extend(chosen)
        per_generator_counts[gen_name] = len(chosen)
    train_indices = train_real_indices + train_ai_indices
    train_row_labels = ([0] * len(train_real_indices)) + ([1] * len(train_ai_indices))
    train_subset = HFImageDataset(train_base, train_indices, train_transform, train_row_labels)

    if all_generators_mode:
        # For the deployable model, train on every AI generator available in the
        # training split. Split the official validation split into disjoint
        # validation and test pools, stratified by generator to preserve diversity.
        split_rng = random.Random(seed + 5000)
        val_real_pool = [i for i, y in enumerate(val_labels) if int(y) == 0]
        split_rng.shuffle(val_real_pool)
        real_mid = len(val_real_pool) // 2
        val_real_candidates = val_real_pool[:real_mid]
        test_real_candidates = val_real_pool[real_mid:]

        val_ai_candidates, test_ai_candidates = [], []
        val_ai_by_generator: dict[int, list[int]] = {}
        for i, (y, g) in enumerate(zip(val_labels, val_generators)):
            if int(y) == 1:
                val_ai_by_generator.setdefault(int(g), []).append(i)
        for gen_id in sorted(val_ai_by_generator):
            group = list(val_ai_by_generator[gen_id])
            split_rng.shuffle(group)
            mid = len(group) // 2
            val_ai_candidates.extend(group[:mid])
            test_ai_candidates.extend(group[mid:])

        val_count = min(len(val_real_candidates), len(val_ai_candidates), max_val_per_class)
        test_count = min(len(test_real_candidates), len(test_ai_candidates), max_test_per_class)
        if val_count < 1 or test_count < 1:
            raise ValueError("Not enough images to make disjoint validation and test splits.")

        val_real_selected = _subsample(val_real_candidates, val_count, seed + 2001)
        val_ai_selected = _subsample(val_ai_candidates, val_count, seed + 2002)
        test_real_selected = _subsample(test_real_candidates, test_count, seed + 3001)
        test_ai_selected = _subsample(test_ai_candidates, test_count, seed + 3002)

        validation_indices = val_real_selected + val_ai_selected
        validation_labels = [0] * len(val_real_selected) + [1] * len(val_ai_selected)
        validation_subset = HFImageDataset(
            val_base, validation_indices, eval_transform, validation_labels
        )
        test_indices = test_real_selected + test_ai_selected
        test_labels = [0] * len(test_real_selected) + [1] * len(test_ai_selected)
        holdout_test_subset = HFImageDataset(
            val_base, test_indices, eval_transform, test_labels
        )
        evaluation_protocol = "all_generators_seen_during_training; validation/test images are disjoint but test generators are not held out"
    else:
        val_ai_indices = [
            i for i, (y, g) in enumerate(zip(val_labels, val_generators))
            if int(y) == 1 and int(g) != holdout_id
        ]
        val_count = min(len(val_real_indices), len(val_ai_indices), max_val_per_class)
        if val_count < 1:
            raise ValueError("Not enough validation real/AI examples for validation.")
        val_real_selected = _subsample(val_real_indices, val_count, seed + 2001)
        val_ai_selected = _subsample(val_ai_indices, val_count, seed + 2002)
        validation_subset = ConcatDataset([
            HFImageDataset(train_base, val_real_selected, eval_transform, [0] * len(val_real_selected)),
            HFImageDataset(val_base, val_ai_selected, eval_transform, [1] * len(val_ai_selected)),
        ])

        test_real_all = [i for i, y in enumerate(val_labels) if int(y) == 0]
        test_ai_holdout = [
            i for i, (y, g) in enumerate(zip(val_labels, val_generators))
            if int(y) == 1 and int(g) == holdout_id
        ]
        test_count = min(len(test_real_all), len(test_ai_holdout), max_test_per_class)
        if test_count < 1:
            raise ValueError(
                f"No holdout test examples found for {resolved_name!r}. Check dataset version."
            )
        test_real_selected = _subsample(test_real_all, test_count, seed + 3001)
        test_ai_selected = _subsample(test_ai_holdout, test_count, seed + 3002)
        test_row_labels = ([0] * len(test_real_selected)) + ([1] * len(test_ai_selected))
        holdout_test_subset = HFImageDataset(
            val_base, test_real_selected + test_ai_selected, eval_transform, test_row_labels
        )
        evaluation_protocol = f"generator-held-out test: {resolved_name} fake images were excluded from training"

    counts = {
        "train_real": len(train_real_indices),
        "train_ai": len(train_ai_indices),
        "validation_real": len(val_real_selected),
        "validation_ai": len(val_ai_selected),
        "holdout_test_real": len(test_real_selected),
        "holdout_test_ai": len(test_ai_selected),
        "per_generator_train_ai": per_generator_counts,
        "evaluation_protocol": evaluation_protocol,
    }
    return DataSplits(
        train=train_subset,
        validation=validation_subset,
        holdout_test=holdout_test_subset,
        generator_names=generator_names,
        holdout_generator=resolved_name,
        counts=counts,
    )
