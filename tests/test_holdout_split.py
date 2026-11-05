import unittest
from PIL import Image

from hf_tiny_data import build_splits


class FakeClassLabel:
    def __init__(self, names):
        self.names = names


class FakeSplit:
    def __init__(self, rows, generator_names):
        self.rows = rows
        self.features = {"generator": FakeClassLabel(generator_names)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def make_rows(real_count, fake_by_gen, generator_names, seed_color):
    rows = []
    for i in range(real_count):
        rows.append({"image": Image.new("RGB", (16, 16), (i % 255, 20, 20)), "label": 0, "generator": 0})
    for gen_id, count in fake_by_gen.items():
        for i in range(count):
            rows.append({"image": Image.new("RGB", (16, 16), (seed_color, gen_id, i % 255)), "label": 1, "generator": gen_id})
    return rows


class HoldoutSplitTests(unittest.TestCase):
    def test_heldout_generator_is_not_in_training(self):
        names = ["Real", "Midjourney", "ADM"]
        train = FakeSplit(make_rows(20, {1: 10, 2: 10}, names, 50), names)
        validation = FakeSplit(make_rows(10, {1: 8, 2: 8}, names, 100), names)
        splits = build_splits(
            {"train": train, "validation": validation},
            train_transform=lambda image: image,
            eval_transform=lambda image: image,
            holdout_generator="Midjourney",
            max_train_per_generator=10,
            max_val_per_class=10,
            max_test_per_class=10,
            seed=1,
        )
        train_gen = [train.rows[i]["generator"] for i in splits.train.indices]
        train_label = list(splits.train.labels)
        heldout_id = names.index("Midjourney")
        for gen, label in zip(train_gen, train_label):
            if label == 1:
                self.assertNotEqual(gen, heldout_id)
        self.assertEqual(len(splits.validation), splits.counts["validation_real"] + splits.counts["validation_ai"])
        self.assertEqual(len(splits.holdout_test), splits.counts["holdout_test_real"] + splits.counts["holdout_test_ai"])
        self.assertEqual(splits.counts["holdout_test_real"], splits.counts["holdout_test_ai"])


    def test_all_generators_mode_includes_midjourney_in_training_and_disjoint_test(self):
        names = ["Real", "Midjourney", "ADM", "SD14"]
        train = FakeSplit(make_rows(40, {1: 20, 2: 20, 3: 20}, names, 40), names)
        validation = FakeSplit(make_rows(40, {1: 20, 2: 20, 3: 20}, names, 80), names)
        splits = build_splits(
            {"train": train, "validation": validation},
            train_transform=lambda image: image,
            eval_transform=lambda image: image,
            holdout_generator="all",
            max_train_per_generator=20,
            max_val_per_class=15,
            max_test_per_class=15,
            seed=11,
        )
        train_generators = [train.rows[i]["generator"] for i in splits.train.indices]
        train_labels = list(splits.train.labels)
        self.assertIn(1, [g for g, y in zip(train_generators, train_labels) if y == 1])
        # Every validation/test sample comes from the official validation split,
        # and their indices must not overlap.
        val_indices = set(splits.validation.indices)
        test_indices = set(splits.holdout_test.indices)
        self.assertFalse(val_indices & test_indices)
        self.assertEqual(splits.counts["validation_real"], splits.counts["validation_ai"])
        self.assertEqual(splits.counts["holdout_test_real"], splits.counts["holdout_test_ai"])


if __name__ == "__main__":
    unittest.main()
