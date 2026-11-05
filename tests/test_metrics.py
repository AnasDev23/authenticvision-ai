import unittest

from metrics_utils import calculate_metrics, calibrate_threshold


class MetricsTests(unittest.TestCase):
    def test_metrics_perfect_predictions(self):
        y = [0, 0, 1, 1]
        p = [0.05, 0.20, 0.80, 0.95]
        result = calculate_metrics(y, p, threshold=0.5)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["false_positive_rate_real_as_ai"], 0.0)
        self.assertAlmostEqual(result["false_negative_rate_ai_as_real"], 0.0)
        self.assertAlmostEqual(result["roc_auc"], 1.0)

    def test_threshold_uses_only_real_scores(self):
        y = [0, 0, 0, 0, 1, 1]
        p = [0.10, 0.20, 0.30, 0.40, 0.70, 0.90]
        threshold = calibrate_threshold(y, p, target_fpr=0.25)
        self.assertGreaterEqual(threshold, 0.4)

    def test_empty_metrics(self):
        self.assertEqual(calculate_metrics([], [], 0.5)["n_samples"], 0)


if __name__ == "__main__":
    unittest.main()
