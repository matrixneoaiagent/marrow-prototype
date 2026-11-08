import json
from pathlib import Path
import tempfile
import unittest

from marrow.features import FEATURE_NAMES
from marrow.model import (
    FeatureNormalizer,
    ProfileModel,
    choose_threshold,
    roc_auc,
    wilson_interval,
)


def row(offset: float) -> list[float | None]:
    values: list[float | None] = []
    for index, name in enumerate(FEATURE_NAMES):
        if name.startswith("f_"):
            values.append(1.0)
        else:
            values.append(offset + 0.01 * (index % 7))
    return values


class ModelTests(unittest.TestCase):
    def test_profile_separates_and_round_trips(self):
        positives = [row(2.0 + 0.02 * i) for i in range(8)]
        negatives = [row(0.0 + 0.02 * i) for i in range(8)]
        normalizer = FeatureNormalizer.fit(positives + negatives)
        model = ProfileModel.fit(positives, negatives, normalizer=normalizer)
        model.set_anomaly_gate(positives)
        positive_scores = [model.score_features(r) for r in positives]
        negative_scores = [model.score_features(r) for r in negatives]
        self.assertGreater(min(positive_scores), max(negative_scores))
        self.assertGreater(roc_auc(positive_scores, negative_scores), 0.9)

        selection = choose_threshold(positive_scores, negative_scores, 0.05)
        self.assertLessEqual(selection["validation_far"], 0.05)
        self.assertEqual(selection["validation_frr"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            model.save(path)
            loaded = ProfileModel.load(path)
            self.assertEqual(loaded.threshold, model.threshold)
            self.assertEqual(len(loaded.weights), len(FEATURE_NAMES))
            self.assertEqual(loaded.score_features(positives[0]), model.score_features(positives[0]))

    def test_threshold_is_deterministic(self):
        positives = [0.91, 0.82, 0.77, 0.66]
        negatives = [0.21, 0.14, 0.09, 0.03]
        first = choose_threshold(positives, negatives, 0.05)
        second = choose_threshold(positives, negatives, 0.05)
        self.assertEqual(first, second)
        self.assertEqual(first["validation_far"], 0.0)
        self.assertEqual(first["validation_frr"], 0.0)

    def test_wilson_interval(self):
        low, high = wilson_interval(0, 36)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.2)


if __name__ == "__main__":
    unittest.main()
