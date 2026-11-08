"""Profile fitting, scoring and the pre-registered threshold rule.

The model is a diagonal linear discriminant fitted to one person against a
background of other people.  It is deliberately small and interpretable:

* every feature is robustly standardised against the training population;
* a feature's weight is its standardised mean difference between the enrolled
  person and everyone else, shrunk towards zero and clipped;
* the score is a logistic transform of the linear discriminant;
* the accept/reject threshold is chosen on a validation split only, by a rule
  fixed in code, and written to a lock file before any test session is scored.

No network, no third-party library, no random initialisation at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .features import BINARY_FEATURES, FEATURE_NAMES

PROFILE_FORMAT = "marrow-profile-v1"
THRESHOLD_RULE = "lowest error trade-off: maximise TPR subject to validation FAR <= 0.05; ties go to the higher threshold"
TARGET_FAR = 0.05

# A mimic can copy the owner's averages and still differ in dispersion, path
# shape or burst rhythm.  This panel is deliberately two-sided: it rejects a
# session that is too regular as well as one that is too noisy, because both
# are departures from the enrolled person's own range.
ANOMALY_FEATURES: tuple[str, ...] = (
    "p_straight_med",
    "p_curvature_med_log",
    "p_jerk_med_log",
    "p_dwell_cv",
    "p_pause_cv",
    "p_stroke_speed_cv",
    "p_speed_cv",
    "p_dirchange_per_stroke",
    "t_typed_intra_cv",
    "t_typed_burst_chars_cv",
    "t_onscreen_intra_cv",
    "t_onscreen_burst_chars_cv",
    "t_soft_gap_cv",
    "t_soft_chars_cv",
    "t_correction_ratio",
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _var(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def _std(values: Sequence[float]) -> float:
    return math.sqrt(max(0.0, _var(values)))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return float(ordered[n // 2])
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(60.0, x))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-60.0, x))
    return z / (1.0 + z)


@dataclass
class FeatureNormalizer:
    """Robust standardisation, with one set of statistics per input modality.

    A mouse user and a switch user differ enormously in absolute speed and
    timing.  Comparing both to one global mean would make "uses a mouse" look
    like "is Alice".  The normalizer therefore groups training sessions by the
    *observed text-arrival modality* (typed, on-screen, software) and
    standardises inside that group.  The group is inferred from the recording
    itself, never from an OS or device field.  A recording with no text falls
    back to the global statistics.
    """

    median: list[float]
    scale: list[float]
    binary: list[bool]
    groups: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    @staticmethod
    def _group_from_row(row: Sequence[float | None]) -> str:
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}

        def value(name: str) -> float:
            raw = row[index[name]] if index[name] < len(row) else None
            return 0.0 if raw is None else float(raw)

        typed = max(value("t_modality_typed"), value("f_has_typed"))
        onscreen = max(value("t_modality_onscreen"), value("f_has_onscreen"))
        software = max(value("t_modality_software"), value("f_has_software"))
        if typed + onscreen + software <= 0.0:
            return "global"
        if software >= typed and software >= onscreen:
            return "software"
        if onscreen >= typed:
            return "onscreen"
        return "typed"

    @classmethod
    def fit(cls, matrix: Sequence[Sequence[float | None]]) -> "FeatureNormalizer":
        if not matrix:
            raise ValueError("cannot fit a normalizer on an empty matrix")
        width = len(FEATURE_NAMES)
        median: list[float] = []
        scale: list[float] = []
        binary: list[bool] = []
        for column in range(width):
            name = FEATURE_NAMES[column]
            values = [float(row[column]) for row in matrix if row[column] is not None]
            is_binary = name in BINARY_FEATURES
            if is_binary or len(values) < 2:
                median.append(0.0 if is_binary else _median(values))
                scale.append(1.0)
                binary.append(is_binary)
                continue
            med = _median(values)
            iqr = _quantile(values, 0.75) - _quantile(values, 0.25)
            robust = iqr / 1.349
            if robust < 1e-9:
                robust = _std(values)
            if robust < 1e-9:
                robust = 1.0
            median.append(med)
            scale.append(robust)
            binary.append(False)

        groups: dict[str, dict[str, list[float]]] = {}
        buckets: dict[str, list[Sequence[float | None]]] = {}
        for row in matrix:
            buckets.setdefault(cls._group_from_row(row), []).append(row)
        for group, rows in buckets.items():
            if len(rows) < 6:
                continue
            gmedian: list[float] = []
            gscale: list[float] = []
            for column in range(width):
                name = FEATURE_NAMES[column]
                if name in BINARY_FEATURES:
                    gmedian.append(0.0)
                    gscale.append(1.0)
                    continue
                values = [float(row[column]) for row in rows if row[column] is not None]
                if len(values) < 2:
                    gmedian.append(median[column])
                    gscale.append(scale[column])
                    continue
                med = _median(values)
                iqr = _quantile(values, 0.75) - _quantile(values, 0.25)
                robust = iqr / 1.349
                if robust < 1e-9:
                    robust = _std(values)
                if robust < 1e-9:
                    robust = 1.0
                gmedian.append(med)
                gscale.append(robust)
            groups[group] = {"median": gmedian, "scale": gscale}
        return cls(median=median, scale=scale, binary=binary, groups=groups)

    def transform_row(self, row: Sequence[float | None], group: str | None = None) -> list[float]:
        if group is None:
            group = self._group_from_row(row)
        stats = self.groups.get(group)
        medians = stats["median"] if stats else self.median
        scales = stats["scale"] if stats else self.scale
        out: list[float] = []
        for value, med, scale, is_binary in zip(row, medians, scales, self.binary):
            if value is None:
                out.append(0.0)
                continue
            if is_binary:
                out.append(1.0 if float(value) >= 0.5 else 0.0)
                continue
            z = (float(value) - med) / scale
            out.append(max(-8.0, min(8.0, z)))
        return out

    def transform(self, matrix: Sequence[Sequence[float | None]]) -> list[list[float]]:
        return [self.transform_row(row) for row in matrix]

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "scale": self.scale,
            "binary": self.binary,
            "groups": self.groups,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            median=[float(v) for v in value["median"]],
            scale=[float(v) for v in value["scale"]],
            binary=[bool(v) for v in value["binary"]],
            groups={
                str(group): {
                    "median": [float(v) for v in stats["median"]],
                    "scale": [float(v) for v in stats["scale"]],
                }
                for group, stats in value.get("groups", {}).items()
            },
        )


@dataclass
class ProfileModel:
    weights: list[float]
    bias: float
    center: float
    calibration_scale: float
    normalizer: FeatureNormalizer
    threshold: float = 0.5
    anomaly_stats: list[dict[str, float]] = field(default_factory=list)
    anomaly_gate: float | None = None
    anomaly_scale: float = 0.40
    metadata: dict[str, Any] | None = None

    @classmethod
    def fit(
        cls,
        positive_matrix: Sequence[Sequence[float | None]],
        negative_matrix: Sequence[Sequence[float | None]],
        normalizer: FeatureNormalizer | None = None,
        max_features: int = 16,
        shrinkage: float = 0.15,
    ) -> "ProfileModel":
        if len(positive_matrix) < 2:
            raise ValueError("at least two positive (enrolment) sessions are required")
        if len(negative_matrix) < 2:
            raise ValueError("at least two background sessions are required to fit a profile")
        if normalizer is None:
            normalizer = FeatureNormalizer.fit(list(positive_matrix) + list(negative_matrix))
        pos_raw = [list(row) for row in positive_matrix]
        neg_raw = [list(row) for row in negative_matrix]
        pos = normalizer.transform(pos_raw)
        neg = normalizer.transform(neg_raw)

        width = len(FEATURE_NAMES)
        effects: list[tuple[float, int]] = []
        for column in range(width):
            p_all = [row[column] for row in pos_raw]
            n_all = [row[column] for row in neg_raw]
            p_present = [float(v) for v in p_all if v is not None]
            n_present = [float(v) for v in n_all if v is not None]
            # A feature that is missing in a large part of either class is not
            # evidence about a person; it is evidence about which events
            # happened to be captured.
            if len(p_present) < 0.80 * len(pos) or len(n_present) < 0.80 * len(neg):
                continue
            p = [row[column] for row in pos]
            n = [row[column] for row in neg]
            mp, mn = _mean(p), _mean(n)
            vp, vn = _var(p), _var(n)
            pooled = math.sqrt(max(1e-6, vp + vn + shrinkage))
            effect = (mp - mn) / pooled
            effect *= len(pos) / (len(pos) + 6.0)
            if abs(effect) >= 0.10:
                effects.append((effect, column))

        effects.sort(key=lambda item: abs(item[0]), reverse=True)
        chosen: list[int] = []
        # Greedy de-correlation: two nearly identical features should not get
        # two votes merely because they are both present.
        for effect, column in effects:
            if len(chosen) >= max_features:
                break
            okay = True
            for other in chosen:
                xs = [row[column] for row in pos + neg]
                ys = [row[other] for row in pos + neg]
                mx, my = _mean(xs), _mean(ys)
                sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
                sy = math.sqrt(sum((y - my) ** 2 for y in ys))
                if sx < 1e-12 or sy < 1e-12:
                    continue
                corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
                if abs(corr) > 0.82:
                    okay = False
                    break
            if okay:
                chosen.append(column)
        if len(chosen) < 4:
            chosen = [column for _, column in effects[:4]]

        weights = [0.0] * width
        for effect, column in effects:
            if column in chosen:
                # Shrink very large effects; a single weird session should not
                # dominate the profile.
                weights[column] = max(-2.0, min(2.0, effect / (1.0 + 0.15 * abs(effect))))
        bias = 0.0
        for column in chosen:
            p = [row[column] for row in pos]
            n = [row[column] for row in neg]
            bias -= 0.5 * weights[column] * (_mean(p) + _mean(n))

        raw_pos = [sum(w * x for w, x in zip(weights, row)) + bias for row in pos]
        raw_neg = [sum(w * x for w, x in zip(weights, row)) + bias for row in neg]
        center = 0.5 * (_mean(raw_pos) + _mean(raw_neg))
        spread = _std(raw_pos + raw_neg)
        if spread < 1e-6:
            spread = 1.0

        anomaly_stats: list[dict[str, float]] = []
        for name in ANOMALY_FEATURES:
            column = FEATURE_NAMES.index(name)
            values = [float(row[column]) for row in pos_raw if row[column] is not None]
            if len(values) < 0.80 * len(pos_raw):
                continue
            med = _median(values)
            mad = _median([abs(v - med) for v in values]) * 1.4826
            if mad < 1e-9:
                mad = _std(values)
            if mad < 1e-9:
                continue
            anomaly_stats.append({"index": float(column), "median": med, "scale": mad})

        return cls(
            weights=weights,
            bias=bias,
            center=center,
            calibration_scale=spread,
            normalizer=normalizer,
            anomaly_stats=anomaly_stats,
            metadata={
                "positive_sessions": len(positive_matrix),
                "negative_sessions": len(negative_matrix),
                "selected_features": [FEATURE_NAMES[c] for c in chosen],
                "max_features": max_features,
                "shrinkage": shrinkage,
                "presence_filter": 0.80,
                "correlation_prune": 0.82,
            },
        )

    def _top2_z(self, features: Sequence[float | None]) -> float | None:
        if not self.anomaly_stats:
            return None
        zs: list[float] = []
        for stat in self.anomaly_stats:
            index = int(stat["index"])
            if index >= len(features) or features[index] is None:
                continue
            z = abs((float(features[index]) - stat["median"]) / max(1e-9, stat["scale"]))
            zs.append(min(20.0, z))
        if not zs:
            return None
        zs.sort(reverse=True)
        return sum(zs[:2]) / min(2, len(zs))

    def set_anomaly_gate(self, validation_matrix: Sequence[Sequence[float | None]], floor: float = 3.8) -> float:
        values = [self._top2_z(row) for row in validation_matrix]
        values = [v for v in values if v is not None]
        if not values:
            self.anomaly_gate = None
            return float("inf")
        values.sort()
        q90 = _quantile(values, 0.90)
        self.anomaly_gate = max(floor, q90 + 0.40)
        return self.anomaly_gate

    def score_features(self, features: Sequence[float | None]) -> float:
        z = self.normalizer.transform_row(features)
        raw = sum(w * x for w, x in zip(self.weights, z)) + self.bias
        calibrated = (raw - self.center) / max(1e-6, self.calibration_scale) * 1.8
        lda_score = sigmoid(calibrated)
        if self.anomaly_gate is None:
            return lda_score
        top2 = self._top2_z(features)
        if top2 is None:
            return lda_score
        anomaly_score = sigmoid((self.anomaly_gate - top2) / max(0.05, self.anomaly_scale))
        return min(lda_score, anomaly_score)

    def accept(self, features: Sequence[float | None], threshold: float | None = None) -> bool:
        return self.score_features(features) >= (self.threshold if threshold is None else threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROFILE_FORMAT,
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights,
            "bias": self.bias,
            "center": self.center,
            "calibration_scale": self.calibration_scale,
            "normalizer": self.normalizer.to_dict(),
            "threshold": self.threshold,
            "anomaly_stats": self.anomaly_stats,
            "anomaly_gate": self.anomaly_gate,
            "anomaly_scale": self.anomaly_scale,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProfileModel":
        if value.get("format") != PROFILE_FORMAT:
            raise ValueError(f"profile format must be {PROFILE_FORMAT!r}")
        if list(value.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ValueError("profile feature list does not match this build")
        return cls(
            weights=[float(v) for v in value["weights"]],
            bias=float(value["bias"]),
            center=float(value["center"]),
            calibration_scale=float(value["calibration_scale"]),
            normalizer=FeatureNormalizer.from_dict(value["normalizer"]),
            threshold=float(value.get("threshold", 0.5)),
            anomaly_stats=[
                {"index": float(stat["index"]), "median": float(stat["median"]), "scale": float(stat["scale"])}
                for stat in value.get("anomaly_stats", [])
            ],
            anomaly_gate=(
                None if value.get("anomaly_gate") is None else float(value["anomaly_gate"])
            ),
            anomaly_scale=float(value.get("anomaly_scale", 0.40)),
            metadata=dict(value.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProfileModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def choose_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    target_far: float = TARGET_FAR,
) -> dict[str, Any]:
    """Choose a threshold from validation scores only.

    Accept iff score >= threshold.  Candidates sit just above each observed
    score, so a candidate never accidentally accepts the validation session that
    defined it.  The rule is fixed before test data is opened.
    """

    if not positive_scores or not negative_scores:
        raise ValueError("threshold selection needs both positive and negative validation scores")
    scores = sorted({min(1.0, max(0.0, float(s))) for s in list(positive_scores) + list(negative_scores)})
    candidates = {0.0, 1.0, 0.5}
    candidates.update(scores)
    # Midpoints are the useful separators: a threshold just below the lowest
    # positive and just above the highest negative accepts every positive
    # without accepting the negative that defined the boundary.
    for a, b in zip(scores, scores[1:]):
        candidates.add(0.5 * (a + b))
    rows: list[dict[str, float]] = []
    for threshold in sorted(candidates):
        far = sum(1 for s in negative_scores if s >= threshold) / len(negative_scores)
        frr = sum(1 for s in positive_scores if s < threshold) / len(positive_scores)
        tpr = 1.0 - frr
        rows.append({"threshold": threshold, "far": far, "frr": frr, "tpr": tpr})

    feasible = [row for row in rows if row["far"] <= target_far + 1e-12]
    if feasible:
        # Maximise TPR; among ties choose the higher threshold (more conservative).
        best = max(feasible, key=lambda row: (round(row["tpr"], 12), row["threshold"]))
    else:
        # No validation threshold meets the target.  Pick the smallest FAR, then
        # the highest TPR; report the miss rather than pretending it passed.
        best = min(rows, key=lambda row: (row["far"], -row["tpr"], -row["threshold"]))
    return {
        "threshold": best["threshold"],
        "target_far": target_far,
        "validation_far": best["far"],
        "validation_frr": best["frr"],
        "validation_tpr": best["tpr"],
        "feasible": bool(feasible),
        "rule": THRESHOLD_RULE,
        "candidates": len(rows),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def roc_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.5
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_threshold(
    path: str | Path,
    selection: dict[str, Any],
    validation_files: Sequence[str | Path],
    code_files: Sequence[str | Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "format": "marrow-threshold-lock-v1",
        "selection": selection,
        "swap_selections": extra or {},
        "validation_sha256": {
            Path(p).as_posix(): file_sha256(p) for p in validation_files
        },
        "code_sha256": {
            Path(p).as_posix(): file_sha256(p) for p in code_files
        },
        "note": "Written after validation-only threshold selection and before any test score is computed.",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
