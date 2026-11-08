#!/usr/bin/env python3
"""Offline train/test CLI for recordings in the Marrow v1 format.

This is the generic entry point.  It does not know about the shipped fake
dataset, does not capture anything, and never opens a network socket.

Examples
--------
Enrol from one person's sessions and a background of other people:

    python3 tools/train_test.py train \
      --owner recordings/alice/ \
      --background recordings/others/ \
      --validation-owner recordings/alice-val/ \
      --validation-background recordings/others-val/ \
      --out alice.profile.json

Score new recordings:

    python3 tools/train_test.py test \
      --profile alice.profile.json \
      --input recordings/new-session.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marrow.features import FEATURE_NAMES, extract_features  # noqa: E402
from marrow.format import Recording, load_many  # noqa: E402
from marrow.model import (  # noqa: E402
    FeatureNormalizer,
    ProfileModel,
    TARGET_FAR,
    choose_threshold,
)


def _matrix(recordings: list[Recording]) -> list[list[float | None]]:
    return [[extract_features(recording).get(name) for name in FEATURE_NAMES] for recording in recordings]


def _load(paths: list[str], name: str) -> list[Recording]:
    recordings = load_many(paths)
    if not recordings:
        raise SystemExit(f"{name}: no .json/.jsonl recordings found in {paths}")
    return recordings


def command_train(args: argparse.Namespace) -> int:
    owner = _load(args.owner, "owner")
    background = _load(args.background, "background")
    if len(owner) < 2:
        raise SystemExit("owner: at least two enrolment sessions are required")
    if len(background) < 2:
        raise SystemExit("background: at least two other-person sessions are required")

    normalizer = FeatureNormalizer.fit(_matrix(owner + background))
    model = ProfileModel.fit(
        _matrix(owner),
        _matrix(background),
        normalizer=normalizer,
        max_features=args.max_features,
    )

    threshold = args.threshold
    validation_note = "no validation split supplied; threshold is the fixed default"
    if args.validation_owner and args.validation_background:
        validation_owner = _load(args.validation_owner, "validation-owner")
        validation_background = _load(args.validation_background, "validation-background")
        model.set_anomaly_gate(_matrix(validation_owner))
        positive_scores = [model.score_features(row) for row in _matrix(validation_owner)]
        negative_scores = [model.score_features(row) for row in _matrix(validation_background)]
        selection = choose_threshold(positive_scores, negative_scores, args.target_far)
        threshold = float(selection["threshold"])
        validation_note = (
            f"validation FAR {selection['validation_far']:.2%}, "
            f"FRR {selection['validation_frr']:.2%}"
        )
    else:
        model.set_anomaly_gate(_matrix(owner))
        if threshold is None:
            threshold = 0.5

    model.threshold = float(threshold)
    model.metadata = dict(model.metadata or {})
    model.metadata["training_note"] = validation_note
    model.save(args.out)
    summary = {
        "profile": str(args.out),
        "owner_sessions": len(owner),
        "background_sessions": len(background),
        "threshold": model.threshold,
        "anomaly_gate": model.anomaly_gate,
        "selected_features": (model.metadata or {}).get("selected_features", []),
        "validation": validation_note,
    }
    print(json.dumps(summary, indent=2))
    return 0


def command_test(args: argparse.Namespace) -> int:
    model = ProfileModel.load(args.profile)
    if args.threshold is not None:
        model.threshold = float(args.threshold)
    recordings = _load(args.input, "input")
    rows = []
    accepted_count = 0
    for recording in recordings:
        features = [extract_features(recording).get(name) for name in FEATURE_NAMES]
        score = model.score_features(features)
        accepted = score >= model.threshold
        accepted_count += int(accepted)
        rows.append({
            "session_id": recording.session_id,
            "file": recording.path,
            "score": round(score, 6),
            "threshold": model.threshold,
            "accepted": accepted,
        })
    if args.json:
        print(json.dumps({
            "profile": str(args.profile),
            "threshold": model.threshold,
            "accepted": accepted_count,
            "total": len(rows),
            "rows": rows,
        }, indent=2))
    else:
        for row in rows:
            mark = "ACCEPT" if row["accepted"] else "REJECT"
            print(f"{mark}  score={row['score']:.6f}  threshold={row['threshold']:.6f}  {row['session_id']}")
        print(f"\n{accepted_count}/{len(rows)} accepted")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit a profile from owner and background recordings")
    train.add_argument("--owner", nargs="+", required=True, help="owner enrolment files or directories")
    train.add_argument("--background", nargs="+", required=True, help="other-person files or directories")
    train.add_argument("--validation-owner", nargs="*", default=None, help="held-out owner files/directories")
    train.add_argument("--validation-background", nargs="*", default=None, help="held-out other-person files/directories")
    train.add_argument("--out", required=True, help="output profile JSON")
    train.add_argument("--threshold", type=float, default=None, help="fixed threshold when no validation split is supplied")
    train.add_argument("--target-far", type=float, default=TARGET_FAR, help="validation FAR target (default 0.05)")
    train.add_argument("--max-features", type=int, default=16, help="maximum selected features (default 16)")
    train.set_defaults(func=command_train)

    test = sub.add_parser("test", help="score recordings against a profile")
    test.add_argument("--profile", required=True, help="profile JSON from train")
    test.add_argument("--input", nargs="+", required=True, help="recording files or directories")
    test.add_argument("--threshold", type=float, default=None, help="override the profile threshold")
    test.add_argument("--json", action="store_true", help="print machine-readable JSON")
    test.set_defaults(func=command_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
