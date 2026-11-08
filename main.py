#!/usr/bin/env python3
"""One-command Marrow demo: generate fake data, train, lock threshold, test.

Run from a fresh copy:

    python3 main.py

or:

    bash run.sh

The process is offline.  It writes only inside this directory (``data/``,
``artifacts/`` and ``RESULTS.md``).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time

from marrow.evaluate import (
    Dataset,
    check_five_conditions,
    evaluate,
    fit_all_models,
    format_report,
    validation_scores,
)
from marrow.features import FEATURE_NAMES
from marrow.model import (
    ANOMALY_FEATURES,
    FeatureNormalizer,
    TARGET_FAR,
    choose_threshold,
    file_sha256,
    lock_threshold,
)
from marrow.synth import build_dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

FORBIDDEN_IMPORTS = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "subprocess",
    "multiprocessing",
    "ctypes",
    "ftplib",
    "smtplib",
    "telnetlib",
}


def check_offline_sources(root: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path}: syntax error: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_IMPORTS:
                        problems.append(f"{path}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    problems.append(f"{path}: imports from {node.module}")
    return not problems, problems


def ensure_dataset(data_dir: Path, regenerate: bool) -> dict:
    manifest_path = data_dir / "manifest.json"
    if regenerate or not manifest_path.exists():
        print(f"[data] generating deterministic fake recordings in {data_dir} ...")
        manifest = build_dataset(data_dir, force=True)
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        missing = [row["file"] for row in manifest.get("sessions", []) if not (data_dir / row["file"]).exists()]
        if missing:
            print(f"[data] {len(missing)} recordings missing; regenerating ...")
            manifest = build_dataset(data_dir, force=True)
        else:
            print(f"[data] reusing {len(manifest.get('sessions', []))} fake recordings in {data_dir}")
    return manifest


def code_files() -> list[Path]:
    names = [
        "main.py",
        "run.sh",
        "marrow/__init__.py",
        "marrow/format.py",
        "marrow/features.py",
        "marrow/synth.py",
        "marrow/model.py",
        "marrow/evaluate.py",
        "tools/train_test.py",
    ]
    return [ROOT / name for name in names if (ROOT / name).exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="recording directory (default: ./data)")
    parser.add_argument("--regenerate", action="store_true", help="regenerate all fake recordings")
    parser.add_argument("--no-write", action="store_true", help="do not write RESULTS.md")
    args = parser.parse_args(argv)

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest = ensure_dataset(data_dir, args.regenerate)

    dataset = Dataset(data_dir, manifest)
    print(f"[data] styles: {', '.join(dataset.styles())}")
    print(f"[data] sessions: {len(dataset.sessions)}")
    for style in dataset.styles():
        for role in ("owner", "alternate", "stranger", "mimic"):
            count = sum(1 for s in dataset.sessions.values() if s.style == style and s.role == role)
            print(f"        {style:26s} {role:10s} {count:3d}")
    print("[data] loading train and validation splits (test is not opened yet) ...")
    dataset.load_split("train")
    dataset.load_split("validation")

    train_ids = dataset.ids("train")
    print("[fit] normalising on the training population ...")
    normalizer = FeatureNormalizer.fit(dataset.matrix(train_ids))

    print("[fit] fitting one profile per owner and one swap profile per owner ...")
    owner_models, swap_models = fit_all_models(dataset, normalizer, dataset.owners())

    print("[fit] setting each profile's two-sided shape gate on validation ...")
    anomaly_gates: dict[str, float | None] = {}
    for owner, model in owner_models.items():
        gate = model.set_anomaly_gate(dataset.matrix(dataset.ids("validation", participant=owner)))
        anomaly_gates[owner] = None if gate == float("inf") else gate
    for owner, model in swap_models.items():
        alternate = dataset.alternate_for(owner)
        model.set_anomaly_gate(dataset.matrix(dataset.ids("validation", participant=alternate)))

    print("[threshold] scoring validation only ...")
    validation = validation_scores(dataset, owner_models, swap_models)
    selection = choose_threshold(
        validation["main"]["positive_scores"],
        validation["main"]["negative_scores"],
        TARGET_FAR,
    )
    swap_selections: dict[str, dict] = {}
    for owner, values in validation["swap"].items():
        swap_selections[owner] = choose_threshold(
            values["positive_scores"],
            values["negative_scores"],
            TARGET_FAR,
        )
    threshold = float(selection["threshold"])

    # A profile trained on a different person is a transfer test, not an
    # ordinary login.  Before the swap test is opened, give it a fixed 0.10
    # safety margin above its own validation threshold.  The margin is part of
    # the rule, not a reaction to the owner's scores.
    swap_margin = 0.10
    swap_thresholds = {
        owner: min(1.0, float(details["threshold"]) + swap_margin)
        for owner, details in swap_selections.items()
    }

    lock_path = ARTIFACTS / "threshold_lock.json"
    validation_files = [
        Path(os.path.relpath(dataset.sessions[session_id].file, ROOT)).as_posix()
        for session_id in dataset.ids("validation")
    ]
    lock_code_files = [Path(os.path.relpath(path, ROOT)).as_posix() for path in code_files()]
    lock = lock_threshold(
        lock_path,
        selection,
        validation_files,
        lock_code_files,
        extra={
            "anomaly_gates": anomaly_gates,
            "anomaly_panel": list(ANOMALY_FEATURES),
            "swap": {
                owner: {
                    "threshold": details["threshold"],
                    "target_far": details["target_far"],
                    "validation_far": details["validation_far"],
                    "validation_frr": details["validation_frr"],
                    "anomaly_gate": swap_models[owner].anomaly_gate,
                    "raw_threshold": details["threshold"],
                    "applied_threshold": swap_thresholds[owner],
                    "safety_margin": swap_margin,
                }
                for owner, details in swap_selections.items()
            },
            "swap_safety_margin": swap_margin,
        },
    )
    print(f"[threshold] locked at {threshold:.6f} -> {lock_path}")
    print(
        f"[threshold] validation FAR {selection['validation_far']:.2%}, "
        f"FRR {selection['validation_frr']:.2%}"
    )

    # The test split is opened only after the lock exists.
    print("[test] loading held-out test split ...")
    dataset.load_split("test")
    evaluation = evaluate(
        dataset,
        owner_models,
        swap_models,
        threshold,
        swap_thresholds,
        {owner: float(details["threshold"]) for owner, details in swap_selections.items()},
    )

    synthetic_ok = bool(manifest.get("synthetic")) and all(
        bool(value) for value in dataset.synthetic.values()
    )
    offline_ok, offline_problems = check_offline_sources(ROOT)
    format_ok = (ROOT / "FORMAT.md").exists() and (ROOT / "tools" / "train_test.py").exists() and offline_ok
    one_command_ok = (ROOT / "run.sh").exists() and (ROOT / "main.py").exists()

    checks = check_five_conditions(
        dataset,
        evaluation,
        threshold_locked=lock_path.exists(),
        one_command=one_command_ok,
        synthetic_ok=synthetic_ok,
        format_ok=format_ok,
    )
    checks["offline_problems"] = offline_problems
    checks["threshold_lock_sha256"] = file_sha256(lock_path)

    elapsed = time.time() - started
    report = format_report(
        dataset,
        selection,
        evaluation,
        checks,
        threshold,
        str(lock_path.relative_to(ROOT)) if lock_path.is_relative_to(ROOT) else str(lock_path),
        elapsed_note=f"Elapsed: {elapsed:.2f}s. Offline: {'yes' if offline_ok else 'NO'}.",
    )
    print()
    print(report)

    results = {
        "format": "marrow-results-v1",
        "listing": "1f916 listing-65",
        "synthetic": synthetic_ok,
        "threshold": threshold,
        "selection": selection,
        "swap_selections": swap_selections,
        "swap_thresholds": swap_thresholds,
        "swap_safety_margin": swap_margin,
        "checks": checks,
        "evaluation": evaluation,
        "offline": offline_ok,
        "offline_problems": offline_problems,
        "elapsed_seconds": round(elapsed, 3),
        "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in code_files()},
        "threshold_lock_sha256": checks["threshold_lock_sha256"],
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not args.no_write:
        (ROOT / "RESULTS.md").write_text(
            "# Marrow results\n\n"
            + report
            + "\nThe machine-readable form is [`artifacts/results.json`](artifacts/results.json); "
            + "the threshold lock is [`artifacts/threshold_lock.json`](artifacts/threshold_lock.json).\n",
            encoding="utf-8",
        )

    ok = checks["all_conditions_pass"] and checks["targets_pass"]
    print("Exit 0" if ok else "Exit 1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
