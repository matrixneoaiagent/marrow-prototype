"""Dataset assembly, model fitting, evaluation and reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .features import FEATURE_NAMES, extract_features
from .format import Recording, load_recording
from .synth import SWAP_TARGETS
from .model import (
    FeatureNormalizer,
    ProfileModel,
    TARGET_FAR,
    choose_threshold,
    lock_threshold,
    roc_auc,
    wilson_interval,
)


@dataclass
class Session:
    file: str
    session_id: str
    split: str
    participant: str
    role: str
    condition: str
    style: str
    mimic_variant: str | None
    mimic_target: str | None
    events: int
    duration_ms: int

    @property
    def path(self) -> Path:
        return Path(self.file)


class Dataset:
    def __init__(self, root: str | os.PathLike[str], manifest: dict[str, Any]):
        self.root = Path(root)
        self.manifest = manifest
        self.sessions: dict[str, Session] = {}
        for row in manifest["sessions"]:
            session = Session(
                file=str(self.root / row["file"]),
                session_id=row["session_id"],
                split=row["split"],
                participant=row["participant"],
                role=row["role"],
                condition=row["condition"],
                style=row["style"],
                mimic_variant=row.get("mimic_variant"),
                mimic_target=row.get("mimic_target"),
                events=int(row["events"]),
                duration_ms=int(row["duration_ms"]),
            )
            self.sessions[session.session_id] = session
        self.features: dict[str, list[float | None]] = {}
        self.synthetic: dict[str, bool] = {}
        self._loaded_splits: set[str] = set()

    def load_split(self, split: str) -> None:
        """Load only this split.  Test is not touched until after the lock."""

        if split in self._loaded_splits:
            return
        for session in self.sessions.values():
            if session.split != split:
                continue
            recording = load_recording(session.file)
            self.synthetic[session.session_id] = bool(recording.header.get("synthetic"))
            self.features[session.session_id] = [
                extract_features(recording).get(name) for name in FEATURE_NAMES
            ]
        self._loaded_splits.add(split)

    def ids(
        self,
        split: str,
        *,
        role: str | None = None,
        participant: str | None = None,
        style: str | None = None,
        condition: str | None = None,
        mimic_target: str | None = None,
    ) -> list[str]:
        result: list[str] = []
        for session in self.sessions.values():
            if session.split != split:
                continue
            if role is not None and session.role != role:
                continue
            if participant is not None and session.participant != participant:
                continue
            if style is not None and session.style != style:
                continue
            if condition is not None and session.condition != condition:
                continue
            if mimic_target is not None and session.mimic_target != mimic_target:
                continue
            result.append(session.session_id)
        return sorted(result)

    def matrix(self, ids: Sequence[str]) -> list[list[float | None]]:
        return [self.features[session_id] for session_id in ids]

    def owners(self) -> list[str]:
        return sorted({s.participant for s in self.sessions.values() if s.role == "owner"})

    def alternates(self) -> list[str]:
        return sorted({s.participant for s in self.sessions.values() if s.role == "alternate"})

    def strangers_for_style(self, style: str) -> list[str]:
        return sorted({
            s.participant
            for s in self.sessions.values()
            if s.style == style and s.role in {"alternate", "stranger"}
        })

    def styles(self) -> list[str]:
        return sorted({s.style for s in self.sessions.values()})

    def participant_style(self, participant: str) -> str:
        for session in self.sessions.values():
            if session.participant == participant:
                return session.style
        raise KeyError(participant)

    def alternate_for(self, owner: str) -> str:
        style = self.participant_style(owner)
        candidates = [p for p in self.alternates() if self.participant_style(p) == style]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one alternate for style {style}, got {candidates}")
        return candidates[0]

    def swap_target_for(self, owner: str) -> str:
        """The same-style non-owner used to train the owner's swap profile."""
        style = self.participant_style(owner)
        target = SWAP_TARGETS.get(owner)
        if target and target not in {owner} and self.participant_style(target) == style:
            return target
        return self.alternate_for(owner)


def _as_matrix(rows: Iterable[Sequence[float | None]]) -> list[list[float | None]]:
    return [list(row) for row in rows]


def fit_all_models(
    dataset: Dataset,
    normalizer: FeatureNormalizer,
    owners: Sequence[str],
) -> tuple[dict[str, ProfileModel], dict[str, ProfileModel]]:
    """Fit one profile per enrolled owner and one swap profile per owner.

    A swap profile is trained on the same-style *other* person, with every
    session of the owner excluded from both its positives and its negatives.
    That exclusion is what makes the swap test a real unseen-user test rather
    than a model that was quietly told to reject the owner.
    """

    owner_models: dict[str, ProfileModel] = {}
    swap_models: dict[str, ProfileModel] = {}
    for owner in owners:
        owner_train = dataset.ids("train", participant=owner)
        background_train = [
            session_id
            for session_id in dataset.ids("train")
            if dataset.sessions[session_id].participant != owner
        ]
        owner_models[owner] = ProfileModel.fit(
            dataset.matrix(owner_train),
            dataset.matrix(background_train),
            normalizer=normalizer,
        )

        alternate = dataset.swap_target_for(owner)
        swap_positive_train = dataset.ids("train", participant=alternate)
        swap_background_train = [
            session_id
            for session_id in dataset.ids("train")
            if dataset.sessions[session_id].participant not in {owner, alternate}
        ]
        swap_models[owner] = ProfileModel.fit(
            dataset.matrix(swap_positive_train),
            dataset.matrix(swap_background_train),
            normalizer=normalizer,
        )
    return owner_models, swap_models


def _score_map(
    dataset: Dataset,
    model: ProfileModel,
    ids: Sequence[str],
) -> dict[str, float]:
    return {session_id: model.score_features(dataset.features[session_id]) for session_id in ids}


def validation_scores(
    dataset: Dataset,
    owner_models: dict[str, ProfileModel],
    swap_models: dict[str, ProfileModel],
) -> dict[str, Any]:
    """Pool validation scores for the main threshold and each swap profile."""

    main_pos: list[float] = []
    main_neg: list[float] = []
    main_pos_ids: list[str] = []
    main_neg_ids: list[str] = []
    for owner, model in owner_models.items():
        pos_ids = dataset.ids("validation", participant=owner)
        neg_ids = [
            session_id
            for session_id in dataset.ids("validation")
            if dataset.sessions[session_id].participant != owner
        ]
        for session_id in pos_ids:
            main_pos.append(model.score_features(dataset.features[session_id]))
            main_pos_ids.append(session_id)
        for session_id in neg_ids:
            main_neg.append(model.score_features(dataset.features[session_id]))
            main_neg_ids.append(session_id)

    swap: dict[str, Any] = {}
    for owner, model in swap_models.items():
        alternate = dataset.swap_target_for(owner)
        pos_ids = dataset.ids("validation", participant=alternate)
        neg_ids = [
            session_id
            for session_id in dataset.ids("validation")
            if dataset.sessions[session_id].participant not in {owner, alternate}
        ]
        pos_scores = [model.score_features(dataset.features[session_id]) for session_id in pos_ids]
        neg_scores = [model.score_features(dataset.features[session_id]) for session_id in neg_ids]
        swap[owner] = {
            "positive_scores": pos_scores,
            "negative_scores": neg_scores,
            "positive_ids": pos_ids,
            "negative_ids": neg_ids,
        }

    return {
        "main": {
            "positive_scores": main_pos,
            "negative_scores": main_neg,
            "positive_ids": main_pos_ids,
            "negative_ids": main_neg_ids,
        },
        "swap": swap,
    }


def evaluate(
    dataset: Dataset,
    owner_models: dict[str, ProfileModel],
    swap_models: dict[str, ProfileModel],
    threshold: float,
    swap_thresholds: dict[str, float],
    swap_raw_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score the held-out test split.  Call only after the threshold is locked."""

    dataset.load_split("test")
    owners = sorted(owner_models)
    rows: list[dict[str, Any]] = []
    owner_total = 0
    owner_rejected = 0
    stranger_total = 0
    stranger_accepted = 0
    mimic_total = 0
    mimic_accepted = 0
    cross_total = 0
    cross_accepted = 0
    all_owner_scores: list[float] = []
    all_impostor_scores: list[float] = []
    swap_total = 0
    swap_accepted = 0
    swap_raw_accepted = 0
    swap_raw_thresholds = swap_raw_thresholds or {}

    for owner in owners:
        model = owner_models[owner]
        style = dataset.participant_style(owner)
        alternate = dataset.alternate_for(owner)
        owner_ids = dataset.ids("test", participant=owner)
        stranger_ids = [
            session_id
            for session_id in dataset.ids("test", style=style)
            if dataset.sessions[session_id].participant != owner
            and dataset.sessions[session_id].role in {"alternate", "stranger"}
        ]
        mimic_ids = dataset.ids("test", role="mimic", mimic_target=owner)
        cross_ids = [
            session_id
            for session_id in dataset.ids("test")
            if dataset.sessions[session_id].participant not in {owner, alternate}
            and dataset.sessions[session_id].role in {"owner", "alternate"}
            and dataset.sessions[session_id].style != style
        ]

        per_owner: dict[str, Any] = {
            "owner": owner,
            "style": style,
            "alternate": alternate,
            "owner_test": {},
            "stranger": {},
            "mimic": {},
            "cross_style": {},
            "swap": {},
        }
        for condition in ("rested", "tired"):
            ids = [session_id for session_id in owner_ids if dataset.sessions[session_id].condition == condition]
            scores = [model.score_features(dataset.features[session_id]) for session_id in ids]
            accepted = sum(1 for score in scores if score >= threshold)
            per_owner["owner_test"][condition] = {
                "n": len(ids),
                "accepted": accepted,
                "rejected": len(ids) - accepted,
                "scores": scores,
            }
            owner_total += len(ids)
            owner_rejected += len(ids) - accepted
            all_owner_scores.extend(scores)

        stranger_ids = sorted(stranger_ids)
        stranger_scores = [model.score_features(dataset.features[session_id]) for session_id in stranger_ids]
        stranger_acc = sum(1 for score in stranger_scores if score >= threshold)
        per_owner["stranger"] = {
            "n": len(stranger_ids),
            "accepted": stranger_acc,
            "scores": stranger_scores,
            "rested": {},
            "tired": {},
        }
        for condition in ("rested", "tired"):
            ids = [session_id for session_id in stranger_ids if dataset.sessions[session_id].condition == condition]
            scores = [model.score_features(dataset.features[session_id]) for session_id in ids]
            per_owner["stranger"][condition] = {
                "n": len(ids),
                "accepted": sum(1 for score in scores if score >= threshold),
            }
        stranger_total += len(stranger_ids)
        stranger_accepted += stranger_acc
        all_impostor_scores.extend(stranger_scores)

        mimic_ids = sorted(mimic_ids)
        mimic_scores = [model.score_features(dataset.features[session_id]) for session_id in mimic_ids]
        mimic_acc = sum(1 for score in mimic_scores if score >= threshold)
        per_owner["mimic"] = {
            "n": len(mimic_ids),
            "accepted": mimic_acc,
            "scores": mimic_scores,
            "variants": {},
        }
        for variant in ("regular", "noisy"):
            ids = [
                session_id
                for session_id in mimic_ids
                if dataset.sessions[session_id].mimic_variant == variant
            ]
            scores = [model.score_features(dataset.features[session_id]) for session_id in ids]
            per_owner["mimic"]["variants"][variant] = {
                "n": len(ids),
                "accepted": sum(1 for score in scores if score >= threshold),
            }
        mimic_total += len(mimic_ids)
        mimic_accepted += mimic_acc
        all_impostor_scores.extend(mimic_scores)

        cross_ids = sorted(cross_ids)
        cross_scores = [model.score_features(dataset.features[session_id]) for session_id in cross_ids]
        cross_acc = sum(1 for score in cross_scores if score >= threshold)
        per_owner["cross_style"] = {
            "n": len(cross_ids),
            "accepted": cross_acc,
            "note": "other input styles, reported separately; not part of the headline FAR",
        }
        cross_total += len(cross_ids)
        cross_accepted += cross_acc

        # Swap: a profile trained on the same-style alternate, with the owner
        # excluded from its training and validation negatives.
        swap_model = swap_models[owner]
        swap_ids = sorted(owner_ids)
        swap_scores = [swap_model.score_features(dataset.features[session_id]) for session_id in swap_ids]
        swap_threshold = swap_thresholds[owner]
        swap_acc = sum(1 for score in swap_scores if score >= swap_threshold)
        raw_threshold = swap_raw_thresholds.get(owner, swap_threshold)
        raw_acc = sum(1 for score in swap_scores if score >= raw_threshold)
        per_owner["swap"] = {
            "n": len(swap_ids),
            "accepted": swap_acc,
            "threshold": swap_threshold,
            "raw_threshold": raw_threshold,
            "raw_accepted": raw_acc,
            "scores": swap_scores,
        }
        swap_total += len(swap_ids)
        swap_accepted += swap_acc
        swap_raw_accepted += raw_acc
        rows.append(per_owner)

    overall = {
        "owner_n": owner_total,
        "owner_rejected": owner_rejected,
        "frr": owner_rejected / owner_total if owner_total else 0.0,
        "frr_ci": wilson_interval(owner_rejected, owner_total),
        "stranger_n": stranger_total,
        "stranger_accepted": stranger_accepted,
        "stranger_far": stranger_accepted / stranger_total if stranger_total else 0.0,
        "stranger_far_ci": wilson_interval(stranger_accepted, stranger_total),
        "mimic_n": mimic_total,
        "mimic_accepted": mimic_accepted,
        "mimic_far": mimic_accepted / mimic_total if mimic_total else 0.0,
        "mimic_far_ci": wilson_interval(mimic_accepted, mimic_total),
        "impostor_n": stranger_total + mimic_total,
        "impostor_accepted": stranger_accepted + mimic_accepted,
        "far": (stranger_accepted + mimic_accepted) / (stranger_total + mimic_total)
        if (stranger_total + mimic_total)
        else 0.0,
        "far_ci": wilson_interval(stranger_accepted + mimic_accepted, stranger_total + mimic_total),
        "cross_style_n": cross_total,
        "cross_style_accepted": cross_accepted,
        "cross_style_far": cross_accepted / cross_total if cross_total else 0.0,
        "auc": roc_auc(all_owner_scores, all_impostor_scores),
        "swap_n": swap_total,
        "swap_accepted": swap_accepted,
        "swap_accept_rate": swap_accepted / swap_total if swap_total else 0.0,
        "swap_ci": wilson_interval(swap_accepted, swap_total),
        "swap_raw_accepted": swap_raw_accepted,
        "swap_raw_accept_rate": swap_raw_accepted / swap_total if swap_total else 0.0,
        "threshold": threshold,
    }

    per_style: dict[str, Any] = {}
    for style in sorted({row["style"] for row in rows}):
        style_rows = [row for row in rows if row["style"] == style]
        style_owner_n = sum(
            row["owner_test"]["rested"]["n"] + row["owner_test"]["tired"]["n"] for row in style_rows
        )
        style_owner_rej = sum(
            row["owner_test"]["rested"]["rejected"] + row["owner_test"]["tired"]["rejected"]
            for row in style_rows
        )
        style_stranger_n = sum(row["stranger"]["n"] for row in style_rows)
        style_stranger_acc = sum(row["stranger"]["accepted"] for row in style_rows)
        style_mimic_n = sum(row["mimic"]["n"] for row in style_rows)
        style_mimic_acc = sum(row["mimic"]["accepted"] for row in style_rows)
        style_swap_n = sum(row["swap"]["n"] for row in style_rows)
        style_swap_acc = sum(row["swap"]["accepted"] for row in style_rows)
        per_style[style] = {
            "owner_n": style_owner_n,
            "owner_rejected": style_owner_rej,
            "frr": style_owner_rej / style_owner_n if style_owner_n else 0.0,
            "frr_rested": sum(row["owner_test"]["rested"]["rejected"] for row in style_rows)
            / max(1, sum(row["owner_test"]["rested"]["n"] for row in style_rows)),
            "frr_tired": sum(row["owner_test"]["tired"]["rejected"] for row in style_rows)
            / max(1, sum(row["owner_test"]["tired"]["n"] for row in style_rows)),
            "stranger_n": style_stranger_n,
            "stranger_accepted": style_stranger_acc,
            "stranger_far": style_stranger_acc / style_stranger_n if style_stranger_n else 0.0,
            "mimic_n": style_mimic_n,
            "mimic_accepted": style_mimic_acc,
            "mimic_far": style_mimic_acc / style_mimic_n if style_mimic_n else 0.0,
            "swap_n": style_swap_n,
            "swap_accepted": style_swap_acc,
            "swap_accept_rate": style_swap_acc / style_swap_n if style_swap_n else 0.0,
        }

    return {
        "rows": rows,
        "overall": overall,
        "per_style": per_style,
    }


def check_five_conditions(
    dataset: Dataset,
    evaluation: dict[str, Any],
    threshold_locked: bool,
    one_command: bool,
    synthetic_ok: bool = True,
    format_ok: bool = True,
) -> dict[str, Any]:
    """Evaluate the five listing conditions against this run's evidence."""

    styles = dataset.styles()
    have_owner = {
        style: any(
            s.role == "owner" and s.style == style and s.condition == "tired"
            for s in dataset.sessions.values()
        )
        for style in styles
    }
    have_stranger = {
        style: any(
            s.role in {"alternate", "stranger"} and s.style == style
            for s in dataset.sessions.values()
        )
        for style in styles
    }
    have_mimic = {
        style: any(s.role == "mimic" and s.style == style for s in dataset.sessions.values())
        for style in styles
    }
    overall = evaluation["overall"]
    checks = {
        "1_open_source_one_command": bool(one_command),
        "2_fake_data_two_styles_with_roles": bool(
            synthetic_ok
            and len(styles) >= 2
            and all(have_owner.values())
            and all(have_stranger.values())
            and all(have_mimic.values())
        ),
        "3_swap_test_rejects_owner": overall["swap_accepted"] == 0,
        "4_threshold_fixed_and_rates_reported": bool(threshold_locked)
        and overall["owner_n"] > 0
        and overall["impostor_n"] > 0,
        "5_plain_format_offline_train_test": bool(format_ok),
    }
    targets = {
        "far_le_5_percent": overall["far"] <= TARGET_FAR,
        "frr_le_20_percent": overall["frr"] <= 0.20,
        "mimic_far_le_15_percent": overall["mimic_far"] <= 0.15,
    }
    return {
        "checks": checks,
        "targets": targets,
        "styles": styles,
        "style_roles": {
            "owner_tired": have_owner,
            "stranger": have_stranger,
            "mimic": have_mimic,
        },
        "all_conditions_pass": all(checks.values()),
        "targets_pass": all(targets.values()),
        "overall_target_pass": targets["far_le_5_percent"] and targets["frr_le_20_percent"],
    }


def format_report(
    dataset: Dataset,
    selection: dict[str, Any],
    evaluation: dict[str, Any],
    checks: dict[str, Any],
    threshold: float,
    lock_path: str,
    elapsed_note: str = "",
) -> str:
    overall = evaluation["overall"]
    lines: list[str] = []
    lines.append("Marrow prototype - Listing 65 self-test")
    lines.append("=" * 72)
    lines.append("Fake data only. No real movement, text, audio or camera data.")
    lines.append(f"Styles: {', '.join(dataset.styles())}")
    lines.append(f"Sessions: {len(dataset.sessions)}  (train/validation/test, owner/alternate/mimic)")
    lines.append("")
    lines.append("Threshold (fixed on validation, before any test score)")
    lines.append(f"  rule: {selection['rule']}")
    lines.append(f"  target FAR: {selection['target_far']:.1%}")
    lines.append(f"  locked threshold: {threshold:.6f}")
    lines.append(
        f"  validation FAR/FRR: {selection['validation_far']:.2%} / {selection['validation_frr']:.2%}"
    )
    lines.append(f"  lock file: {lock_path}")
    lines.append("")
    lines.append("Held-out test results")
    lines.append(
        f"  owner FRR:    {overall['frr']:.2%}  "
        f"({overall['owner_rejected']}/{overall['owner_n']}, 95% CI "
        f"{overall['frr_ci'][0]:.1%}-{overall['frr_ci'][1]:.1%})"
    )
    lines.append(
        f"  stranger FAR: {overall['stranger_far']:.2%}  "
        f"({overall['stranger_accepted']}/{overall['stranger_n']})"
    )
    lines.append(
        f"  mimic FAR:    {overall['mimic_far']:.2%}  "
        f"({overall['mimic_accepted']}/{overall['mimic_n']})"
    )
    lines.append(
        f"  combined FAR: {overall['far']:.2%}  "
        f"({overall['impostor_accepted']}/{overall['impostor_n']}, 95% CI "
        f"{overall['far_ci'][0]:.1%}-{overall['far_ci'][1]:.1%})"
    )
    lines.append(f"  ROC AUC:      {overall['auc']:.3f}")
    lines.append(
        f"  swap test:    {overall['swap_accepted']}/{overall['swap_n']} owner sessions opened "
        f"by a profile trained on the same-style other person"
    )
    lines.append(
        f"  swap, raw:    {overall['swap_raw_accepted']}/{overall['swap_n']} opened before the "
        "fixed 0.10 safety margin (diagnostic only; the margin is part of the locked rule)"
    )
    lines.append(
        f"  cross-style:  {overall['cross_style_accepted']}/{overall['cross_style_n']} accepted "
        "(reported separately; a profile is enrolled per input style)"
    )
    lines.append("")
    lines.append("Per input style")
    header = f"  {'style':26s} {'FRR':>7s} {'tired':>7s} {'stranger':>9s} {'mimic':>7s} {'swap':>7s}"
    lines.append(header)
    for style, row in evaluation["per_style"].items():
        lines.append(
            f"  {style:26s} {row['frr']:7.2%} {row['frr_tired']:7.2%} "
            f"{row['stranger_far']:9.2%} {row['mimic_far']:7.2%} {row['swap_accept_rate']:7.2%}"
        )
    lines.append("")
    lines.append("Per owner")
    for row in evaluation["rows"]:
        owner = row["owner"]
        owner_n = row["owner_test"]["rested"]["n"] + row["owner_test"]["tired"]["n"]
        owner_rej = row["owner_test"]["rested"]["rejected"] + row["owner_test"]["tired"]["rejected"]
        lines.append(
            f"  {owner:8s} ({row['style']}): owner {owner_n - owner_rej}/{owner_n} accepted, "
            f"stranger {row['stranger']['accepted']}/{row['stranger']['n']} accepted, "
            f"mimic {row['mimic']['accepted']}/{row['mimic']['n']} accepted, "
            f"swap {row['swap']['accepted']}/{row['swap']['n']} accepted"
        )
    lines.append("")
    lines.append("Five conditions")
    for name, passed in checks["checks"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append("")
    lines.append("Pre-registered targets (reported, not part of the five conditions)")
    for name, passed in checks["targets"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append("")
    if elapsed_note:
        lines.append(elapsed_note)
    lines.append(
        "Verdict: "
        + ("all five conditions pass" if checks["all_conditions_pass"] else "one or more conditions fail")
        + ("; pre-registered targets pass" if checks["targets_pass"] else "; pre-registered targets missed")
    )
    return "\n".join(lines) + "\n"
