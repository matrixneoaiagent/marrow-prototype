"""Feature extraction for Marrow recordings.

Everything in this module is derived from event *timing*, *geometry* and
*modality*.  Text content is never inspected: an insert made of ``"hello"`` and
an insert made of ``"xxxxx"`` produce the same feature value.  That keeps the
detector usable on private recordings and prevents it from learning what a
person writes instead of how they write.

Two design choices matter for device independence:

* pointer features describe shape, dispersion and change, not just absolute
  speed;
* text features separate the rhythm *inside* a typing burst from the pauses
  *between* bursts.  A dictation phrase and a typed word have very different
  absolute intervals, but both have a within-burst rhythm and a between-burst
  pause distribution.

The feature set is broad and redundant.  A profile is fitted to one person and
selects a small, regularised subset.  Missing groups (a keyboard-only session,
a pointer-only session, a switch user with few samples) are represented as
``None`` and imputed at normalisation time.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Sequence

from .format import Recording

# The order is part of the saved profile format.  Append new features at the
# end; never reorder, or an old profile would silently read the wrong weights.
FEATURE_NAMES: tuple[str, ...] = (
    # session shape
    "s_duration_log",
    "s_event_rate",
    "s_pointer_rate",
    "s_click_rate",
    "s_char_rate",
    "s_pointer_share",
    "s_text_share",
    "s_text_per_click",
    # pointer kinematics
    "p_speed_med_log",
    "p_speed_p90_log",
    "p_speed_cv",
    "p_speed_skew",
    "p_speed_entropy",
    "p_accel_med_log",
    "p_jerk_med_log",
    "p_dirchange_rate",
    "p_dir_entropy",
    "p_stroke_n_per_min",
    "p_stroke_dur_med_log",
    "p_stroke_speed_med_log",
    "p_stroke_speed_cv",
    "p_straight_med",
    "p_straight_cv",
    "p_curvature_med_log",
    "p_dirchange_per_stroke",
    "p_pause_frac",
    "p_pause_med_log",
    "p_pause_cv",
    "p_pause_tail",
    "p_dwell_med_log",
    "p_dwell_cv",
    "p_preclick_pause_med_log",
    "p_click_interval_cv",
    "p_click_pos_std",
    "p_overshoot_frac",
    "p_speed_trend",
    "p_speed_ratio",
    # text, all modalities
    "t_insert_rate",
    "t_delete_rate",
    "t_correction_ratio",
    "t_delete_burst_med_log",
    "t_delete_burst_cv",
    "t_multi_delete_frac",
    "t_correction_latency_med_log",
    "t_correction_after_insert_frac",
    "t_modality_typed",
    "t_modality_onscreen",
    "t_modality_software",
    "t_modality_switch_rate",
    "t_pause_frac",
    "t_pause_med_log",
    "t_pause_cv",
    "t_burstiness",
    "t_autocorr",
    "t_interval_entropy",
    "t_chars_per_insert_med_log",
    "t_chars_per_insert_cv",
    "t_interval_trend",
    "t_first_text_delay_log",
    # per-modality rhythm
    "t_typed_intra_med_log",
    "t_typed_intra_cv",
    "t_typed_intra_autocorr",
    "t_typed_intra_tail",
    "t_typed_burst_chars_med_log",
    "t_typed_burst_chars_cv",
    "t_typed_pause_frac",
    "t_typed_pause_med_log",
    "t_onscreen_intra_med_log",
    "t_onscreen_intra_cv",
    "t_onscreen_intra_autocorr",
    "t_onscreen_intra_tail",
    "t_onscreen_burst_chars_med_log",
    "t_onscreen_burst_chars_cv",
    "t_onscreen_pause_frac",
    "t_onscreen_pause_med_log",
    "t_soft_gap_med_log",
    "t_soft_gap_cv",
    "t_soft_gap_autocorr",
    "t_soft_gap_tail",
    "t_soft_chars_med_log",
    "t_soft_chars_cv",
    "t_soft_pause_frac",
    "t_soft_pause_med_log",
    # evidence flags
    "f_has_pointer",
    "f_has_clicks",
    "f_has_text",
    "f_has_typed",
    "f_has_onscreen",
    "f_has_software",
)

BINARY_FEATURES = frozenset(
    {
        "f_has_pointer",
        "f_has_clicks",
        "f_has_text",
        "f_has_typed",
        "f_has_onscreen",
        "f_has_software",
    }
)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return float(ordered[middle])
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _var(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def _std(values: Sequence[float]) -> float:
    return math.sqrt(max(0.0, _var(values)))


def _cv(values: Sequence[float]) -> float:
    m = _mean(values)
    if abs(m) < 1e-12:
        return 0.0
    return _std(values) / abs(m)


def _skew(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    m = _mean(values)
    s = _std(values)
    if s < 1e-12:
        return 0.0
    return sum(((v - m) / s) ** 3 for v in values) / len(values)


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


def _entropy(values: Sequence[float], bins: int = 8) -> float:
    if len(values) < 2:
        return 0.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return 0.0
    counts = [0] * bins
    for value in values:
        index = int((value - lo) / (hi - lo) * bins)
        counts[min(bins - 1, max(0, index))] += 1
    total = float(len(values))
    result = 0.0
    for count in counts:
        if count:
            p = count / total
            result -= p * math.log(p, 2)
    return result / math.log(bins, 2)


def _autocorr(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    m = _mean(values)
    denom = sum((v - m) ** 2 for v in values)
    if denom < 1e-12:
        return 0.0
    numer = sum((values[i] - m) * (values[i - 1] - m) for i in range(1, len(values)))
    return numer / denom


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _safe_log(value: float | None) -> float | None:
    if value is None:
        return None
    return math.log1p(max(0.0, float(value)))


def _divide(numer: float, denom: float, default: float = 0.0) -> float:
    return numer / denom if abs(denom) > 1e-12 else default


def _insert_length(event: dict[str, Any]) -> int:
    if "text" in event:
        return len(str(event["text"]))
    return int(event.get("text_len", 1))


def _delete_count(event: dict[str, Any]) -> int:
    return int(event.get("count", 1))


def _angles(points: Sequence[tuple[float, float]]) -> list[float]:
    angles: list[float] = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) + abs(dy) > 1e-12:
            angles.append(math.atan2(dy, dx))
    return angles


def _angle_delta(a: float, b: float) -> float:
    delta = (b - a + math.pi) % (2 * math.pi) - math.pi
    return abs(delta)


def _stroke_points(points: Sequence[tuple[int, float, float]], gap_s: float) -> list[list[tuple[int, float, float]]]:
    if not points:
        return []
    strokes: list[list[tuple[int, float, float]]] = [[points[0]]]
    for previous, current in zip(points, points[1:]):
        if (current[0] - previous[0]) / 1000.0 > gap_s:
            strokes.append([current])
        else:
            strokes[-1].append(current)
    return [s for s in strokes if len(s) >= 2]


def _pointer_features(rec: Recording, features: dict[str, float | None]) -> None:
    points = [(int(e["t"]), float(e["x"]), float(e["y"])) for e in rec.events if e["type"] == "pointer"]
    clicks = [e for e in rec.events if e["type"] == "click"]
    duration = max(1.0, rec.duration_ms / 1000.0)

    features["f_has_pointer"] = 1.0 if len(points) >= 4 else 0.0
    features["f_has_clicks"] = 1.0 if clicks else 0.0

    if len(points) < 4:
        return

    times = [p[0] for p in points]
    dts: list[float] = []
    speeds: list[float] = []
    for (t1, x1, y1), (t2, x2, y2) in zip(points, points[1:]):
        dt = (t2 - t1) / 1000.0
        if dt <= 0:
            continue
        dist = math.hypot(x2 - x1, y2 - y1)
        dts.append(dt)
        speeds.append(dist / dt)

    if not speeds:
        return

    features["p_speed_med_log"] = _safe_log(_median(speeds))
    features["p_speed_p90_log"] = _safe_log(_quantile(speeds, 0.9))
    features["p_speed_cv"] = _cv(speeds)
    features["p_speed_skew"] = _skew([math.log1p(s) for s in speeds])
    features["p_speed_entropy"] = _entropy([math.log1p(s) for s in speeds]) if len(speeds) >= 3 else 0.0

    accels: list[float] = []
    for i, (s1, s2) in enumerate(zip(speeds, speeds[1:])):
        dt = dts[i] if i < len(dts) else 0.0
        if dt > 0:
            accels.append(abs(s2 - s1) / dt)
    jerks: list[float] = []
    for a1, a2, dt in zip(accels, accels[1:], dts[1:]):
        if dt > 0:
            jerks.append(abs(a2 - a1) / dt)
    features["p_accel_med_log"] = _safe_log(_median(accels))
    features["p_jerk_med_log"] = _safe_log(_median(jerks))

    angles = _angles([(p[1], p[2]) for p in points])
    changes = [_angle_delta(a, b) for a, b in zip(angles, angles[1:])]
    features["p_dirchange_rate"] = _divide(sum(changes), duration) if changes else 0.0
    features["p_dir_entropy"] = _entropy(angles) if len(angles) >= 3 else 0.0

    # Strokes: a pause is a gap much longer than the session's normal sampling
    # interval.  The 0.55 s floor keeps a 5 Hz eye-gaze stream in one stroke
    # while still splitting ordinary mouse movement at a natural pause.
    median_dt = _median(dts)
    gap_s = max(0.55, 3.0 * median_dt)
    strokes = _stroke_points(points, gap_s)
    features["p_stroke_n_per_min"] = len(strokes) / duration * 60.0
    if strokes:
        stroke_durations: list[float] = []
        stroke_speeds: list[float] = []
        straightness: list[float] = []
        curvatures: list[float] = []
        stroke_changes: list[float] = []
        for stroke in strokes:
            st = (stroke[-1][0] - stroke[0][0]) / 1000.0
            if st <= 0:
                continue
            path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(stroke, stroke[1:]))
            displacement = math.hypot(stroke[-1][1] - stroke[0][1], stroke[-1][2] - stroke[0][2])
            if path <= 1e-9:
                continue
            stroke_durations.append(st)
            stroke_speeds.append(path / st)
            straightness.append(min(1.0, displacement / path))
            curvatures.append(max(0.0, path / max(displacement, 1e-6) - 1.0))
            s_angles = _angles([(p[1], p[2]) for p in stroke])
            if len(s_angles) >= 2:
                stroke_changes.append(sum(_angle_delta(a, b) for a, b in zip(s_angles, s_angles[1:])))
        if stroke_durations:
            features["p_stroke_dur_med_log"] = _safe_log(_median(stroke_durations))
            features["p_stroke_speed_med_log"] = _safe_log(_median(stroke_speeds))
            features["p_stroke_speed_cv"] = _cv(stroke_speeds) if len(stroke_speeds) >= 3 else None
            features["p_straight_med"] = _median(straightness)
            features["p_straight_cv"] = _cv(straightness) if len(straightness) >= 3 else None
            features["p_curvature_med_log"] = _safe_log(_median(curvatures))
            features["p_dirchange_per_stroke"] = _divide(sum(stroke_changes), len(stroke_changes)) if stroke_changes else 0.0

    pauses = [dt for dt in dts if dt > 0.55]
    features["p_pause_frac"] = len(pauses) / len(dts) if dts else 0.0
    if len(pauses) >= 2:
        features["p_pause_med_log"] = _safe_log(_median(pauses))
        features["p_pause_cv"] = _cv(pauses)
        features["p_pause_tail"] = sum(1 for p in pauses if p > 2.0) / len(pauses)

    # Click dynamics.
    downs = [e for e in clicks if e.get("action") == "down"]
    ups = [e for e in clicks if e.get("action") == "up"]
    pairs: list[tuple[int, int, float, float]] = []
    used: set[int] = set()
    for down in downs:
        dt = int(down["t"])
        for index, up in enumerate(ups):
            if index in used or int(up["t"]) < dt:
                continue
            if up.get("button", "left") != down.get("button", "left"):
                continue
            if int(up["t"]) - dt <= 3000:
                pairs.append((dt, int(up["t"]), float(down["x"]), float(down["y"])))
                used.add(index)
                break

    if len(pairs) >= 2:
        dwells = [(up_t - down_t) / 1000.0 for down_t, up_t, _, _ in pairs]
        features["p_dwell_med_log"] = _safe_log(_median(dwells))
        features["p_dwell_cv"] = _cv(dwells) if len(dwells) >= 3 else None
        intervals = [(b[0] - a[0]) / 1000.0 for a, b in zip(pairs, pairs[1:])]
        if len(intervals) >= 3:
            features["p_click_interval_cv"] = _cv(intervals)
        cxs = [p[2] for p in pairs]
        cys = [p[3] for p in pairs]
        features["p_click_pos_std"] = 0.5 * (_std(cxs) + _std(cys)) if len(pairs) >= 3 else None

        preclick_pauses: list[float] = []
        overshoots: list[int] = []
        for down_t, _, cx, cy in pairs:
            prior = [p for p in points if p[0] <= down_t]
            if not prior:
                continue
            last = prior[-1]
            preclick_pauses.append((down_t - last[0]) / 1000.0)
            window = [p for p in prior if down_t - p[0] <= 650]
            if len(window) >= 4:
                path = sum(
                    math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(window, window[1:])
                )
                net = math.hypot(window[-1][1] - window[0][1], window[-1][2] - window[0][2])
                w_angles = _angles([(p[1], p[2]) for p in window])
                max_turn = max(
                    (_angle_delta(a, b) for a, b in zip(w_angles, w_angles[1:])),
                    default=0.0,
                )
                overshoots.append(1 if path > 1.35 * max(net, 1e-6) and max_turn > math.radians(100) else 0)
        if preclick_pauses:
            features["p_preclick_pause_med_log"] = _safe_log(_median(preclick_pauses))
        if len(overshoots) >= 2:
            features["p_overshoot_frac"] = sum(overshoots) / len(overshoots)

    # Within-session drift: fatigue is not just "slower on average"; a tired
    # person usually slows down across the session.  Binning keeps this robust
    # against single long pauses.
    bins = 5
    bin_speeds: list[list[float]] = [[] for _ in range(bins)]
    start = times[0]
    span = max(1, times[-1] - start)
    for t, s in zip(times, speeds):
        idx = min(bins - 1, int((t - start) / span * bins))
        bin_speeds[idx].append(s)
    bin_medians = [math.log1p(_median(v)) for v in bin_speeds if len(v) >= 3]
    if len(bin_medians) >= 3:
        features["p_speed_trend"] = _slope(list(range(len(bin_medians))), bin_medians) / max(1.0, len(bin_medians))
    if len(bin_medians) >= 2:
        first = _median([v for v in bin_speeds[0] if v] or [0.0])
        last = _median([v for v in bin_speeds[-1] if v] or [0.0])
        features["p_speed_ratio"] = _divide(last, first, 1.0)


def _bursts(events: Sequence[dict[str, Any]], gap_s: float) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for event in events:
        if result and (int(event["t"]) - int(result[-1][-1]["t"])) / 1000.0 <= gap_s:
            result[-1].append(event)
        else:
            result.append([event])
    return result


def _source_features(
    events: Sequence[dict[str, Any]],
    source: str,
    features: dict[str, float | None],
) -> None:
    inserts = [e for e in events if e.get("source") == source]
    if len(inserts) < 3:
        return
    intervals = [(int(b["t"]) - int(a["t"])) / 1000.0 for a, b in zip(inserts, inserts[1:])]
    intervals = [i for i in intervals if i > 0]

    if source in {"typed", "onscreen"}:
        prefix = "t_typed" if source == "typed" else "t_onscreen"
        burst_gap = 2.0
        groups = _bursts(inserts, burst_gap)
        intra: list[float] = []
        for group in groups:
            for a, b in zip(group, group[1:]):
                gap = (int(b["t"]) - int(a["t"])) / 1000.0
                if gap > 0:
                    intra.append(gap)
        if len(intra) >= 3:
            logs = [math.log1p(i) for i in intra]
            features[f"{prefix}_intra_med_log"] = _safe_log(_median(intra))
            features[f"{prefix}_intra_cv"] = _cv(intra)
            features[f"{prefix}_intra_autocorr"] = _autocorr(logs)
            features[f"{prefix}_intra_tail"] = sum(1 for i in intra if i > 4 * _median(intra)) / len(intra)
        chars = [sum(_insert_length(e) for e in group) for group in groups]
        if len(chars) >= 2:
            features[f"{prefix}_burst_chars_med_log"] = _safe_log(_median(chars))
            features[f"{prefix}_burst_chars_cv"] = _cv(chars)
        pauses = [
            (int(b["t"]) - int(a["t"])) / 1000.0
            for group_a, group_b in zip(groups, groups[1:])
            for a in [group_a[-1]]
            for b in [group_b[0]]
        ]
        features[f"{prefix}_pause_frac"] = len(pauses) / len(groups) if groups else 0.0
        if len(pauses) >= 2:
            features[f"{prefix}_pause_med_log"] = _safe_log(_median(pauses))

    elif source == "software":
        if len(intervals) >= 3:
            logs = [math.log1p(i) for i in intervals]
            features["t_soft_gap_med_log"] = _safe_log(_median(intervals))
            features["t_soft_gap_cv"] = _cv(intervals)
            features["t_soft_gap_autocorr"] = _autocorr(logs)
            features["t_soft_gap_tail"] = sum(1 for i in intervals if i > 4 * _median(intervals)) / len(intervals)
        lengths = [_insert_length(e) for e in inserts]
        if len(lengths) >= 2:
            features["t_soft_chars_med_log"] = _safe_log(_median(lengths))
            features["t_soft_chars_cv"] = _cv(lengths)
        groups = _bursts(inserts, 6.0)
        pauses = [
            (int(b["t"]) - int(a["t"])) / 1000.0
            for group_a, group_b in zip(groups, groups[1:])
            for a in [group_a[-1]]
            for b in [group_b[0]]
        ]
        features["t_soft_pause_frac"] = len(pauses) / len(groups) if groups else 0.0
        if len(pauses) >= 2:
            features["t_soft_pause_med_log"] = _safe_log(_median(pauses))


def _text_features(rec: Recording, features: dict[str, float | None]) -> None:
    inserts = [e for e in rec.events if e["type"] == "text" and e.get("op") == "insert"]
    deletes = [e for e in rec.events if e["type"] == "text" and e.get("op") == "delete"]
    duration = max(1.0, rec.duration_ms / 1000.0)
    first_t = rec.events[0]["t"] if rec.events else 0

    features["f_has_text"] = 1.0 if inserts else 0.0
    for source in ("typed", "onscreen", "software"):
        features[f"f_has_{source}"] = 1.0 if any(e.get("source") == source for e in inserts) else 0.0

    if not inserts:
        return

    inserted_chars = sum(_insert_length(e) for e in inserts)
    deleted_chars = sum(_delete_count(e) for e in deletes)
    features["t_insert_rate"] = len(inserts) / duration * 60.0
    features["t_delete_rate"] = len(deletes) / duration * 60.0
    features["t_correction_ratio"] = min(5.0, _divide(deleted_chars, max(1, inserted_chars)))
    features["t_multi_delete_frac"] = (
        sum(1 for e in deletes if _delete_count(e) > 1) / len(deletes) if deletes else 0.0
    )

    bursts: list[list[dict[str, Any]]] = []
    for event in deletes:
        if bursts and int(event["t"]) - int(bursts[-1][-1]["t"]) <= 450:
            bursts[-1].append(event)
        else:
            bursts.append([event])
    if len(bursts) >= 2:
        sizes = [sum(_delete_count(e) for e in burst) for burst in bursts]
        features["t_delete_burst_med_log"] = _safe_log(_median(sizes))
        features["t_delete_burst_cv"] = _cv(sizes) if len(sizes) >= 3 else None
        latencies: list[float] = []
        after_insert = 0
        for burst in bursts:
            prior = [e for e in inserts if int(e["t"]) <= int(burst[0]["t"])]
            if prior:
                latency = (int(burst[0]["t"]) - int(prior[-1]["t"])) / 1000.0
                latencies.append(latency)
                if latency <= 3.0:
                    after_insert += 1
        if latencies:
            features["t_correction_latency_med_log"] = _safe_log(_median(latencies))
        features["t_correction_after_insert_frac"] = after_insert / len(bursts)

    chars_by_source: Counter[str] = Counter()
    for event in inserts:
        chars_by_source[str(event["source"])] += _insert_length(event)
    total_chars = max(1, sum(chars_by_source.values()))
    features["t_modality_typed"] = chars_by_source.get("typed", 0) / total_chars
    features["t_modality_onscreen"] = chars_by_source.get("onscreen", 0) / total_chars
    features["t_modality_software"] = chars_by_source.get("software", 0) / total_chars
    switches = sum(1 for a, b in zip(inserts, inserts[1:]) if a["source"] != b["source"])
    features["t_modality_switch_rate"] = switches / duration * 60.0

    intervals = [(int(b["t"]) - int(a["t"])) / 1000.0 for a, b in zip(inserts, inserts[1:])]
    positive_intervals = [i for i in intervals if i > 0]
    features["t_chars_per_insert_med_log"] = _safe_log(_median([_insert_length(e) for e in inserts]))
    features["t_chars_per_insert_cv"] = _cv([_insert_length(e) for e in inserts]) if len(inserts) >= 2 else 0.0
    features["t_first_text_delay_log"] = _safe_log((int(inserts[0]["t"]) - int(first_t)) / 1000.0)
    if len(positive_intervals) >= 3:
        features["t_autocorr"] = _autocorr([math.log1p(i) for i in positive_intervals])
        features["t_interval_entropy"] = _entropy([math.log1p(i) for i in positive_intervals])
        features["t_pause_frac"] = sum(1 for i in positive_intervals if i > 2.0) / len(positive_intervals)
        pauses = [i for i in positive_intervals if i > 2.0]
        if len(pauses) >= 2:
            features["t_pause_med_log"] = _safe_log(_median(pauses))
            features["t_pause_cv"] = _cv(pauses)
    if len(positive_intervals) >= 4:
        logs = [math.log1p(i) for i in positive_intervals]
        features["t_interval_trend"] = _slope(list(range(len(logs))), logs) / max(1.0, len(logs))

    span = max(1.0, (int(inserts[-1]["t"]) - int(inserts[0]["t"])) / 1000.0)
    buckets = [0] * max(1, int(math.ceil(span)))
    for event in inserts:
        idx = min(len(buckets) - 1, int((int(event["t"]) - int(inserts[0]["t"])) / 1000.0))
        buckets[idx] += _insert_length(event)
    features["t_burstiness"] = _cv(buckets)

    click_count = sum(1 for e in rec.events if e["type"] == "click" and e.get("action") == "down")
    features["s_text_per_click"] = _divide(len(inserts), click_count) if click_count else None

    for source in ("typed", "onscreen", "software"):
        _source_features(inserts, source, features)


def extract_features(recording: Recording) -> dict[str, float | None]:
    """Return one feature dictionary for a whole session.

    A verification decision uses one session.  The prototype does not need a
    camera, audio, a particular OS, or a particular input device; if a group of
    events is absent the corresponding features are simply ``None``.
    """

    features: dict[str, float | None] = {name: None for name in FEATURE_NAMES}
    duration = max(1.0, recording.duration_ms / 1000.0)
    n_events = len(recording.events)
    n_pointer = sum(1 for e in recording.events if e["type"] == "pointer")
    n_clicks = sum(1 for e in recording.events if e["type"] == "click" and e.get("action") == "down")
    n_chars = sum(
        _insert_length(e) for e in recording.events if e["type"] == "text" and e.get("op") == "insert"
    )

    features["s_duration_log"] = math.log1p(duration)
    features["s_event_rate"] = n_events / duration * 60.0
    features["s_pointer_rate"] = n_pointer / duration * 60.0
    features["s_click_rate"] = n_clicks / duration * 60.0
    features["s_char_rate"] = n_chars / duration * 60.0
    features["s_pointer_share"] = n_pointer / n_events if n_events else 0.0
    features["s_text_share"] = sum(1 for e in recording.events if e["type"] == "text") / n_events if n_events else 0.0

    _pointer_features(recording, features)
    _text_features(recording, features)
    return features


def feature_matrix(recordings: Iterable[Recording]) -> list[list[float | None]]:
    return [[row.get(name) for name in FEATURE_NAMES] for row in (extract_features(r) for r in recordings)]
