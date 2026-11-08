"""Read, validate and write OS-neutral Marrow recordings.

A recording is either:

* a JSON Lines file: the first line is a header object, every later line is one
  event object, or
* a single JSON object with the header fields and an ``events`` array.

The format is intentionally small.  It carries only information an operating
system already has: millisecond timestamps, pointer positions as a fraction of
the screen, click transitions, and text events labelled by how the text arrived.
No window titles, process names, key codes, device names, audio or camera data
belong in a recording.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

FORMAT = "marrow-recording-v1"
EVENT_TYPES = {"pointer", "click", "text"}
TEXT_OPS = {"insert", "delete"}
TEXT_SOURCES = {"typed", "onscreen", "software"}
CLICK_ACTIONS = {"down", "up"}
BUTTONS = {"left", "right", "middle"}


class RecordingError(ValueError):
    """A recording is missing a required field or carries an invalid value."""


@dataclass(frozen=True)
class Recording:
    header: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    path: str = ""

    @property
    def session_id(self) -> str:
        return str(self.header.get("session_id", ""))

    @property
    def duration_ms(self) -> int:
        if not self.events:
            return 0
        return int(self.events[-1]["t"] - self.events[0]["t"])

    @property
    def labels(self) -> dict[str, Any]:
        value = self.header.get("labels", {})
        return value if isinstance(value, dict) else {}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RecordingError(f"{name} must be a finite number")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordingError(f"{name} must be an integer")
    return value


def validate_header(header: dict[str, Any]) -> None:
    if not isinstance(header, dict):
        raise RecordingError("header must be an object")
    if header.get("format") != FORMAT:
        raise RecordingError(f"header.format must be {FORMAT!r}")
    if not isinstance(header.get("session_id"), str) or not header["session_id"]:
        raise RecordingError("header.session_id must be a non-empty string")
    if header.get("clock", "ms") != "ms":
        raise RecordingError("header.clock must be 'ms'")
    if header.get("pointer_units", "screen_fraction") != "screen_fraction":
        raise RecordingError("header.pointer_units must be 'screen_fraction'")
    if "duration_ms" in header:
        _integer(header["duration_ms"], "header.duration_ms")
    if "synthetic" in header and not isinstance(header["synthetic"], bool):
        raise RecordingError("header.synthetic must be a boolean")
    labels = header.get("labels")
    if labels is not None and not isinstance(labels, dict):
        raise RecordingError("header.labels must be an object when present")


def validate_event(event: dict[str, Any], index: int, previous_t: int | None) -> None:
    where = f"event {index}"
    if not isinstance(event, dict):
        raise RecordingError(f"{where} must be an object")
    if "t" not in event:
        raise RecordingError(f"{where}.t is required")
    t = _integer(event["t"], f"{where}.t")
    if t < 0:
        raise RecordingError(f"{where}.t must be non-negative")
    if previous_t is not None and t < previous_t:
        raise RecordingError(f"{where}.t is earlier than the previous event")
    kind = event.get("type")
    if kind not in EVENT_TYPES:
        raise RecordingError(f"{where}.type must be one of {sorted(EVENT_TYPES)}")
    if kind == "pointer":
        for axis in ("x", "y"):
            if axis not in event:
                raise RecordingError(f"{where}.{axis} is required for a pointer event")
            value = _finite_number(event[axis], f"{where}.{axis}")
            if not 0.0 <= value <= 1.0:
                raise RecordingError(f"{where}.{axis} must be a screen fraction in [0, 1]")
    elif kind == "click":
        action = event.get("action")
        if action not in CLICK_ACTIONS:
            raise RecordingError(f"{where}.action must be 'down' or 'up'")
        button = event.get("button", "left")
        if button not in BUTTONS:
            raise RecordingError(f"{where}.button must be left, right or middle")
        for axis in ("x", "y"):
            if axis not in event:
                raise RecordingError(f"{where}.{axis} is required for a click event")
            value = _finite_number(event[axis], f"{where}.{axis}")
            if not 0.0 <= value <= 1.0:
                raise RecordingError(f"{where}.{axis} must be a screen fraction in [0, 1]")
    elif kind == "text":
        op = event.get("op")
        if op not in TEXT_OPS:
            raise RecordingError(f"{where}.op must be 'insert' or 'delete'")
        source = event.get("source")
        if source not in TEXT_SOURCES:
            raise RecordingError(f"{where}.source must be typed, onscreen or software")
        if op == "insert":
            has_text = "text" in event
            has_len = "text_len" in event
            if has_text == has_len:
                raise RecordingError(f"{where} must carry exactly one of text or text_len")
            if has_text:
                if not isinstance(event["text"], str) or not event["text"]:
                    raise RecordingError(f"{where}.text must be a non-empty string")
            else:
                length = _integer(event["text_len"], f"{where}.text_len")
                if length <= 0:
                    raise RecordingError(f"{where}.text_len must be positive")
        else:
            count = event.get("count", 1)
            count = _integer(count, f"{where}.count")
            if count <= 0:
                raise RecordingError(f"{where}.count must be positive")


def validate_recording(header: dict[str, Any], events: Sequence[dict[str, Any]]) -> None:
    validate_header(header)
    previous: int | None = None
    for index, event in enumerate(events):
        validate_event(event, index, previous)
        previous = int(event["t"])


def from_object(obj: dict[str, Any], path: str = "") -> Recording:
    if not isinstance(obj, dict):
        raise RecordingError(f"{path or 'recording'} must be a JSON object")
    if "events" not in obj or not isinstance(obj["events"], list):
        raise RecordingError(f"{path or 'recording'} must carry an events array")
    header = {key: value for key, value in obj.items() if key != "events"}
    events = tuple(obj["events"])
    validate_recording(header, events)
    return Recording(header=header, events=events, path=path)


def load_recording(path: str | os.PathLike[str]) -> Recording:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecordingError(f"cannot read {p}: {exc}") from exc
    if not raw.strip():
        raise RecordingError(f"{p} is empty")
    text = raw.lstrip()
    if text.startswith("{") and "\n" not in text.strip():
        obj = json.loads(text)
        return from_object(obj, str(p))
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise RecordingError(f"{p} is empty")
    try:
        header = json.loads(lines[0])
        events = [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as exc:
        raise RecordingError(f"{p}: invalid JSON: {exc}") from exc
    validate_recording(header, events)
    return Recording(header=header, events=tuple(events), path=str(p))


def dump_recording(recording: Recording) -> str:
    lines = [json.dumps(recording.header, ensure_ascii=False, separators=(",", ":"))]
    for event in recording.events:
        lines.append(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def write_recording(path: str | os.PathLike[str], header: dict[str, Any], events: Iterable[dict[str, Any]]) -> None:
    rec = Recording(header=dict(header), events=tuple(dict(e) for e in events))
    validate_recording(rec.header, rec.events)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_recording(rec), encoding="utf-8")


def iter_recording_paths(root: str | os.PathLike[str]) -> Iterator[Path]:
    p = Path(root)
    if p.is_file():
        if p.suffix.lower() in {".json", ".jsonl"}:
            yield p
        return
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".json", ".jsonl"}:
                yield child


def load_many(paths: Iterable[str | os.PathLike[str]]) -> list[Recording]:
    recordings: list[Recording] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for child in iter_recording_paths(p):
                recordings.append(load_recording(child))
        else:
            recordings.append(load_recording(p))
    return recordings
