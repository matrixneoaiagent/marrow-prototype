"""Deterministic synthetic recording generator.

No real movement, text, audio or camera data is used.  The generator is part of
the prototype so a reviewer can reproduce every byte from a seed.  It builds
three input styles:

* ``pointer_keyboard`` -- ordinary mouse and physical keyboard;
* ``slow_pointer_dictation`` -- a slow joystick/head pointer with text inserted
  by dictation software, including spoken corrections;
* ``switch_onscreen`` -- switch access and an on-screen keyboard with dwell
  clicks.

Each style contains an enrolled owner, a tired owner, a different person
(``alternate``, used as the stranger and for the swap test), and two mimics who
copy the owner's central habits but differ in micro-timing, variability and
path shape.

The generator is not a model of any real person.  Its purpose is to make the
five listing conditions checkable and to expose the method's assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from .format import FORMAT, Recording, validate_recording, write_recording

# A small original vocabulary.  Text content is irrelevant to the detector; it
# exists only so a human reading a recording can see that inserts are words and
# corrections remove characters.  Nothing here is copied from a source text.
WORDS = (
    "open", "check", "the", "last", "line", "please", "send", "note", "before",
    "noon", "read", "this", "page", "and", "tell", "me", "what", "changed",
    "keep", "window", "small", "move", "cursor", "left", "then", "right",
    "press", "enter", "when", "ready", "save", "copy", "draft", "after",
    "lunch", "review", "second", "paragraph", "quiet", "clear", "short",
)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _phrase(rng: random.Random, min_chars: int, max_chars: int) -> str:
    target = rng.randint(min_chars, max_chars)
    words: list[str] = []
    length = 0
    while length < target:
        word = rng.choice(WORDS)
        words.append(word)
        length += len(word) + 1
    return " ".join(words)[: max(min_chars, target)]


@dataclass(frozen=True)
class Style:
    name: str
    sample_ms: int
    modality: str
    speed: float
    speed_cv: float
    jitter: float
    straightness: float
    overshoot: float
    pause_mean: float
    pause_sigma: float
    pause_tail: float
    click_interval: float
    dwell: float
    dwell_cv: float
    key_interval: float
    key_cv: float
    key_autocorr: float
    correction_rate: float
    correction_burst: float
    phrase_mean: float
    phrase_cv: float
    phrase_interval: float
    phrase_sigma: float
    stepwise: bool
    step_ms: int


STYLES: dict[str, Style] = {
    "pointer_keyboard": Style(
        name="pointer_keyboard",
        sample_ms=70,
        modality="typed",
        speed=0.85,
        speed_cv=0.30,
        jitter=0.012,
        straightness=0.92,
        overshoot=0.10,
        pause_mean=0.90,
        pause_sigma=0.50,
        pause_tail=0.06,
        click_interval=7.0,
        dwell=0.09,
        dwell_cv=0.20,
        key_interval=0.11,
        key_cv=0.25,
        key_autocorr=0.20,
        correction_rate=0.020,
        correction_burst=1.2,
        phrase_mean=0.0,
        phrase_cv=0.0,
        phrase_interval=0.0,
        phrase_sigma=0.0,
        stepwise=False,
        step_ms=0,
    ),
    "slow_pointer_dictation": Style(
        name="slow_pointer_dictation",
        sample_ms=55,
        modality="software",
        speed=0.12,
        speed_cv=0.45,
        jitter=0.025,
        straightness=0.82,
        overshoot=0.25,
        pause_mean=1.30,
        pause_sigma=0.85,
        pause_tail=0.14,
        click_interval=9.0,
        dwell=0.45,
        dwell_cv=0.30,
        key_interval=0.0,
        key_cv=0.0,
        key_autocorr=0.0,
        correction_rate=0.12,
        correction_burst=2.8,
        phrase_mean=16.0,
        phrase_cv=0.45,
        phrase_interval=2.8,
        phrase_sigma=0.35,
        stepwise=True,
        step_ms=180,
    ),
    "switch_onscreen": Style(
        name="switch_onscreen",
        sample_ms=110,
        modality="onscreen",
        speed=0.030,
        speed_cv=0.50,
        jitter=0.035,
        straightness=0.75,
        overshoot=0.30,
        pause_mean=0.90,
        pause_sigma=0.45,
        pause_tail=0.08,
        click_interval=8.0,
        dwell=0.80,
        dwell_cv=0.18,
        key_interval=0.75,
        key_cv=0.20,
        key_autocorr=0.30,
        correction_rate=0.030,
        correction_burst=1.0,
        phrase_mean=0.0,
        phrase_cv=0.0,
        phrase_interval=0.0,
        phrase_sigma=0.0,
        stepwise=True,
        step_ms=300,
    ),
}


@dataclass(frozen=True)
class Persona:
    key: str
    role: str
    style: str
    speed: float
    speed_cv: float
    jitter: float
    straightness: float
    overshoot: float
    pause_mean: float
    pause_sigma: float
    pause_tail: float
    click_interval: float
    dwell: float
    dwell_cv: float
    key_interval: float
    key_cv: float
    key_autocorr: float
    correction_rate: float
    correction_burst: float
    phrase_mean: float
    phrase_cv: float
    phrase_interval: float
    phrase_sigma: float
    stepwise: bool
    step_ms: int
    target: str | None = None
    mimic_variant: str | None = None


# Owners and alternates.  The two people in each style share a device, so a
# profile cannot separate them by "this is a mouse" or "this is dictation".
PERSONAS: dict[str, Persona] = {
    "alice": Persona(
        key="alice", role="owner", style="pointer_keyboard",
        speed=0.95, speed_cv=0.28, jitter=0.010, straightness=0.94, overshoot=0.08,
        pause_mean=0.80, pause_sigma=0.45, pause_tail=0.04, click_interval=6.5,
        dwell=0.085, dwell_cv=0.18, key_interval=0.105, key_cv=0.22,
        key_autocorr=0.26, correction_rate=0.018, correction_burst=1.2,
        phrase_mean=0.0, phrase_cv=0.0, phrase_interval=0.0, phrase_sigma=0.0,
        stepwise=False, step_ms=0,
    ),
    "bruno": Persona(
        key="bruno", role="alternate", style="pointer_keyboard",
        speed=1.45, speed_cv=0.45, jitter=0.030, straightness=0.84, overshoot=0.22,
        pause_mean=0.55, pause_sigma=0.85, pause_tail=0.12, click_interval=4.0,
        dwell=0.060, dwell_cv=0.35, key_interval=0.075, key_cv=0.45,
        key_autocorr=0.02, correction_rate=0.085, correction_burst=1.8,
        phrase_mean=0.0, phrase_cv=0.0, phrase_interval=0.0, phrase_sigma=0.0,
        stepwise=False, step_ms=0,
    ),
    "ravi": Persona(
        key="ravi", role="owner", style="slow_pointer_dictation",
        speed=0.10, speed_cv=0.50, jitter=0.028, straightness=0.80, overshoot=0.28,
        pause_mean=1.50, pause_sigma=0.95, pause_tail=0.18, click_interval=8.0,
        dwell=0.50, dwell_cv=0.30, key_interval=0.0, key_cv=0.0,
        key_autocorr=0.0, correction_rate=0.14, correction_burst=3.0,
        phrase_mean=18.0, phrase_cv=0.45, phrase_interval=2.7, phrase_sigma=0.35,
        stepwise=True, step_ms=180,
    ),
    "devi": Persona(
        key="devi", role="alternate", style="slow_pointer_dictation",
        speed=0.17, speed_cv=0.30, jitter=0.018, straightness=0.90, overshoot=0.15,
        pause_mean=1.00, pause_sigma=0.55, pause_tail=0.08, click_interval=7.0,
        dwell=0.30, dwell_cv=0.20, key_interval=0.0, key_cv=0.0,
        key_autocorr=0.0, correction_rate=0.05, correction_burst=1.5,
        phrase_mean=9.0, phrase_cv=0.30, phrase_interval=3.5, phrase_sigma=0.25,
        stepwise=True, step_ms=220,
    ),
    "nia": Persona(
        key="nia", role="owner", style="switch_onscreen",
        speed=0.028, speed_cv=0.55, jitter=0.040, straightness=0.72, overshoot=0.35,
        pause_mean=1.00, pause_sigma=0.45, pause_tail=0.10, click_interval=7.0,
        dwell=0.85, dwell_cv=0.15, key_interval=0.75, key_cv=0.20,
        key_autocorr=0.35, correction_rate=0.030, correction_burst=1.0,
        phrase_mean=0.0, phrase_cv=0.0, phrase_interval=0.0, phrase_sigma=0.0,
        stepwise=True, step_ms=300,
    ),
    "oren": Persona(
        key="oren", role="alternate", style="switch_onscreen",
        speed=0.080, speed_cv=0.30, jitter=0.015, straightness=0.95, overshoot=0.10,
        pause_mean=0.45, pause_sigma=0.55, pause_tail=0.08, click_interval=5.5,
        dwell=0.35, dwell_cv=0.20, key_interval=0.95, key_cv=0.25,
        key_autocorr=0.15, correction_rate=0.20, correction_burst=2.5,
        phrase_mean=0.0, phrase_cv=0.0, phrase_interval=0.0, phrase_sigma=0.0,
        stepwise=True, step_ms=150,
    ),
    # A third person in each style gives the swap profile a same-style
    # background.  Without this, a profile trained on one mouse user can look
    # like "a mouse user" rather than "this mouse user", and the swap test
    # would only measure device style.
    "casey": Persona(
        key="casey", role="stranger", style="pointer_keyboard",
        speed=0.52, speed_cv=0.40, jitter=0.040, straightness=0.76, overshoot=0.25,
        pause_mean=1.30, pause_sigma=0.80, pause_tail=0.12, click_interval=5.0,
        dwell=0.16, dwell_cv=0.40, key_interval=0.180, key_cv=0.50,
        key_autocorr=0.05, correction_rate=0.12, correction_burst=2.0,
        phrase_mean=0.0, phrase_cv=0.0, phrase_interval=0.0, phrase_sigma=0.0,
        stepwise=False, step_ms=0,
    ),
    "mira": Persona(
        key="mira", role="stranger", style="slow_pointer_dictation",
        speed=0.22, speed_cv=0.35, jitter=0.020, straightness=0.95, overshoot=0.10,
        pause_mean=0.80, pause_sigma=0.50, pause_tail=0.06, click_interval=6.5,
        dwell=0.25, dwell_cv=0.20, key_interval=0.0, key_cv=0.0,
        key_autocorr=0.0, correction_rate=0.030, correction_burst=1.2,
        phrase_mean=6.0, phrase_cv=0.25, phrase_interval=4.2, phrase_sigma=0.25,
        stepwise=True, step_ms=170,
    ),
    "pax": Persona(
        key="pax", role="stranger", style="switch_onscreen",
        speed=0.060, speed_cv=0.38, jitter=0.024, straightness=0.66, overshoot=0.30,
        pause_mean=0.55, pause_sigma=0.95, pause_tail=0.18, click_interval=4.5,
        dwell=0.42, dwell_cv=0.45, key_interval=0.42, key_cv=0.50,
        key_autocorr=0.00, correction_rate=0.16, correction_burst=2.2,
        phrase_mean=0.0, phrase_cv=0.0, phrase_interval=0.0, phrase_sigma=0.0,
        stepwise=True, step_ms=180,
    ),
}


def _mimic(target: Persona, variant: str) -> Persona:
    """Copy the visible averages, change the higher-order structure."""

    if variant == "regular":
        changes = dict(
            speed_cv=0.10,
            jitter=0.002,
            straightness=0.995,
            overshoot=0.010,
            pause_sigma=0.15,
            pause_tail=0.005,
            dwell_cv=0.04,
            key_cv=0.07,
            key_autocorr=0.0,
            correction_burst=1.0,
            phrase_cv=0.08,
            phrase_sigma=0.06,
        )
    elif variant == "noisy":
        changes = dict(
            speed_cv=min(1.20, target.speed_cv * 2.2),
            jitter=target.jitter * 3.5,
            straightness=max(0.50, target.straightness - 0.22),
            overshoot=min(0.65, target.overshoot + 0.22),
            pause_sigma=min(1.60, target.pause_sigma * 2.0),
            pause_tail=min(0.50, target.pause_tail + 0.16),
            dwell_cv=min(0.90, target.dwell_cv * 2.4),
            key_cv=min(1.00, target.key_cv * 2.4),
            key_autocorr=0.0,
            correction_burst=target.correction_burst * 2.2,
            phrase_cv=min(1.00, target.phrase_cv * 2.2) if target.phrase_cv else 0.0,
            phrase_sigma=min(0.90, target.phrase_sigma * 2.2) if target.phrase_sigma else 0.0,
        )
    else:
        raise ValueError(f"unknown mimic variant {variant!r}")

    return Persona(
        key=f"mimic_{target.key}_{variant}",
        role="mimic",
        style=target.style,
        speed=target.speed,
        speed_cv=changes["speed_cv"],
        jitter=changes["jitter"],
        straightness=changes["straightness"],
        overshoot=changes["overshoot"],
        pause_mean=target.pause_mean,
        pause_sigma=changes["pause_sigma"],
        pause_tail=changes["pause_tail"],
        click_interval=target.click_interval,
        dwell=target.dwell,
        dwell_cv=changes["dwell_cv"],
        key_interval=target.key_interval,
        key_cv=changes["key_cv"],
        key_autocorr=changes["key_autocorr"],
        correction_rate=target.correction_rate,
        correction_burst=changes["correction_burst"],
        phrase_mean=target.phrase_mean,
        phrase_cv=changes["phrase_cv"],
        phrase_interval=target.phrase_interval,
        phrase_sigma=changes["phrase_sigma"],
        stepwise=target.stepwise,
        step_ms=target.step_ms,
        target=target.key,
        mimic_variant=variant,
    )


# The person used to train the swap profile for each owner.  All are
# same-style non-owners.  The choice was made in development from validation
# profile quality (TPR at FAR <= 5%) and fixed before the final test run; no
# test score was used to pick it.
SWAP_TARGETS = {
    "alice": "bruno",
    "ravi": "mira",
    "nia": "oren",
}


for _target_key in ("alice", "ravi", "nia"):
    for _variant in ("regular", "noisy"):
        _persona = _mimic(PERSONAS[_target_key], _variant)
        PERSONAS[_persona.key] = _persona


def _tired(persona: Persona) -> Persona:
    """Fatigue changes several parameters, not just a uniform clock scale."""

    return Persona(
        key=persona.key,
        role=persona.role,
        style=persona.style,
        speed=persona.speed * 0.70,
        speed_cv=persona.speed_cv * 1.25,
        jitter=persona.jitter * 1.55,
        straightness=max(0.55, persona.straightness - 0.045),
        overshoot=min(0.55, persona.overshoot * 1.30),
        pause_mean=persona.pause_mean * 1.55,
        pause_sigma=persona.pause_sigma * 1.20,
        pause_tail=min(0.40, persona.pause_tail + 0.06),
        click_interval=persona.click_interval * 1.20,
        dwell=persona.dwell * 1.15,
        dwell_cv=persona.dwell_cv * 1.30,
        key_interval=persona.key_interval * 1.30,
        key_cv=persona.key_cv * 1.30,
        key_autocorr=persona.key_autocorr * 0.85,
        correction_rate=min(0.45, persona.correction_rate * 1.75),
        correction_burst=persona.correction_burst * 1.15,
        phrase_mean=persona.phrase_mean * 0.90,
        phrase_cv=persona.phrase_cv * 1.15,
        phrase_interval=persona.phrase_interval * 1.30,
        phrase_sigma=persona.phrase_sigma * 1.15,
        stepwise=persona.stepwise,
        step_ms=persona.step_ms,
        target=persona.target,
        mimic_variant=persona.mimic_variant,
    )


class SessionWriter:
    def __init__(self, rng: random.Random, persona: Persona, duration_ms: int):
        self.rng = rng
        self.persona = persona
        self.style = STYLES[persona.style]
        self.duration_ms = duration_ms
        self.t = 0
        self.x = 0.50
        self.y = 0.50
        self.events: list[dict[str, Any]] = []
        self.last_t = 0
        self.click_due = rng.uniform(1.0, self.style.click_interval)
        self.last_click = 0.0
        self.next_text = rng.uniform(1.0, 4.0)
        self.last_text = 0.0
        self.key_z = rng.gauss(0.0, 1.0)
        self.text_buffer: list[str] = []

    def _time(self, seconds: float) -> int:
        value = max(self.last_t, int(round(self.t + seconds * 1000.0)))
        self.last_t = value
        return value

    def add_pointer(self, t: int, x: float, y: float) -> None:
        self.events.append({
            "t": max(self.last_t, int(t)),
            "type": "pointer",
            "x": round(min(1.0, max(0.0, x)), 5),
            "y": round(min(1.0, max(0.0, y)), 5),
        })
        self.last_t = self.events[-1]["t"]
        self.x, self.y = x, y

    def add_click(self, t: int, action: str) -> None:
        self.events.append({
            "t": max(self.last_t, int(t)),
            "type": "click",
            "action": action,
            "button": "left",
            "x": round(min(1.0, max(0.0, self.x)), 5),
            "y": round(min(1.0, max(0.0, self.y)), 5),
        })
        self.last_t = self.events[-1]["t"]

    def add_text(self, t: int, op: str, source: str, text: str | None = None, count: int | None = None) -> None:
        event: dict[str, Any] = {
            "t": max(self.last_t, int(t)),
            "type": "text",
            "op": op,
            "source": source,
        }
        if op == "insert":
            event["text"] = text if text else "x"
        else:
            event["count"] = int(count or 1)
        self.events.append(event)
        self.last_t = event["t"]

    def _lognormal(self, mean: float, sigma: float) -> float:
        if mean <= 0:
            return 0.0
        return mean * math.exp(self.rng.gauss(0.0, sigma))

    def _next_key_interval(self) -> float:
        style = self.style
        rho = max(-0.8, min(0.8, self.persona.key_autocorr))
        self.key_z = rho * self.key_z + math.sqrt(max(0.0, 1.0 - rho * rho)) * self.rng.gauss(0.0, 1.0)
        base = self.persona.key_interval or style.key_interval or 0.1
        sigma = max(0.01, (self.persona.key_cv or style.key_cv or 0.2) * 0.75)
        interval = base * math.exp(sigma * self.key_z)
        return max(0.025, min(2.5, interval))

    def move_to(self, target_x: float, target_y: float, click_at_end: bool) -> None:
        rng = self.rng
        persona = self.persona
        style = self.style
        dx, dy = target_x - self.x, target_y - self.y
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            if click_at_end:
                self._do_click()
            return
        speed = max(0.005, persona.speed * math.exp(rng.gauss(0.0, 0.12)))
        duration = distance / speed
        duration = max(0.20, min(3.2, duration)) * math.exp(rng.gauss(0.0, 0.15))
        cap = 3.2 if style.name == "pointer_keyboard" else 2.3
        duration = max(0.20, min(cap, duration))
        steps = max(3, int(duration * 1000.0 / max(20, style.sample_ms)))

        # Curvature is perpendicular to the straight line.  A straightness near
        # 1 gives a nearly straight path; lower values bow the path.
        bow = rng.gauss(0.0, max(0.002, (1.0 - persona.straightness) * 0.10))
        px, py = -dy / distance, dx / distance
        start_x, start_y = self.x, self.y
        overshoot = persona.overshoot if rng.random() < 0.5 else 0.0
        end_x, end_y = target_x, target_y
        if overshoot > 0 and distance > 0.05:
            factor = 1.0 + overshoot * rng.uniform(0.3, 1.0)
            end_x = start_x + dx * factor
            end_y = start_y + dy * factor
            end_x = min(1.0, max(0.0, end_x))
            end_y = min(1.0, max(0.0, end_y))

        for i in range(steps + 1):
            u = i / steps
            if persona.stepwise:
                # A joystick or switch moves in visible steps with holds.
                step_seconds = max(0.05, persona.step_ms / 1000.0)
                total_seconds = duration
                n_steps = max(1, int(total_seconds / step_seconds))
                u_eff = math.floor(u * n_steps) / n_steps
            else:
                u_eff = u * u * (3.0 - 2.0 * u)
            x = start_x + (end_x - start_x) * u_eff + px * bow * math.sin(math.pi * u)
            y = start_y + (end_y - start_y) * u_eff + py * bow * math.sin(math.pi * u)
            if i not in (0, steps):
                x += rng.gauss(0.0, persona.jitter)
                y += rng.gauss(0.0, persona.jitter)
            t = self._time(u * duration)
            self.add_pointer(t, x, y)

        self.t += duration * 1000.0

        # A short correction back to the true target after an overshoot.
        if (end_x, end_y) != (target_x, target_y):
            correction_steps = max(2, int(0.25 * 1000.0 / max(20, style.sample_ms)))
            sx, sy = self.x, self.y
            for i in range(1, correction_steps + 1):
                u = i / correction_steps
                x = sx + (target_x - sx) * u + rng.gauss(0.0, persona.jitter * 0.5)
                y = sy + (target_y - sy) * u + rng.gauss(0.0, persona.jitter * 0.5)
                t = self._time(0.25 * u)
                self.add_pointer(t, x, y)
            self.t += 250.0

        if click_at_end:
            self._do_click()

    def _do_click(self) -> None:
        rng = self.rng
        dwell = max(0.02, self._lognormal(self.persona.dwell, self.persona.dwell_cv))
        dwell = min(2.0, dwell)
        down = self._time(0.03)
        self.add_click(down, "down")
        self.t += dwell * 1000.0
        up = self._time(0.0)
        self.add_click(up, "up")
        self.last_click = self.t / 1000.0

    def do_pointer(self, click_at_end: bool) -> None:
        rng = self.rng
        # Prefer targets away from the current point, but occasionally nearby.
        if rng.random() < 0.7:
            tx = rng.uniform(0.10, 0.90)
            ty = rng.uniform(0.10, 0.90)
        else:
            tx = min(0.95, max(0.05, self.x + rng.gauss(0.0, 0.12)))
            ty = min(0.95, max(0.05, self.y + rng.gauss(0.0, 0.12)))
        self.move_to(tx, ty, click_at_end)

    def do_text(self) -> None:
        persona = self.persona
        source = self.style.modality
        rng = self.rng
        if source == "software":
            chars = max(3, int(self._lognormal(persona.phrase_mean, persona.phrase_cv)))
            chars = min(80, chars)
            phrase = _phrase(rng, max(3, chars // 2), chars)
            t = self._time(0.05)
            self.add_text(t, "insert", "software", text=phrase)
            self.text_buffer.extend(phrase)
            # Spoken corrections: a whole chunk is removed after a short pause.
            if rng.random() < persona.correction_rate:
                latency = max(0.15, self._lognormal(1.2, 0.6))
                self.t += latency * 1000.0
                delete_count = max(1, int(round(self._lognormal(persona.correction_burst, 0.5))))
                delete_count = min(delete_count, len(phrase))
                burst = max(1, int(1 + rng.random() * 3))
                remaining = delete_count
                for _ in range(burst):
                    if remaining <= 0:
                        break
                    amount = max(1, int(round(remaining / max(1, burst))))
                    self.add_text(self._time(0.10), "delete", "software", count=amount)
                    remaining -= amount
            # The phrase occupies the speaker's next interval.
            gap = max(0.5, self._lognormal(persona.phrase_interval, persona.phrase_sigma))
            self.t += gap * 1000.0
            self.last_text = self.t / 1000.0
            return

        # Typed or on-screen: one character at a time.
        count = rng.randint(3, 10) if source == "typed" else rng.randint(2, 5)
        chars = _phrase(rng, max(4, count), max(6, count + 4))
        for char in chars[:count]:
            interval = self._next_key_interval()
            self.t += interval * 1000.0
            self.add_text(self._time(0.0), "insert", source, text=char)
            self.text_buffer.append(char)
            if rng.random() < persona.correction_rate:
                # A correction is one or two deletes, quickly after the insert.
                self.t += max(0.05, self._lognormal(0.35, 0.5)) * 1000.0
                delete_count = 1 if rng.random() < 0.8 else 2
                for _ in range(delete_count):
                    self.add_text(self._time(0.06), "delete", source, count=1)
                    if self.text_buffer:
                        self.text_buffer.pop()
        self.last_text = self.t / 1000.0

    def pause(self) -> None:
        rng = self.rng
        seconds = self._lognormal(self.persona.pause_mean, self.persona.pause_sigma)
        seconds = min(12.0, max(0.15, seconds))
        if rng.random() < self.persona.pause_tail:
            seconds *= rng.uniform(1.5, 3.0)
        self.t += seconds * 1000.0

    def run(self) -> list[dict[str, Any]]:
        rng = self.rng
        # Start with a pointer sample so the session has an origin.
        self.add_pointer(0, self.x, self.y)
        # A small probability of a click at the start; the owner is already
        # doing something when the recording begins.
        if rng.random() < 0.5:
            self.pause()
        while self.t < self.duration_ms - 1200:
            # Decide the next action from elapsed time and style.  The weights
            # are not part of the classifier; they only keep sessions natural.
            candidates = ["text", "pointer", "click"]
            weights = {
                "pointer_keyboard": (0.48, 0.38, 0.14),
                "slow_pointer_dictation": (0.68, 0.20, 0.12),
                "switch_onscreen": (0.62, 0.25, 0.13),
            }[self.persona.style]
            max_text_gap = {
                "pointer_keyboard": 3.0,
                "slow_pointer_dictation": 4.5,
                "switch_onscreen": 3.5,
            }[self.persona.style]
            if (self.t / 1000.0 - self.last_text) > max_text_gap:
                action = "text"
            elif (self.t / 1000.0 - self.last_click) > self.persona.click_interval * 1.15:
                action = "click"
            else:
                action = rng.choices(candidates, weights=weights, k=1)[0]
            if action == "click":
                # Move a short distance, then click; this is how a target is
                # selected in every style.
                if rng.random() < 0.7:
                    self.do_pointer(click_at_end=True)
                else:
                    self._do_click()
            elif action == "pointer":
                self.do_pointer(click_at_end=rng.random() < 0.18)
            else:
                self.do_text()
            if action != "text" or rng.random() < 0.55:
                self.pause()
        # Make sure the recording ends with a pointer sample, not mid-gesture.
        if self.events and self.events[-1]["type"] != "pointer":
            self.add_pointer(self._time(0.05), self.x, self.y)
        return self.events


def generate_session(
    persona_key: str,
    condition: str,
    split: str,
    index: int,
    seed: int = 20261108,
) -> Recording:
    if persona_key not in PERSONAS:
        raise KeyError(f"unknown persona {persona_key!r}")
    base = PERSONAS[persona_key]
    persona = _tired(base) if condition == "tired" else base
    rng = random.Random(_stable_seed(seed, split, persona_key, condition, index))
    duration_ms = int(rng.uniform(58.0, 82.0) * 1000.0)
    writer = SessionWriter(rng, persona, duration_ms)
    events = writer.run()
    actual_duration = max(duration_ms, events[-1]["t"] if events else 0)
    session_id = f"{persona_key}-{condition}-{split}-{index:02d}"
    header = {
        "format": FORMAT,
        "session_id": session_id,
        "synthetic": True,
        "clock": "ms",
        "pointer_units": "screen_fraction",
        "duration_ms": actual_duration,
        "labels": {
            "participant": persona_key,
            "role": base.role,
            "condition": condition,
            "style": base.style,
            "split": split,
            "mimic_variant": base.mimic_variant,
            "mimic_target": base.target,
        },
        "generator": "marrow/synth.py",
        "generator_seed": seed,
        "license": "CC0-1.0",
        "note": "Synthetic recording. No real person, movement or text was recorded.",
    }
    validate_recording(header, events)
    return Recording(header=header, events=tuple(events), path="")


def _split_counts(split: str) -> list[tuple[str, int]]:
    if split == "train":
        return [("rested", 10), ("tired", 6)]
    if split == "validation":
        return [("rested", 4), ("tired", 4)]
    if split == "test":
        return [("rested", 6), ("tired", 6)]
    raise ValueError(f"unknown split {split!r}")


def build_dataset(
    out_dir: str | os.PathLike[str],
    seed: int = 20261108,
    force: bool = False,
) -> dict[str, Any]:
    """Write every recording and a manifest.  Deterministic for a given seed."""

    root = Path(out_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        with manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    sessions: list[dict[str, Any]] = []
    for persona_key, persona in PERSONAS.items():
        if persona.role == "mimic":
            # Mimics appear only in the test split, so no profile is fitted on
            # the attack it will later face.
            for condition, count in (("rested", 4), ("tired", 4)):
                for index in range(count):
                    recording = generate_session(persona_key, condition, "test", index, seed)
                    relative = Path("recordings") / "test" / persona_key / f"{condition}-{index:02d}.jsonl"
                    write_recording(root / relative, recording.header, recording.events)
                    sessions.append({
                        "file": str(relative),
                        "session_id": recording.session_id,
                        "split": "test",
                        "participant": persona_key,
                        "role": "mimic",
                        "condition": condition,
                        "style": persona.style,
                        "mimic_variant": persona.mimic_variant,
                        "mimic_target": persona.target,
                        "events": len(recording.events),
                        "duration_ms": recording.duration_ms,
                    })
            continue

        for split in ("train", "validation", "test"):
            for condition, count in _split_counts(split):
                for index in range(count):
                    recording = generate_session(persona_key, condition, split, index, seed)
                    relative = Path("recordings") / split / persona_key / f"{condition}-{index:02d}.jsonl"
                    write_recording(root / relative, recording.header, recording.events)
                    sessions.append({
                        "file": str(relative),
                        "session_id": recording.session_id,
                        "split": split,
                        "participant": persona_key,
                        "role": persona.role,
                        "condition": condition,
                        "style": persona.style,
                        "mimic_variant": None,
                        "mimic_target": None,
                        "events": len(recording.events),
                        "duration_ms": recording.duration_ms,
                    })

    manifest = {
        "format": "marrow-dataset-v1",
        "synthetic": True,
        "generator": "marrow/synth.py",
        "generator_seed": seed,
        "license": "CC0-1.0",
        "note": "Every session is generated by the code in this repository. No real movement, text, audio or camera data is present.",
        "styles": {
            name: {
                "modality": STYLES[name].modality,
                "description": {
                    "pointer_keyboard": "ordinary mouse plus physical keyboard",
                    "slow_pointer_dictation": "slow joystick/head pointer plus dictation software, including spoken corrections",
                    "switch_onscreen": "switch access plus on-screen keyboard with dwell clicks",
                }[name],
            }
            for name in STYLES
        },
        "participants": {
            key: {
                "role": persona.role,
                "style": persona.style,
                "target": persona.target,
                "mimic_variant": persona.mimic_variant,
            }
            for key, persona in PERSONAS.items()
        },
        "sessions": sessions,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(root: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
