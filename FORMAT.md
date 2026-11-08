# Recording format (Marrow v1)

This is the whole format.  It is plain UTF-8 text, has no OS-specific fields,
and can be produced by a browser, a Mac, a Windows program, a Linux daemon, or a
synthetic generator.  Nothing in the detector branches on where a recording was
captured.

A recording is one session.  It is either:

* **JSON Lines** (preferred): the first line is a header object, every later
  line is one event object; or
* **one JSON object** with the header fields and an `events` array.

Blank lines are ignored in JSONL.

## Header

```json
{"format":"marrow-recording-v1","session_id":"alice-2026-11-08-01","synthetic":false,"clock":"ms","pointer_units":"screen_fraction","duration_ms":61234,"labels":{"participant":"alice","condition":"rested"}}
```

| field | required | meaning |
|---|---|---|
| `format` | yes | exactly `marrow-recording-v1` |
| `session_id` | yes | non-empty string; unique within your corpus |
| `synthetic` | recommended | `true` when no real person produced the events |
| `clock` | optional | `ms` (the only accepted value) |
| `pointer_units` | optional | `screen_fraction` (the only accepted value) |
| `duration_ms` | optional | integer session duration |
| `labels` | optional | free-form object for your own splits/evaluation; the detector ignores it |

`labels` is where a researcher puts `participant`, `condition`, `role`, etc.
It is never read by `marrow.features` or `marrow.model`.  Keeping labels out of
the event stream means a recording can be shared without revealing who made it.

## Events

All events have an integer `t`: milliseconds since the start of the session.
Timestamps must be non-decreasing.  Pointer positions are fractions of the
coordinate space, not pixels: `x = 0.0` is the left edge, `x = 1.0` the right
edge; `y = 0.0` is the top edge, `y = 1.0` the bottom edge.  Use the same
coordinate space for the whole session (full screen or the application window;
either is fine, as long as it is consistent).

### Pointer

```json
{"t":120,"type":"pointer","x":0.5123,"y":0.4410}
```

`x` and `y` are finite numbers in `[0, 1]`.  A pointer event is a sample of the
pointer position.  Sampling may be event-driven and irregular; the format does
not require a fixed rate.  A station-keeping sample (same position as the
previous one) is allowed.

### Click

```json
{"t":180,"type":"click","action":"down","button":"left","x":0.5123,"y":0.4410}
{"t":271,"type":"click","action":"up","button":"left","x":0.5123,"y":0.4410}
```

`action` is `down` or `up`.  `button` is `left`, `right` or `middle` (default
`left`).  `x`/`y` are the screen fractions at the click.  A dwell click (eye
gaze, head mouse, switch) is simply a long gap between `down` and `up`; no
special event type is needed.  A click without a matching `up` is ignored by
the feature extractor.

### Text

A text event is an insertion or a deletion.  `source` says how the text
arrived:

* `typed` — a physical keyboard;
* `onscreen` — an on-screen keyboard, switch scan, or other pointer-driven
  character entry;
* `software` — dictation, autocomplete, paste, or another program inserting
  text.  A dictated correction is a `delete` with `source: "software"`.

Insert:

```json
{"t":340,"type":"text","op":"insert","source":"typed","text":"t"}
{"t":5120,"type":"text","op":"insert","source":"software","text":"please check the last line"}
```

The detector only uses the length of `text`, never its content.  For a
content-free recording (recommended for real users), send `text_len` instead:

```json
{"t":5120,"type":"text","op":"insert","source":"software","text_len":26}
```

Exactly one of `text` and `text_len` must be present on an insert.

Delete:

```json
{"t":410,"type":"text","op":"delete","source":"typed","count":1}
{"t":8200,"type":"text","op":"delete","source":"software","count":4}
```

`count` is a positive integer (default `1`).  A single backspace is `count: 1`;
a dictated phrase correction may remove several characters at once.

## Complete example

```jsonl
{"format":"marrow-recording-v1","session_id":"demo-01","synthetic":true,"clock":"ms","pointer_units":"screen_fraction","duration_ms":4200,"labels":{"participant":"demo","condition":"rested"}}
{"t":0,"type":"pointer","x":0.50,"y":0.50}
{"t":80,"type":"text","op":"insert","source":"typed","text":"o"}
{"t":190,"type":"text","op":"insert","source":"typed","text":"k"}
{"t":260,"type":"pointer","x":0.53,"y":0.48}
{"t":330,"type":"text","op":"insert","source":"typed","text":" "}
{"t":420,"type":"text","op":"insert","source":"typed","text":"n"}
{"t":520,"type":"text","op":"delete","source":"typed","count":1}
{"t":610,"type":"text","op":"insert","source":"typed","text":"n"}
{"t":900,"type":"click","action":"down","button":"left","x":0.60,"y":0.40}
{"t":980,"type":"click","action":"up","button":"left","x":0.60,"y":0.40}
{"t":1200,"type":"text","op":"insert","source":"software","text":"open the note"}
{"t":2400,"type":"text","op":"delete","source":"software","count":4}
```

## Validation rules

`marrow.format.load_recording` rejects:

* a missing or wrong `format`;
* a missing `session_id`;
* a timestamp that is not an integer or goes backwards;
* a pointer/click coordinate outside `[0, 1]`;
* a click without `down`/`up`;
* a text event with an unknown `source` or `op`;
* an insert with both `text` and `text_len`, or neither;
* an empty/zero-length insert or a non-positive delete count.

The loader accepts a `.json` object with an `events` array, or `.jsonl`.
Directories are read recursively.

## Offline train/test

`tools/train_test.py` is the generic CLI.  It reads recordings in this format,
fits a profile, and scores new recordings.  It never opens a network socket.

```bash
# Enrol on one person's sessions, using other people as background.
python3 tools/train_test.py train \
  --owner recordings/alice/ \
  --background recordings/others/ \
  --validation-owner recordings/alice-val/ \
  --validation-background recordings/others-val/ \
  --out alice.profile.json

# Score any recordings in the format.
python3 tools/train_test.py test \
  --profile alice.profile.json \
  --input recordings/new-session.jsonl
```

The profile JSON contains only feature names, robust normalisation statistics,
weights, the threshold and the shape-gate statistics.  It contains no text and
no raw events.

## Privacy notes

* The detector never reads `text` content; `text_len` is enough.
* Do not include window titles, process names, device names, key codes, audio,
  camera frames, or file paths.
* A recording can reveal behaviour even without content.  Treat it as
  sensitive; the shipped `data/` corpus is synthetic and licensed CC0.
* This prototype is not a production authenticator.  It is a testable method
  for comparing a session with an enrolled profile.
