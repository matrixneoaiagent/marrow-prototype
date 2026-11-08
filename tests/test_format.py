import json
from pathlib import Path
import tempfile
import unittest

from marrow.format import RecordingError, load_recording, write_recording


HEADER = {
    "format": "marrow-recording-v1",
    "session_id": "test-session",
    "synthetic": True,
    "clock": "ms",
    "pointer_units": "screen_fraction",
    "labels": {"participant": "tester"},
}


class FormatTests(unittest.TestCase):
    def test_round_trip_jsonl(self):
        events = [
            {"t": 0, "type": "pointer", "x": 0.5, "y": 0.5},
            {"t": 100, "type": "text", "op": "insert", "source": "typed", "text": "a"},
            {"t": 220, "type": "text", "op": "delete", "source": "typed", "count": 1},
            {"t": 300, "type": "click", "action": "down", "button": "left", "x": 0.6, "y": 0.4},
            {"t": 390, "type": "click", "action": "up", "button": "left", "x": 0.6, "y": 0.4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            write_recording(path, HEADER, events)
            loaded = load_recording(path)
        self.assertEqual(loaded.session_id, "test-session")
        self.assertEqual(len(loaded.events), len(events))
        self.assertTrue(loaded.header["synthetic"])
        self.assertEqual(loaded.labels["participant"], "tester")

    def test_text_len_is_equivalent_to_text(self):
        events = [
            {"t": 0, "type": "text", "op": "insert", "source": "software", "text_len": 7},
            {"t": 900, "type": "text", "op": "delete", "source": "software", "count": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            write_recording(path, HEADER, events)
            loaded = load_recording(path)
        self.assertEqual(loaded.events[0]["text_len"], 7)
        self.assertEqual(loaded.events[1]["count"], 2)

    def test_json_object_form(self):
        obj = dict(HEADER)
        obj["events"] = [{"t": 0, "type": "pointer", "x": 0.1, "y": 0.2}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text(json.dumps(obj), encoding="utf-8")
            loaded = load_recording(path)
        self.assertEqual(len(loaded.events), 1)

    def test_bad_coordinate_is_rejected(self):
        events = [{"t": 0, "type": "pointer", "x": 1.2, "y": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            with self.assertRaises(RecordingError):
                write_recording(path, HEADER, events)

    def test_backwards_time_is_rejected(self):
        events = [
            {"t": 100, "type": "pointer", "x": 0.2, "y": 0.2},
            {"t": 50, "type": "pointer", "x": 0.3, "y": 0.3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            with self.assertRaises(RecordingError):
                write_recording(path, HEADER, events)

    def test_unknown_text_source_is_rejected(self):
        events = [{"t": 0, "type": "text", "op": "insert", "source": "telepathy", "text": "x"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            with self.assertRaises(RecordingError):
                write_recording(path, HEADER, events)


if __name__ == "__main__":
    unittest.main()
