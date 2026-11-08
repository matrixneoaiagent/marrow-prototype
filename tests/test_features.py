import unittest

from marrow.features import FEATURE_NAMES, extract_features
from marrow.format import Recording
from marrow.synth import generate_session


class FeatureTests(unittest.TestCase):
    def test_every_style_produces_evidence(self):
        for persona, style in (("alice", "pointer_keyboard"), ("ravi", "slow_pointer_dictation"), ("nia", "switch_onscreen")):
            recording = generate_session(persona, "rested", "test", 0)
            features = extract_features(recording)
            self.assertEqual(len(features), len(FEATURE_NAMES))
            self.assertGreater(features["s_event_rate"], 0)
            self.assertEqual(features["f_has_pointer"], 1.0)
            self.assertEqual(features["f_has_text"], 1.0)
            self.assertEqual(recording.labels["style"], style)

    def test_text_content_does_not_change_features(self):
        base = generate_session("alice", "rested", "test", 0)
        events = []
        for event in base.events:
            if event["type"] == "text" and event["op"] == "insert":
                event = dict(event)
                event["text"] = "z" * len(event["text"])
            events.append(event)
        altered = Recording(header=base.header, events=tuple(events))
        first = extract_features(base)
        second = extract_features(altered)
        for name in FEATURE_NAMES:
            if name.startswith("f_"):
                continue
            self.assertEqual(first[name], second[name], name)

    def test_pointer_only_session(self):
        header = {
            "format": "marrow-recording-v1",
            "session_id": "pointer-only",
            "synthetic": True,
        }
        events = tuple(
            {"t": i * 50, "type": "pointer", "x": 0.1 + i * 0.01, "y": 0.2 + i * 0.005}
            for i in range(40)
        )
        features = extract_features(Recording(header=header, events=events))
        self.assertEqual(features["f_has_pointer"], 1.0)
        self.assertEqual(features["f_has_text"], 0.0)
        self.assertIsNone(features["t_insert_rate"])

    def test_text_only_session(self):
        header = {
            "format": "marrow-recording-v1",
            "session_id": "text-only",
            "synthetic": True,
        }
        events = tuple(
            {"t": i * 300, "type": "text", "op": "insert", "source": "onscreen", "text": "x"}
            for i in range(30)
        )
        features = extract_features(Recording(header=header, events=events))
        self.assertEqual(features["f_has_pointer"], 0.0)
        self.assertEqual(features["f_has_text"], 1.0)
        self.assertEqual(features["f_has_onscreen"], 1.0)
        self.assertIsNone(features["p_speed_med_log"])

    def test_generator_is_deterministic(self):
        first = generate_session("ravi", "tired", "test", 3)
        second = generate_session("ravi", "tired", "test", 3)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.header, second.header)


if __name__ == "__main__":
    unittest.main()
