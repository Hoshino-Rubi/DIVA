"""Directed period adapter checks; no files, models or external audio."""
from array import array
import math
import unittest
from unittest.mock import patch

from any2xsampa.local.render_period import analyze_period


class RenderPeriodTests(unittest.TestCase):
    def test_counts_and_rate_cannot_be_implicitly_reinterpreted(self):
        for pcm, rate in (([0.1] * 1000, 16000), (array("f", [0.1]), 16000),
                          (array("h", [1]), True), (array("h", [1]), 3200)):
            with self.assertRaises(ValueError):
                analyze_period(pcm, rate)

    def test_silence_and_short_window_do_not_load_optional_runtime(self):
        with patch("any2xsampa.local.render_period._dependencies", side_effect=AssertionError):
            silence = analyze_period(array("h", [0]) * 1600, 16000)
            short = analyze_period(array("h", [100]) * 799, 16000)
        self.assertEqual(silence["f0_status"], "unknown_digital_silence")
        self.assertEqual(short["f0_status"], "unknown_no_complete_frames")
        self.assertEqual(short["unframed_tail_samples"], 799)

    def test_missing_dependency_is_explicitly_unknown(self):
        with patch("any2xsampa.local.render_period._dependencies", side_effect=ImportError("fixture")):
            result = analyze_period(array("h", [1000]) * 1600, 16000)
        self.assertEqual(result["f0_status"], "unavailable")
        self.assertIn("fixture", result["f0_error"])
        self.assertIsNone(result["pitch_truth"])

    def test_long_window_keeps_a_bound_without_loading_numerical_runtime(self):
        with patch("any2xsampa.local.render_period._dependencies", side_effect=AssertionError):
            result = analyze_period(array("h", [100]) * 320001, 16000)
        self.assertEqual(result["f0_status"], "unknown_window_exceeds_20_seconds")
        self.assertIsNone(result["f0_quantiles_hz"])

    def test_known_220hz_period_in_existing_runtime(self):
        rate = 16000
        pcm = array("h", (round(6000 * math.sin(2 * math.pi * 220 * i / rate))
                          for i in range(3201)))
        before = pcm.tobytes()
        result = analyze_period(pcm, rate)
        self.assertEqual(result["f0_status"], "period_estimate_unqualified", result)
        self.assertAlmostEqual(result["f0_quantiles_hz"]["p50"], 220, delta=2.2)
        self.assertEqual(result["unframed_tail_samples"], 1)
        self.assertEqual(pcm.tobytes(), before)
        self.assertIsNone(result["pronunciation_correct"])
        self.assertIsNone(result["voicing_truth"])


if __name__ == "__main__":
    unittest.main()
