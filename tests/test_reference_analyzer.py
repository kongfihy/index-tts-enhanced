import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts_dubbing.audio.reference_analyzer import (
    MAX_REFERENCE_SECONDS,
    analyze_reference_audio,
    prepare_reference_audio,
)


def tone_with_silence(
    sample_rate: int = 16000,
    leading: float = 0.5,
    speech: float = 3.0,
    trailing: float = 0.6,
    amplitude: float = 0.2,
) -> np.ndarray:
    speech_samples = np.arange(int(speech * sample_rate), dtype=np.float32)
    tone = amplitude * np.sin(2.0 * np.pi * 220.0 * speech_samples / sample_rate)
    return np.concatenate(
        [
            np.zeros(int(leading * sample_rate), dtype=np.float32),
            tone,
            np.zeros(int(trailing * sample_rate), dtype=np.float32),
        ]
    )


class ReferenceAnalyzerTests(unittest.TestCase):
    def test_analysis_reports_duration_level_and_edge_silence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.wav"
            sf.write(path, tone_with_silence(), 16000, subtype="PCM_16")

            analysis = analyze_reference_audio(path)

            self.assertTrue(analysis.usable)
            self.assertAlmostEqual(analysis.duration_seconds, 4.1, places=1)
            self.assertGreater(analysis.leading_silence_seconds, 0.4)
            self.assertGreater(analysis.trailing_silence_seconds, 0.5)
            self.assertLess(analysis.peak_dbfs, -10.0)
            self.assertIn("峰值", analysis.summary())

    def test_long_reference_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "long.wav"
            audio = tone_with_silence(leading=0.0, speech=MAX_REFERENCE_SECONDS + 1.0, trailing=0.0)
            sf.write(path, audio, 16000, subtype="PCM_16")

            analysis = analyze_reference_audio(path)

            self.assertFalse(analysis.usable)
            self.assertTrue(any("超过 15 秒" in item for item in analysis.errors))

    def test_prepare_trims_edges_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            output_dir = root / "prepared"
            sf.write(source, tone_with_silence(), 16000, subtype="PCM_16")
            original_bytes = source.read_bytes()

            prepared = prepare_reference_audio(source, output_dir)
            output_analysis = analyze_reference_audio(prepared.output_path)

            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertTrue(Path(prepared.output_path).is_file())
            self.assertLess(prepared.output_duration_seconds, 3.4)
            self.assertGreater(prepared.removed_leading_seconds, 0.3)
            self.assertGreater(prepared.removed_trailing_seconds, 0.4)
            self.assertTrue(output_analysis.usable)

    def test_silent_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "silent.wav"
            sf.write(path, np.zeros(3 * 16000, dtype=np.float32), 16000, subtype="PCM_16")

            analysis = analyze_reference_audio(path)

            self.assertFalse(analysis.usable)
            self.assertTrue(any("没有检测到清晰人声" in item for item in analysis.errors))


if __name__ == "__main__":
    unittest.main()
