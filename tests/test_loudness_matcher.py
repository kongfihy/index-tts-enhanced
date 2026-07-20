import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts_dubbing.audio.loudness_matcher import (
    active_rms_dbfs,
    match_waveform_to_reference_level,
    write_loudness_matched_copy,
)


def tone(amplitude: float, seconds: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    timeline = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * timeline)).astype(np.float32)


class LoudnessMatcherTests(unittest.TestCase):
    def test_quiet_output_moves_toward_reference_with_bounded_linear_gain(self):
        generated = tone(0.08)
        reference_level = active_rms_dbfs(tone(0.20), 16000)

        processed, stats = match_waveform_to_reference_level(
            generated,
            16000,
            reference_level,
        )

        self.assertAlmostEqual(stats.applied_gain_db, 6.0, places=5)
        self.assertTrue(stats.gain_limited)
        self.assertFalse(stats.peak_limited)
        self.assertAlmostEqual(
            stats.output_active_rms_dbfs - stats.input_active_rms_dbfs,
            6.0,
            places=4,
        )
        nonzero = np.abs(generated) > 1e-6
        np.testing.assert_allclose(
            processed[nonzero] / generated[nonzero],
            10 ** (6.0 / 20.0),
            rtol=1e-5,
        )

    def test_peak_headroom_prevents_clipping(self):
        generated = tone(0.88)
        reference_level = active_rms_dbfs(tone(0.95), 16000)

        processed, stats = match_waveform_to_reference_level(
            generated,
            16000,
            reference_level,
            target_peak_dbfs=-1.0,
        )

        self.assertTrue(stats.peak_limited)
        self.assertLessEqual(float(np.max(np.abs(processed))), 10 ** (-1.0 / 20.0) + 1e-6)
        self.assertAlmostEqual(stats.output_peak_dbfs, -1.0, places=4)

    def test_louder_output_is_attenuated_toward_reference(self):
        generated = tone(0.4)
        reference_level = active_rms_dbfs(tone(0.2), 16000)

        _processed, stats = match_waveform_to_reference_level(
            generated,
            16000,
            reference_level,
        )

        self.assertAlmostEqual(stats.applied_gain_db, -6.0206, places=3)
        self.assertAlmostEqual(stats.output_active_rms_dbfs, reference_level, places=3)

    def test_attenuation_is_bounded_for_an_extremely_quiet_reference(self):
        generated = tone(0.5)
        reference_level = active_rms_dbfs(tone(0.01), 16000)

        _processed, stats = match_waveform_to_reference_level(
            generated,
            16000,
            reference_level,
        )

        self.assertAlmostEqual(stats.applied_gain_db, -12.0, places=5)
        self.assertTrue(stats.gain_limited)

    def test_silent_generated_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "没有检测到可用的人声"):
            match_waveform_to_reference_level(
                np.zeros(16000, dtype=np.float32),
                16000,
                -20.0,
            )

    def test_write_copy_preserves_source_and_sample_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dry.wav"
            reference = root / "reference.wav"
            destination = root / "level-matched.wav"
            sf.write(source, tone(0.08), 16000, subtype="PCM_16")
            sf.write(reference, tone(0.20, seconds=3.0), 16000, subtype="PCM_16")
            source_bytes = source.read_bytes()

            output, stats = write_loudness_matched_copy(source, reference, destination)

            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(output, str(destination))
            self.assertEqual(sf.info(output).samplerate, 16000)
            self.assertGreater(stats.output_active_rms_dbfs, stats.input_active_rms_dbfs)
            self.assertIn("峰值", stats.summary())


if __name__ == "__main__":
    unittest.main()
