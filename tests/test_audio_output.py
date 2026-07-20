import math
import unittest

import numpy as np

from indextts.utils.audio_output import finalize_waveform


class AudioOutputTests(unittest.TestCase):
    def test_over_range_waveform_is_reduced_without_hard_clipping(self):
        waveform = np.array([[-1.2, -0.6, 0.0, 0.6, 1.2]], dtype=np.float32)

        pcm, stats = finalize_waveform(waveform, target_peak_dbfs=-1.0)

        target_peak = 10 ** (-1.0 / 20.0)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(pcm.shape, waveform.shape)
        self.assertAlmostEqual(stats.output_peak, target_peak, places=6)
        self.assertLess(stats.gain, 1.0)
        self.assertEqual(stats.samples_above_full_scale, 2)
        # Linear gain preserves the 2:1 relationship instead of flattening both
        # positive samples to the same clipped value.
        self.assertAlmostEqual(int(pcm[0, 4]) / int(pcm[0, 3]), 2.0, places=3)
        self.assertLess(np.max(np.abs(pcm.astype(np.int32))), 32767)

    def test_quiet_waveform_is_not_amplified(self):
        waveform = np.array([[0.25, -0.25]], dtype=np.float32)

        pcm, stats = finalize_waveform(waveform)

        self.assertEqual(stats.gain, 1.0)
        np.testing.assert_array_equal(pcm, np.array([[8192, -8192]], dtype=np.int16))


    def test_quiet_waveform_can_use_safe_peak_headroom(self):
        waveform = np.array([[0.25, -0.25]], dtype=np.float32)

        pcm, stats = finalize_waveform(
            waveform,
            target_peak_dbfs=-1.0,
            normalize_peak=True,
            max_amplification_db=6.0,
        )

        self.assertTrue(stats.peak_amplified)
        self.assertAlmostEqual(stats.gain_db, 6.0, places=5)
        self.assertAlmostEqual(stats.output_peak, 0.25 * (10 ** (6.0 / 20.0)), places=6)
        self.assertGreater(stats.output_rms_dbfs, stats.input_rms_dbfs)
        self.assertLess(np.max(np.abs(pcm.astype(np.int32))), 32767)

    def test_peak_normalization_reaches_target_when_within_gain_cap(self):
        waveform = np.array([[0.7, -0.7]], dtype=np.float32)

        _pcm, stats = finalize_waveform(waveform, normalize_peak=True)

        self.assertAlmostEqual(stats.output_peak_dbfs, -1.0, places=5)
        self.assertGreater(stats.gain, 1.0)

    def test_non_finite_samples_are_replaced_with_silence(self):
        waveform = np.array([[np.nan, np.inf, -np.inf, 0.5]], dtype=np.float32)

        pcm, stats = finalize_waveform(waveform)

        self.assertEqual(stats.non_finite_samples, 3)
        np.testing.assert_array_equal(pcm[0, :3], np.zeros(3, dtype=np.int16))
        self.assertGreater(pcm[0, 3], 0)

    def test_silence_has_negative_infinite_peak_dbfs(self):
        pcm, stats = finalize_waveform(np.zeros((1, 8), dtype=np.float32))

        self.assertTrue(math.isinf(stats.input_peak_dbfs))
        self.assertLess(stats.input_peak_dbfs, 0)
        self.assertTrue(np.all(pcm == 0))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            finalize_waveform(np.array([], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "at or below 0"):
            finalize_waveform(np.zeros((1, 2), dtype=np.float32), target_peak_dbfs=1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            finalize_waveform(np.zeros((1, 2), dtype=np.float32), target_peak_dbfs=np.nan)


if __name__ == "__main__":
    unittest.main()
