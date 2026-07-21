import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts_webui_helpers import (
    generation_readiness,
    generation_request_key,
    matched_output_details,
    normalize_advanced_generation_args,
    normalize_choice_index,
    normalize_generation_text,
    reference_audio_quality_message,
    validate_audio_file,
    validate_generation_inputs,
)


def write_reference_wav(path: Path, duration: float = 3.0, sample_rate: int = 16000) -> None:
    samples = np.arange(int(duration * sample_rate), dtype=np.float32)
    audio = 0.2 * np.sin(2.0 * np.pi * 220.0 * samples / sample_rate)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


class WebUIHelperTests(unittest.TestCase):
    def test_normalize_generation_text_strips_outer_whitespace(self):
        self.assertEqual(normalize_generation_text("  测试文本\n"), "测试文本")
        self.assertEqual(normalize_generation_text(None), "")

    def test_readiness_reports_missing_inputs(self):
        self.assertEqual(
            generation_readiness(None, ""),
            (False, "请先上传参考音频，并填写需要生成的文本。"),
        )
        self.assertFalse(generation_readiness(None, "已有文本")[0])
        self.assertFalse(generation_readiness("/tmp/reference.wav", "")[0])

    def test_readiness_accepts_audio_and_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "reference.wav"
            write_reference_wav(audio_path)
            ready, message = generation_readiness(audio_path, "测试文本")
            self.assertTrue(ready)
            self.assertIn("准备就绪", message)

    def test_reference_quality_message_contains_basic_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "reference.wav"
            write_reference_wav(audio_path)
            message = reference_audio_quality_message(audio_path)
            self.assertIn("3.0 秒", message)
            self.assertIn("16.0 kHz", message)
            self.assertIn("峰值", message)

    def test_request_key_tracks_the_full_snapshot(self):
        snapshot = {"text": "测试", "advanced_args": [True, 0.8]}
        same_snapshot = {"advanced_args": [True, 0.8], "text": "测试"}
        changed_snapshot = {"text": "测试", "advanced_args": [True, 0.7]}
        self.assertEqual(
            generation_request_key(snapshot),
            generation_request_key(same_snapshot),
        )
        self.assertNotEqual(
            generation_request_key(snapshot),
            generation_request_key(changed_snapshot),
        )

    def test_normalize_choice_index_accepts_index_and_label(self):
        choices = ["默认", "参考音频", "情感向量"]
        self.assertEqual(normalize_choice_index(1, choices), 1)
        self.assertEqual(normalize_choice_index("情感向量", choices), 2)
        with self.assertRaisesRegex(ValueError, "重新选择"):
            normalize_choice_index("未知选项", choices)
        with self.assertRaisesRegex(ValueError, "失效"):
            normalize_choice_index(9, choices)

    def test_validation_requires_existing_audio_file(self):
        with self.assertRaisesRegex(ValueError, "重新上传"):
            validate_generation_inputs("/tmp/missing-reference.wav", "测试文本")

    def test_audio_validation_accepts_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "emotion.wav"
            audio_path.touch()
            self.assertEqual(
                validate_audio_file(audio_path, "情感参考音频"),
                str(audio_path),
            )

    def test_matched_output_details_distinguish_delivery_and_level_only_copies(self):
        source = Path("/tmp/candidate-01-seed-7.wav")
        delivery_path, delivery_label, delivery_variant = matched_output_details(
            source, 1, True
        )
        self.assertEqual(delivery_path.name, "candidate-01-seed-7-delivery-matched.wav")
        self.assertEqual(delivery_label, "候选 1 · 匹配交付版")
        self.assertEqual(delivery_variant, "level_and_format_matched")

        level_path, level_label, level_variant = matched_output_details(source, 1, False)
        self.assertEqual(level_path.name, "candidate-01-seed-7-level-matched.wav")
        self.assertEqual(level_label, "候选 1 · 安全响度匹配")
        self.assertEqual(level_variant, "level_matched")

    def test_advanced_args_are_normalized_and_old_snapshots_are_supported(self):
        self.assertEqual(
            normalize_advanced_generation_args(
                [True, 0.8, 30.0, 0.8, 0.0, 3.0, 10.0, 1500.0, True, True]
            ),
            [True, 0.8, 30, 0.8, 0.0, 3, 10.0, 1500, True, True],
        )
        self.assertEqual(
            normalize_advanced_generation_args(
                [True, 0.8, 30.0, 0.8, 0.0, 3.0, 10.0, 1500.0, True]
            )[-2:],
            [True, False],
        )
        self.assertEqual(
            normalize_advanced_generation_args(
                [True, 0.8, 30.0, 0.8, 0.0, 3.0, 10.0, 1500.0]
            )[-2:],
            [False, False],
        )
        with self.assertRaisesRegex(ValueError, "不完整"):
            normalize_advanced_generation_args([True])
        with self.assertRaisesRegex(ValueError, "0 到 1"):
            normalize_advanced_generation_args(
                [True, 1.2, 30, 0.8, 0.0, 3, 10.0, 1500, True, True]
            )

    def test_validation_returns_clean_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "reference.wav"
            write_reference_wav(audio_path)
            self.assertEqual(
                validate_generation_inputs(audio_path, "  测试文本  "),
                "测试文本",
            )


if __name__ == "__main__":
    unittest.main()
