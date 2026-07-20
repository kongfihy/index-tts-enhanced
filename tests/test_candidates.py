import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from indextts_dubbing.candidates import (
    build_candidate_plan,
    candidate_output_paths,
    read_candidate_manifest,
    write_candidate_manifest,
)


class CandidateHelpersTests(unittest.TestCase):
    def test_candidate_plan_uses_consecutive_reproducible_seeds(self):
        plan = build_candidate_plan(20, 3)
        self.assertEqual(plan.base_seed, 20)
        self.assertEqual(plan.seeds, (20, 21, 22))

    def test_auto_seed_is_resolved_once_before_building_sequence(self):
        with patch("indextts.utils.randomness.secrets.randbelow", return_value=99):
            plan = build_candidate_plan(-1, 2)
        self.assertEqual(plan.seeds, (99, 100))

    def test_manifest_round_trip_ignores_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = candidate_output_paths(root, "job-1", [7, 8])
            paths[0].parent.mkdir(parents=True)
            paths[0].touch()
            write_candidate_manifest(root, "job-1", paths, [7, 8])

            candidates = read_candidate_manifest(root, "job-1")

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["seed"], 7)
            self.assertEqual(candidates[0]["path"], str(paths[0]))

    def test_manifest_preserves_result_labels_and_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = candidate_output_paths(root, "job-2", [9])[0]
            matched = path.with_name(path.stem + "-level-matched.wav")
            path.parent.mkdir(parents=True)
            path.touch()
            matched.touch()

            write_candidate_manifest(
                root,
                "job-2",
                [path, matched],
                [9, 9],
                labels=["候选 1 · 原始干声", "候选 1 · 安全响度匹配"],
                variants=["dry", "level_matched"],
            )

            candidates = read_candidate_manifest(root, "job-2")
            self.assertEqual([item["variant"] for item in candidates], ["dry", "level_matched"])
            self.assertEqual(candidates[1]["label"], "候选 1 · 安全响度匹配")


if __name__ == "__main__":
    unittest.main()
