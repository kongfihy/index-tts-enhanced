import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from indextts_task_center import (
    ADMIN_HTML,
    JobAlreadyRunning,
    JobCancelled,
    JobManager,
    JobProgress,
    is_management_client,
    task_center_host_for_webui,
)


class JobManagerQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "jobs.sqlite3"
        self.output_root = self.root / "outputs" / "tasks"

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_manager(self, dedupe_window_seconds=3):
        return JobManager(
            db_path=self.db_path,
            dedupe_window_seconds=dedupe_window_seconds,
            output_root=self.output_root,
        )

    @staticmethod
    def create_job(manager, text="测试文本", prompt="reference.m4a"):
        return manager.create_job(
            project_name="默认项目",
            job_type="tts",
            input_summary=text,
            parameters={"prompt_audio": prompt},
        )

    def test_duplicate_submit_reuses_active_job(self):
        manager = self.make_manager()
        first_job_id = self.create_job(manager)
        should_run, existing_output = manager.start_job(first_job_id)

        duplicate_job_id = self.create_job(manager)

        self.assertTrue(should_run)
        self.assertIsNone(existing_output)
        self.assertEqual(duplicate_job_id, first_job_id)
        self.assertEqual(manager.status()["active_jobs"], 1)
        self.assertEqual(manager.status()["queued_jobs"], 0)


    def test_different_seeds_are_not_deduplicated(self):
        manager = self.make_manager()
        first_job_id = manager.create_job(
            project_name="默认项目",
            job_type="tts",
            input_summary="测试文本",
            parameters={"prompt_audio": "reference.wav"},
            seed=10,
        )
        second_job_id = manager.create_job(
            project_name="默认项目",
            job_type="tts",
            input_summary="测试文本",
            parameters={"prompt_audio": "reference.wav"},
            seed=11,
        )

        self.assertNotEqual(first_job_id, second_job_id)

    def test_duplicate_generation_cannot_claim_same_job_twice(self):
        manager = self.make_manager()
        job_id = self.create_job(manager)
        manager.start_job(job_id)

        with self.assertRaises(JobAlreadyRunning):
            manager.start_job(job_id)

    def test_completed_duplicate_returns_existing_output(self):
        manager = self.make_manager()
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        manager.complete_job(job_id, "outputs/tasks/example.wav")

        should_run, existing_output = manager.start_job(job_id)

        self.assertFalse(should_run)
        self.assertEqual(existing_output, "outputs/tasks/example.wav")

    def test_queued_job_is_exposed_as_current_job(self):
        manager = self.make_manager()
        job_id = self.create_job(manager)

        status = manager.status()

        self.assertEqual(status["queued_jobs"], 1)
        self.assertEqual(status["active_jobs"], 0)
        self.assertEqual(status["current_job"]["job_id"], job_id)
        self.assertEqual(status["current_job"]["status"], "queued")


    def test_completed_job_cannot_be_cancelled(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        manager.complete_job(job_id, "outputs/tasks/example.wav")

        self.assertFalse(manager.request_cancel(job_id))
        job = manager.get_job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["message"], "生成完成")
        self.assertFalse(job["cancel_requested"])

    def test_cancelled_running_job_cannot_be_marked_completed(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)

        self.assertTrue(manager.request_cancel(job_id))
        with self.assertRaises(JobCancelled):
            manager.complete_job(job_id, "outputs/tasks/should-not-survive.wav")

        job = manager.get_job(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["message"], "任务已取消")
        self.assertTrue(job["cancel_requested"])
        self.assertIsNone(job["output_path"])

    def test_queued_cancel_is_terminal_and_idempotent(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)

        self.assertTrue(manager.request_cancel(job_id))
        self.assertFalse(manager.request_cancel(job_id))
        job = manager.get_job(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["message"], "排队任务已取消")


    def test_late_failure_does_not_overwrite_completed_job(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        manager.complete_job(job_id, "outputs/tasks/example.wav")

        manager.fail_job(job_id, RuntimeError("late callback"))

        job = manager.get_job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["output_path"], "outputs/tasks/example.wav")
        self.assertIsNone(job["error"])

    def test_invalid_list_limit_falls_back_safely(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        self.create_job(manager)

        jobs = manager.list_jobs(limit="not-a-number")

        self.assertEqual(len(jobs), 1)


    def test_seed_is_saved_with_job(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = manager.create_job(
            project_name="默认项目",
            job_type="tts",
            input_summary="测试文本",
            parameters={"candidate_count": 2},
            seed=123,
        )

        job = manager.get_job(job_id)

        self.assertEqual(job["seed"], "123")
        self.assertEqual(job["parameters"]["candidate_count"], 2)

    def test_candidate_progress_is_scaled_without_moving_backwards(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        progress = JobProgress(
            manager,
            job_id,
            progress_start=0.5,
            progress_span=0.5,
            description_prefix="候选 2/2 · ",
        )

        progress(0.4, desc="semantic generation")
        job = manager.get_job(job_id)

        self.assertAlmostEqual(job["progress"], 0.7)
        self.assertEqual(job["stage"], "semantic_generation")
        self.assertEqual(job["message"], "候选 2/2 · semantic generation")

    def test_restart_recovers_active_and_queued_jobs(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        queued_job_id = self.create_job(manager, text="排队任务")
        active_job_id = self.create_job(manager, text="生成任务")
        manager.start_job(active_job_id)

        recovered_manager = self.make_manager(dedupe_window_seconds=0)

        queued_job = recovered_manager.get_job(queued_job_id)
        active_job = recovered_manager.get_job(active_job_id)
        self.assertEqual(queued_job["status"], "cancelled")
        self.assertEqual(queued_job["message"], "服务重启，排队任务已取消")
        self.assertEqual(active_job["status"], "failed")
        self.assertEqual(active_job["message"], "服务重启后未完成")

    def test_multiple_outputs_are_saved_without_overwriting_previous_version(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        first_job = self.create_job(manager, text="第一版")
        manager.start_job(first_job)
        first_path = self.output_root / first_job / "candidate-01.wav"
        first_path.parent.mkdir(parents=True)
        first_path.write_bytes(b"first")
        manager.complete_job(first_job, [first_path])

        second_job = self.create_job(manager, text="第二版")
        manager.start_job(second_job)
        second_paths = [
            self.output_root / second_job / "candidate-01.wav",
            self.output_root / second_job / "candidate-02.wav",
        ]
        for index, path in enumerate(second_paths, start=1):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"second-{index}".encode())
        manager.complete_job(second_job, second_paths)

        jobs = manager.list_jobs(project_id=manager.get_job(first_job)["project_id"])
        self.assertEqual([job["project_version"] for job in jobs], [2, 1])
        self.assertEqual(manager.get_job(first_job)["output_paths"], [str(first_path)])
        self.assertEqual(manager.get_job(second_job)["output_paths"], [str(path) for path in second_paths])
        self.assertTrue(first_path.exists())

    def test_old_candidate_manifest_restores_all_downloads(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        paths = [self.output_root / job_id / f"candidate-{index:02d}.wav" for index in (1, 2)]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")
        (self.output_root / job_id / "candidates.json").write_text(json.dumps({
            "candidates": [{"path": str(path)} for path in paths],
        }))
        manager.complete_job(job_id, str(paths[0]))
        with manager._connect() as conn:
            conn.execute("UPDATE jobs SET outputs_json='[]' WHERE job_id=?", (job_id,))

        public = manager.public_job(manager.get_job(job_id))

        self.assertEqual(public["output_count"], 2)
        self.assertTrue(all(item["available"] for item in public["outputs"]))
        self.assertNotIn(str(self.root), json.dumps(public, ensure_ascii=False))

    def test_completed_output_can_be_marked_as_preferred_and_keeps_manifest_label(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        paths = [self.output_root / job_id / f"candidate-{index:02d}.wav" for index in (1, 2)]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")
        (self.output_root / job_id / "candidates.json").write_text(json.dumps({
            "candidates": [
                {"path": str(paths[0]), "label": "候选 1 · 原始干声", "seed": 20, "variant": "dry"},
                {"path": str(paths[1]), "label": "候选 2 · 原始干声", "seed": 21, "variant": "dry"},
            ],
        }))
        manager.complete_job(job_id, paths)

        self.assertTrue(manager.select_output(job_id, 2))
        job = manager.get_job(job_id)
        public = manager.public_job(job)

        self.assertEqual(job["selected_output_index"], 2)
        self.assertFalse(public["outputs"][0]["selected"])
        self.assertTrue(public["outputs"][1]["selected"])
        self.assertEqual(public["outputs"][1]["label"], "候选 2 · 原始干声")
        self.assertEqual(public["outputs"][1]["seed"], 21)
        self.assertEqual(public["selected_output"]["index"], 2)

    def test_preferred_output_must_exist_and_be_available(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        path = self.output_root / job_id / "candidate-01.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"wav")
        manager.complete_job(job_id, [path])

        with self.assertRaisesRegex(ValueError, "生成结果不存在"):
            manager.select_output(job_id, 2)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "生成结果文件不可用"):
            manager.select_output(job_id, 1)

    def test_output_download_rejects_paths_outside_task_root(self):
        manager = self.make_manager(dedupe_window_seconds=0)
        outside = self.root / "private.wav"
        outside.write_bytes(b"secret")
        job_id = self.create_job(manager)
        manager.start_job(job_id)
        manager.complete_job(job_id, [outside])

        public = manager.public_job(manager.get_job(job_id))

        self.assertFalse(public["outputs"][0]["available"])
        self.assertIsNone(manager.download_path(job_id, 1))

    def test_existing_database_is_migrated_with_output_columns(self):
        legacy_db = self.root / "legacy.sqlite3"
        with sqlite3.connect(legacy_db) as conn:
            conn.execute("""CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, project_name TEXT NOT NULL,
                job_type TEXT NOT NULL, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                stage TEXT NOT NULL DEFAULT '', message TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, input_summary TEXT NOT NULL DEFAULT '', output_path TEXT,
                error TEXT, seed TEXT, parameters_json TEXT NOT NULL DEFAULT '{}',
                cancel_requested INTEGER NOT NULL DEFAULT 0)""")
        manager = JobManager(legacy_db, dedupe_window_seconds=0, output_root=self.output_root)
        with manager._connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        self.assertIn("outputs_json", columns)
        self.assertIn("selected_output_index", columns)

    def test_lan_access_is_read_only_but_loopback_can_manage(self):
        self.assertTrue(is_management_client("127.0.0.1", "127.0.0.1"))
        self.assertTrue(is_management_client("192.0.2.10", "192.0.2.10"))
        self.assertFalse(is_management_client("192.0.2.20", "192.0.2.10"))

    def test_task_center_host_follows_webui_exposure(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INDEXTTS_TASK_CENTER_HOST", None)
            self.assertEqual(task_center_host_for_webui("0.0.0.0"), "0.0.0.0")
            self.assertEqual(task_center_host_for_webui("127.0.0.1"), "127.0.0.1")
        with patch.dict(os.environ, {"INDEXTTS_TASK_CENTER_HOST": "192.0.2.10"}):
            self.assertEqual(task_center_host_for_webui("127.0.0.1"), "192.0.2.10")

    def test_task_center_page_contains_grouped_versions_and_downloads(self):
        self.assertIn("projectGroup", ADMIN_HTML)
        self.assertIn("版本 ${x.project_version}", ADMIN_HTML)
        self.assertIn("设为最佳", ADMIN_HTML)
        self.assertIn("selectOutput", ADMIN_HTML)
        self.assertIn("${resultCount} 个结果", ADMIN_HTML)
        self.assertIn("打包下载全部", ADMIN_HTML)
        self.assertIn("局域网只读", ADMIN_HTML)


if __name__ == "__main__":
    unittest.main()
