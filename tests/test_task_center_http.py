import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

from indextts_task_center import JobManager, TaskCenterHandler


class TaskCenterHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.output_root = root / "outputs" / "tasks"
        self.manager = JobManager(
            db_path=root / "jobs.sqlite3",
            dedupe_window_seconds=0,
            output_root=self.output_root,
        )
        handler = type("IsolatedTaskCenterHandler", (TaskCenterHandler,), {})
        handler.manager = self.manager
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def create_completed_job(self):
        job_id = self.manager.create_job(
            project_name="测试项目",
            job_type="tts",
            input_summary="这是一个公开任务摘要",
            parameters={"candidate_count": 2},
            seed=10,
        )
        self.manager.start_job(job_id)
        job_dir = self.output_root / job_id
        job_dir.mkdir(parents=True)
        paths = [job_dir / "candidate-01.wav", job_dir / "candidate-02.wav"]
        paths[0].write_bytes(b"RIFF-first")
        paths[1].write_bytes(b"RIFF-second")
        (job_dir / "candidates.json").write_text(json.dumps({
            "candidates": [
                {"path": str(paths[0]), "label": "候选 1 · 原始干声", "seed": 10},
                {"path": str(paths[1]), "label": "候选 2 · 原始干声", "seed": 11},
            ],
        }))
        self.manager.complete_job(job_id, paths)
        return job_id, paths

    def get_json(self, path):
        with urllib.request.urlopen(self.base_url + path, timeout=3) as response:
            return response.status, json.loads(response.read())

    def test_public_jobs_api_hides_absolute_paths_and_exposes_downloads(self):
        job_id, _ = self.create_completed_job()

        status, payload = self.get_json("/api/jobs")

        self.assertEqual(status, 200)
        self.assertEqual(len(payload["jobs"]), 1)
        job = payload["jobs"][0]
        self.assertEqual(job["job_id"], job_id)
        self.assertEqual(job["project_version"], 1)
        self.assertEqual(job["output_count"], 2)
        self.assertEqual(len(job["outputs"]), 2)
        self.assertTrue(all(output["available"] for output in job["outputs"]))
        self.assertEqual([output["label"] for output in job["outputs"]], [
            "候选 1 · 原始干声",
            "候选 2 · 原始干声",
        ])
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(str(self.output_root), serialized)
        self.assertNotIn("output_path", serialized)
        self.assertNotIn("outputs_json", serialized)

    def test_single_output_download_returns_registered_file(self):
        job_id, paths = self.create_completed_job()

        with urllib.request.urlopen(
            f"{self.base_url}/api/jobs/{job_id}/outputs/2/download",
            timeout=3,
        ) as response:
            body = response.read()
            disposition = response.headers.get("Content-Disposition")

        self.assertEqual(body, paths[1].read_bytes())
        self.assertIn("candidate-02.wav", disposition)
        self.assertEqual(response.headers.get_content_type(), "audio/x-wav")

    def test_loopback_client_can_mark_and_change_preferred_output(self):
        job_id, _ = self.create_completed_job()

        for index in (2, 1):
            request = urllib.request.Request(
                f"{self.base_url}/api/jobs/{job_id}/outputs/{index}/select",
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["selected_output_index"], index)

        _, job = self.get_json(f"/api/jobs/{job_id}")
        self.assertEqual(job["selected_output_index"], 1)
        self.assertTrue(job["outputs"][0]["selected"])
        self.assertFalse(job["outputs"][1]["selected"])
        self.assertEqual(job["selected_output"]["index"], 1)

    def test_download_all_returns_zip_with_every_candidate(self):
        job_id, paths = self.create_completed_job()

        with urllib.request.urlopen(
            f"{self.base_url}/api/jobs/{job_id}/download-all",
            timeout=3,
        ) as response:
            body = response.read()

        self.assertEqual(response.headers.get_content_type(), "application/zip")
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(path.name for path in paths))
            self.assertEqual(archive.read(paths[0].name), paths[0].read_bytes())
            self.assertEqual(archive.read(paths[1].name), paths[1].read_bytes())

    def test_unknown_or_unregistered_output_cannot_be_downloaded(self):
        job_id, _ = self.create_completed_job()

        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(
                f"{self.base_url}/api/jobs/{job_id}/outputs/99/download",
                timeout=3,
            )

        self.assertEqual(context.exception.code, 404)

    def test_loopback_client_can_cancel_queued_job(self):
        job_id = self.manager.create_job("测试项目", "tts", "排队任务")
        request = urllib.request.Request(
            f"{self.base_url}/api/jobs/{job_id}/cancel",
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read())

        self.assertTrue(payload["ok"])
        self.assertEqual(self.manager.get_job(job_id)["status"], "cancelled")

    def test_read_only_handler_allows_get_but_rejects_cancel(self):
        job_id = self.manager.create_job("远程项目", "tts", "只读任务")

        handler = type("ReadOnlyTaskCenterHandler", (TaskCenterHandler,), {
            "_can_manage": lambda self: False,
        })
        handler.manager = self.manager
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base_url + "/api/status", timeout=3) as response:
                status_payload = json.loads(response.read())
            self.assertFalse(status_payload["can_manage"])

            request = urllib.request.Request(
                f"{base_url}/api/jobs/{job_id}/cancel",
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(request, timeout=3)
            self.assertEqual(context.exception.code, 403)
            self.assertEqual(self.manager.get_job(job_id)["status"], "queued")

            completed_job_id, _ = self.create_completed_job()
            select_request = urllib.request.Request(
                f"{base_url}/api/jobs/{completed_job_id}/outputs/1/select",
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as select_context:
                urllib.request.urlopen(select_request, timeout=3)
            self.assertEqual(select_context.exception.code, 403)
            self.assertIsNone(self.manager.get_job(completed_job_id)["selected_output_index"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
