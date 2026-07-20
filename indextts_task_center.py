from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import sqlite3
import tempfile
import threading
import traceback
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "IndexTTSMenuBar"
DB_PATH = Path(os.environ.get("INDEXTTS_TASK_CENTER_DB", str(APP_SUPPORT / "jobs.sqlite3")))
TASK_CENTER_HOST = os.environ.get("INDEXTTS_TASK_CENTER_HOST", "127.0.0.1")
TASK_CENTER_PORT = int(os.environ.get("INDEXTTS_TASK_CENTER_PORT", "7861"))
OUTPUT_ROOT = Path(os.environ.get("INDEXTTS_OUTPUT_ROOT", "outputs/tasks"))
JOB_DEDUPE_WINDOW_SECONDS = max(0.0, float(os.environ.get("INDEXTTS_JOB_DEDUPE_SECONDS", "3")))

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"preparing", "running", "postprocessing"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def task_center_host_for_webui(webui_host: str | None) -> str:
    """Expose the task center whenever the WebUI itself is exposed to the LAN."""
    configured = os.environ.get("INDEXTTS_TASK_CENTER_HOST")
    if configured:
        return configured
    clean_host = (webui_host or "").strip().lower()
    return "0.0.0.0" if clean_host in {"0.0.0.0", "::", "[::]"} else "127.0.0.1"


def is_management_client(client_ip: str, server_ip: str | None = None) -> bool:
    client = str(client_ip or "").split("%", 1)[0]
    server = str(server_ip or "").split("%", 1)[0]
    try:
        if ipaddress.ip_address(client).is_loopback:
            return True
    except ValueError:
        return False
    return bool(server and server not in {"0.0.0.0", "::"} and client == server)


class JobCancelled(RuntimeError):
    pass


class JobAlreadyRunning(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        dedupe_window_seconds: float | None = None,
        output_root: Path = OUTPUT_ROOT,
    ):
        self.db_path = Path(db_path)
        self.output_root = Path(output_root)
        self.dedupe_window_seconds = (
            JOB_DEDUPE_WINDOW_SECONDS
            if dedupe_window_seconds is None
            else max(0.0, float(dedupe_window_seconds))
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
        self._recover_interrupted_jobs()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    input_summary TEXT NOT NULL DEFAULT '',
                    output_path TEXT,
                    outputs_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    seed TEXT,
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "outputs_json" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN outputs_json TEXT NOT NULL DEFAULT '[]'")

    def _recover_interrupted_jobs(self):
        recovered_at = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE jobs
                   SET status='failed', stage='failed', finished_at=?,
                       error='IndexTTS 服务重启，任务已中断', message='服务重启后未完成'
                   WHERE status IN ('preparing','running','postprocessing')""",
                (recovered_at,),
            )
            conn.execute(
                """UPDATE jobs
                   SET status='cancelled', stage='cancelled', finished_at=?, cancel_requested=1,
                       message='服务重启，排队任务已取消'
                   WHERE status='queued'""",
                (recovered_at,),
            )

    def create_job(self, project_name: str, job_type: str, input_summary: str, parameters=None, seed=None) -> str:
        clean_name = (project_name or "默认项目").strip()[:80] or "默认项目"
        project_id = uuid.uuid5(uuid.NAMESPACE_URL, clean_name).hex[:12]
        clean_summary = (input_summary or "")[:300]
        clean_parameters = parameters or {}
        parameters_json = json.dumps(clean_parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        created_at = datetime.now(timezone.utc)

        with self._lock, self._connect() as conn:
            if self.dedupe_window_seconds > 0:
                rows = conn.execute(
                    """SELECT job_id, created_at, seed, parameters_json
                       FROM jobs
                       WHERE project_id=? AND job_type=? AND input_summary=?
                         AND status IN ('queued','preparing','running','postprocessing')
                       ORDER BY created_at DESC LIMIT 20""",
                    (project_id, job_type, clean_summary),
                ).fetchall()
                for row in rows:
                    try:
                        existing_created_at = datetime.fromisoformat(row["created_at"])
                        age_seconds = (created_at - existing_created_at).total_seconds()
                        existing_parameters = json.loads(row["parameters_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if age_seconds > self.dedupe_window_seconds:
                        break
                    requested_seed = str(seed) if seed is not None else None
                    if (
                        age_seconds >= 0
                        and existing_parameters == clean_parameters
                        and row["seed"] == requested_seed
                    ):
                        return row["job_id"]

            job_id = uuid.uuid4().hex[:12]
            conn.execute(
                """INSERT INTO jobs
                   (job_id, project_id, project_name, job_type, status, progress, stage,
                    message, created_at, input_summary, seed, parameters_json)
                   VALUES (?, ?, ?, ?, 'queued', 0, 'queued', '等待生成', ?, ?, ?, ?)""",
                (
                    job_id,
                    project_id,
                    clean_name,
                    job_type,
                    created_at.isoformat(timespec="seconds"),
                    clean_summary,
                    str(seed) if seed is not None else None,
                    parameters_json,
                ),
            )
        return job_id

    def update_job(self, job_id: str, **fields):
        allowed = {
            "status", "progress", "stage", "message", "started_at", "finished_at",
            "output_path", "outputs_json", "error", "cancel_requested",
        }
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        if "progress" in values:
            values["progress"] = max(0.0, min(1.0, float(values["progress"])))
        sql = ", ".join(f"{key}=?" for key in values)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {sql} WHERE job_id=?", (*values.values(), job_id))

    def start_job(self, job_id: str) -> tuple[bool, str | None]:
        """Atomically claim a queued job and suppress duplicate Gradio events."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise RuntimeError("任务记录不存在")

            status = row["status"]
            if row["cancel_requested"] or status == "cancelled":
                conn.execute(
                    """UPDATE jobs SET status='cancelled', stage='cancelled', finished_at=?,
                       message='任务已取消' WHERE job_id=?""",
                    (utc_now(), job_id),
                )
                raise JobCancelled("任务已取消")
            if status == "completed":
                return False, row["output_path"]
            if status in ACTIVE_STATUSES:
                raise JobAlreadyRunning("相同任务已在生成，请勿重复提交")
            if status == "failed":
                raise RuntimeError("任务已经失败，请重新提交")
            if status != "queued":
                raise RuntimeError(f"任务状态异常：{status}")

            cursor = conn.execute(
                """UPDATE jobs
                   SET status='preparing', progress=0.01, stage='preparing',
                       message='准备参考音频', started_at=?
                   WHERE job_id=? AND status='queued'""",
                (utc_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise JobAlreadyRunning("相同任务已被其他生成流程接管")
        return True, None

    def complete_job(self, job_id: str, output_paths):
        if isinstance(output_paths, (str, os.PathLike)):
            clean_outputs = [str(output_paths)]
        else:
            clean_outputs = [str(path) for path in (output_paths or []) if path]
        primary_output = clean_outputs[0] if clean_outputs else None
        outputs_json = json.dumps(clean_outputs, ensure_ascii=False)
        finished_at = utc_now()
        cancelled = False
        error_message = None
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """UPDATE jobs
                   SET status='completed', progress=1, stage='completed', message='生成完成',
                       finished_at=?, output_path=?, outputs_json=?
                   WHERE job_id=? AND cancel_requested=0
                     AND status IN ('preparing','running','postprocessing')""",
                (finished_at, primary_output, outputs_json, job_id),
            )
            if cursor.rowcount == 1:
                return

            row = conn.execute(
                "SELECT status, cancel_requested FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row and (row["cancel_requested"] or row["status"] == "cancelled"):
                conn.execute(
                    """UPDATE jobs
                       SET status='cancelled', stage='cancelled', message='任务已取消',
                           finished_at=?, cancel_requested=1
                       WHERE job_id=?""",
                    (finished_at, job_id),
                )
                cancelled = True
            elif row is None:
                error_message = "任务记录不存在"
            else:
                error_message = f"无法完成当前状态的任务：{row['status']}"

        if cancelled:
            raise JobCancelled("任务已取消")
        if error_message:
            raise RuntimeError(error_message)

    def fail_job(self, job_id: str, exc: BaseException):
        finished_at = utc_now()
        with self._lock, self._connect() as conn:
            if isinstance(exc, JobCancelled):
                conn.execute(
                    """UPDATE jobs
                       SET status='cancelled', stage='cancelled', message='任务已取消',
                           finished_at=?, cancel_requested=1
                       WHERE job_id=?
                         AND status IN ('queued','preparing','running','postprocessing','cancelled')""",
                    (finished_at, job_id),
                )
                return

            error_text = f"{type(exc).__name__}: {exc}"[:1200]
            conn.execute(
                """UPDATE jobs
                   SET status='failed', stage='failed', message='生成失败', error=?, finished_at=?
                   WHERE job_id=?
                     AND status IN ('queued','preparing','running','postprocessing')""",
                (error_text, finished_at, job_id),
            )

    def request_cancel(self, job_id: str) -> bool:
        requested_at = utc_now()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """UPDATE jobs
                   SET status='cancelled', stage='cancelled', message='排队任务已取消',
                       finished_at=?, cancel_requested=1
                   WHERE job_id=? AND status='queued'""",
                (requested_at, job_id),
            )
            if cursor.rowcount == 1:
                return True

            cursor = conn.execute(
                """UPDATE jobs
                   SET cancel_requested=1, message='正在安全取消，将在当前阶段结束后停止'
                   WHERE job_id=? AND cancel_requested=0
                     AND status IN ('preparing','running','postprocessing')""",
                (job_id,),
            )
            return cursor.rowcount == 1

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT cancel_requested, status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return bool(row and (row["cancel_requested"] or row["status"] == "cancelled"))

    @staticmethod
    def _version_select(alias="j") -> str:
        return (
            f"(SELECT COUNT(*) FROM jobs j2 WHERE j2.project_id={alias}.project_id "
            f"AND j2.rowid<={alias}.rowid) AS project_version"
        )

    def get_job(self, job_id: str):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT j.*, {self._version_select()} FROM jobs j WHERE job_id=?",
                (job_id,),
            ).fetchone()
        return self._serialize(row) if row else None

    def list_jobs(self, limit=100, status=None, project_id=None):
        try:
            clean_limit = min(max(int(limit), 1), 5000)
        except (TypeError, ValueError):
            clean_limit = 100

        clauses, params = [], []
        if status == "active":
            clauses.append("j.status IN ('preparing','running','postprocessing')")
        elif status:
            clauses.append("j.status=?")
            params.append(status)
        if project_id:
            clauses.append("j.project_id=?")
            params.append(project_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT j.*, {self._version_select()} FROM jobs j{where} "
                "ORDER BY j.created_at DESC, j.rowid DESC LIMIT ?",
                (*params, clean_limit),
            ).fetchall()
        return [self._serialize(row) for row in rows]

    def list_projects(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT project_id, project_name, COUNT(*) AS job_count,
                          MAX(created_at) AS updated_at,
                          SUM(CASE WHEN status IN ('queued','preparing','running','postprocessing') THEN 1 ELSE 0 END) AS active_jobs
                   FROM jobs GROUP BY project_id, project_name ORDER BY updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def status(self):
        with self._lock, self._connect() as conn:
            counts = {
                row["status"]: row["count"]
                for row in conn.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status")
            }
            current = conn.execute(
                f"SELECT j.*, {self._version_select()} FROM jobs j "
                "WHERE status IN ('preparing','running','postprocessing') ORDER BY started_at ASC LIMIT 1"
            ).fetchone()
            if current is None:
                current = conn.execute(
                    f"SELECT j.*, {self._version_select()} FROM jobs j "
                    "WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
        return {
            "service": "running",
            "active_jobs": sum(counts.get(status, 0) for status in ACTIVE_STATUSES),
            "queued_jobs": counts.get("queued", 0),
            "failed_jobs": counts.get("failed", 0),
            "current_job": self._serialize(current) if current else None,
            "updated_at": utc_now(),
        }

    def _manifest_output_paths(self, job_id: str) -> list[str]:
        manifest = self.output_root / job_id / "candidates.json"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            return []
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return []
        return [str(item.get("path")) for item in candidates if isinstance(item, dict) and item.get("path")]

    def output_paths_for_job(self, job) -> list[str]:
        if not job:
            return []
        raw_outputs = job.get("output_paths")
        if not raw_outputs:
            raw_outputs = self._manifest_output_paths(str(job.get("job_id") or ""))
        if not raw_outputs and job.get("output_path"):
            raw_outputs = [job["output_path"]]
        seen = set()
        outputs = []
        for path in raw_outputs or []:
            clean = str(path)
            if clean and clean not in seen:
                outputs.append(clean)
                seen.add(clean)
        return outputs

    def resolve_output_path(self, raw_path: str | os.PathLike | None) -> Path | None:
        if not raw_path:
            return None
        root = self.output_root.resolve()
        path = Path(raw_path).expanduser()
        candidates = [path.resolve()] if path.is_absolute() else [
            (Path.cwd() / path).resolve(),
            (root / path).resolve(),
        ]
        for candidate in candidates:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return candidate
        return None

    def output_entries(self, job) -> list[dict]:
        entries = []
        for index, raw_path in enumerate(self.output_paths_for_job(job), start=1):
            resolved = self.resolve_output_path(raw_path)
            entries.append({
                "index": index,
                "name": resolved.name if resolved else Path(raw_path).name or f"候选-{index}.wav",
                "available": bool(resolved and resolved.is_file()),
                "download_url": f"/api/jobs/{job['job_id']}/outputs/{index}/download",
            })
        return entries

    def download_path(self, job_id: str, output_index: int) -> Path | None:
        job = self.get_job(job_id)
        paths = self.output_paths_for_job(job)
        if output_index < 1 or output_index > len(paths):
            return None
        resolved = self.resolve_output_path(paths[output_index - 1])
        return resolved if resolved and resolved.is_file() else None

    def public_job(self, job):
        if not job:
            return None
        hidden = {"output_path", "output_paths", "outputs_json", "error"}
        public = {key: value for key, value in job.items() if key not in hidden}
        public["outputs"] = self.output_entries(job)
        public["output_count"] = len(public["outputs"])
        public["download_all_url"] = f"/api/jobs/{job['job_id']}/download-all" if public["outputs"] else None
        if job.get("error"):
            public["error_message"] = "生成失败，请在运行服务的电脑上查看日志"
        return public

    @staticmethod
    def _serialize(row):
        data = dict(row)
        data["progress"] = round(float(data.get("progress") or 0), 4)
        data["cancel_requested"] = bool(data.get("cancel_requested"))
        try:
            data["parameters"] = json.loads(data.pop("parameters_json") or "{}")
        except json.JSONDecodeError:
            data["parameters"] = {}
            data.pop("parameters_json", None)
        try:
            outputs = json.loads(data.get("outputs_json") or "[]")
            data["output_paths"] = outputs if isinstance(outputs, list) else []
        except json.JSONDecodeError:
            data["output_paths"] = []
        data["project_version"] = int(data.get("project_version") or 1)
        return data


class JobProgress:
    def __init__(
        self,
        manager: JobManager,
        job_id: str,
        gradio_progress=None,
        *,
        progress_start: float = 0.0,
        progress_span: float = 1.0,
        description_prefix: str = "",
    ):
        self.manager = manager
        self.job_id = job_id
        self.gradio_progress = gradio_progress
        self.progress_start = max(0.0, min(1.0, float(progress_start)))
        self.progress_span = max(0.0, min(1.0 - self.progress_start, float(progress_span)))
        self.description_prefix = str(description_prefix or "")

    def __call__(self, value, desc=""):
        if self.manager.is_cancel_requested(self.job_id):
            raise JobCancelled("任务已取消")
        local_progress = max(0.0, min(1.0, float(value or 0)))
        progress = self.progress_start + local_progress * self.progress_span
        full_desc = self.description_prefix + (desc or "生成中")
        stage = self._stage(desc, local_progress)
        status = "postprocessing" if stage == "saving" and progress >= 0.99 else "running"
        self.manager.update_job(self.job_id, status=status, progress=progress, stage=stage, message=full_desc)
        if self.gradio_progress is not None:
            self.gradio_progress(progress, desc=full_desc)

    @staticmethod
    def _stage(desc: str, value: float) -> str:
        text = (desc or "").lower()
        if "text" in text: return "text_processing"
        if "speaker" in text or "prompt" in text: return "voice_encoding"
        if "emotion" in text or "emo" in text: return "emotion_encoding"
        if "gpt" in text or "semantic" in text: return "semantic_generation"
        if "s2mel" in text or "mel" in text: return "acoustic_generation"
        if "bigvgan" in text or "vocoder" in text: return "waveform_generation"
        if "save" in text: return "saving"
        if value < 0.1: return "preparing"
        return "running"


ADMIN_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IndexTTS 任务中心</title>
<style>
:root{color-scheme:light dark;--bg:#f4f6f8;--panel:#fff;--text:#17202a;--muted:#667085;--line:#e4e7ec;--accent:#2563eb;--accent-soft:#eff6ff;--good:#15803d;--warn:#b45309;--bad:#b42318;--shadow:0 10px 35px rgba(16,24,40,.08)}
@media(prefers-color-scheme:dark){:root{--bg:#101317;--panel:#181d23;--text:#eef2f6;--muted:#9aa4b2;--line:#2d3640;--accent:#6ea8fe;--accent-soft:#18283d;--good:#5bd486;--warn:#f7b955;--bad:#ff8178;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}.wrap{max-width:1180px;margin:auto;padding:28px 20px 56px}header{display:flex;justify-content:space-between;gap:16px;align-items:flex-end;margin-bottom:20px}h1,h2,h3,p{margin:0}.sub{color:var(--muted)}.access{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:var(--panel)}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.panel,.stat,.projectGroup{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.stat{padding:16px}.stat b{display:block;font-size:24px;margin-top:4px}.current{display:none;padding:18px;margin-bottom:14px}.current.show{display:block}.currentTop,.toolbar,.projectHeader,.versionTop,.actions{display:flex;align-items:center;justify-content:space-between;gap:12px}.toolbar{padding:12px;margin-bottom:14px}.filters{display:flex;gap:8px;flex-wrap:wrap}select,button,.download{font:inherit;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--text);padding:8px 11px;text-decoration:none;cursor:pointer}button:hover,.download:hover{border-color:var(--accent);color:var(--accent)}button.cancel{color:var(--bad)}.wideProgress,.progress{height:7px;background:var(--line);border-radius:999px;overflow:hidden}.wideProgress{margin-top:14px}.bar{height:100%;background:var(--accent);transition:width .25s}.projectsList{display:grid;gap:12px}.projectGroup{overflow:hidden}.projectGroup>summary{list-style:none;cursor:pointer;padding:17px 18px}.projectGroup>summary::-webkit-details-marker{display:none}.projectHeader:before{content:'›';font-size:25px;color:var(--muted);transition:transform .2s}.projectGroup[open] .projectHeader:before{transform:rotate(90deg)}.projectTitle{flex:1}.projectTitle h2{font-size:17px}.count{white-space:nowrap;color:var(--muted)}.versions{border-top:1px solid var(--line)}.version{padding:17px 18px;border-top:1px solid var(--line)}.version:first-child{border-top:0}.versionTitle{font-weight:700}.summary{margin:8px 0;color:var(--text);white-space:pre-wrap}.meta{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:13px}.badge{display:inline-block;border-radius:999px;padding:3px 8px;background:var(--accent-soft);color:var(--accent);font-size:12px}.badge.completed{color:var(--good)}.badge.failed,.badge.cancelled{color:var(--bad)}.badge.queued{color:var(--warn)}.actions{justify-content:flex-start;flex-wrap:wrap;margin-top:12px}.download.unavailable{pointer-events:none;opacity:.45}.empty{padding:36px;text-align:center}.manageOnly{display:none}.canManage .manageOnly{display:inline-flex}@media(max-width:720px){.wrap{padding:18px 12px}.stats{grid-template-columns:repeat(2,1fr)}header{align-items:flex-start;flex-direction:column}.toolbar{align-items:stretch;flex-direction:column}.currentTop,.versionTop{align-items:flex-start;flex-direction:column}.hide-sm{display:none}}
</style></head><body><div class="wrap">
<header><div><h1>任务中心</h1><p class="sub">同一项目自动归档为多个版本，所有生成结果都可随时下载。</p></div><div id="access" class="access">正在连接…</div></header>
<section class="stats"><div class="stat"><span class="sub">运行中</span><b id="active">0</b></div><div class="stat"><span class="sub">排队</span><b id="queued">0</b></div><div class="stat"><span class="sub">失败</span><b id="failed">0</b></div><div class="stat"><span class="sub">项目</span><b id="projectCount">0</b></div></section>
<section id="current" class="panel current"></section>
<section class="panel toolbar"><div class="filters"><select id="status"><option value="">全部状态</option><option value="completed">已完成</option><option value="queued">排队中</option><option value="active">生成中</option><option value="failed">失败</option><option value="cancelled">已取消</option></select><select id="project"><option value="">全部项目</option></select></div><span id="updated" class="sub">准备刷新</span></section>
<main id="projectsList" class="projectsList"><div class="panel empty sub">正在读取任务…</div></main>
</div><script>
const names={queued:'排队中',preparing:'准备中',running:'生成中',postprocessing:'保存中',completed:'已完成',failed:'失败',cancelled:'已取消'};
const stages={queued:'等待生成',preparing:'准备参考音频',text_processing:'处理文本',voice_encoding:'提取音色',emotion_encoding:'处理情感',semantic_generation:'生成语义',acoustic_generation:'生成声学特征',waveform_generation:'合成波形',saving:'保存文件',completed:'生成完成',failed:'生成失败',cancelled:'已取消'};
let canManage=false,openProjects=new Set(),initialized=false;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const timeText=s=>{if(!s)return '-';let d=new Date(s);return isNaN(d)?esc(s):d.toLocaleString('zh-CN',{hour12:false})};
async function cancelJob(id){if(!canManage||!confirm('确定取消这个任务吗？'))return;await fetch(`/api/jobs/${id}/cancel`,{method:'POST'});refresh()}
function outputButtons(x){if(!x.outputs?.length)return x.status==='completed'?'<span class="sub">未找到输出文件</span>':'';let items=x.outputs.map(o=>`<a class="download ${o.available?'':'unavailable'}" href="${o.available?o.download_url:'#'}">下载候选 ${o.index}</a>`);if(x.outputs.filter(o=>o.available).length>1)items.push(`<a class="download" href="${x.download_all_url}">打包下载全部</a>`);return items.join('')}
function renderVersion(x){let pc=Math.round((x.progress||0)*100),active=!['completed','failed','cancelled'].includes(x.status),candidateCount=x.parameters?.candidate_count||x.output_count||1;return `<article class="version"><div class="versionTop"><div><span class="versionTitle">版本 ${x.project_version}</span> <span class="badge ${x.status}">${names[x.status]||esc(x.status)}</span></div><span class="sub">${timeText(x.created_at)}</span></div><div class="summary">${esc(x.input_summary||x.job_type)}</div><div class="meta"><span>${esc(stages[x.stage]||x.message||'')}</span>${x.seed!==null&&x.seed!==undefined?`<span>种子 ${esc(x.seed)}</span>`:''}<span>${candidateCount} 个候选</span><span>${pc}%</span></div>${active?`<div class="progress" style="margin-top:10px"><div class="bar" style="width:${pc}%"></div></div>`:''}<div class="actions">${outputButtons(x)}${active?`<button class="cancel manageOnly" onclick="cancelJob('${x.job_id}')">取消任务</button>`:''}</div></article>`}
function renderProjects(jobs){let groups=new Map();for(let x of jobs){if(!groups.has(x.project_id))groups.set(x.project_id,{name:x.project_name,jobs:[]});groups.get(x.project_id).jobs.push(x)}let root=document.querySelector('#projectsList');if(!groups.size){root.innerHTML='<div class="panel empty sub">没有符合筛选条件的任务</div>';return}if(!initialized){openProjects.add(groups.keys().next().value);initialized=true}root.innerHTML=[...groups].map(([id,g])=>`<details class="projectGroup" data-id="${esc(id)}" ${openProjects.has(id)?'open':''}><summary><div class="projectHeader"><div class="projectTitle"><h2>${esc(g.name)}</h2><div class="sub">最近更新 ${timeText(g.jobs[0].created_at)}</div></div><span class="count">${g.jobs.length} 个版本</span></div></summary><div class="versions">${g.jobs.map(renderVersion).join('')}</div></details>`).join('');root.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>d.open?openProjects.add(d.dataset.id):openProjects.delete(d.dataset.id)))}
async function refresh(){try{let status=document.querySelector('#status').value,project=document.querySelector('#project').value,q=new URLSearchParams({limit:'5000'});if(status)q.set('status',status);if(project)q.set('project_id',project);let [sr,jr,pr]=await Promise.all([fetch('/api/status'),fetch('/api/jobs?'+q),fetch('/api/projects')]);let s=await sr.json(),j=await jr.json(),p=await pr.json();canManage=!!s.can_manage;document.body.classList.toggle('canManage',canManage);document.querySelector('#access').textContent=canManage?'本机管理模式':'局域网只读 · 可查看和下载';document.querySelector('#active').textContent=s.active_jobs;document.querySelector('#queued').textContent=s.queued_jobs;document.querySelector('#failed').textContent=s.failed_jobs;document.querySelector('#projectCount').textContent=p.projects.length;document.querySelector('#updated').textContent='刚刚更新';let ps=document.querySelector('#project'),old=ps.value;ps.innerHTML='<option value="">全部项目</option>'+p.projects.map(x=>`<option value="${esc(x.project_id)}">${esc(x.project_name)} (${x.job_count})</option>`).join('');ps.value=old;let c=document.querySelector('#current');if(s.current_job){let x=s.current_job,pc=Math.round(x.progress*100);c.className='panel current show';c.innerHTML=`<div class="currentTop"><div><h3>${esc(x.project_name)} · 版本 ${x.project_version}</h3><div class="sub">${esc(stages[x.stage]||x.message||'生成中')} · ${pc}%</div></div><button class="cancel manageOnly" onclick="cancelJob('${x.job_id}')">取消当前任务</button></div><div class="wideProgress"><div class="bar" style="width:${pc}%"></div></div>`}else{c.className='panel current';c.innerHTML=''}renderProjects(j.jobs)}catch(e){document.querySelector('#updated').textContent='连接失败'}}
document.querySelector('#status').onchange=refresh;document.querySelector('#project').onchange=refresh;refresh();setInterval(refresh,2500);
</script></body></html>'''


class TaskCenterHandler(BaseHTTPRequestHandler):
    manager: JobManager = None

    def log_message(self, fmt, *args):
        return

    def _can_manage(self) -> bool:
        try:
            server_ip = self.connection.getsockname()[0]
        except OSError:
            server_ip = None
        return is_management_client(self.client_address[0], server_ip)

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name, safe='')}")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "private, no-store")
        self.end_headers()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def _send_zip(self, job_id: str):
        job = self.manager.get_job(job_id)
        if not job:
            self._json({"error": "not found"}, 404)
            return
        files = []
        for raw_path in self.manager.output_paths_for_job(job):
            resolved = self.manager.resolve_output_path(raw_path)
            if resolved and resolved.is_file():
                files.append(resolved)
        if not files:
            self._json({"error": "没有可下载的输出"}, 404)
            return
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as archive:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                used_names = set()
                for index, path in enumerate(files, start=1):
                    name = path.name
                    if name in used_names:
                        name = f"candidate-{index:02d}-{name}"
                    used_names.add(name)
                    bundle.write(path, arcname=name)
            size = archive.tell()
            archive.seek(0)
            filename = f"{job['project_name']}-版本{job['project_version']}-{job_id}.zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(filename, safe='')}")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "private, no-store")
            self.end_headers()
            while chunk := archive.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = ADMIN_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            payload = self.manager.status()
            payload["can_manage"] = self._can_manage()
            if payload.get("current_job"):
                payload["current_job"] = self.manager.public_job(payload["current_job"])
            self._json(payload)
            return
        if parsed.path == "/api/jobs":
            query = parse_qs(parsed.query)
            jobs = self.manager.list_jobs(
                query.get("limit", [100])[0],
                query.get("status", [None])[0] or None,
                query.get("project_id", [None])[0] or None,
            )
            self._json({"jobs": [self.manager.public_job(job) for job in jobs]})
            return
        if parsed.path == "/api/projects":
            self._json({"projects": self.manager.list_projects()})
            return

        parts = parsed.path.strip("/").split("/")
        if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "outputs" and parts[5] == "download":
            try:
                output_index = int(parts[4])
            except ValueError:
                self._json({"error": "invalid output index"}, 400)
                return
            path = self.manager.download_path(parts[2], output_index)
            if not path:
                self._json({"error": "not found"}, 404)
                return
            self._send_file(path)
            return
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "download-all":
            self._send_zip(parts[2])
            return
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            job = self.manager.get_job(parts[2])
            self._json(self.manager.public_job(job) if job else {"error": "not found"}, 200 if job else 404)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
            if not self._can_manage():
                self._json({"error": "局域网访问为只读模式"}, 403)
                return
            ok = self.manager.request_cancel(parts[2])
            self._json({"ok": ok}, 200 if ok else 409)
            return
        self._json({"error": "not found"}, 404)


def start_task_center_server(manager: JobManager, host=TASK_CENTER_HOST, port=TASK_CENTER_PORT):
    TaskCenterHandler.manager = manager

    def serve():
        try:
            server = ThreadingHTTPServer((host, port), TaskCenterHandler)
            display_host = "127.0.0.1" if host == "0.0.0.0" else host
            print(f">> Task Center: http://{display_host}:{port} (listen {host})")
            server.serve_forever()
        except OSError as exc:
            print(f">> Task Center failed to start: {exc}")
            traceback.print_exc()

    thread = threading.Thread(target=serve, name="IndexTTSTaskCenter", daemon=True)
    thread.start()
    return thread
