"""Planning and manifest helpers for multi-candidate TTS generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from indextts.utils.randomness import (
    candidate_seed_sequence,
    normalize_candidate_count,
    normalize_inference_seed,
)


@dataclass(frozen=True)
class CandidatePlan:
    base_seed: int
    seeds: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.seeds)


def build_candidate_plan(seed_value, candidate_count) -> CandidatePlan:
    base_seed = normalize_inference_seed(seed_value)
    count = normalize_candidate_count(candidate_count)
    return CandidatePlan(base_seed=base_seed, seeds=tuple(candidate_seed_sequence(base_seed, count)))


def candidate_directory(output_root: str | Path, job_id: str) -> Path:
    safe_job_id = "".join(character for character in str(job_id) if character.isalnum() or character in "-_")
    if not safe_job_id:
        raise ValueError("任务编号无效")
    return Path(output_root) / safe_job_id


def candidate_output_paths(output_root: str | Path, job_id: str, seeds) -> list[Path]:
    root = candidate_directory(output_root, job_id)
    return [
        root / f"candidate-{index:02d}-seed-{int(seed)}.wav"
        for index, seed in enumerate(seeds, start=1)
    ]


def write_candidate_manifest(
    output_root: str | Path,
    job_id: str,
    paths,
    seeds,
    *,
    labels=None,
    variants=None,
) -> Path:
    root = candidate_directory(output_root, job_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "candidates.json"
    paths = list(paths)
    seeds = list(seeds)
    if len(paths) != len(seeds):
        raise ValueError("生成结果与随机种子数量不一致")
    labels = list(labels) if labels is not None else [None] * len(paths)
    variants = list(variants) if variants is not None else [None] * len(paths)
    if len(labels) != len(paths) or len(variants) != len(paths):
        raise ValueError("生成结果元数据数量不一致")

    candidates = []
    for index, (seed, path, label, variant) in enumerate(
        zip(seeds, paths, labels, variants),
        start=1,
    ):
        item = {"index": index, "seed": int(seed), "path": str(Path(path))}
        if label:
            item["label"] = str(label)
        if variant:
            item["variant"] = str(variant)
        candidates.append(item)
    payload = {"job_id": str(job_id), "candidates": candidates}
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest)
    return manifest


def read_candidate_manifest(output_root: str | Path, job_id: str) -> list[dict]:
    manifest = candidate_directory(output_root, job_id) / "candidates.json"
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    valid = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        path = Path(str(candidate.get("path") or ""))
        if path.is_file():
            valid.append(candidate)
    return valid
