"""Hugging Face cache selection and offline-first loading helpers.

The WebUI imports several libraries that can initialize Hugging Face cache
constants very early.  Configure the environment before those imports and
always pass an explicit cache directory when loading remote components.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, TypeVar

T = TypeVar("T")

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _deduplicate(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.expanduser().resolve(strict=False))
        if normalized not in seen:
            seen.add(normalized)
            result.append(Path(normalized))
    return result


def persistent_hub_cache(
    *,
    home: Path | None = None,
    platform: str | None = None,
) -> Path:
    """Return the stable default cache used for future downloads."""

    home = (home or Path.home()).expanduser()
    platform = platform or sys.platform
    if platform == "darwin":
        return home / "Library" / "Caches" / "IndexTTS" / "huggingface" / "hub"
    return home / ".cache" / "indextts" / "huggingface" / "hub"


def hub_cache_candidates(
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
    temp_root: Path = Path("/private/tmp"),
) -> list[Path]:
    """List caches in lookup order.

    The first entry is the stable write target.  Older project, user and
    temporary cache locations remain readable so a cached model is not missed
    merely because an earlier launcher used a different environment.
    """

    environ = os.environ if environ is None else environ
    home = (home or Path.home()).expanduser()
    project_root = project_root or Path(__file__).resolve().parents[2]

    explicit = environ.get("INDEXTTS_HF_HUB_CACHE") or environ.get("HF_HUB_CACHE")
    xdg_root = Path(environ.get("XDG_CACHE_HOME", home / ".cache"))
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))

    paths.extend(
        [
            persistent_hub_cache(home=home, platform=platform),
            project_root / "checkpoints" / "hf_cache",
            home / ".cache" / "huggingface" / "hub",
            xdg_root / "huggingface" / "hub",
            temp_root / "indextts-cache" / "huggingface" / "hub",
        ]
    )
    return _deduplicate(paths)


def configure_huggingface_environment(
    *,
    project_root: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> Path:
    """Configure Hugging Face before importing transformers/huggingface_hub."""

    environ = environ if environ is not None else os.environ
    primary = hub_cache_candidates(
        project_root=project_root,
        environ=environ,
        home=home,
        platform=platform,
    )[0]
    primary.mkdir(parents=True, exist_ok=True)

    # Set all commonly consulted variables.  This must run before importing
    # transformers, gradio or huggingface_hub because they cache these values.
    environ["INDEXTTS_HF_HUB_CACHE"] = str(primary)
    environ["HF_HUB_CACHE"] = str(primary)
    environ.setdefault("HF_HOME", str(primary.parent))
    return primary


def offline_mode(environ: Mapping[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return any(
        environ.get(name, "").strip().lower() in _TRUE_VALUES
        for name in ("INDEXTTS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _cache_has_content(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def load_pretrained_offline_first(
    loader: Callable[..., T],
    model_id: str,
    *,
    component_name: str | None = None,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    **kwargs,
) -> T:
    """Load from every known local cache before allowing a network request."""

    environ = os.environ if environ is None else environ
    candidates = hub_cache_candidates(project_root=project_root, environ=environ)
    local_errors: list[tuple[Path, Exception]] = []

    for cache_dir in candidates:
        if not _cache_has_content(cache_dir):
            continue
        try:
            value = loader(
                model_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
                **kwargs,
            )
            print(f">> {component_name or model_id} loaded from local cache: {cache_dir}")
            return value
        except Exception as error:  # Each loader uses different local-miss types.
            local_errors.append((cache_dir, error))

    label = component_name or model_id
    searched = ", ".join(str(path) for path in candidates)
    if offline_mode(environ):
        detail = f" Last local error: {local_errors[-1][1]}" if local_errors else ""
        raise RuntimeError(
            f"{label} is missing from local caches while offline mode is enabled. "
            f"Searched: {searched}.{detail}"
        )

    primary = candidates[0]
    try:
        print(f">> {label} not found locally; downloading into: {primary}")
        return loader(model_id, cache_dir=str(primary), **kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load {label}. Local caches were checked first, then the "
            f"Hugging Face download failed. Searched: {searched}. Cause: {error}"
        ) from error


def download_file_offline_first(
    downloader: Callable[..., T],
    repo_id: str,
    *,
    filename: str,
    component_name: str | None = None,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    **kwargs,
) -> T:
    """Offline-first wrapper for huggingface_hub.hf_hub_download."""

    return load_pretrained_offline_first(
        lambda model_id, **load_kwargs: downloader(
            repo_id=model_id,
            filename=filename,
            **load_kwargs,
        ),
        repo_id,
        component_name=component_name or f"{repo_id}/{filename}",
        project_root=project_root,
        environ=environ,
        **kwargs,
    )
