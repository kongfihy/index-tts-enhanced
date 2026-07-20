"""Random-seed helpers for reproducible IndexTTS inference."""

from __future__ import annotations

import random
import secrets

MAX_INFERENCE_SEED = 2**31 - 1


def normalize_inference_seed(value, *, randomize_negative: bool = True) -> int:
    """Normalize a UI/API seed into the range supported by all RNG backends."""

    if value is None or (isinstance(value, str) and not value.strip()):
        seed = -1
    else:
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("随机种子必须是整数") from exc

    if seed < 0:
        if not randomize_negative:
            raise ValueError("随机种子不能小于 0")
        return secrets.randbelow(MAX_INFERENCE_SEED + 1)
    if seed > MAX_INFERENCE_SEED:
        raise ValueError(f"随机种子不能大于 {MAX_INFERENCE_SEED}")
    return seed


def normalize_candidate_count(value, *, maximum: int = 3) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("候选数量必须是整数") from exc
    if not 1 <= count <= maximum:
        raise ValueError(f"候选数量必须在 1 到 {maximum} 之间")
    return count


def candidate_seed_sequence(base_seed: int, count: int) -> list[int]:
    clean_seed = normalize_inference_seed(base_seed, randomize_negative=False)
    clean_count = normalize_candidate_count(count)
    modulus = MAX_INFERENCE_SEED + 1
    return [(clean_seed + index) % modulus for index in range(clean_count)]


def apply_inference_seed(seed: int) -> int:
    """Seed Python, NumPy and Torch immediately before one inference candidate."""

    clean_seed = normalize_inference_seed(seed, randomize_negative=False)
    random.seed(clean_seed)

    import numpy as np
    import torch

    np.random.seed(clean_seed)
    torch.manual_seed(clean_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(clean_seed)
    return clean_seed
