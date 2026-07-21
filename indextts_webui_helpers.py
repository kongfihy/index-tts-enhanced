from __future__ import annotations

import hashlib
import json
from pathlib import Path

from indextts_dubbing.audio.reference_analyzer import (
    analyze_reference_audio,
    prepare_reference_audio,
)


def normalize_generation_text(text: str | None) -> str:
    return (text or "").strip()


def matched_output_details(
    output_path: str | Path,
    candidate_number: int,
    match_reference_format: bool,
) -> tuple[Path, str, str]:
    """Return the path, UI label and manifest variant for the A/B copy."""

    source = Path(output_path)
    if match_reference_format:
        return (
            source.with_name(source.stem + "-delivery-matched.wav"),
            f"候选 {candidate_number} · 匹配交付版",
            "level_and_format_matched",
        )
    return (
        source.with_name(source.stem + "-level-matched.wav"),
        f"候选 {candidate_number} · 安全响度匹配",
        "level_matched",
    )


def normalize_choice_index(value, choices) -> int:
    nested_value = getattr(value, "value", value)
    if isinstance(nested_value, int) and not isinstance(nested_value, bool):
        if 0 <= nested_value < len(choices):
            return nested_value
        raise ValueError("当前选项已经失效，请重新选择")
    if nested_value in choices:
        return choices.index(nested_value)
    raise ValueError("无法识别当前选项，请重新选择")


def generation_request_key(request_snapshot: dict) -> str:
    payload = json.dumps(
        request_snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def validate_audio_file(audio_path: str | Path | None, label: str = "参考音频") -> str:
    if audio_path is None or not str(audio_path).strip():
        raise ValueError(f"请选择{label}")

    normalized_path = Path(str(audio_path))
    if not normalized_path.is_file():
        raise ValueError(f"{label}已经失效，请重新上传")
    return str(normalized_path)


def generation_readiness(prompt_audio: str | Path | None, text: str | None) -> tuple[bool, str]:
    has_audio = bool(str(prompt_audio).strip()) if prompt_audio is not None else False
    clean_text = normalize_generation_text(text)

    if not has_audio and not clean_text:
        return False, "请先上传参考音频，并填写需要生成的文本。"
    if not has_audio:
        return False, f"还需要上传参考音频 · 当前文本 {len(clean_text)} 个字符"
    try:
        analysis = analyze_reference_audio(validate_audio_file(prompt_audio, "音色参考音频"))
    except ValueError as exc:
        return False, str(exc)
    if not analysis.usable:
        return False, "；".join(analysis.errors)
    if not clean_text:
        return False, "参考音频已就绪 · 还需要填写生成文本"
    return True, f"准备就绪 · 文本 {len(clean_text)} 个字符"


def reference_audio_quality_message(prompt_audio: str | Path | None) -> str:
    if prompt_audio is None or not str(prompt_audio).strip():
        return "上传后会自动检查时长、音量、削波和静音比例。"
    try:
        audio_path = validate_audio_file(prompt_audio, "音色参考音频")
        return analyze_reference_audio(audio_path).summary()
    except ValueError as exc:
        return f"参考音频不可用\n\n{exc}"


def prepare_reference_audio_file(
    prompt_audio: str | Path | None,
    output_dir: str | Path = "prompts/prepared",
) -> tuple[str, str]:
    audio_path = validate_audio_file(prompt_audio, "音色参考音频")
    prepared = prepare_reference_audio(audio_path, output_dir)
    analysis = analyze_reference_audio(prepared.output_path)
    if not analysis.usable:
        raise ValueError("；".join(analysis.errors))
    return prepared.output_path, prepared.summary() + "\n\n" + analysis.summary()


def validate_generation_inputs(prompt_audio: str | Path | None, text: str | None) -> str:
    clean_text = normalize_generation_text(text)
    audio_path = validate_audio_file(prompt_audio, "音色参考音频")
    analysis = analyze_reference_audio(audio_path)
    if not analysis.usable:
        raise ValueError("；".join(analysis.errors))
    if not clean_text:
        raise ValueError("请输入需要生成的文本")
    return clean_text


def normalize_advanced_generation_args(values) -> list[bool | float | int]:
    values = list(values)
    if len(values) == 8:
        # Snapshots from before A/B output matching existed.
        values.extend([False, False])
    elif len(values) == 9:
        # Snapshots with loudness A/B but before delivery-format matching.
        values.append(False)
    if len(values) != 10:
        raise ValueError("高级生成参数不完整，请刷新页面后重试")

    (
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
        create_loudness_match,
        match_reference_format,
    ) = values
    if not all(
        isinstance(value, bool)
        for value in (do_sample, create_loudness_match, match_reference_format)
    ):
        raise ValueError("生成开关状态异常，请刷新页面后重试")

    normalized = [
        do_sample,
        float(top_p),
        int(top_k),
        float(temperature),
        float(length_penalty),
        int(num_beams),
        float(repetition_penalty),
        int(max_mel_tokens),
        create_loudness_match,
        match_reference_format,
    ]
    (
        _,
        clean_top_p,
        clean_top_k,
        clean_temperature,
        _,
        clean_num_beams,
        clean_repetition_penalty,
        clean_max_mel_tokens,
        _,
        _,
    ) = normalized
    if not 0.0 <= clean_top_p <= 1.0:
        raise ValueError("核心采样概率必须在 0 到 1 之间")
    if clean_top_k < 0:
        raise ValueError("候选词数量不能小于 0")
    if clean_temperature <= 0:
        raise ValueError("生成温度必须大于 0")
    if clean_num_beams < 1:
        raise ValueError("束搜索数量不能小于 1")
    if clean_repetition_penalty <= 0:
        raise ValueError("重复惩罚必须大于 0")
    if clean_max_mel_tokens <= 0:
        raise ValueError("最大音频 Token 数必须大于 0")
    return normalized
