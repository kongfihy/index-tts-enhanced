"""Reference-audio quality checks and non-destructive preparation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf


MAX_REFERENCE_SECONDS = 15.0
MAX_ANALYSIS_SECONDS = 30.0
MAX_PREPARATION_SOURCE_SECONDS = 60.0
SILENCE_THRESHOLD_DBFS = -45.0
CLIPPING_THRESHOLD = 0.999


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


def _format_db(value: float) -> str:
    return f"{value:.1f}" if math.isfinite(value) else "-∞"


def _channel_label(channels: int) -> str:
    if channels == 1:
        return "单声道"
    if channels == 2:
        return "双声道"
    return f"{channels} 声道"


@dataclass(frozen=True)
class ReferenceAudioAnalysis:
    path: str
    duration_seconds: float
    analyzed_seconds: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    active_rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    longest_silence_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "不可用"
        if self.warnings:
            return "需要注意"
        return "良好"

    def summary(self) -> str:
        facts = (
            f"参考音频{self.status} · {self.duration_seconds:.1f} 秒 · "
            f"{self.sample_rate / 1000:.1f} kHz · {_channel_label(self.channels)} · "
            f"峰值 {_format_db(self.peak_dbfs)} dBFS"
        )
        details = self.errors or self.warnings
        if details:
            return facts + "\n\n" + "；".join(details)
        return facts + "\n\n音量、时长和静音比例适合直接作为音色参考。"


@dataclass(frozen=True)
class PreparedReferenceAudio:
    source_path: str
    output_path: str
    original_duration_seconds: float
    output_duration_seconds: float
    removed_leading_seconds: float
    removed_trailing_seconds: float
    peak_gain_db: float

    def summary(self) -> str:
        removed = self.removed_leading_seconds + self.removed_trailing_seconds
        message = (
            f"参考音频已整理 · {self.original_duration_seconds:.1f} 秒 → "
            f"{self.output_duration_seconds:.1f} 秒"
        )
        if removed >= 0.05:
            message += f" · 去除首尾静音 {removed:.1f} 秒"
        else:
            message += " · 未发现需要去除的明显首尾静音"
        if abs(self.peak_gain_db) >= 0.05:
            message += f" · 峰值保护 {self.peak_gain_db:.1f} dB"
        return message + "\n\n原文件未被覆盖，生成时将使用整理后的独立 WAV。"


def _frame_rms(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    frame_length = max(1, int(round(sample_rate * 0.02)))
    hop_length = max(1, int(round(sample_rate * 0.01)))
    if mono.size <= frame_length:
        padded = np.pad(mono, (0, frame_length - mono.size))
        return np.array([float(np.sqrt(np.mean(np.square(padded))))]), frame_length, hop_length

    values = []
    for start in range(0, mono.size - frame_length + 1, hop_length):
        frame = mono[start:start + frame_length]
        values.append(float(np.sqrt(np.mean(np.square(frame)))))
    return np.asarray(values, dtype=np.float64), frame_length, hop_length


def _silence_metrics(
    mono: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, int, int, float, float, float, float]:
    rms_values, frame_length, hop_length = _frame_rms(mono, sample_rate)
    threshold = 10.0 ** (SILENCE_THRESHOLD_DBFS / 20.0)
    active = rms_values > threshold
    silence_ratio = float(np.mean(~active)) if active.size else 1.0

    if not np.any(active):
        analyzed_seconds = mono.size / sample_rate
        return active, frame_length, hop_length, silence_ratio, analyzed_seconds, analyzed_seconds, analyzed_seconds

    first_active = int(np.argmax(active))
    last_active = int(len(active) - 1 - np.argmax(active[::-1]))
    leading = min(mono.size, first_active * hop_length) / sample_rate
    active_end = min(mono.size, last_active * hop_length + frame_length)
    trailing = max(0, mono.size - active_end) / sample_rate

    longest_frames = 0
    current_frames = 0
    for is_active in active:
        if is_active:
            longest_frames = max(longest_frames, current_frames)
            current_frames = 0
        else:
            current_frames += 1
    longest_frames = max(longest_frames, current_frames)
    longest = (longest_frames * hop_length + (frame_length if longest_frames else 0)) / sample_rate
    return active, frame_length, hop_length, silence_ratio, longest, leading, trailing


def _read_audio(path: str, max_seconds: float | None = None) -> tuple[np.ndarray, int, int, float]:
    try:
        info = sf.info(path)
    except (RuntimeError, OSError) as exc:
        raise ValueError("无法读取参考音频，请重新上传WAV、MP3 或 FLAC 文件") from exc

    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("参考音频没有可读取的声音内容")

    frames = info.frames
    if max_seconds is not None:
        frames = min(frames, int(round(info.samplerate * max_seconds)))
    try:
        audio, sample_rate = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError("参考音频解码失败，请重新导出后上传") from exc

    if audio.size == 0:
        raise ValueError("参考音频没有可读取的声音内容")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio, int(sample_rate), int(info.channels), float(info.duration)


def _analysis_cache_key(path: str) -> tuple[str, int, int]:
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ValueError("参考音频已经失效，请重新上传") from exc
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def analyze_reference_audio(path: str | Path) -> ReferenceAudioAnalysis:
    normalized_path, size, modified_ns = _analysis_cache_key(str(path))
    return _analyze_reference_audio_cached(normalized_path, size, modified_ns)


@lru_cache(maxsize=32)
def _analyze_reference_audio_cached(
    path: str,
    _size: int,
    _modified_ns: int,
) -> ReferenceAudioAnalysis:
    audio, sample_rate, channels, duration = _read_audio(path, MAX_ANALYSIS_SECONDS)
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    analyzed_seconds = mono.size / sample_rate

    absolute = np.abs(mono)
    peak = float(np.max(absolute))
    peak_dbfs = _to_dbfs(peak)
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    rms_dbfs = _to_dbfs(rms)
    clipping_ratio = float(np.mean(absolute >= CLIPPING_THRESHOLD))

    active, frame_length, hop_length, silence_ratio, longest, leading, trailing = _silence_metrics(
        mono,
        sample_rate,
    )
    if np.any(active):
        active_samples = []
        for frame_index in np.flatnonzero(active):
            start = int(frame_index) * hop_length
            active_samples.append(mono[start:min(mono.size, start + frame_length)])
        joined_active = np.concatenate(active_samples) if active_samples else mono
        active_rms = float(np.sqrt(np.mean(np.square(joined_active, dtype=np.float64))))
    else:
        active_rms = 0.0
    active_rms_dbfs = _to_dbfs(active_rms)

    warnings = []
    errors = []
    if duration > MAX_REFERENCE_SECONDS:
        errors.append(f"参考音频超过 {MAX_REFERENCE_SECONDS:.0f} 秒，请先选择一段较短、语气明确的人声")
    if duration < 1.0:
        errors.append("参考音频短于 1 秒，无法稳定提取音色")
    elif duration < 2.0:
        warnings.append("参考音频偏短，建议使用 3 到 10 秒的完整自然语句")
    if not np.any(active):
        errors.append("没有检测到清晰人声，请检查文件内容或录音音量")
    elif silence_ratio >= 0.65:
        warnings.append("静音占比较高，建议使用“整理参考音频”去除首尾空白")
    if longest >= 2.0:
        warnings.append(f"存在约 {longest:.1f} 秒的连续静音，可能削弱音色和语气稳定性")
    if sample_rate < 16000:
        warnings.append("采样率低于 16 kHz，声音细节可能不足")
    if peak_dbfs < -24.0:
        warnings.append("录音峰值较低，可能同时放大底噪")
    if active_rms_dbfs < -35.0:
        warnings.append("有效人声音量较低，建议重新选择更清晰的录音")
    if clipping_ratio >= 0.0005 or peak_dbfs >= -0.05:
        warnings.append("检测到接近满刻度的采样，参考音频可能已经削波")
    if channels > 2:
        warnings.append("多声道音频会在生成前合并为单声道")

    return ReferenceAudioAnalysis(
        path=path,
        duration_seconds=duration,
        analyzed_seconds=analyzed_seconds,
        sample_rate=sample_rate,
        channels=channels,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        active_rms_dbfs=active_rms_dbfs,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        longest_silence_seconds=longest,
        leading_silence_seconds=leading,
        trailing_silence_seconds=trailing,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def prepare_reference_audio(
    path: str | Path,
    output_dir: str | Path,
    *,
    padding_seconds: float = 0.12,
    fade_seconds: float = 0.01,
) -> PreparedReferenceAudio:
    """Create a trimmed mono WAV without overwriting the uploaded source."""

    source = Path(path).expanduser().resolve()
    try:
        source_info = sf.info(str(source))
    except (RuntimeError, OSError) as exc:
        raise ValueError("无法读取参考音频，请重新上传 WAV、MP3 或 FLAC 文件") from exc
    if source_info.duration > MAX_PREPARATION_SOURCE_SECONDS:
        raise ValueError("参考音频超过 60 秒，请先在本地选取需要的短句")

    audio, sample_rate, _channels, duration = _read_audio(str(source), None)
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    active, frame_length, hop_length, _ratio, _longest, _leading, _trailing = _silence_metrics(
        mono,
        sample_rate,
    )
    if not np.any(active):
        raise ValueError("没有检测到清晰人声，无法自动整理")

    first_active = int(np.argmax(active))
    last_active = int(len(active) - 1 - np.argmax(active[::-1]))
    padding = int(round(max(0.0, padding_seconds) * sample_rate))
    start = max(0, first_active * hop_length - padding)
    end = min(mono.size, last_active * hop_length + frame_length + padding)
    prepared = mono[start:end].copy()
    prepared_duration = prepared.size / sample_rate
    if prepared_duration > MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"去除首尾静音后仍有 {prepared_duration:.1f} 秒，请选择不超过 {MAX_REFERENCE_SECONDS:.0f} 秒的片段"
        )

    fade_samples = min(int(round(max(0.0, fade_seconds) * sample_rate)), prepared.size // 2)
    if fade_samples > 1:
        fade = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
        prepared[:fade_samples] *= fade
        prepared[-fade_samples:] *= fade[::-1]

    peak = float(np.max(np.abs(prepared))) if prepared.size else 0.0
    peak_gain = 1.0
    target_peak = 10.0 ** (-1.0 / 20.0)
    if peak > target_peak:
        peak_gain = target_peak / peak
        prepared *= peak_gain

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{start}:{end}".encode("utf-8")
    output_path = output_root / f"reference-{hashlib.sha256(identity).hexdigest()[:20]}.wav"
    sf.write(str(output_path), prepared, sample_rate, subtype="PCM_16")

    return PreparedReferenceAudio(
        source_path=str(source),
        output_path=str(output_path),
        original_duration_seconds=duration,
        output_duration_seconds=prepared_duration,
        removed_leading_seconds=start / sample_rate,
        removed_trailing_seconds=max(0, mono.size - end) / sample_rate,
        peak_gain_db=_to_dbfs(peak_gain),
    )
