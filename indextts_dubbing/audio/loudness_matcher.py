"""Conservative reference-level and delivery-format matching for generated speech."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from indextts_dubbing.audio.reference_analyzer import (
    SILENCE_THRESHOLD_DBFS,
    analyze_reference_audio,
)


@dataclass(frozen=True)
class LoudnessMatchStats:
    reference_active_rms_dbfs: float
    input_active_rms_dbfs: float
    output_active_rms_dbfs: float
    input_peak_dbfs: float
    output_peak_dbfs: float
    requested_gain_db: float
    applied_gain_db: float
    peak_limited: bool
    gain_limited: bool
    input_sample_rate: int
    output_sample_rate: int
    input_channels: int
    output_channels: int
    reference_sample_rate: int | None = None
    reference_channels: int | None = None
    format_matched: bool = False

    def summary(self) -> str:
        limits = []
        if self.gain_limited:
            limits.append("增益上限")
        if self.peak_limited:
            limits.append("峰值保护")
        limit_text = f" · 受{'、'.join(limits)}限制" if limits else ""
        format_text = ""
        if self.format_matched:
            format_text = (
                f" · 格式 {self.input_sample_rate / 1000:.1f}→"
                f"{self.output_sample_rate / 1000:.1f} kHz · "
                f"{self.input_channels}→{self.output_channels} 声道"
            )
        return (
            f"参考有效人声 {self.reference_active_rms_dbfs:.1f} dBFS · "
            f"输出 {self.input_active_rms_dbfs:.1f} → {self.output_active_rms_dbfs:.1f} dBFS · "
            f"增益 {self.applied_gain_db:+.1f} dB · 峰值 {self.output_peak_dbfs:.1f} dBFS"
            f"{format_text}{limit_text}"
        )


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


def _as_audio_matrix(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("音频没有可处理的采样")
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if samples.ndim == 1:
        return samples[:, None]
    if samples.ndim == 2:
        return samples
    raise ValueError("音频维度不受支持")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    samples = _as_audio_matrix(audio)
    return np.mean(samples, axis=1, dtype=np.float64).astype(np.float32)


def _channel_count(audio: np.ndarray) -> int:
    return int(_as_audio_matrix(audio).shape[1])


def active_rms_dbfs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure RMS from speech-active 20 ms frames using a fixed noise gate."""

    if int(sample_rate) <= 0:
        raise ValueError("采样率必须大于 0")
    mono = _to_mono(audio)
    frame_length = max(1, int(round(sample_rate * 0.02)))
    hop_length = max(1, int(round(sample_rate * 0.01)))
    threshold = 10.0 ** (SILENCE_THRESHOLD_DBFS / 20.0)

    if mono.size <= frame_length:
        frames = [np.pad(mono, (0, frame_length - mono.size))]
    else:
        frames = [
            mono[start:start + frame_length]
            for start in range(0, mono.size - frame_length + 1, hop_length)
        ]

    active_frames = []
    for frame in frames:
        frame_rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        if frame_rms > threshold:
            active_frames.append(frame)
    if not active_frames:
        return float("-inf")

    active_samples = np.concatenate(active_frames)
    active_rms = float(np.sqrt(np.mean(np.square(active_samples, dtype=np.float64))))
    return _to_dbfs(active_rms)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = _as_audio_matrix(audio)
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("采样率必须大于 0")
    if source_rate == target_rate:
        return samples.copy()
    divisor = math.gcd(source_rate, target_rate)
    converted = resample_poly(
        samples,
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
    )
    return np.asarray(converted, dtype=np.float32)


def _match_channel_count(audio: np.ndarray, target_channels: int) -> np.ndarray:
    samples = _as_audio_matrix(audio)
    target_channels = int(target_channels)
    if target_channels not in (1, 2):
        raise ValueError("当前仅支持匹配单声道或双声道")
    current_channels = samples.shape[1]
    if current_channels == target_channels:
        return samples.copy()
    mono = np.mean(samples, axis=1, dtype=np.float64).astype(np.float32)[:, None]
    if target_channels == 1:
        return mono
    return np.repeat(mono, 2, axis=1)


def _apply_peak_ceiling(
    audio: np.ndarray,
    target_peak_dbfs: float,
) -> tuple[np.ndarray, float, bool]:
    samples = _as_audio_matrix(audio)
    peak = float(np.max(np.abs(samples)))
    target_peak = 10.0 ** (float(target_peak_dbfs) / 20.0)
    if peak <= 0.0 or peak <= target_peak:
        return samples, 0.0, False
    gain = target_peak / peak
    gain_db = _to_dbfs(gain)
    protected = samples * gain
    return np.clip(protected, -target_peak, target_peak).astype(np.float32), gain_db, True


def match_waveform_to_reference_level(
    waveform: np.ndarray,
    sample_rate: int,
    reference_active_rms_dbfs: float,
    *,
    target_peak_dbfs: float = -1.0,
    max_amplification_db: float = 6.0,
    max_attenuation_db: float = 12.0,
) -> tuple[np.ndarray, LoudnessMatchStats]:
    """Apply one bounded linear gain; never compress or hard-clip the signal."""

    sample_rate = int(sample_rate)
    target_peak_dbfs = float(target_peak_dbfs)
    max_amplification_db = float(max_amplification_db)
    max_attenuation_db = float(max_attenuation_db)
    reference_active_rms_dbfs = float(reference_active_rms_dbfs)
    if not math.isfinite(reference_active_rms_dbfs):
        raise ValueError("参考音频没有可用的有效人声音量")
    if not math.isfinite(target_peak_dbfs) or target_peak_dbfs > 0.0:
        raise ValueError("目标峰值必须是有限且不高于 0 dBFS 的数值")
    if max_amplification_db < 0.0 or max_attenuation_db < 0.0:
        raise ValueError("增益限制不能为负数")

    original_ndim = np.asarray(waveform).ndim
    samples = _as_audio_matrix(waveform)
    input_active_rms_dbfs = active_rms_dbfs(samples, sample_rate)
    if not math.isfinite(input_active_rms_dbfs):
        raise ValueError("生成音频没有检测到可用的人声")
    input_peak = float(np.max(np.abs(samples)))
    input_peak_dbfs = _to_dbfs(input_peak)

    requested_gain_db = reference_active_rms_dbfs - input_active_rms_dbfs
    bounded_gain_db = min(
        max(requested_gain_db, -max_attenuation_db),
        max_amplification_db,
    )
    peak_headroom_db = target_peak_dbfs - input_peak_dbfs
    applied_gain_db = min(bounded_gain_db, peak_headroom_db)
    gain = 10.0 ** (applied_gain_db / 20.0)

    processed = samples * gain
    target_peak = 10.0 ** (target_peak_dbfs / 20.0)
    # Numerical safety only. The headroom calculation above should already
    # keep the waveform inside the ceiling without waveform-shape clipping.
    processed = np.clip(processed, -target_peak, target_peak).astype(np.float32, copy=False)
    output_peak_dbfs = _to_dbfs(float(np.max(np.abs(processed))))
    output_active_rms_dbfs = active_rms_dbfs(processed, sample_rate)
    channels = int(samples.shape[1])

    stats = LoudnessMatchStats(
        reference_active_rms_dbfs=reference_active_rms_dbfs,
        input_active_rms_dbfs=input_active_rms_dbfs,
        output_active_rms_dbfs=output_active_rms_dbfs,
        input_peak_dbfs=input_peak_dbfs,
        output_peak_dbfs=output_peak_dbfs,
        requested_gain_db=requested_gain_db,
        applied_gain_db=applied_gain_db,
        peak_limited=applied_gain_db < bounded_gain_db - 1e-6,
        gain_limited=abs(bounded_gain_db - requested_gain_db) > 1e-6,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        input_channels=channels,
        output_channels=channels,
    )
    if original_ndim == 1:
        return processed[:, 0], stats
    return processed, stats


def write_loudness_matched_copy(
    source_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    target_peak_dbfs: float = -1.0,
    max_amplification_db: float = 6.0,
    max_attenuation_db: float = 12.0,
    match_reference_format: bool = False,
) -> tuple[str, LoudnessMatchStats]:
    """Write a non-destructive level or delivery-matched WAV beside the original."""

    source = Path(source_path)
    reference = analyze_reference_audio(reference_path)
    if not reference.usable:
        raise ValueError("参考音频不适合进行响度匹配：" + "；".join(reference.errors))

    try:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError("无法读取刚生成的音频，不能创建响度匹配版本") from exc

    processed, stats = match_waveform_to_reference_level(
        audio,
        int(sample_rate),
        reference.active_rms_dbfs,
        target_peak_dbfs=target_peak_dbfs,
        max_amplification_db=max_amplification_db,
        max_attenuation_db=max_attenuation_db,
    )
    output_sample_rate = int(sample_rate)
    if match_reference_format:
        output_sample_rate = int(reference.sample_rate)
        target_channels = 1 if int(reference.channels) == 1 else 2
        processed = _resample_audio(processed, int(sample_rate), output_sample_rate)
        processed = _match_channel_count(processed, target_channels)
        processed, safety_gain_db, resample_peak_limited = _apply_peak_ceiling(
            processed,
            target_peak_dbfs,
        )
        stats = replace(
            stats,
            output_active_rms_dbfs=active_rms_dbfs(processed, output_sample_rate),
            output_peak_dbfs=_to_dbfs(float(np.max(np.abs(processed)))),
            applied_gain_db=stats.applied_gain_db + safety_gain_db,
            peak_limited=stats.peak_limited or resample_peak_limited,
            output_sample_rate=output_sample_rate,
            output_channels=int(processed.shape[1]),
            reference_sample_rate=int(reference.sample_rate),
            reference_channels=int(reference.channels),
            format_matched=True,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), processed, output_sample_rate, subtype="PCM_16")
    return str(destination), stats
