"""Final audio output processing for IndexTTS waveforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AudioOutputStats:
    input_peak: float
    input_peak_dbfs: float
    output_peak: float
    output_peak_dbfs: float
    input_rms_dbfs: float
    output_rms_dbfs: float
    gain: float
    gain_db: float
    samples_above_full_scale: int
    non_finite_samples: int

    @property
    def peak_reduced(self) -> bool:
        return self.gain < 1.0

    @property
    def peak_amplified(self) -> bool:
        return self.gain > 1.0


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def finalize_waveform(
    waveform,
    *,
    target_peak_dbfs: float = -1.0,
    normalize_peak: bool = False,
    max_amplification_db: float = 6.0,
) -> tuple[np.ndarray, AudioOutputStats]:
    """Safely convert a floating-point waveform to PCM int16.

    The whole generated waveform is processed at once. Signals above the target
    peak are reduced with one linear gain, preserving their shape instead of
    hard-clipping individual segments. When ``normalize_peak`` is enabled,
    quieter generated audio can use otherwise-unused headroom, capped by
    ``max_amplification_db``. This is still a single linear gain: it does not
    compress the signal or change its dynamics.
    """

    target_peak_dbfs = float(target_peak_dbfs)
    max_amplification_db = float(max_amplification_db)
    if not math.isfinite(target_peak_dbfs) or target_peak_dbfs > 0.0:
        raise ValueError("target_peak_dbfs must be a finite value at or below 0 dBFS")
    if not math.isfinite(max_amplification_db) or max_amplification_db < 0.0:
        raise ValueError("max_amplification_db must be a finite non-negative value")

    samples = np.asarray(waveform)
    if samples.size == 0:
        raise ValueError("waveform must contain at least one sample")

    samples = samples.astype(np.float32, copy=True)
    finite_mask = np.isfinite(samples)
    non_finite_samples = int(samples.size - np.count_nonzero(finite_mask))
    if non_finite_samples:
        samples[~finite_mask] = 0.0

    absolute = np.abs(samples)
    input_peak = float(np.max(absolute))
    input_rms = _rms(samples)
    samples_above_full_scale = int(np.count_nonzero(absolute > 1.0))
    target_peak = 10.0 ** (target_peak_dbfs / 20.0)

    if input_peak > 0.0:
        peak_gain = target_peak / input_peak
        if normalize_peak:
            max_gain = 10.0 ** (max_amplification_db / 20.0)
            gain = min(peak_gain, max_gain)
        else:
            gain = min(1.0, peak_gain)
    else:
        gain = 1.0

    processed = samples * gain
    # Numerical safety only: the linear gain above should already keep all
    # samples within the target peak without waveform-shape clipping.
    processed = np.clip(processed, -target_peak, target_peak)
    output_peak = float(np.max(np.abs(processed)))
    output_rms = _rms(processed)

    pcm16 = np.rint(processed * 32767.0).astype(np.int16)
    stats = AudioOutputStats(
        input_peak=input_peak,
        input_peak_dbfs=_to_dbfs(input_peak),
        output_peak=output_peak,
        output_peak_dbfs=_to_dbfs(output_peak),
        input_rms_dbfs=_to_dbfs(input_rms),
        output_rms_dbfs=_to_dbfs(output_rms),
        gain=gain,
        gain_db=_to_dbfs(gain),
        samples_above_full_scale=samples_above_full_scale,
        non_finite_samples=non_finite_samples,
    )
    return pcm16, stats


def describe_output_stats(stats: AudioOutputStats) -> str:
    input_db = f"{stats.input_peak_dbfs:.2f}" if math.isfinite(stats.input_peak_dbfs) else "-inf"
    output_db = f"{stats.output_peak_dbfs:.2f}" if math.isfinite(stats.output_peak_dbfs) else "-inf"
    input_rms = f"{stats.input_rms_dbfs:.2f}" if math.isfinite(stats.input_rms_dbfs) else "-inf"
    output_rms = f"{stats.output_rms_dbfs:.2f}" if math.isfinite(stats.output_rms_dbfs) else "-inf"
    return (
        f"input peak {input_db} dBFS, output peak {output_db} dBFS, "
        f"input RMS {input_rms} dBFS, output RMS {output_rms} dBFS, "
        f"gain {stats.gain_db:.2f} dB, "
        f"samples above full scale {stats.samples_above_full_scale}, "
        f"non-finite samples {stats.non_finite_samples}"
    )
