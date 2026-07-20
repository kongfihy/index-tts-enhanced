"""Audio analysis, preparation, and conservative output matching helpers."""

from .reference_analyzer import (
    PreparedReferenceAudio,
    ReferenceAudioAnalysis,
    analyze_reference_audio,
    prepare_reference_audio,
)
from .loudness_matcher import (
    LoudnessMatchStats,
    active_rms_dbfs,
    match_waveform_to_reference_level,
    write_loudness_matched_copy,
)

__all__ = [
    "LoudnessMatchStats",
    "PreparedReferenceAudio",
    "ReferenceAudioAnalysis",
    "active_rms_dbfs",
    "analyze_reference_audio",
    "match_waveform_to_reference_level",
    "prepare_reference_audio",
    "write_loudness_matched_copy",
]
