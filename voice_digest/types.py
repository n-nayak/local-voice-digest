"""Core data models for transcription, diarization, and prosodic analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """Single aligned word with timestamps in seconds."""

    text: str
    start: float
    end: float
    score: float = 1.0
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ASRSegment:
    """Raw ASR segment with optional word-level forced alignment."""

    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class DiarizationSegment:
    """Speaker diarization span in seconds."""

    start: float
    end: float
    speaker: str


@dataclass
class TurnProsody:
    """Acoustic and temporal biomarkers computed for a single speaker turn."""

    wpm: float
    syllables_per_sec: float
    is_fast: bool = False
    block_count: int = 0
    prolongation_count: int = 0
    repetition_count: int = 0
    stress_score: float = 0.0
    state: str = "calm"  # "calm", "elevated", or "tense"
    ser_anxious_score: Optional[float] = None
    f0_mean: Optional[float] = None
    jitter: Optional[float] = None
    spectral_flux: Optional[float] = None
    rapid_breath_intakes: int = 0

    @property
    def total_disfluencies(self) -> int:
        return self.block_count + self.prolongation_count + self.repetition_count

    @property
    def is_elevated_stress(self) -> bool:
        return self.state in ("elevated", "tense")


@dataclass
class Turn:
    """Merged continuous speaker turn."""

    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    prosody: Optional[TurnProsody] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerSummary:
    """Session-level aggregated acoustic metrics for a single speaker."""

    speaker: str
    avg_wpm: float
    pace_label: str  # "Slow", "Normal", "Fast"
    total_disfluencies: int
    elevated_stress_pct: float
    turn_count: int


@dataclass
class SessionSummary:
    """Session-level acoustic summary header."""

    duration_seconds: float
    speaker_count: int
    speakers: list[SpeakerSummary] = field(default_factory=list)
