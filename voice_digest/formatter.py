"""Transcript formatting for base mode and --prosody metadata modes."""

from __future__ import annotations

from typing import Optional, Sequence

from local_transcribe.types import SessionSummary, Turn, TurnProsody

VALID_METADATA_MODES = ("both", "summary-only", "inline-only")


def format_duration(seconds: float) -> str:
    """Format duration as '58m 14s' or '1h 02m 14s'."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _format_disfluency_tag(p: TurnProsody) -> Optional[str]:
    parts: list[str] = []
    if p.block_count > 0:
        unit = "block" if p.block_count == 1 else "blocks"
        parts.append(f"{p.block_count} {unit}")
    if p.prolongation_count > 0:
        unit = "prolongation" if p.prolongation_count == 1 else "prolongations"
        parts.append(f"{p.prolongation_count} {unit}")
    if p.repetition_count > 0:
        unit = "repetition" if p.repetition_count == 1 else "repetitions"
        parts.append(f"{p.repetition_count} {unit}")
    if not parts:
        return None
    return "disfluency: " + ", ".join(parts)


def format_inline_metadata(p: TurnProsody) -> str:
    """Render inline turn metadata matching Spec Section 5.2."""
    wpm_int = int(round(p.wpm))
    pace_str = f"pace: {wpm_int} WPM (fast)" if p.is_fast else f"pace: {wpm_int} WPM"
    fields = [pace_str]
    disf = _format_disfluency_tag(p)
    if disf:
        fields.append(disf)
    fields.append(f"state: {p.state}")
    return "[" + " | ".join(fields) + "]"


def format_session_summary(summary: SessionSummary) -> str:
    """Render the session acoustic summary block matching Spec Section 5.2."""
    lines = [
        "=== SESSION ACOUSTIC SUMMARY ===",
        f"Duration: {format_duration(summary.duration_seconds)} | Speakers: {summary.speaker_count}",
    ]
    pace_col_width = max(
        (len(f"{int(round(s.avg_wpm))} WPM ({s.pace_label})") for s in summary.speakers),
        default=0,
    )
    for spk in summary.speakers:
        pace_val = f"{int(round(spk.avg_wpm))} WPM ({spk.pace_label})"
        lines.append(
            f"{spk.speaker}: Avg Pace: {pace_val:<{pace_col_width}} | "
            f"Acoustic Disfluencies: {spk.total_disfluencies} | "
            f"Elevated Stress Turns: {int(round(spk.elevated_stress_pct))}%"
        )
    lines.append("================================")
    return "\n".join(lines)


def format_ai_summary(summary_text: str) -> str:
    """Render the local AI summary header block."""
    body = summary_text.strip()
    return "\n".join([
        "=== AI SUMMARY ===",
        body,
        "==================",
    ])


def format_transcript(
    turns: Sequence[Turn],
    prosody: bool = False,
    metadata_mode: str = "both",
    summary: Optional[SessionSummary] = None,
    ai_summary: Optional[str] = None,
) -> str:
    """Format turns into the final .txt transcript string."""
    if metadata_mode not in VALID_METADATA_MODES:
        raise ValueError(
            f"Invalid metadata_mode '{metadata_mode}'. Choose from {VALID_METADATA_MODES}."
        )

    include_summary = prosody and metadata_mode in ("both", "summary-only") and summary is not None
    include_inline = prosody and metadata_mode in ("both", "inline-only")

    sections: list[str] = []
    if ai_summary and ai_summary.strip():
        sections.append(format_ai_summary(ai_summary))
    if include_summary and summary is not None:
        sections.append(format_session_summary(summary))

    turn_lines: list[str] = []
    for turn in turns:
        text = turn.text.strip()
        if not text:
            continue
        if include_inline and turn.prosody is not None:
            tag = format_inline_metadata(turn.prosody)
            turn_lines.append(f"{turn.speaker} {tag}: {text}")
        else:
            turn_lines.append(f"{turn.speaker}: {text}")

    if turn_lines:
        sections.append("\n".join(turn_lines))

    return "\n\n".join(sections) + ("\n" if sections else "")
