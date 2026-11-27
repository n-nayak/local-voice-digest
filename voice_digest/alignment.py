"""Speaker-word alignment, first-appearance speaker relabeling, and turn merging."""

from __future__ import annotations

import re
from typing import Sequence

from voice_digest.types import ASRSegment, DiarizationSegment, Turn, Word

_LEADING_PUNCT_RE = re.compile(r"^[,.;:!?)\]}]+")
_SENTENCE_END_CHARS = (".", "?", "!")


def _smooth_sentence_speakers(
    raw_labeled_words: list[tuple[Word, str]],
    sentence_gap_sec: float = 1.0,
) -> list[tuple[Word, str]]:
    """Assign a single majority-vote speaker per sentence so sentences aren't torn in half."""
    if len(raw_labeled_words) <= 1:
        return raw_labeled_words

    # Partition into sentence groups at . ? ! or pauses >= sentence_gap_sec
    groups: list[list[tuple[Word, str]]] = []
    current: list[tuple[Word, str]] = [raw_labeled_words[0]]
    for item in raw_labeled_words[1:]:
        prev_word, _ = current[-1]
        curr_word, _ = item
        gap = curr_word.start - prev_word.end
        if prev_word.text.strip().endswith(_SENTENCE_END_CHARS) or gap >= sentence_gap_sec:
            groups.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        groups.append(current)

    smoothed: list[tuple[Word, str]] = []
    for grp in groups:
        # Duration-weighted majority vote per sentence
        weights: dict[str, float] = {}
        for word, spk in grp:
            weights[spk] = weights.get(spk, 0.0) + max(0.05, word.duration)
        winner = max(weights.items(), key=lambda kv: kv[1])[0]
        for word, _ in grp:
            smoothed.append((word, winner))
    return smoothed


def _compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _assign_raw_speaker(
    word: Word, diarization_segments: Sequence[DiarizationSegment]
) -> str:
    """Assign the best-matching diarization speaker to a word by overlap or proximity."""
    if not diarization_segments:
        return "SPEAKER_00"

    best_speaker = ""
    best_overlap = 0.0
    for seg in diarization_segments:
        ov = _compute_overlap(word.start, word.end, seg.start, seg.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = seg.speaker

    if best_overlap > 0.0:
        return best_speaker

    # Fallback: nearest diarization segment to the word midpoint
    midpoint = 0.5 * (word.start + word.end)
    nearest = min(
        diarization_segments,
        key=lambda seg: (
            0.0
            if seg.start <= midpoint <= seg.end
            else min(abs(midpoint - seg.start), abs(midpoint - seg.end))
        ),
    )
    return nearest.speaker


def _join_word_texts(words: Sequence[Word]) -> str:
    """Join word tokens cleanly without extraneous spaces before punctuation."""
    parts: list[str] = []
    for word in words:
        tok = word.text.strip()
        if not tok:
            continue
        if parts and _LEADING_PUNCT_RE.match(tok):
            parts[-1] = parts[-1] + tok
        else:
            parts.append(tok)
    return " ".join(parts)


def align_and_merge_turns(
    asr_segments: Sequence[ASRSegment],
    diarization_segments: Sequence[DiarizationSegment],
    pause_threshold: float = 2.0,
) -> list[Turn]:
    """Map ASR words to diarized speakers, relabel by first appearance, and merge turns.

    Speakers are assigned consistent labels ('speaker 1', 'speaker 2', ..., 'speaker N')
    strictly ordered by their first appearance in the audio. Consecutive words from the
    same speaker are merged into a single turn unless separated by a silence gap exceeding
    `pause_threshold` seconds.
    """
    # Flatten all words in chronological order
    all_words: list[Word] = []
    for seg in asr_segments:
        if seg.words:
            all_words.extend(seg.words)
        elif seg.text.strip():
            tokens = seg.text.strip().split()
            dur = max(0.01, seg.end - seg.start)
            step = dur / len(tokens)
            for idx, tok in enumerate(tokens):
                all_words.append(
                    Word(
                        text=tok,
                        start=seg.start + idx * step,
                        end=seg.start + (idx + 1) * step,
                    )
                )

    all_words.sort(key=lambda w: (w.start, w.end))
    if not all_words:
        return []

    # Assign raw speakers per word, then smooth to sentence boundaries
    raw_pairs = [
        (word, _assign_raw_speaker(word, diarization_segments))
        for word in all_words
    ]
    smoothed_pairs = _smooth_sentence_speakers(raw_pairs)

    # Map raw diarizer IDs to 'speaker 1', 'speaker 2', ... by first appearance
    speaker_map: dict[str, str] = {}
    labeled_words: list[Word] = []
    for word, raw_speaker in smoothed_pairs:
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"speaker {len(speaker_map) + 1}"
        labeled_words.append(
            Word(
                text=word.text,
                start=word.start,
                end=word.end,
                score=word.score,
                speaker=speaker_map[raw_speaker],
            )
        )

    # Merge consecutive words from the same speaker unless gap > pause_threshold
    turns: list[Turn] = []
    current_words: list[Word] = [labeled_words[0]]

    for word in labeled_words[1:]:
        prev = current_words[-1]
        gap = word.start - prev.end
        same_speaker = word.speaker == prev.speaker
        if same_speaker and gap <= pause_threshold:
            current_words.append(word)
        else:
            turns.append(
                Turn(
                    speaker=current_words[0].speaker or "speaker 1",
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=_join_word_texts(current_words),
                    words=current_words,
                )
            )
            current_words = [word]

    if current_words:
        turns.append(
            Turn(
                speaker=current_words[0].speaker or "speaker 1",
                start=current_words[0].start,
                end=current_words[-1].end,
                text=_join_word_texts(current_words),
                words=current_words,
            )
        )

    return turns
