"""Tests for speaker-word alignment, first-appearance ordering, and turn merging."""

from __future__ import annotations

import unittest

from voice_digest.alignment import align_and_merge_turns
from voice_digest.types import ASRSegment, DiarizationSegment, Word


class TestAlignmentAndMerging(unittest.TestCase):
    def test_first_appearance_speaker_relabeling(self) -> None:
        # SPEAKER_02 speaks first -> must become 'speaker 1'
        # SPEAKER_00 speaks second -> must become 'speaker 2'
        asr = [
            ASRSegment(
                text="Hello there.",
                start=0.0,
                end=1.0,
                words=[
                    Word("Hello", 0.0, 0.4),
                    Word("there.", 0.5, 1.0),
                ],
            ),
            ASRSegment(
                text="Hi back.",
                start=1.5,
                end=2.5,
                words=[
                    Word("Hi", 1.5, 1.9),
                    Word("back.", 2.0, 2.5),
                ],
            ),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=1.1, speaker="SPEAKER_02"),
            DiarizationSegment(start=1.4, end=2.6, speaker="SPEAKER_00"),
        ]

        turns = align_and_merge_turns(asr, diar, pause_threshold=2.0)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker, "speaker 1")
        self.assertEqual(turns[0].text, "Hello there.")
        self.assertEqual(turns[1].speaker, "speaker 2")
        self.assertEqual(turns[1].text, "Hi back.")

    def test_same_speaker_merges_within_pause_threshold_and_splits_above(self) -> None:
        asr = [
            ASRSegment(
                text="Part one.",
                start=0.0,
                end=1.0,
                words=[Word("Part", 0.0, 0.4), Word("one.", 0.5, 1.0)],
            ),
            # 1.2s gap -> should merge (<= 2.0s)
            ASRSegment(
                text="Part two.",
                start=2.2,
                end=3.2,
                words=[Word("Part", 2.2, 2.6), Word("two.", 2.7, 3.2)],
            ),
            # 2.5s gap -> should split (> 2.0s)
            ASRSegment(
                text="After long pause.",
                start=5.7,
                end=7.0,
                words=[
                    Word("After", 5.7, 6.0),
                    Word("long", 6.1, 6.4),
                    Word("pause.", 6.5, 7.0),
                ],
            ),
        ]
        diar = [DiarizationSegment(start=0.0, end=7.5, speaker="SPEAKER_00")]

        turns = align_and_merge_turns(asr, diar, pause_threshold=2.0)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].text, "Part one. Part two.")
        self.assertEqual(turns[1].text, "After long pause.")

    def test_sentence_smoothing_prevents_mid_sentence_speaker_flip(self) -> None:
        # "Nice to meet you." has "you." slightly overlapping SPEAKER_01,
        # but sentence-level majority vote must keep "Nice to meet you." intact on speaker 1!
        asr = [
            ASRSegment(
                text="Nice to meet you.",
                start=0.0,
                end=1.2,
                words=[
                    Word("Nice", 0.0, 0.3),
                    Word("to", 0.32, 0.5),
                    Word("meet", 0.52, 0.9),
                    Word("you.", 0.92, 1.2),
                ],
            ),
            ASRSegment(
                text="Nice to meet you too.",
                start=1.25,
                end=2.5,
                words=[
                    Word("Nice", 1.25, 1.5),
                    Word("to", 1.52, 1.7),
                    Word("meet", 1.72, 2.0),
                    Word("you", 2.02, 2.2),
                    Word("too.", 2.22, 2.5),
                ],
            ),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=0.95, speaker="SPEAKER_00"),
            DiarizationSegment(start=0.95, end=2.6, speaker="SPEAKER_01"),
        ]

        turns = align_and_merge_turns(asr, diar, pause_threshold=2.0)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker, "speaker 1")
        self.assertEqual(turns[0].text, "Nice to meet you.")
        self.assertEqual(turns[1].speaker, "speaker 2")
        self.assertEqual(turns[1].text, "Nice to meet you too.")

    def test_llm_repair_json_parser_and_turn_remerging(self) -> None:
        from voice_digest.llm_repair import _merge_repaired_sentences, _parse_speaker_json

        mapping = _parse_speaker_json(
            '```json\n{"0": 1, "1": "speaker 2", "2": 99}\n```', num_speakers=2
        )
        self.assertEqual(mapping, {0: 1, 1: 2})

        merged = _merge_repaired_sentences(
            [
                ("speaker 1", [Word("How", 0.0, 0.2), Word("are", 0.25, 0.4), Word("you?", 0.45, 0.7)]),
                ("speaker 2", [Word("Good!", 0.8, 1.1)]),
            ],
            pause_threshold=2.0,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].speaker, "speaker 1")
        self.assertEqual(merged[1].speaker, "speaker 2")


if __name__ == "__main__":
    unittest.main()
