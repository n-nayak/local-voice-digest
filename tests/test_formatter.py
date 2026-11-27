"""Tests for base and --prosody output formatting compliance."""

from __future__ import annotations

import unittest

from voice_digest.formatter import format_transcript
from voice_digest.types import SessionSummary, SpeakerSummary, Turn, TurnProsody


class TestOutputFormatter(unittest.TestCase):
    def setUp(self) -> None:
        self.turns = [
            Turn(
                speaker="speaker 1",
                start=0.0,
                end=4.0,
                text="Good morning, thanks for making time to go over the project timeline today.",
                prosody=TurnProsody(wpm=135.2, syllables_per_sec=3.1, is_fast=False, state="calm"),
            ),
            Turn(
                speaker="speaker 2",
                start=4.2,
                end=8.5,
                text="Happy to be here. I have a few concerns about the deployment window we discussed last week.",
                prosody=TurnProsody(
                    wpm=192.4,
                    syllables_per_sec=4.6,
                    is_fast=True,
                    block_count=2,
                    state="tense",
                ),
            ),
        ]
        self.summary = SessionSummary(
            duration_seconds=58 * 60 + 14,
            speaker_count=2,
            speakers=[
                SpeakerSummary("speaker 1", 138.0, "Normal", 2, 5.0, 20),
                SpeakerSummary("speaker 2", 184.0, "Fast", 11, 34.0, 20),
            ],
        )

    def test_base_format_compliance(self) -> None:
        out = format_transcript(self.turns, prosody=False)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("speaker 1: Good morning"))
        self.assertTrue(lines[1].startswith("speaker 2: Happy to be here."))
        self.assertNotIn("SESSION ACOUSTIC SUMMARY", out)
        self.assertNotIn("[pace:", out)

    def test_prosody_both_mode(self) -> None:
        out = format_transcript(
            self.turns, prosody=True, metadata_mode="both", summary=self.summary
        )
        self.assertIn("=== SESSION ACOUSTIC SUMMARY ===", out)
        self.assertIn("Duration: 58m 14s | Speakers: 2", out)
        self.assertIn("speaker 1: Avg Pace: 138 WPM (Normal)", out)
        self.assertIn("Acoustic Disfluencies: 11 | Elevated Stress Turns: 34%", out)
        self.assertIn(
            "speaker 1 [pace: 135 WPM | state: calm]: Good morning",
            out,
        )
        self.assertIn(
            "speaker 2 [pace: 192 WPM (fast) | disfluency: 2 blocks | state: tense]: Happy to be here.",
            out,
        )

    def test_prosody_summary_only_and_inline_only(self) -> None:
        summary_only = format_transcript(
            self.turns, prosody=True, metadata_mode="summary-only", summary=self.summary
        )
        self.assertIn("=== SESSION ACOUSTIC SUMMARY ===", summary_only)
        self.assertNotIn("[pace:", summary_only)
        self.assertIn("speaker 1: Good morning", summary_only)

        inline_only = format_transcript(
            self.turns, prosody=True, metadata_mode="inline-only", summary=self.summary
        )
        self.assertNotIn("=== SESSION ACOUSTIC SUMMARY ===", inline_only)
        self.assertIn("speaker 2 [pace: 192 WPM (fast) | disfluency: 2 blocks | state: tense]:", inline_only)

    def test_ai_summary_prepended_and_map_reduce_chunking(self) -> None:
        from voice_digest.llm_repair import _chunk_text_for_summary

        out = format_transcript(
            self.turns,
            prosody=False,
            ai_summary="- Agreed to review deployment window.\n- Follow-up on timeline blockers.",
        )
        self.assertTrue(out.startswith("=== AI SUMMARY ===\n"))
        self.assertIn("- Agreed to review deployment window.", out)
        self.assertIn("speaker 1: Good morning", out)

        chunks = _chunk_text_for_summary([f"speaker 1: sentence {i}" for i in range(100)], max_chars=250)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c), 250)


if __name__ == "__main__":
    unittest.main()
