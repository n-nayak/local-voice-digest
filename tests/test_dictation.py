"""Unit tests for the local dictation engine, Qwen3 helpers, and model paths."""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from voice_digest.dictation import (
    DictationEngine,
    build_dictation_messages,
    handle_daemon_request,
    parse_custom_vocab,
    sanitize_dictation_output,
)
from voice_digest.diarization import resolve_diarizer
from voice_digest.llm_repair import apply_chat_template_no_think, strip_thinking_blocks
from voice_digest.models import (
    DICTATION_ASR_REPO_ID,
    DICTATION_LLM_REPO_ID,
    LLM_REPAIR_REPO_ID,
    SORTFORMER_REPO_ID,
    dictation_asr_model_path,
    dictation_llm_model_path,
    llm_repair_model_path,
    sortformer_model_path,
)


def _write_test_wav(path: Path, amplitude: float, duration_sec: float = 0.5) -> None:
    sr = 16_000
    n_samples = int(sr * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            val = int(max(-1.0, min(1.0, amplitude * math.sin(2.0 * math.pi * 440.0 * i / sr))) * 32767)
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))


class DictationTests(unittest.TestCase):
    def test_model_paths_and_ungated_repo_ids(self) -> None:
        root = Path("/tmp/voicedigest-models")
        self.assertIn("Qwen3.5-4B", LLM_REPAIR_REPO_ID)
        self.assertIn("sortformer", SORTFORMER_REPO_ID)
        self.assertIn("parakeet-tdt-0.6b-v3", DICTATION_ASR_REPO_ID)
        self.assertIn("Qwen3-1.7B", DICTATION_LLM_REPO_ID)
        self.assertEqual(llm_repair_model_path(root), root / "llm" / "qwen3.5-4b-mlx-4bit")
        self.assertEqual(sortformer_model_path(root), root / "sortformer" / "sortformer-4spk-v2.1")
        self.assertEqual(dictation_asr_model_path(root), root / "dictation" / "parakeet-tdt-0.6b-v3")
        self.assertEqual(dictation_llm_model_path(root), root / "llm" / "qwen3-1.7b-4bit")

    def test_sortformer_auto_falls_back_to_sherpa_above_four_speakers(self) -> None:
        self.assertEqual(resolve_diarizer("sortformer", 2), "sortformer")
        self.assertEqual(resolve_diarizer("sortformer", 4), "sortformer")
        self.assertEqual(resolve_diarizer("sortformer", 5), "sherpa-onnx")
        self.assertEqual(resolve_diarizer("sherpa-onnx", 2), "sherpa-onnx")

    def test_sortformer_streaming_merges_chunks_and_reports_progress(self) -> None:
        import sys
        import types
        from unittest import mock

        from voice_digest.diarization import _diarize_sortformer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_path = root / "sample.wav"
            _write_test_wav(wav_path, amplitude=0.2, duration_sec=1.0)
            sf_dir = sortformer_model_path(root)
            sf_dir.mkdir(parents=True, exist_ok=True)
            (sf_dir / "config.json").write_text("{}", encoding="utf-8")

            class FakeSeg:
                def __init__(self, start: float, end: float, speaker: int) -> None:
                    self.start = start
                    self.end = end
                    self.speaker = speaker

            class FakeOut:
                def __init__(self, segments: list[FakeSeg]) -> None:
                    self.segments = segments

            class FakeSortformer:
                def generate_stream(self, _path: str, **_kwargs: object):
                    yield FakeOut([FakeSeg(0.0, 5.0, 0)])
                    yield FakeOut([FakeSeg(5.05, 8.0, 0), FakeSeg(8.5, 10.0, 1)])

            fake_vad = types.ModuleType("mlx_audio.vad")
            fake_vad.load = lambda _dir: FakeSortformer()  # type: ignore[attr-defined]
            fake_pkg = types.ModuleType("mlx_audio")
            fake_pkg.vad = fake_vad  # type: ignore[attr-defined]

            progress_calls: list[tuple[float, str]] = []
            with mock.patch.dict(sys.modules, {"mlx_audio": fake_pkg, "mlx_audio.vad": fake_vad}):
                segs = _diarize_sortformer(
                    wav_path=wav_path,
                    num_speakers=2,
                    model_dir=root,
                    progress_cb=lambda f, msg: progress_calls.append((f, msg)),
                )

            self.assertEqual(len(segs), 2)
            self.assertEqual(segs[0].speaker, "SPEAKER_00")
            self.assertAlmostEqual(segs[0].start, 0.0)
            self.assertAlmostEqual(segs[0].end, 8.0)
            self.assertEqual(segs[1].speaker, "SPEAKER_01")
            self.assertTrue(len(progress_calls) >= 2)

    def test_clamp_to_top_speakers_and_spherical_kmeans(self) -> None:
        from voice_digest.diarization import _clamp_to_top_speakers, _spherical_kmeans
        from voice_digest.types import DiarizationSegment

        segs = [
            DiarizationSegment(0.0, 10.0, "SPEAKER_00"),
            DiarizationSegment(10.5, 11.2, "SPEAKER_02"),  # phantom 3rd slot
            DiarizationSegment(12.0, 22.0, "SPEAKER_01"),
        ]
        clamped = _clamp_to_top_speakers(segs, num_speakers=2)
        self.assertEqual({s.speaker for s in clamped}, {"SPEAKER_00", "SPEAKER_01"})
        self.assertEqual(clamped[1].speaker, "SPEAKER_00")

        # Two clearly separated voiceprint clusters should separate cleanly
        embs = [
            [1.0, 0.0, 0.0],
            [0.98, 0.1, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.99, 0.0],
        ]
        labels = _spherical_kmeans(embs, k=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])

    def test_qwen3_chat_template_no_think_and_strip(self) -> None:
        class ModernTok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                return f"thinking={enable_thinking}|{messages[-1]['content']}"

        class LegacyTok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return f"legacy|{messages[-1]['content']}"

        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(apply_chat_template_no_think(ModernTok(), msgs, "fb"), "thinking=False|hello")
        self.assertEqual(apply_chat_template_no_think(LegacyTok(), msgs, "fb"), "legacy|hello")
        self.assertEqual(strip_thinking_blocks("<think>internal plan</think>\nFinal text."), "Final text.")

    def test_parse_custom_vocab_dedupes_and_splits(self) -> None:
        terms = parse_custom_vocab("VoiceDigest, sherpa-onnx\nmlx-whisper; voicedigest,  ")
        self.assertEqual(terms, ["VoiceDigest", "sherpa-onnx", "mlx-whisper"])

    def test_build_dictation_messages_includes_context_and_vocab(self) -> None:
        msgs = build_dictation_messages(
            raw_text="um let's ship voice digest with sherpa on x",
            app_name="Slack",
            window_title="#engineering - Workspace",
            vocab="VoiceDigest, sherpa-onnx",
        )
        self.assertEqual(len(msgs), 2)
        self.assertIn("Active app: Slack", msgs[0]["content"])
        self.assertIn("Window: #engineering - Workspace", msgs[0]["content"])
        self.assertIn("VoiceDigest, sherpa-onnx", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "um let's ship voice digest with sherpa on x")

    def test_sanitize_dictation_output_guards_hallucinations(self) -> None:
        raw = "Let's meet at 3 actually make it 4."
        self.assertEqual(sanitize_dictation_output(raw, '"Let\'s meet at 4."'), "Let's meet at 4.")
        self.assertEqual(sanitize_dictation_output(raw, "<think>hmm</think>Let's meet at 4."), "Let's meet at 4.")
        # Hallucinated long essay falls back to raw
        self.assertEqual(sanitize_dictation_output(raw, "A " * 200), raw)
        # Catastrophic truncation falls back to raw
        self.assertEqual(sanitize_dictation_output(raw, "OK"), raw)

    def test_dictation_engine_and_daemon_rpc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            silent_wav = Path(tmp) / "silent.wav"
            speech_wav = Path(tmp) / "speech.wav"
            _write_test_wav(silent_wav, amplitude=0.0)
            _write_test_wav(speech_wav, amplitude=0.25)

            seen_messages: list[list[dict[str, str]]] = []

            def fake_asr(_path: Path) -> str:
                return "um let's meet at 3 actually make it 4"

            def fake_llm(messages: list[dict[str, str]], _max_tokens: int) -> str:
                seen_messages.append(messages)
                return "Let's meet at 4."

            engine = DictationEngine(model_dir=Path(tmp), asr_fn=fake_asr, llm_fn=fake_llm)

            # Silent clip should short-circuit without calling ASR/LLM
            silent_res = engine.transcribe_clip(silent_wav)
            self.assertEqual(silent_res.text, "")
            self.assertFalse(silent_res.used_llm)

            # Normal speech clip via daemon RPC handler
            resp, should_exit = handle_daemon_request(
                engine,
                {
                    "id": 7,
                    "cmd": "transcribe",
                    "wav": str(speech_wav),
                    "cleanup": True,
                    "app_name": "Mail",
                    "window_title": "Re: Launch",
                    "vocab": "VoiceDigest",
                },
            )
            self.assertFalse(should_exit)
            self.assertTrue(resp["ok"])
            self.assertEqual(resp["id"], 7)
            self.assertEqual(resp["text"], "Let's meet at 4.")
            self.assertEqual(resp["raw_text"], "um let's meet at 3 actually make it 4")
            self.assertTrue(resp["used_llm"])
            self.assertEqual(len(seen_messages), 1)
            self.assertIn("Active app: Mail", seen_messages[0][0]["content"])

            # Ping & quit commands
            pong, exit_ping = handle_daemon_request(engine, {"id": 8, "cmd": "ping"})
            self.assertTrue(pong["ok"])
            self.assertFalse(exit_ping)
            bye, exit_quit = handle_daemon_request(engine, {"id": 9, "cmd": "quit"})
            self.assertTrue(bye["ok"])
            self.assertTrue(exit_quit)


if __name__ == "__main__":
    unittest.main()
