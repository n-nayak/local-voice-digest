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
from voice_digest.llm_repair import apply_chat_template_no_think, strip_thinking_blocks
from voice_digest.models import (
    DICTATION_ASR_REPO_ID,
    DICTATION_LLM_REPO_ID,
    LLM_REPAIR_REPO_ID,
    dictation_asr_model_path,
    dictation_llm_model_path,
    llm_repair_model_path,
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
        self.assertIn("Qwen3-4B", LLM_REPAIR_REPO_ID)
        self.assertIn("parakeet-tdt-0.6b-v3", DICTATION_ASR_REPO_ID)
        self.assertIn("Qwen3-1.7B", DICTATION_LLM_REPO_ID)
        self.assertEqual(llm_repair_model_path(root), root / "llm" / "qwen3-4b-instruct-4bit")
        self.assertEqual(dictation_asr_model_path(root), root / "dictation" / "parakeet-tdt-0.6b-v3")
        self.assertEqual(dictation_llm_model_path(root), root / "llm" / "qwen3-1.7b-4bit")

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
