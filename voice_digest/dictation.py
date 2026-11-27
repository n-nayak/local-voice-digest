"""Ultra-low-latency (<500ms) local system-wide dictation engine and persistent JSON-RPC daemon.

Uses public, ungated Apple Silicon MLX models:
- ASR: mlx-community/parakeet-tdt-0.6b-v3 (FastConformer TDT, ~90-160ms on M-series)
- Optional Cleanup LLM: mlx-community/Qwen3-1.7B-4bit (non-thinking mode, ~160-240ms)

Runs strictly offline inside `offline_socket_guard()` once models are cached.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from voice_digest.audio import get_wav_rms
from voice_digest.llm_repair import apply_chat_template_no_think, strip_thinking_blocks
from voice_digest.models import (
    dictation_asr_model_path,
    dictation_llm_model_path,
    release_memory,
)
from voice_digest.offline import offline_socket_guard

logger = logging.getLogger(__name__)

_SILENCE_RMS_THRESHOLD = 0.0005
_MAX_VOCAB_TERMS = 60
_MAX_CONTEXT_CHARS = 160
_QUOTE_WRAP_RE = re.compile(r'^["\'`]+|["\'`]+$')


@dataclass(frozen=True)
class DictationResult:
    """Result of a single dictation burst."""

    text: str
    raw_text: str
    asr_ms: int
    llm_ms: int
    total_ms: int
    used_llm: bool


def parse_custom_vocab(vocab: Optional[Sequence[str] | str]) -> list[str]:
    """Normalize a list or comma/newline-separated custom vocabulary string."""
    if vocab is None:
        return []
    if isinstance(vocab, str):
        raw_items = re.split(r"[,\n;]+", vocab)
    else:
        raw_items = list(vocab)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        term = item.strip()
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(term[:64])
        if len(cleaned) >= _MAX_VOCAB_TERMS:
            break
    return cleaned


def build_dictation_messages(
    raw_text: str,
    app_name: str = "",
    window_title: str = "",
    vocab: Optional[Sequence[str] | str] = None,
) -> list[dict[str, str]]:
    """Build a compact system + user prompt for low-latency dictation cleanup."""
    rules = [
        "You are a fast, faithful speech-to-text dictation cleaner.",
        "Clean up the spoken text and output ONLY the final ready-to-paste text (no quotes, no preamble, no explanations).",
        "Rules:",
        "1. Remove filler words (um, uh, er, ah, you know, like when used as a filler) and accidental repeated words/stutters.",
        "2. Resolve spoken self-corrections (e.g., 'meet at 3 actually make it 4' -> 'meet at 4').",
        "3. Apply natural capitalization, punctuation, and formatting.",
        "4. Preserve the speaker's exact meaning, tone, and wording. NEVER add new facts or drop real content.",
    ]

    ctx_parts: list[str] = []
    clean_app = app_name.strip()[:60]
    clean_win = window_title.strip()[:_MAX_CONTEXT_CHARS]
    if clean_app:
        ctx_parts.append(f"Active app: {clean_app}")
    if clean_win:
        ctx_parts.append(f"Window: {clean_win}")
    if ctx_parts:
        rules.append("Context: " + " | ".join(ctx_parts))

    terms = parse_custom_vocab(vocab)
    if terms:
        rules.append("Prefer these exact spellings when phonetically close: " + ", ".join(terms))

    return [
        {"role": "system", "content": "\n".join(rules)},
        {"role": "user", "content": raw_text.strip()},
    ]


def sanitize_dictation_output(raw_text: str, cleaned_text: str) -> str:
    """Guard against empty, wrapped, or hallucinated LLM outputs; fall back to raw ASR if unsafe."""
    raw = raw_text.strip()
    if not raw:
        return ""
    candidate = strip_thinking_blocks(cleaned_text).strip()
    if not candidate:
        return raw

    # Strip accidental wrapping quotes if raw text was not quoted
    if len(candidate) >= 2 and candidate[0] in ('"', "'", "`") and candidate[-1] == candidate[0]:
        if not raw.startswith(candidate[0]):
            candidate = _QUOTE_WRAP_RE.sub("", candidate).strip()

    if not candidate:
        return raw

    # Reject obvious hallucinations (way too long) or catastrophic truncation
    raw_len = len(raw)
    cand_len = len(candidate)
    if raw_len >= 12:
        if cand_len > max(raw_len * 2.2, raw_len + 80):
            logger.warning("Dictation LLM output too long (%d vs %d); falling back to raw ASR.", cand_len, raw_len)
            return raw
        if cand_len < max(3, int(raw_len * 0.25)):
            logger.warning("Dictation LLM output truncated (%d vs %d); falling back to raw ASR.", cand_len, raw_len)
            return raw

    return candidate


class DictationEngine:
    """Warm, strictly offline MLX dictation engine (Parakeet v3 + optional Qwen3-1.7B)."""

    def __init__(
        self,
        model_dir: Path,
        load_llm: bool = True,
        asr_fn: Optional[Callable[[Path], str]] = None,
        llm_fn: Optional[Callable[[list[dict[str, str]], int], str]] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self._load_llm_requested = load_llm
        self._asr_fn = asr_fn
        self._llm_fn = llm_fn
        self._asr_model: object = None
        self._llm_model: object = None
        self._llm_tokenizer: object = None

    @property
    def llm_loaded(self) -> bool:
        return self._llm_fn is not None or self._llm_model is not None

    def load(self) -> None:
        """Load Parakeet v3 and (optionally) Qwen3-1.7B from local offline cache into unified RAM."""
        if self._asr_fn is None:
            asr_dir = dictation_asr_model_path(self.model_dir)
            if not asr_dir.exists() or not any(asr_dir.iterdir()):
                raise FileNotFoundError(
                    f"Parakeet dictation model not found at '{asr_dir}'. "
                    f"Run 'python transcribe.py --download-models --dictation-models --model-dir {self.model_dir}' first."
                )
            try:
                from parakeet_mlx import from_pretrained
            except ImportError as exc:
                raise RuntimeError(
                    "parakeet-mlx is not installed. Run 'uv pip install parakeet-mlx' first."
                ) from exc

            logger.info("Loading warm Parakeet v3 dictation model from %s", asr_dir)
            self._asr_model = from_pretrained(str(asr_dir))

        if self._load_llm_requested and self._llm_fn is None:
            llm_dir = dictation_llm_model_path(self.model_dir)
            if llm_dir.exists() and any(llm_dir.iterdir()):
                try:
                    from mlx_lm import load as mlx_lm_load
                except ImportError as exc:
                    raise RuntimeError(
                        "mlx-lm is not installed. Install it via 'uv pip install mlx-lm'."
                    ) from exc
                logger.info("Loading warm Qwen3-1.7B dictation cleanup model from %s", llm_dir)
                self._llm_model, self._llm_tokenizer = mlx_lm_load(str(llm_dir))
            else:
                logger.warning(
                    "Dictation cleanup LLM not found at '%s'; continuing with raw Parakeet ASR.",
                    llm_dir,
                )

    def close(self) -> None:
        """Release warm models from unified memory."""
        self._asr_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        release_memory()

    def _transcribe_asr(self, wav_path: Path) -> str:
        if self._asr_fn is not None:
            return self._asr_fn(wav_path).strip()
        if self._asr_model is None:
            raise RuntimeError("DictationEngine.load() must be called before transcribe_clip().")
        result = self._asr_model.transcribe(str(wav_path))
        return str(getattr(result, "text", "")).strip()

    def _cleanup_llm(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        if self._llm_fn is not None:
            return self._llm_fn(messages, max_tokens)
        if self._llm_model is None or self._llm_tokenizer is None:
            return ""
        from mlx_lm import generate

        prompt = apply_chat_template_no_think(
            self._llm_tokenizer,
            messages,
            fallback_prompt=f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nCleaned:",
        )
        gen_kwargs: dict[str, object] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": False,
        }
        try:
            from mlx_lm.sample_utils import make_sampler

            gen_kwargs["sampler"] = make_sampler(temp=0.0)
        except Exception:
            pass
        return str(generate(self._llm_model, self._llm_tokenizer, **gen_kwargs))

    def transcribe_clip(
        self,
        wav_path: Path | str,
        cleanup: bool = True,
        app_name: str = "",
        window_title: str = "",
        vocab: Optional[Sequence[str] | str] = None,
    ) -> DictationResult:
        """Transcribe a short WAV clip and optionally polish it with Qwen3-1.7B."""
        t0 = time.perf_counter()
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Dictation audio file not found: {path}")

        # Skip silent / accidental short presses
        if get_wav_rms(path) < _SILENCE_RMS_THRESHOLD:
            total_ms = int((time.perf_counter() - t0) * 1000)
            return DictationResult(
                text="",
                raw_text="",
                asr_ms=0,
                llm_ms=0,
                total_ms=total_ms,
                used_llm=False,
            )

        with offline_socket_guard():
            t_asr = time.perf_counter()
            raw_text = self._transcribe_asr(path)
            asr_ms = int((time.perf_counter() - t_asr) * 1000)

            if not raw_text:
                total_ms = int((time.perf_counter() - t0) * 1000)
                return DictationResult(
                    text="",
                    raw_text="",
                    asr_ms=asr_ms,
                    llm_ms=0,
                    total_ms=total_ms,
                    used_llm=False,
                )

            final_text = raw_text
            llm_ms = 0
            used_llm = False

            if cleanup and self.llm_loaded:
                t_llm = time.perf_counter()
                messages = build_dictation_messages(
                    raw_text=raw_text,
                    app_name=app_name,
                    window_title=window_title,
                    vocab=vocab,
                )
                # Tight token cap proportional to utterance length keeps latency low
                max_tokens = min(512, max(64, int(len(raw_text.split()) * 2.2) + 24))
                cleaned = self._cleanup_llm(messages, max_tokens=max_tokens)
                final_text = sanitize_dictation_output(raw_text, cleaned)
                llm_ms = int((time.perf_counter() - t_llm) * 1000)
                used_llm = True

        total_ms = int((time.perf_counter() - t0) * 1000)
        return DictationResult(
            text=final_text,
            raw_text=raw_text,
            asr_ms=asr_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            used_llm=used_llm,
        )


def handle_daemon_request(engine: DictationEngine, payload: dict[str, object]) -> tuple[dict[str, object], bool]:
    """Handle one JSON-RPC request dict. Returns (response_dict, should_exit)."""
    cmd = str(payload.get("cmd", "")).strip().lower()
    req_id = payload.get("id")

    if cmd == "ping":
        return {"id": req_id, "ok": True, "event": "pong", "llm_loaded": engine.llm_loaded}, False
    if cmd == "quit":
        return {"id": req_id, "ok": True, "event": "bye"}, True
    if cmd == "transcribe":
        wav = str(payload.get("wav", "")).strip()
        if not wav:
            return {"id": req_id, "ok": False, "error": "Missing 'wav' path."}, False
        cleanup = bool(payload.get("cleanup", True))
        app_name = str(payload.get("app_name", "") or "")
        window_title = str(payload.get("window_title", "") or "")
        vocab = payload.get("vocab")
        if not isinstance(vocab, (list, str)):
            vocab = None
        try:
            res = engine.transcribe_clip(
                wav_path=Path(wav),
                cleanup=cleanup,
                app_name=app_name,
                window_title=window_title,
                vocab=vocab,
            )
            return {
                "id": req_id,
                "ok": True,
                "text": res.text,
                "raw_text": res.raw_text,
                "asr_ms": res.asr_ms,
                "llm_ms": res.llm_ms,
                "total_ms": res.total_ms,
                "used_llm": res.used_llm,
            }, False
        except Exception as exc:
            return {"id": req_id, "ok": False, "error": str(exc)}, False

    return {"id": req_id, "ok": False, "error": f"Unknown command: '{cmd}'"}, False


def run_dictation_daemon(model_dir: Path, load_llm: bool = True) -> int:
    """Run a warm persistent newline-delimited JSON-RPC daemon over stdin/stdout."""
    engine = DictationEngine(model_dir=model_dir, load_llm=load_llm)
    try:
        engine.load()
    except Exception as exc:
        print(json.dumps({"event": "error", "ok": False, "error": str(exc)}), flush=True)
        return 1

    print(
        json.dumps({"event": "ready", "ok": True, "llm_loaded": engine.llm_loaded}),
        flush=True,
    )

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"Invalid JSON: {exc}"}), flush=True)
                continue

            response, should_exit = handle_daemon_request(engine, payload)
            print(json.dumps(response), flush=True)
            if should_exit:
                break
    finally:
        engine.close()

    return 0
