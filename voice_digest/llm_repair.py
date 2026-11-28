"""Local MLX LLM semantic speaker-attribution repair (100% offline, preserves exact words & timestamps)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

from voice_digest.alignment import _join_word_texts
from voice_digest.models import llm_repair_model_path, release_memory
from voice_digest.types import Turn, Word

logger = logging.getLogger(__name__)

_SENTENCE_END = (".", "?", "!")
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_PAUSE_CUE_MIN_SEC = 0.8
_SUMMARY_SINGLE_PASS_MAX_CHARS = 12_000
_SUMMARY_CHUNK_CHARS = 10_000
MAX_SUMMARY_FOCUS_CHARS = 160


def apply_chat_template_no_think(
    tokenizer: object,
    messages: list[dict[str, str]],
    fallback_prompt: str,
) -> str:
    """Render a chat template with Qwen3 thinking mode disabled when supported."""
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_fn):
        return fallback_prompt
    try:
        return str(
            apply_fn(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(
            apply_fn(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def strip_thinking_blocks(text: str) -> str:
    """Defensively strip any <think>...</think> reasoning tags from model output."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _split_turns_into_sentences(turns: Sequence[Turn]) -> list[tuple[str, list[Word]]]:
    """Break turns into atomic sentence units while preserving original speaker hint and Word timestamps."""
    sentences: list[tuple[str, list[Word]]] = []
    for turn in turns:
        if not turn.words:
            if turn.text.strip():
                sentences.append((turn.speaker, [Word(turn.text.strip(), turn.start, turn.end, speaker=turn.speaker)]))
            continue

        buf: list[Word] = []
        for w in turn.words:
            buf.append(w)
            if w.text.strip().endswith(_SENTENCE_END):
                sentences.append((turn.speaker, buf))
                buf = []
        if buf:
            sentences.append((turn.speaker, buf))
    return sentences


def _parse_speaker_json(raw_output: str, num_speakers: int) -> dict[int, int]:
    """Safely extract {sentence_idx: speaker_num} from LLM output (accepts int or 'speaker X')."""
    match = _JSON_OBJECT_RE.search(raw_output)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[int, int] = {}
    for k, v in parsed.items():
        try:
            idx = int(str(k).strip())
            if isinstance(v, int):
                spk = v
            else:
                m_num = re.search(r"\d+", str(v))
                if not m_num:
                    continue
                spk = int(m_num.group(0))
            if 1 <= spk <= num_speakers:
                result[idx] = spk
        except (ValueError, TypeError):
            continue
    return result


def _merge_repaired_sentences(
    sentences: Sequence[tuple[str, list[Word]]],
    pause_threshold: float,
) -> list[Turn]:
    """Re-merge sentences with repaired speaker labels into Turns, ordered by first appearance."""
    if not sentences:
        return []

    # Normalize speaker IDs by first appearance
    remap: dict[str, str] = {}
    normalized: list[tuple[str, list[Word]]] = []
    for spk, words in sentences:
        if spk not in remap:
            remap[spk] = f"speaker {len(remap) + 1}"
        new_spk = remap[spk]
        updated_words = [
            Word(w.text, w.start, w.end, w.score, speaker=new_spk) for w in words
        ]
        normalized.append((new_spk, updated_words))

    turns: list[Turn] = []
    cur_spk, cur_words = normalized[0][0], list(normalized[0][1])

    for spk, words in normalized[1:]:
        gap = words[0].start - cur_words[-1].end
        if spk == cur_spk and gap <= pause_threshold:
            cur_words.extend(words)
        else:
            turns.append(
                Turn(
                    speaker=cur_spk,
                    start=cur_words[0].start,
                    end=cur_words[-1].end,
                    text=_join_word_texts(cur_words),
                    words=cur_words,
                )
            )
            cur_spk, cur_words = spk, list(words)

    turns.append(
        Turn(
            speaker=cur_spk,
            start=cur_words[0].start,
            end=cur_words[-1].end,
            text=_join_word_texts(cur_words),
            words=cur_words,
        )
    )
    return turns


def repair_speaker_turns(
    turns: list[Turn],
    num_speakers: int,
    model_dir: Path,
    pause_threshold: float = 2.0,
    chunk_size: int = 24,
    context_overlap: int = 5,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[Turn]:
    """Use a local MLX LLM (Qwen3.5-4B-MLX-4bit) to fix conversational speaker attribution errors.

    Only speaker IDs are updated; transcript words and timestamps are preserved verbatim.
    """
    if not turns or num_speakers < 2:
        return turns

    local_model = llm_repair_model_path(model_dir)
    if not local_model.exists() or not any(local_model.iterdir()):
        raise FileNotFoundError(
            f"Local LLM speaker-repair model not found at '{local_model}'. "
            f"Run 'python transcribe.py --download-models --llm-repair --model-dir {model_dir}' first."
        )

    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is not installed. Install it via 'uv pip install -e \".[mlx]\"'."
        ) from exc

    sentences = _split_turns_into_sentences(turns)
    if len(sentences) < 2:
        return turns

    logger.info(
        "Loading local MLX LLM from %s to repair speaker attribution across %d sentences",
        local_model,
        len(sentences),
    )
    model, tokenizer = load(str(local_model))
    repaired_speakers: list[str] = [spk for spk, _ in sentences]
    total_reassigned = 0

    try:
        # Infer a compact global speaker role profile from the opening ~40 sentences
        # so long 5-minute monologues stay anchored to the right speaker across all chunks.
        speaker_profile = ""
        try:
            opening_lines = [
                f"[{spk}] {_join_word_texts(words)}"
                for spk, words in sentences[:40]
            ]
            profile_sys = (
                f"In 1 short sentence per speaker (1 to {num_speakers}), describe each speaker's role "
                "or identity based on this opening excerpt (e.g., 'Speaker 1: Interviewer asking questions; "
                "Speaker 2: Candidate explaining their background'). Output plain text only, max 40 words."
            )
            speaker_profile = _run_mlx_chat(
                model,
                tokenizer,
                profile_sys,
                "\n".join(opening_lines),
                max_tokens=96,
            ).strip()
        except Exception:
            speaker_profile = ""

        profile_block = f"Global speaker roles: {speaker_profile}\n" if speaker_profile else ""

        total = len(sentences)
        chunk_starts = list(range(0, total, chunk_size))
        for c_idx, start in enumerate(chunk_starts, start=1):
            end = min(total, start + chunk_size)
            ctx_start = max(0, start - context_overlap)

            if progress_cb is not None:
                progress_cb(c_idx / float(len(chunk_starts)), f"Chunk {c_idx}/{len(chunk_starts)}")

            lines: list[str] = []
            for i in range(ctx_start, end):
                text = _join_word_texts(sentences[i][1])
                pause_cue = ""
                if i > ctx_start:
                    gap = sentences[i][1][0].start - sentences[i - 1][1][-1].end
                    if gap >= _PAUSE_CUE_MIN_SEC:
                        pause_cue = f" [pause {gap:.1f}s]"
                if i < start:
                    lines.append(f"context [{repaired_speakers[i]}]: {text}")
                else:
                    lines.append(f"{i}{pause_cue}: {text}")

            system_msg = (
                f"You assign speaker numbers (1 to {num_speakers}) to numbered sentences from a "
                f"{num_speakers}-speaker conversation (such as an interview, podcast, meeting, or call).\n"
                f"{profile_block}"
                "Rules:\n"
                "1. Infer and track each speaker's perspective, pronouns ('I/my/our' vs 'you/your'), and identity/role "
                "across the conversation (e.g., host vs guest, interviewer vs candidate, presenter vs listener). "
                "If Speaker 2 introduced their team or company ('what we're building at X'), a later line saying 'I lead the team at X' is STILL Speaker 2.\n"
                "2. A question and its answer belong to DIFFERENT speakers.\n"
                "3. Reciprocal greetings and brief acknowledgments ('Nice to meet you' / 'Nice to meet you too', "
                "confirming a name, 'Yeah got it') belong to DIFFERENT speakers.\n"
                "4. Keep a continuous multi-sentence or multi-minute explanation, story, or monologue by the same person "
                "strictly on the SAME speaker—even if they pause briefly, ask a rhetorical question, or quote what someone else said. "
                "Do NOT split a monologue across speakers without an explicit conversational handoff or reply.\n"
                "5. A [pause Xs] cue often marks a turn handoff, though a speaker can also pause mid-thought.\n"
                "6. Stay consistent with any preceding 'context [speaker X]' lines and the global speaker roles above.\n"
                f"7. Output ONLY a single compact JSON object mapping EVERY numbered line ID ({start}..{end - 1}) "
                "to an integer speaker number. No markdown, no explanation.\n\n"
                "Example input:\n"
                "0: Hey I'm good how are you?\n"
                "1: Do you pronounce your name Alex?\n"
                "2 [pause 0.9s]: Alex, yeah.\n"
                "3: Okay awesome, nice to meet you.\n"
                "4: Nice to meet you too.\n"
                "5: So I'll give you some context on our team and what we're building.\n"
                "6: We're focusing on the new launch and I'd love to hear your thoughts.\n"
                "7 [pause 1.1s]: Awesome, yeah, let's get started.\n"
                'Example output:\n{"0": 1, "1": 1, "2": 2, "3": 1, "4": 2, "5": 2, "6": 2, "7": 1}'
            )
            user_msg = "\n".join(lines)

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            prompt = apply_chat_template_no_think(
                tokenizer,
                messages,
                fallback_prompt=f"{system_msg}\n\n{user_msg}\n\nJSON:",
            )

            gen_kwargs: dict[str, object] = {"prompt": prompt, "max_tokens": 768, "verbose": False}
            try:
                from mlx_lm.sample_utils import make_sampler

                gen_kwargs["sampler"] = make_sampler(temp=0.0)
            except Exception:
                pass

            response = strip_thinking_blocks(str(generate(model, tokenizer, **gen_kwargs)))
            mapping = _parse_speaker_json(response, num_speakers=num_speakers)
            if not mapping:
                logger.warning("LLM speaker repair chunk %d/%d returned unparseable output.", c_idx, len(chunk_starts))
            for idx in range(start, end):
                if idx in mapping:
                    new_spk = f"speaker {mapping[idx]}"
                    if new_spk != repaired_speakers[idx]:
                        total_reassigned += 1
                    repaired_speakers[idx] = new_spk
    finally:
        del model, tokenizer
        release_memory()

    logger.info(
        "LLM speaker repair reassigned %d / %d sentences.", total_reassigned, len(sentences)
    )
    updated_sentences = [
        (repaired_speakers[i], sentences[i][1]) for i in range(len(sentences))
    ]
    return _merge_repaired_sentences(updated_sentences, pause_threshold=pause_threshold)


def _chunk_text_for_summary(lines: Sequence[str], max_chars: int = _SUMMARY_CHUNK_CHARS) -> list[str]:
    """Split transcript lines into bounded character chunks that safely fit Qwen3.5 4B's context window."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        add_len = len(line) + 1
        if buf and buf_len + add_len > max_chars:
            chunks.append("\n".join(buf))
            buf = [line]
            buf_len = add_len
        else:
            buf.append(line)
            buf_len += add_len
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _run_mlx_chat(
    model: object,
    tokenizer: object,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 600,
) -> str:
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = apply_chat_template_no_think(
        tokenizer,
        messages,
        fallback_prompt=f"{system_msg}\n\n{user_msg}\n\nSummary:",
    )

    gen_kwargs: dict[str, object] = {"prompt": prompt, "max_tokens": max_tokens, "verbose": False}
    try:
        from mlx_lm.sample_utils import make_sampler

        gen_kwargs["sampler"] = make_sampler(temp=0.2)
    except Exception:
        pass
    return strip_thinking_blocks(str(generate(model, tokenizer, **gen_kwargs)))


def summarize_transcript(
    turns: Sequence[Turn],
    model_dir: Path,
    focus_prompt: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Generate a local AI summary using Qwen3.5-4B-MLX-4bit with automatic map-reduce chunking.

    Handles arbitrarily long transcripts (1-3+ hours) without overflowing the context window.
    """
    if not turns:
        return ""

    local_model = llm_repair_model_path(model_dir)
    if not local_model.exists() or not any(local_model.iterdir()):
        raise FileNotFoundError(
            f"Local LLM model not found at '{local_model}'. "
            f"Run 'python transcribe.py --download-models --summary --model-dir {model_dir}' first."
        )

    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is not installed. Install it via 'uv pip install -e \".[mlx]\"'."
        ) from exc

    focus = (focus_prompt or "").strip()[:MAX_SUMMARY_FOCUS_CHARS]
    focus_instruction = (
        f"User focus: {focus}"
        if focus
        else "Provide a concise overview, key takeaways, and any action items or TODOs."
    )

    lines = [f"{t.speaker}: {t.text.strip()}" for t in turns if t.text.strip()]
    full_text = "\n".join(lines)
    if not full_text:
        return ""

    logger.info("Loading local MLX LLM from %s for transcript summarization", local_model)
    model, tokenizer = load(str(local_model))
    try:
        if len(full_text) <= _SUMMARY_SINGLE_PASS_MAX_CHARS:
            if progress_cb is not None:
                progress_cb(0.5, "Generating summary")
            system_msg = (
                "You are a clear, concise assistant summarizing an audio transcript. "
                f"{focus_instruction}\n"
                "Format cleanly in plain text with short bullet points (no markdown code fences). "
                "Base every point strictly on what was said in the transcript."
            )
            return _run_mlx_chat(model, tokenizer, system_msg, full_text, max_tokens=650)

        # Map-reduce for long transcripts (1+ hours) so context window is never exceeded
        chunks = _chunk_text_for_summary(lines, max_chars=_SUMMARY_CHUNK_CHARS)
        partials: list[str] = []
        total_steps = len(chunks) + 1
        for idx, chunk in enumerate(chunks, start=1):
            if progress_cb is not None:
                progress_cb(idx / float(total_steps), f"Summarizing part {idx}/{len(chunks)}")
            map_sys = (
                "Extract the most important points, decisions, takeaways, and details relevant to this focus: "
                f"{focus_instruction}\n"
                "Reply with concise bullet points strictly grounded in this excerpt."
            )
            partials.append(_run_mlx_chat(model, tokenizer, map_sys, chunk, max_tokens=350))

        if progress_cb is not None:
            progress_cb(0.95, "Synthesizing final summary")
        reduce_sys = (
            "You are synthesizing section notes from a long audio transcript into one cohesive final summary. "
            f"{focus_instruction}\n"
            "Deduplicate overlapping bullets, organize clearly in plain text with short headings/bullets, "
            "and stay strictly faithful to the notes."
        )
        combined = "\n\n".join(f"--- Part {i + 1} ---\n{p}" for i, p in enumerate(partials))
        return _run_mlx_chat(model, tokenizer, reduce_sys, combined, max_tokens=750)
    finally:
        del model, tokenizer
        release_memory()
