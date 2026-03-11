#!/usr/bin/env python3
"""
Thinking-policy simultaneous interpretation pipeline (no alignment, no majority-vote).

Per trajectory step:
  1. Accumulate observed English source prefix.
  2. Base model samples N possible future source continuations.
  3. Send to thinking model: observed prefix, list of futures, committed Chinese so far.
  4. Thinking model outputs ONLY the additional Chinese delta that is safe to emit
     (consistent with all possible futures). If no safe delta, output EMPTY → READ.
  5. At end of utterance, force-complete the remaining translation.

No awesome-align, simalign, sentence-path, or legacy majority-vote logic.

File structure:
  - CLI, env, text utils: argument parsing, TSV helpers, normalize_zh, clean_llm_output, etc.
  - Base model: load_base_model(), sample_futures() for N future English continuations.
  - Thinking model: build_thinking_prompt(), call_thinking_model() for safe delta; force_complete_translation() at end.
  - process_one_utterance(): main loop over trajectory chunks; per chunk: sample futures → thinking delta or READ; last chunk → force-complete.
  - I/O: iter_assigned_rows(), get_one_row_by_id(), main() with JSON output.

Assumptions:
  - TSV has columns: id (or --id-column), src_text_full (list or string), src_trajectory (list or string).
  - Thinking model is served via OpenAI-compatible API (e.g. vLLM with chat template).
  - Base model is loadable with vLLM for text generation only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thinking-policy simultaneous interpretation (no alignment)."
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--thinking-api-base", default="http://localhost:8100/v1")
    p.add_argument("--thinking-model-name", default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8")
    p.add_argument("--thinking-tokenizer-path",
                   default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Thinking-2507-FP8")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=6, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=12)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--thinking-temperature", type=float, default=0.3)
    p.add_argument("--thinking-max-tokens", type=int, default=256)

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--id-column", default="id")
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--utt-id", default=None)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# =============================================================================
# Environment
# =============================================================================

def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Text utilities
# =============================================================================

def parse_list_column(raw: Any) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)] if str(parsed).strip() else []


def normalize_zh(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = re.sub(r"\s+", "", text)
    return text


def clean_llm_output(text: str) -> str:
    """Strip thinking tags and stray quotes from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("</think>", "")
    text = (text or "").strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    return text


def clean_continuation(observed: str, raw_output: str, max_words: int = 15) -> str:
    """Extract continuation text after observed prefix, cap length."""
    text = (raw_output or "").strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    obs_lower = (observed or "").lower().strip()
    text_lower = (text or "").lower()
    if obs_lower and text_lower.startswith(obs_lower):
        text = text[len(obs_lower):].strip()
    words = (text or "").split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return (text or "").strip()


_FIRST_SENTENCE_END_RE = re.compile(r'[.!?](?:["\')\]]+)?(?=\s|$)')


def truncate_future_to_first_sentence(text: str) -> str:
    """Keep at most the first English sentence to avoid spillover.
    With future_tokens=12, many samples have no period so this often no-ops; still helps when it does."""
    text = (text or "").strip()
    if not text:
        return ""
    m = _FIRST_SENTENCE_END_RE.search(text)
    if not m:
        return text
    return text[: m.end()].strip()


def strip_committed_suffix_from_delta(committed: str, delta: str) -> str:
    """If delta starts with a suffix of committed (overlap), strip it to avoid double-commit.
    Used for step-level delta and final completion."""
    delta = (delta or "").strip()
    if not delta:
        return ""
    if not committed:
        return normalize_zh(delta)
    committed_norm = normalize_zh(committed)
    delta_norm = normalize_zh(delta)
    max_k = min(len(committed_norm), len(delta_norm))
    for k in range(max_k, 0, -1):
        suffix = committed_norm[-k:]
        if delta_norm.startswith(suffix):
            return normalize_zh(delta_norm[k:].strip())
    return delta_norm


def sanitize_thinking_delta(raw: str) -> str:
    """Stricter sanitize for thinking-model output: no explanation, no '答案：', single-line delta."""
    if not raw:
        return ""
    text = (raw or "").strip()
    # Strip thinking tags first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("</think>", "")
    text = re.sub(r"</?output>", "", text, flags=re.IGNORECASE)
    # Take first line only (avoid multi-line explanation)
    if "\n" in text:
        text = text.split("\n")[0].strip()
    # Strip common answer prefixes (model may output "答案：xxx" or "Answer: xxx")
    for prefix in ("答案：", "答案:", "Answer:", "answer:", "输出：", "输出:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = text.strip().strip('"').strip("'").strip("\u201c\u201d\u2018\u2019")
    return text


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return (safe or "unknown")[:200]


def _vlog(log_file: Optional[Any], msg: str) -> None:
    if log_file is not None:
        log_file.write(msg)
        if not msg.endswith("\n"):
            log_file.write("\n")
        log_file.flush()


class _TeeWriter:
    def __init__(self, file_obj):
        self._f = file_obj

    def write(self, msg):
        self._f.write(msg)
        sys.stdout.write(msg)

    def flush(self):
        self._f.flush()
        sys.stdout.flush()

    def close(self):
        self._f.close()


def _message_text_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _extract_reasoning_text(message: Any) -> str:
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        text = _message_text_to_str(value)
        if text:
            return text
    return ""


def _split_qwen_thinking_content(message: Any) -> Tuple[str, str, str]:
    raw_content = _message_text_to_str(getattr(message, "content", None))
    reasoning_text = _extract_reasoning_text(message)
    content_text = raw_content

    # Fallback for Qwen thinking outputs where only a closing </think> is visible
    # in content and the parser did not populate message.reasoning/message.reasoning_content.
    if "</think>" in raw_content:
        before, after = raw_content.rsplit("</think>", 1)
        fallback_reasoning = before.replace("<think>", "").strip()
        fallback_content = after.strip()
        if not reasoning_text and fallback_reasoning:
            reasoning_text = fallback_reasoning
        content_text = fallback_content

    if content_text.startswith("<think>"):
        content_text = content_text[len("<think>"):].strip()
    if content_text.startswith("</think>"):
        content_text = content_text[len("</think>"):].strip()

    return reasoning_text, content_text, raw_content


def _looks_like_valid_zh_delta(text: str) -> bool:
    if not text:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    return bool(re.fullmatch(r"[，。！？；：“”‘’、,.!?;:()（）\[\]\-—…\s]+", text))


# =============================================================================
# Base model: future sampling
# =============================================================================

def load_base_model(path: str, gpu_memory_utilization: float = 0.85) -> LLM:
    return LLM(
        model=path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def sample_futures(
    base_llm: LLM,
    observed_source: str,
    num_futures: int,
    future_tokens: int,
    temperature: float,
) -> Tuple[List[str], List[str]]:
    """Sample N future English continuations from the base model."""
    if not (observed_source or "").strip():
        return [], []
    params = SamplingParams(
        temperature=temperature,
        max_tokens=future_tokens,
        n=num_futures,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.6,
        stop=["\n"],
    )
    outputs = base_llm.generate([observed_source.strip()], params)
    futures: List[str] = []
    raw_outputs: List[str] = []
    for out in outputs[0].outputs:
        raw = (out.text or "").strip()
        raw_outputs.append(raw)
        cleaned = clean_continuation(observed_source, raw)
        if cleaned:
            cleaned = truncate_future_to_first_sentence(cleaned)
            if cleaned:
                futures.append(cleaned)
    return futures, raw_outputs


# =============================================================================
# Thinking model: safe delta
# =============================================================================

def build_thinking_prompt(
    observed_source: str,
    futures: List[str],
    committed_chinese: str,
) -> str:
    """Build the user prompt for the thinking model (safe delta only)."""
    futures_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(futures) if (f or "").strip())
    committed_block = committed_chinese if committed_chinese else "(none yet)"
    return (
        "You are a professional simultaneous interpreter (English → Chinese). "
        "You see a PARTIAL English source and several POSSIBLE future continuations. "
        "You have already committed a partial Chinese translation (may be empty). "
        "Your task: output ONLY the additional Chinese text that is SAFE to emit now.\n\n"
        "Rules:\n"
        "- First identify the longest semantic Chinese prefix that would remain valid under ALL possible futures.\n"
        "- The new text must be consistent with ALL possible futures (no contradiction).\n"
        "- Do NOT repeat or copy the committed translation.\n"
        "- Prefer the maximal safe continuation, not an overly short fragment.\n"
        "- Do NOT output explanation, reasoning, or abstract summary.\n"
        "- If you are uncertain or no safe new segment exists, output exactly: EMPTY\n\n"
        "Example 1:\n"
        "Observed English: he is\n"
        "Possible futures:\n"
        "  1. a worker at the local school.\n"
        "  2. a teacher who later became a principal.\n"
        "Committed Chinese: (none yet)\n"
        "Safe output: 他是\n\n"
        "Example 2:\n"
        "Observed English: I went to the\n"
        "Possible futures:\n"
        "  1. bank to deposit some cash.\n"
        "  2. beach to watch the sunset.\n"
        "Committed Chinese: 我去了\n"
        "Safe output: EMPTY\n\n"
        "Partial English source so far:\n"
        f"{observed_source}\n\n"
        "Possible future continuations (any of these may follow):\n"
        f"{futures_block}\n\n"
        "Committed Chinese translation so far (do not repeat):\n"
        f"{committed_block}\n\n"
        "Output ONLY the next safe Chinese segment to emit, or EMPTY if none."
    )


def call_thinking_model(
    client: OpenAI,
    model: str,
    user_content: str,
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    """Call thinking model via chat endpoint; return content delta and debug payload."""
    messages = [{"role": "user", "content": user_content}]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_content = _split_qwen_thinking_content(message)
    cleaned = normalize_zh(sanitize_thinking_delta(content_text))
    rejected_reason = ""
    delta = cleaned
    if not cleaned or cleaned.upper() == "EMPTY":
        delta = ""
    elif not _looks_like_valid_zh_delta(cleaned):
        rejected_reason = "non_chinese_content"
        delta = ""
    return delta, {
        "messages": messages,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "raw_content": raw_content,
        "cleaned_content": cleaned,
        "finish_reason": getattr(choice, "finish_reason", None),
        "rejected_reason": rejected_reason,
        "accepted_delta": delta,
    }


# =============================================================================
# End-of-utterance: force-complete remaining translation
# =============================================================================

def build_final_completion_prompt(full_source: str, committed_chinese: str) -> str:
    """Prompt to complete the rest of the translation from committed."""
    return (
        "You are a professional translator. Complete the Chinese translation of the following English text. "
        "A partial Chinese translation is already committed; output ONLY the continuation (the part that comes after).\n\n"
        f"English (full):\n{full_source}\n\n"
        f"Chinese already committed (do NOT repeat):\n{committed_chinese or '(none)'}\n\n"
        "Output ONLY the new Chinese characters that follow the committed part. No explanation."
    )


def force_complete_translation(
    client: OpenAI,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    """Get final translation using chat endpoint; return full translation and debug payload."""
    user_content = build_final_completion_prompt(full_source, committed_chinese)
    messages = [{"role": "user", "content": user_content}]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_content = _split_qwen_thinking_content(message)
    continuation = normalize_zh(sanitize_thinking_delta(content_text))
    rejected_reason = ""
    if continuation and continuation.upper() != "EMPTY" and not _looks_like_valid_zh_delta(continuation):
        rejected_reason = "non_chinese_content"
        continuation = ""
    if not committed_chinese:
        return continuation, {
            "messages": messages,
            "reasoning_text": reasoning_text,
            "content_text": content_text,
            "raw_content": raw_content,
            "cleaned_content": continuation,
            "finish_reason": getattr(choice, "finish_reason", None),
            "rejected_reason": rejected_reason,
            "full_translation": continuation,
        }
    committed_norm = normalize_zh(committed_chinese)
    new_part = strip_committed_suffix_from_delta(committed_chinese, continuation)
    new_part = normalize_zh(new_part)
    full_translation = committed_norm + new_part
    return full_translation, {
        "messages": messages,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "raw_content": raw_content,
        "cleaned_content": continuation,
        "deduped_new_part": new_part,
        "finish_reason": getattr(choice, "finish_reason", None),
        "rejected_reason": rejected_reason,
        "full_translation": full_translation,
    }


# =============================================================================
# Process one utterance
# =============================================================================

def process_one_utterance(
    base_llm: LLM,
    client: OpenAI,
    thinking_model: str,
    utt_id: str,
    sentences: List[str],
    trajectory: List[str],
    row: Dict[str, str],
    args: argparse.Namespace,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run thinking-policy pipeline for one utterance.
    Returns dict with utt_id, source_future_sampling (chunks), target_future_sampling (deltas), actions, etc.
    """
    full_source = " ".join(sentences)
    n_chunks = len(trajectory)
    timing: Dict[str, float] = {
        "step1_future_sampling_s": 0.0,
        "step2_thinking_delta_s": 0.0,
        "step3_final_complete_s": 0.0,
    }

    source_chunks: List[str] = []
    target_deltas: List[str] = []
    actions: List[str] = []
    committed = ""
    accumulated_source = ""

    _vlog(verbose_log_file, f"\n{'#' * 60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text: {full_source}")
    _vlog(verbose_log_file, f"# Chunks: {n_chunks}")
    _vlog(verbose_log_file, f"# Thinking model: {thinking_model}")
    _vlog(verbose_log_file, f"{'#' * 60}")

    for chunk_idx, chunk in enumerate(trajectory):
        chunk_str = (chunk or "").strip()
        if chunk_str:
            accumulated_source = (accumulated_source + " " + chunk_str).strip()
        source_chunks.append(chunk_str)

        is_last = chunk_idx == n_chunks - 1

        _vlog(verbose_log_file, f"\n{'=' * 60}")
        _vlog(verbose_log_file, f"Chunk {chunk_idx + 1}/{n_chunks}: {chunk_str!r}")
        _vlog(verbose_log_file, f"  accumulated_source: {accumulated_source!r}")
        _vlog(verbose_log_file, f"  committed_before:  {committed!r}")

        # --- Last chunk: force-complete remainder ---
        if is_last:
            _vlog(verbose_log_file, f"  [last] force-complete from committed len={len(normalize_zh(committed))}")
            t0 = time.perf_counter()
            full_translation, final_debug = force_complete_translation(
                client, thinking_model, full_source, committed
            )
            timing["step3_final_complete_s"] += time.perf_counter() - t0
            _vlog(verbose_log_file, f"  [last] final_messages: {json.dumps(final_debug['messages'], ensure_ascii=False, indent=2)}")
            _vlog(verbose_log_file, f"  [last] final_reasoning_text: {final_debug['reasoning_text']!r}")
            _vlog(verbose_log_file, f"  [last] final_raw_content: {final_debug['raw_content']!r}")
            _vlog(verbose_log_file, f"  [last] final_content_text: {final_debug['content_text']!r}")
            _vlog(verbose_log_file, f"  [last] final_cleaned_content: {final_debug['cleaned_content']!r}")
            if final_debug.get("rejected_reason"):
                _vlog(verbose_log_file, f"  [last] rejected_reason: {final_debug['rejected_reason']}")
            _vlog(verbose_log_file, f"  [last] final_cleaned_full_translation: {full_translation!r}")
            committed_norm = normalize_zh(committed)
            full_norm = normalize_zh(full_translation)
            if len(full_norm) > len(committed_norm):
                remaining = full_norm[len(committed_norm):]
                target_deltas.append(remaining)
                actions.append("WRITE")
                committed = full_translation
                _vlog(verbose_log_file, f"  -> WRITE (end) delta={remaining!r}")
            else:
                target_deltas.append("")
                actions.append("READ")
                _vlog(verbose_log_file, "  -> READ (end, nothing new)")
            continue

        # --- Future sampling ---
        t1_0 = time.perf_counter()
        futures, future_raw_outputs = sample_futures(
            base_llm,
            accumulated_source,
            args.num_futures,
            args.future_tokens,
            args.sample_temperature,
        )
        timing["step1_future_sampling_s"] += time.perf_counter() - t1_0

        _vlog(verbose_log_file, f"  observed_preview: {accumulated_source[:120]!r}")
        _vlog(verbose_log_file, f"  step1_future_raw_outputs: {json.dumps(future_raw_outputs, ensure_ascii=False, indent=2)}")
        _vlog(verbose_log_file, f"  step1_futures_cleaned: {json.dumps(futures, ensure_ascii=False, indent=2)}")

        if len(futures) < 2:
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        # --- Thinking model: safe delta ---
        user_content = build_thinking_prompt(accumulated_source, futures, committed)
        _vlog(verbose_log_file, f"  step2_thinking_user_prompt:\n{user_content}")
        t2_0 = time.perf_counter()
        delta, thinking_debug = call_thinking_model(
            client,
            thinking_model,
            user_content,
            temperature=args.thinking_temperature,
            max_tokens=args.thinking_max_tokens,
        )
        timing["step2_thinking_delta_s"] += time.perf_counter() - t2_0
        # Guard: strip any overlap with committed (avoid double-commit)
        delta_before_dedup = delta
        delta = strip_committed_suffix_from_delta(committed, delta)
        _vlog(verbose_log_file, f"  step2_reasoning_text: {thinking_debug['reasoning_text']!r}")
        _vlog(verbose_log_file, f"  step2_raw_content: {thinking_debug['raw_content']!r}")
        _vlog(verbose_log_file, f"  step2_content_text: {thinking_debug['content_text']!r}")
        _vlog(verbose_log_file, f"  step2_cleaned_content: {thinking_debug['cleaned_content']!r}")
        if thinking_debug.get("rejected_reason"):
            _vlog(verbose_log_file, f"  step2_rejected_reason: {thinking_debug['rejected_reason']}")
        _vlog(verbose_log_file, f"  step2_delta_before_dedup: {delta_before_dedup!r}")
        _vlog(verbose_log_file, f"  step2_delta_after_dedup: {delta!r}")

        if delta:
            target_deltas.append(delta)
            actions.append("WRITE")
            committed = (committed or "") + delta
            _vlog(verbose_log_file, f"  -> WRITE delta={delta!r}")
        else:
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (empty/invalid content)")

    system_output = "".join(d for d in target_deltas if d)

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "original_text": full_source,
        "input_sentences": sentences,
        "source_future_sampling": source_chunks,
        "target_future_sampling": target_deltas,
        "actions": actions,
        "system_output_text": system_output,
        "config": {
            "version": "thinking_policy",
            "num_futures": args.num_futures,
            "future_tokens": args.future_tokens,
            "thinking_model": thinking_model,
        },
        "timing": timing,
    }

    # Optional: reference and simple metrics if reference exists in row
    ref_text = _get_reference_from_row(row)
    if ref_text is not None:
        result["reference_text"] = ref_text
        result["metrics"] = _simple_metrics(system_output, ref_text)
    return result


def _get_reference_from_row(row: Dict[str, str]) -> Optional[str]:
    for key in ("tgt_text_full", "tgt_text", "target_text", "translation", "ref_text"):
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        vals = parse_list_column(raw)
        if vals:
            text = "".join(str(v).strip() for v in vals if str(v).strip())
        else:
            text = str(raw).strip()
        if text:
            return text
    return None


def _simple_metrics(hypothesis: str, reference: str) -> Dict[str, Any]:
    """Simple character-level metrics (no BLEU/LAAL)."""
    hyp_n = len(normalize_zh(hypothesis))
    ref_n = len(normalize_zh(reference))
    return {
        "hypothesis_chars": hyp_n,
        "reference_chars": ref_n,
        "char_ratio": hyp_n / ref_n if ref_n else float("nan"),
    }


# =============================================================================
# Data I/O
# =============================================================================

def iter_assigned_rows(
    input_tsv: str,
    task_id: int,
    num_tasks: int,
    max_rows: Optional[int] = None,
):
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row
            count += 1
            if max_rows is not None and count >= max_rows:
                break


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    n = 0
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        for row_idx, _ in enumerate(csv.DictReader(f, delimiter="\t")):
            if row_idx % num_tasks == task_id:
                n += 1
    return n


def get_one_row_by_id(
    input_tsv: str,
    utt_id: str,
    id_column: str = "id",
) -> Optional[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        for row_idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            if str(row.get(id_column, "")).strip() == str(utt_id).strip():
                return row_idx, row
    return None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_env()

    os.makedirs(args.output_root, exist_ok=True)

    # Thinking model (OpenAI-compatible)
    client = OpenAI(base_url=args.thinking_api_base, api_key="dummy")
    try:
        _ = client.models.list()
    except Exception as e:
        print(f"ERROR: Cannot connect to thinking API at {args.thinking_api_base}: {e}")
        return

    # Base model for future sampling
    print(f"[Base] Loading {args.base_model_path} ...")
    base_llm = load_base_model(args.base_model_path, args.gpu_memory_utilization)
    print("[Base] Loaded.")

    # Row iteration (test_one => verbose for per-step debug)
    if args.test_one:
        args.verbose = True
        if args.utt_id:
            one = get_one_row_by_id(args.input_tsv, args.utt_id, args.id_column)
            if one is None:
                print(f"utt-id '{args.utt_id}' not found.")
                return
            row_list = [one]
        else:
            row_list = list(iter_assigned_rows(
                args.input_tsv, args.task_id, args.num_tasks, max_rows=1
            ))
        total = len(row_list)
    else:
        total = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        if args.max_rows is not None:
            total = min(total, args.max_rows)
        row_list = list(iter_assigned_rows(
            args.input_tsv, args.task_id, args.num_tasks, max_rows=args.max_rows
        ))

    if not row_list:
        print("No rows to process.")
        return

    print(f"[Task {args.task_id}] Processing {total} rows | thinking={args.thinking_model_name} | N={args.num_futures}")

    written = 0
    for row_idx, row in tqdm(row_list, desc="utterances"):
        utt_id = str(row.get(args.id_column, "")).strip() or f"row_{row_idx}"
        out_path = os.path.join(args.output_root, f"{sanitize_filename(utt_id)}.json")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        sentences = parse_list_column(row.get("src_text_full"))
        trajectory = parse_list_column(row.get("src_trajectory"))
        if not sentences:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": "Empty src_text_full"}, f, ensure_ascii=False, indent=2)
            written += 1
            continue
        if not trajectory:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": "Empty src_trajectory"}, f, ensure_ascii=False, indent=2)
            written += 1
            continue

        try:
            verbose_log_file = None
            if args.verbose:
                verbose_log_path = os.path.join(
                    args.output_root,
                    f"verbose_{sanitize_filename(utt_id)}.log",
                )
                raw_file = open(verbose_log_path, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file)
                _vlog(verbose_log_file, f"[VerboseLog] writing to {verbose_log_path}")
            try:
                result = process_one_utterance(
                    base_llm,
                    client,
                    args.thinking_model_name,
                    utt_id,
                    sentences,
                    trajectory,
                    row,
                    args,
                    verbose_log_file=verbose_log_file,
                )
            finally:
                if verbose_log_file is not None:
                    verbose_log_file.close()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": str(e)}, f, ensure_ascii=False, indent=2)
            written += 1

    print(f"[Task {args.task_id}] Done. Written {written} outputs to {args.output_root}")


if __name__ == "__main__":
    main()
