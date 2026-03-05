#!/usr/bin/env python3
"""
Word alignment via simalign + monotonic alignment (drop-in replacement for awesome-align).

Uses simalign (SentenceAligner, e.g. pvl/labse_bert) for src-tgt word alignment,
then converts crossing alignments to monotonic alignment following GigaSpeech's
build_trajectory_full_mfa.py. Goal: truncate target translation by observed source.

Reference: https://github.com/owaski/GigaSpeech/blob/main/preprocess/build_trajectory_full_mfa.py

Dependencies: simalign, jieba
  pip install simalign jieba
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

# Optional: normalize translation before alignment (strip, first line)
def _normalize_tgt_for_jieba(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # First line only, like clean_translation_for_alignment
    if "\n" in t:
        t = t.split("\n")[0].strip()
    return t


def _translation_prefix_for_norm_char_count(translation: str, norm_char_count: int) -> str:
    """Return prefix of translation (first line, stripped) that has exactly norm_char_count
    non-space chars. So core's translation[:len(result)] aligns with norm[:norm_char_count].
    """
    if norm_char_count <= 0:
        return ""
    trans_work = _normalize_tgt_for_jieba(translation)
    n = 0
    for i, c in enumerate(trans_work):
        if not c.isspace():
            n += 1
            if n >= norm_char_count:
                return trans_work[: i + 1]
    return trans_work


def load_align_model(cache_dir: Optional[str] = None, device: Optional[str] = None):
    """Load simalign SentenceAligner. Returns (aligner, None) for drop-in with core.

    Must be called before loading vLLM if align shares GPU with base.
    If cache_dir is set (e.g. from HF_HOME), model is loaded from that directory.
    """
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    from simalign import SentenceAligner
    if device is None:
        device = "cuda"
    elif isinstance(device, str) and device.startswith("cuda"):
        device = "cuda"  # simalign uses "cuda" or "cpu"
    # "m" = match/intersection (key "inter"); "a"=argmax, "i"=itermax. Use intersection for monotonic.
    aligner = SentenceAligner(
        model="pvl/labse_bert",
        token_type="bpe",
        matching_methods="m",
        device=device,
    )
    print(f"[Align v2] simalign loaded on {device}.")
    return aligner, None


def _to_monotonic_alignments(
    alignments_inter: List[Tuple[int, int]],
    n_src: int,
    n_tgt: int,
) -> List[Tuple[int, int]]:
    """Convert crossing alignments to monotonic (src non-decreasing along tgt).

    Follows GigaSpeech build_trajectory_full_mfa.py:
    - sort by (tgt_idx, src_idx)
    - append (len(src_words)-1, len(tgt_words)-1)
    - merge same tgt: keep last (larger src)
    - ensure src non-decreasing: alignments_r[i] = (max(a[0], alignments_r[i-1][0]), a[1])
    - prepend (-1, -1) for trajectory use; we return without (-1,-1) for truncation.
    """
    if not alignments_inter:
        return []
    alignments = sorted(alignments_inter, key=lambda x: (x[1], x[0]))
    alignments.append((n_src - 1, n_tgt - 1))
    alignments_r = []
    for a in alignments:
        if len(alignments_r) > 0 and alignments_r[-1][1] == a[1]:
            alignments_r[-1] = a  # replace element (same tgt idx -> keep later src)
        else:
            alignments_r.append(a)
    for i in range(1, len(alignments_r)):
        s, t = alignments_r[i]
        s_prev = alignments_r[i - 1][0]
        alignments_r[i] = (max(s, s_prev), t)
    # Return without (-1,-1); caller uses (src_idx, tgt_idx) for truncation
    return alignments_r


def get_word_alignments(
    src_text: str,
    tgt_text: str,
    align_model: Any,
    align_tokenizer: Any,
) -> List[Tuple[int, int]]:
    """Word-level alignment via simalign + monotonic.

    Returns (src_word_idx, tgt_char_idx) so core's len(translation) checks work.
    tgt_char_idx = end character index (inclusive) of that tgt word in normalized text.
    - src: split by whitespace (English words).
    - tgt: jieba (Chinese words). Translation is normalized (first line, strip).
    """
    import jieba
    src_words = src_text.strip().split()
    tgt_raw = _normalize_tgt_for_jieba(tgt_text)
    tgt_words = list(jieba.cut(tgt_raw.replace(" ", ""))) if tgt_raw else []

    if not src_words or not tgt_words:
        return []

    try:
        alignments_dict = align_model.get_word_aligns(src_words, tgt_words)
    except Exception:
        return []

    # intersection: key "inter" (matching_methods="m"); some versions use "mwmf"
    inter = alignments_dict.get("inter") or alignments_dict.get("mwmf")
    if not inter:
        return []

    mono = _to_monotonic_alignments(inter, len(src_words), len(tgt_words))
    # Map tgt_word_idx -> end char index (inclusive) in normalized translation
    norm = "".join(tgt_words)
    cum = 0
    char_end = []
    for w in tgt_words:
        cum += len(w)
        char_end.append(cum - 1)
    return [(s, char_end[t]) for s, t in mono if t < len(char_end)]


# Thresholds for truncation (En->Zh often 1~4 chars/word, e.g. "inevitably"->"不可避免地")
_MAX_TGT_CHARS_PER_OBS_WORD = 5
_ALIGNMENT_SPREAD_THRESHOLD = 12
_ALIGNMENT_VERY_END_MARGIN = 3
_ALIGNMENT_TOO_EARLY_THRESHOLD = 2


def truncate_by_alignment(
    full_src: str,
    observed_src: str,
    translation: str,
    alignments: List[Tuple[int, int]],
) -> str:
    """Truncate translation to the part aligned to observed source.

    alignments: (src_word_idx, tgt_char_idx) from get_word_alignments.
    tgt_char_idx = end character index (inclusive) in normalized (no-space) text.
    Returns a prefix of translation so that core's translation[:len(result)] is
    consistent (norm↔translation offset mapping). Caller typically passes
    already normalized translation (e.g. clean_translation_for_alignment + normalize_zh).
    """
    obs_n = len(observed_src.strip().split())
    if obs_n <= 0 or not translation:
        return ""

    norm = _normalize_tgt_for_jieba(translation).replace(" ", "")
    n_norm = len(norm)
    if not n_norm:
        return ""

    last = obs_n - 1
    last_t = [t for s, t in alignments if s == last]

    def _cap_by_obs_words(raw_len: int) -> int:
        if obs_n <= 0:
            return raw_len
        cap = obs_n * _MAX_TGT_CHARS_PER_OBS_WORD
        return min(raw_len, max(2, cap), n_norm)

    def _fallback_safe_tgt_idx() -> str:
        safe_char_idx = -1
        for s, t in alignments:
            if s < obs_n:
                safe_char_idx = max(safe_char_idx, t)
        if safe_char_idx >= 0:
            raw_len = safe_char_idx + 1
            capped = _cap_by_obs_words(raw_len)
            return _translation_prefix_for_norm_char_count(translation, capped)
        return ""

    def _fallback_ratio() -> str:
        src_total = len(full_src.strip().split())
        if src_total == 0:
            return ""
        ratio = obs_n / src_total
        safe_chars = int(len(norm) * ratio * 0.6)
        safe_chars = min(safe_chars, _cap_by_obs_words(safe_chars))
        return _translation_prefix_for_norm_char_count(translation, safe_chars) if safe_chars >= 1 else ""

    if last_t:
        spread = max(last_t) - min(last_t)
        if spread > _ALIGNMENT_SPREAD_THRESHOLD:
            return _fallback_safe_tgt_idx() or _fallback_ratio()
        t_idx = sorted(last_t)[len(last_t) // 2]
        if t_idx > n_norm - 1 - _ALIGNMENT_VERY_END_MARGIN:
            return _fallback_safe_tgt_idx() or _fallback_ratio()
        if t_idx < _ALIGNMENT_TOO_EARLY_THRESHOLD:
            return _fallback_safe_tgt_idx() or _fallback_ratio()
        raw_len = t_idx + 1
        capped = _cap_by_obs_words(raw_len)
        return _translation_prefix_for_norm_char_count(translation, capped)

    return _fallback_safe_tgt_idx() or _fallback_ratio()


def build_local_alignment_windows(
    full_src: str,
    observed_src: str,
    translation: str,
    committed: str,
    src_left_words: int = 40,
    src_right_words: int = 24,
    tgt_left_chars: int = 120,
) -> Tuple[str, str, str, int]:
    """Build shorter alignment inputs around the observed/future boundary.

    Returns:
      local_full_src, local_observed_src, local_translation, tgt_offset
    """
    src_words = full_src.strip().split()
    observed_count = min(len(observed_src.strip().split()), len(src_words))
    if not src_words or observed_count <= 0:
        return full_src, observed_src, translation, 0

    left = max(1, src_left_words)
    right = max(1, src_right_words)
    start = max(0, observed_count - left)
    end = min(len(src_words), observed_count + right)
    if start >= end:
        return full_src, " ".join(src_words[:observed_count]), translation, 0

    local_full_src = " ".join(src_words[start:end])
    local_observed_src = " ".join(src_words[start:observed_count])

    offset = max(0, len(committed) - max(0, tgt_left_chars))
    local_translation = translation[offset:] if translation else ""
    if len(local_translation) < 8:
        offset = 0
        local_translation = translation

    return local_full_src, local_observed_src, local_translation, offset


def get_word_alignments_batch(
    pairs: List[Tuple[str, str]],
    align_model: Any,
    align_tokenizer: Any,
) -> List[List[Tuple[int, int]]]:
    """Batch: call get_word_alignments for each pair (simalign has no native batch)."""
    return [
        get_word_alignments(src_text, tgt_text, align_model, align_tokenizer)
        for src_text, tgt_text in pairs
    ]


