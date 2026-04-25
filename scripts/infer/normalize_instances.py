#!/usr/bin/env python3
"""Expand per-char delay arrays in a simuleval instances.log to match NFKC-normalized prediction.

omnisteval applies NFKC to `prediction` before splitting into char units, which can grow the
string (e.g. `…` → `...`). The original `delays`/`elapsed` arrays have one entry per char of
the raw prediction, so lengths diverge. Here we rebuild those arrays by repeating each char's
timestamp `len(NFKC(char))` times, then write the normalized prediction alongside.
"""
import argparse
import json
import unicodedata


TIMING_KEYS = ("delays", "elapsed")


def expand_timings(prediction: str, values: list) -> list:
    if len(values) != len(prediction):
        raise ValueError(
            f"timing array length {len(values)} does not match raw prediction length {len(prediction)}"
        )
    out = []
    for char, value in zip(prediction, values):
        out.extend([value] * len(unicodedata.normalize("NFKC", char)))
    return out


def normalize_record(record: dict) -> dict:
    raw = record["prediction"]
    normalized = unicodedata.normalize("NFKC", raw)
    if normalized == raw:
        return record
    for key in TIMING_KEYS:
        if key in record:
            record[key] = expand_timings(raw, record[key])
    record["prediction"] = normalized
    for key in TIMING_KEYS:
        if key in record:
            assert len(record[key]) == len(normalized), (
                f"{key} length {len(record[key])} != normalized prediction length {len(normalized)}"
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to instances.log")
    parser.add_argument("output", help="path to write normalized JSONL")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = normalize_record(json.loads(line))
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
