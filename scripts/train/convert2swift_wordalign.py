import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

LANG_CONFIG = {
    "zh": {"name": "Chinese",  "join": ""},
    "de": {"name": "German",   "join": " "},
    "ja": {"name": "Japanese", "join": ""},
}

CHUNK_SAMPLES = 15360  # 960 ms at 16 kHz

DATASET_ROOT = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech"
DEFAULT_TSV_TEMPLATE = f"{DATASET_ROOT}/manifests/train_xl_case_robust_asr-filtered_{{lang}}_metricx-qe3.0_align.tsv"
DEFAULT_EXCLUDE_TEMPLATE = f"{DATASET_ROOT}/manifests_rag/train_s_{{lang}}_origin.jsonl"
DEFAULT_OUTPUT_TEMPLATE = f"{DATASET_ROOT}/manifests_rag/train_s_{{lang}}_origin_2.jsonl"
DEFAULT_AUDIO_CLIPS_TEMPLATE = f"{DATASET_ROOT}/audio_clips_{{lang}}/"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build train_s_{lang}_origin_2.jsonl directly from the align TSV, "
                    "excluding utt_ids already used in train_s_{lang}_origin.jsonl."
    )
    p.add_argument("--lang", required=True, choices=sorted(LANG_CONFIG.keys()))
    p.add_argument("--tsv-path", default=None)
    p.add_argument("--exclude-jsonl", default=None)
    p.add_argument("--output-path", default=None)
    p.add_argument("--audio-clips-root", default=None)
    p.add_argument("--num-samples", type=int, default=12500)
    p.add_argument("--multiplier-upper", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.tsv_path is None:
        args.tsv_path = DEFAULT_TSV_TEMPLATE.format(lang=args.lang)
    if args.exclude_jsonl is None:
        args.exclude_jsonl = DEFAULT_EXCLUDE_TEMPLATE.format(lang=args.lang)
    if args.output_path is None:
        args.output_path = DEFAULT_OUTPUT_TEMPLATE.format(lang=args.lang)
    if args.audio_clips_root is None:
        args.audio_clips_root = DEFAULT_AUDIO_CLIPS_TEMPLATE.format(lang=args.lang)
    assert args.multiplier_upper >= 1
    return args


def load_used_utt_ids(jsonl_path, lang):
    marker = f"audio_clips_{lang}"
    used = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            parts = obj["audios"][0].split("/")
            i = parts.index(marker)
            used.add(f"{parts[i + 1]}_{parts[i + 2]}")
    return used


def join_chunk(chunk, join_str):
    if join_str:
        chunk = [s.strip() for s in chunk if s.strip()]
    return join_str.join(chunk)


def main():
    args = parse_args()
    lang_cfg = LANG_CONFIG[args.lang]
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.audio_clips_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    used_ids = load_used_utt_ids(args.exclude_jsonl, args.lang)
    print(f"Loaded {len(used_ids)} used utt_ids from {args.exclude_jsonl}")

    df = pd.read_csv(args.tsv_path, sep="\t")
    print(f"TSV rows: {len(df)}")

    df = df[~df["id"].isin(used_ids)].reset_index(drop=True)
    print(f"Available non-overlapping rows: {len(df)}")

    if args.num_samples > len(df):
        raise ValueError(
            f"--num-samples {args.num_samples} exceeds available {len(df)} non-overlapping rows"
        )

    sample_idx = rng.choice(len(df), size=args.num_samples, replace=False)
    df = df.iloc[sample_idx].reset_index(drop=True)

    system_prompt = (
        "You are a professional simultaneous interpreter. "
        f"You will be given chunks of English audio and you need to translate the audio into {lang_cfg['name']} text."
    )

    instances = []
    n_skip_parse = 0
    n_skip_mismatch = 0

    pbar = tqdm(range(len(df)), desc="Processing")
    for idx in pbar:
        row = df.iloc[idx]
        utt_id = row["id"]

        try:
            trajectory = ast.literal_eval(row["trajectory"])
        except (ValueError, SyntaxError, TypeError):
            n_skip_parse += 1
            continue
        if not isinstance(trajectory, list) or len(trajectory) == 0:
            n_skip_parse += 1
            continue

        audio_path, start, duration = row["audio"].split(":")
        wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))
        assert sr == 16000

        multiplier = int(rng.integers(1, args.multiplier_upper + 1))
        stepsize = CHUNK_SAMPLES * multiplier

        audio_id, segment_id = utt_id.split("_")
        audio_clips_dir = os.path.join(args.audio_clips_root, audio_id, segment_id)
        os.makedirs(audio_clips_dir, exist_ok=True)

        audio_clip_paths = []
        for clip_idx, i in enumerate(range(0, wav.shape[0], stepsize)):
            wav_clip = wav[i : i + stepsize]
            clip_path = os.path.join(audio_clips_dir, f"{clip_idx}.wav")
            sf.write(clip_path, wav_clip, sr)
            audio_clip_paths.append(clip_path)

        targets = []
        for i in range(0, len(trajectory), multiplier):
            targets.append(join_chunk(trajectory[i : i + multiplier], lang_cfg["join"]))

        if len(audio_clip_paths) != len(targets):
            n_skip_mismatch += 1
            continue

        messages = [{"role": "system", "content": system_prompt}]
        for target in targets:
            messages.append({"role": "user", "content": "<audio>"})
            messages.append({"role": "assistant", "content": target})

        instances.append({
            "messages": messages,
            "audios": audio_clip_paths,
            "merge_multiplier": multiplier,
        })

    with open(args.output_path, "w") as f:
        for instance in instances:
            f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(instances)} instances to {args.output_path} "
        f"(skipped {n_skip_parse} parse, {n_skip_mismatch} chunk/target mismatch)."
    )


if __name__ == "__main__":
    main()

"""
for lang in zh de ja; do
    python scripts/train/convert2swift_wordalign.py --lang ${lang}
done
"""