import argparse
import os
import json
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm

LANG_CONFIG = {
    'zh': {'subdir': 'hibiki-13k',    'name': 'Chinese',  'join': ''},
    'de': {'subdir': 'hibiki_de_13k', 'name': 'German',   'join': ' '},
    'ja': {'subdir': 'hibiki_ja_13k', 'name': 'Japanese', 'join': ''},
}

parser = argparse.ArgumentParser()
parser.add_argument('--lang', choices=sorted(LANG_CONFIG.keys()), default='zh',
                    help='Target language for translation.')
parser.add_argument('--multiplier-upper', type=int, default=12,
                    help='Inclusive upper bound for the random multiplier (sampled from [1, multiplier_upper]).')
args = parser.parse_args()
assert args.multiplier_upper >= 1, "--multiplier-upper must be >= 1"

lang_cfg = LANG_CONFIG[args.lang]

tsv_path = '/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv'
orig_manifest = pd.read_csv(tsv_path, sep='\t')

manifest_root = f"/data/group_data/li_lab/haolingp/data_synthesis/gigaspeech/hibiki/{lang_cfg['subdir']}"
audio_clips_root = f"/data/group_data/li_lab/siqiouya/datasets/gigaspeech/audio_clips_{args.lang}_hibiki/"
output_filename = f"train_s_{args.lang}-hibiki_maxm{args.multiplier_upper}.jsonl"

output_root = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/"
os.makedirs(audio_clips_root, exist_ok=True)
os.makedirs(output_root, exist_ok=True)

json_files = [f for f in os.listdir(manifest_root) if f.endswith('.json')]

instances = []
n_skip = 0

pbar = tqdm(json_files, desc="Processing, skipped 0 instances")
for json_file in pbar:
    with open(os.path.join(manifest_root, json_file), 'r') as f:
        data = json.load(f)

    for item in data:
        utt_id = item['id']
        target_trajectory = item['target_trajectory']

        rows = orig_manifest[orig_manifest['id'] == utt_id]
        if len(rows) == 0:
            n_skip += 1
            pbar.set_description(f"Processing, skipped {n_skip} instances")
            continue

        audio_path, start, duration = rows.iloc[0]['audio'].split(':')
        wav, sr = sf.read(audio_path, start=int(start), frames=int(duration))

        assert sr == 16000
        multiplier = np.random.randint(1, args.multiplier_upper + 1)
        stepsize = 15360 * multiplier

        audio_id, segment_id = utt_id.split('_')

        audio_clips_dir = os.path.join(audio_clips_root, audio_id, segment_id, f"multiplier_{multiplier}")
        os.makedirs(audio_clips_dir, exist_ok=True)
        audio_clip_paths = []

        for idx, i in enumerate(range(0, wav.shape[0], stepsize)):
            wav_clip = wav[i : i + stepsize]
            clip_path = os.path.join(audio_clips_dir, f"{idx}.wav")
            sf.write(clip_path, wav_clip, sr)
            audio_clip_paths.append(clip_path)

        targets = []
        for i in range(0, len(target_trajectory), multiplier):
            chunk = target_trajectory[i:i+multiplier]
            if lang_cfg['join']:
                chunk = [s.strip() for s in chunk if s.strip()]
            targets.append(lang_cfg['join'].join(chunk))

        if len(audio_clip_paths) != len(targets):
            n_skip += 1
            pbar.set_description(f"Processing, skipped {n_skip} instances")
            continue

        messages = [
            {"role": "system", "content": f"You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into {lang_cfg['name']} text."},
        ]
        for target in targets:
            messages.append({"role": "user", "content": "<audio>"})
            messages.append({"role": "assistant", "content": target})
        instance = {
            "messages": messages,
            "audios": audio_clip_paths,
            "multiplier": multiplier,
        }
        instances.append(instance)

with open(os.path.join(output_root, output_filename), 'w') as f:
    for instance in instances:
        f.write(json.dumps(instance, ensure_ascii=False) + "\n")

"""
for lang in zh de ja; do
    python scripts/train/convert2swift_hibiki.py \
        --multiplier-upper 1 \
        --lang ${lang}
done
"""