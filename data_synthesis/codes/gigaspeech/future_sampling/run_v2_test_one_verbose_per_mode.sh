#!/usr/bin/env bash
# 对每个 selection-mode 用 core_v2 各跑一条 test-one + verbose，看不同 mode 下 word alignment 效果。
# Align 与 base 都在 GPU 0 上。不传参数则用下面默认 path 跑全部 mode。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_TSV="${1:-/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered.tsv}"
OUTPUT_ROOT="${2:-/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/out_v2_test_2}"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "ERROR: input TSV not found: $INPUT_TSV"
  echo "Usage: $0 [input.tsv] [output_root]"
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/data/user_data/haolingp/models}"

for MODE in semantic_merge_vote majority_vote lcp70_llm lcp70_code lcp_code; do
  OUT="$OUTPUT_ROOT/v2_verbose_${MODE}"
  mkdir -p "$OUT"
  echo "========== mode=$MODE -> $OUT =========="
  python llm_future_sampling_final_v2.py \
    --input-tsv "$INPUT_TSV" \
    --output-root "$OUT" \
    --test-one \
    --verbose \
    --selection-mode "$MODE" \
    --consensus-ratio "0.7" \
    --overwrite
  echo ""
done

echo "Done. Verbose logs: $OUTPUT_ROOT/v2_verbose_*/verbose_*.log"
echo "Input: $INPUT_TSV"
echo "Output root: $OUTPUT_ROOT"
