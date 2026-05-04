#!/bin/bash
# Run LLMServingSim baseline for v6e × Llama-3.1-8B against real measurements in
# inference_result/Llama-3.1-8B-Instruct__bs{1,2,4}_tp1_seq2048/
#
# Real measurements: simple vLLM script, all 50 requests sent in a burst (no
# external rate limiting). vLLM internally batches up to max_num_seqs.
# JSONL is set up with arrival_ns=0 for all to mirror this.
#
# Run from inside LLMServingSim_ispass26/ where main.py lives. Needs astra-sim
# binary built (docker.sh + compile.sh from a prior session).
set -euo pipefail

REPO=/path/to/Lens-baseline   # adjust for VM
SIM=$REPO/LLMServingSim_ispass26
DATA=$SIM/dataset/v6e_8b_baseline
OUT=$SIM/output/v6e_8b_baseline
CFG=cluster_config/single_node_tpuv6e_8b.json

mkdir -p "$OUT"
cd "$SIM"

for ds in sharegpt cnn writing_prompts; do
  for bs in 1 2 4; do
    echo "=== ${ds}  bs=${bs} ==="
    python3 main.py \
      --cluster-config "$CFG" \
      --dataset "$DATA/${ds}.jsonl" \
      --output "$OUT/${ds}_bs${bs}.csv" \
      --max-batch "$bs" \
      --max-num-batched-tokens 2048 \
      --num-req 50 \
      --log-level WARNING
  done
done

echo "=== done ==="
ls "$OUT"
