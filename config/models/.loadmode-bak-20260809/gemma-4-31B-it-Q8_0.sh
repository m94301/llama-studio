#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Gemma 4 31b It Q8 0 - unsloth, 23.4tg (beats Q6_K)
# block_count: 60
# max_context: 262144
# kv_cache_multiplier: 483
# === llama-studio:meta END ===
#
# Q8_0, NOT Q6_K -- see ga100_perf.txt [23]. Controlled sweep on this exact
# model (1 card, identical settings) found Q8_0 DOMINATES Q6_K:
#     quant    GB/tok    tg     eff GB/s   pp512
#     Q4_K_M    18.31  28.82        528    774.0
#     Q6_K      25.19  19.53        492    690.6
#     Q8_0      32.62  23.44        765    766.6
#   Q8_0 reads +29% MORE bytes than Q6_K yet decodes +20% FASTER, because
#   Q8_0 dequant is trivial (int8 * fp16 scale, byte-aligned) while K-quants
#   unpack sub-byte fields across super-blocks. Q6_K is worst of the three --
#   6-bit fields are not byte-aligned -- and is slowest on PREFILL too.
#   So Q8_0 is faster AND higher quality than Q6_K. Q6_K was deleted.
#   gemma-4-31B-it-Q4_K_M.sh is kept as the raw-speed option (28.8 tg).
#
# FULL 262144 ctx VERIFIED on ONE card: 55968 MiB (54.7 GiB), 9.3 GiB spare.
#   No context sacrifice vs the old Q6_K config.
#
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=1

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8501 \
  --ctx-size 262144 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --batch-size 512 \
  --threads 12 \
  --no-mmap \
  -dio
