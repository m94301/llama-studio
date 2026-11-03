#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Gemma 4 31b It Q4 K M - unsloth, 28.8tg (fastest 31B)
# block_count: 60
# max_context: 262144
# kv_cache_multiplier: 483
# === llama-studio:meta END ===
#
# The RAW SPEED option for gemma-4-31B. See ga100_perf.txt [23]:
#   Q4_K_M 28.82 tg | Q6_K 19.53 (deleted, dominated) | Q8_0 23.44
# Q4_K_M is fastest simply because it reads the fewest bytes (18.31 GB/tok).
# Its effective bandwidth (528 GB/s) is K-quant-typical -- the win is bytes,
# not efficiency. Use gemma-4-31B-it-Q8_0.sh when quality matters more:
# Q8_0 is only -19% tg but a much better quant.
#
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=1

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8502 \
  --ctx-size 262144 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --batch-size 512 \
  --threads 12 \
  --no-mmap \
  -dio
