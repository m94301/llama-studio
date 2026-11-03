#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Gemma 4 26b A4b It Ud Q6 K - 262k reasoning off
# block_count: 30
# max_context: 262144
# kv_cache_multiplier: 239
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=3

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --host 0.0.0.0 \
  --port 8500 \
  --ctx-size 262144 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --threads 12 \
  --reasoning off \
  --no-mmap \
  -dio
