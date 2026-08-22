#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: MiniMax M2.7 IQ4 XS 2card - 38tg 800pp q8kv 160k
# block_count: 62
# max_context: 196608
# kv_cache_multiplier: 1024
# health_timeout: 600
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=1,2

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  --host 0.0.0.0 \
  --port 8130 \
  --ctx-size 160768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --gpu-layers 999 \
  --split-mode layer \
  --threads 12 \
  -fa on \
  --no-mmap \
  -dio
