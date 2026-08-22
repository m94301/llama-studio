#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Gpt Oss 120b Mxfp4  - (1card) 78tg 1244pp q8kv  (2card) 85tg 1870pp 131k
# block_count: 36
# max_context: 131072
# kv_cache_multiplier: 360
# health_timeout: 300
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=2

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive2/lmstudio-community/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port 8120 \
  --ctx-size 131072 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --gpu-layers 999 \
  --threads 12 \
  -fa on \
  --no-mmap \
  -dio
