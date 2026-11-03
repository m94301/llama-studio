#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: GLM-4.5-Air Q6_K 2card - 39tg 1181pp f16kv 128k
# block_count: 47
# max_context: 131072
# kv_cache_multiplier: 1024
# health_timeout: 300
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0,1

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive2/unsloth/GLM-4.5-Air-GGUF/GLM-4.5-Air-Q6_K-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port 8140 \
  --ctx-size 131072 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --split-mode layer \
  --threads 12 \
  -fa on \
  --no-mmap \
  --direct-io
