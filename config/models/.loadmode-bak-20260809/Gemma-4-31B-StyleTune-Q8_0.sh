#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Gemma-4 31B StyleTune Q8_0 - 2card SWA-FULL - 131k q8kv, 50ms warm TTFT, 23tg
# block_count: 60
# max_context: 262144
# kv_cache_multiplier: 688
# health_timeout: 300
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/bartowski/Gryphe_Gemma-4-31B-StyleTune-GGUF/Gryphe_Gemma-4-31B-StyleTune-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8500 \
  --ctx-size 65536 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --gpu-layers 999 \
  --batch-size 512 \
  --threads 12 \
  -fa on \
  --reasoning off \
  --swa-full \
  --no-mmap \
  --direct-io
