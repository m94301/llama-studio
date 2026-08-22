#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Laguna S 2.1 Q4 K M
# block_count: 48
# max_context: 262144
# kv_cache_multiplier: 2560
# === llama-studio:meta END ===

LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive2/poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --gpu-layers 999 \
  --ctx-size 4096 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --batch-size 512 \
  --threads 12
