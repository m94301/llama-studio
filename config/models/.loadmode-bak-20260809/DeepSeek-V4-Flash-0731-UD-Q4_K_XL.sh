#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: DeepSeek V4-Flash 0731 UD-Q4_K_XL - 284B/13B, 4card, 1M ctx
# block_count: 43
# max_context: 1048576
# kv_cache_multiplier: 85
# health_timeout: 900
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0,1,2,3

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf \
  --host 0.0.0.0 \
  --port 8134 \
  --ctx-size 1048576 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --split-mode layer \
  --batch-size 2048 \
  --ubatch-size 1024 \
  --threads 12 \
  --parallel 1 \
  --no-mmap \
  -dio
