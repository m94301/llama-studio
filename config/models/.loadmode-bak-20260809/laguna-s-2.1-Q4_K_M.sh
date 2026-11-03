#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Laguna S 2.1 Q4_K_M 2card - poolside fork (laguna branch)
# block_count: 48
# max_context: 262144
# kv_cache_multiplier: 512
# health_timeout: 600
# === llama-studio:meta END ===

# NOTE: this model requires poolside's llama.cpp fork (branch `laguna`) --
# the `laguna` GGUF architecture is not supported by mainline llama.cpp yet.
# Do NOT point this at ~/projects/llama.cpp/build-cuda-avx1 (the daily-driver
# build) -- it will not recognize the arch.
LLAMA_BIN=/home/m6/projects/llama.cpp-laguna/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0,1

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive2/poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8160 \
  --ctx-size 262144 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --split-mode layer \
  --threads 12 \
  -fa on \
  --no-mmap \
  -dio
