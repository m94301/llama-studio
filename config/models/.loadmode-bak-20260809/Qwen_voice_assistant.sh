#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Qwen Voice Assistant - 35b A3b Q6_K MTP - card3 :8301 - thinking OFF, English
# block_count: 41
# max_context: 262144
# kv_cache_multiplier: 273
# health_timeout: 180
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=3

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf \
  --host 0.0.0.0 \
  --port 8301 \
  --ctx-size 262144 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --gpu-layers 999 \
  --batch-size 2048 \
  --threads 1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --draft-p-min 0.75 \
  --parallel 1 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --no-mmap \
  -dio \
  -fit off
