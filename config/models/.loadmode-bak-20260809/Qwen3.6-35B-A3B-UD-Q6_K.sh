#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Qwen3.6 35b A3b Ud Q6 K MTP - unsloth - ~63-110tg 1700pp 262k f16 (repl IQ4_NL)
# block_count: 41
# max_context: 262144
# kv_cache_multiplier: 186
# health_timeout: 180
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=3

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf \
  --host 0.0.0.0 \
  --port 8300 \
  --ctx-size 262144 \
  --cache-type-k f16 \
  --cache-type-v f16 \
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
