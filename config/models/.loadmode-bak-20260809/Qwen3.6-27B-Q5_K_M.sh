#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Qwen3.6 27b Q5 K M MTP - 47tg 812pp 262k
# block_count: 65
# max_context: 262144
# kv_cache_multiplier: 319
# === llama-studio:meta END ===
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q5_K_M.gguf \
  --host 0.0.0.0 \
  --port 8202 \
  --ctx-size 262144 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --gpu-layers 999 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --threads 12 \
  --parallel 1 \
  -fa on \
  --spec-type draft-mtp \
  --spec-draft-n-max 6 \
  --draft-p-min 0.75 \
  --jinja \
  --chat-template-file /home/m6/projects/llama_studio/templates/chat_template.jinja \
  --no-mmap \
  --direct-io \
  --fit off
