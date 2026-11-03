#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: DeepSeek V4-Flash 0731 UD-Q4_K_XL +DSpark spec - 284B/13B, 3card, 1M ctx
# block_count: 43
# max_context: 1048576
# kv_cache_multiplier: 85
# health_timeout: 900
# === llama-studio:meta END ===
# DSpark speculation: am17an's native 0731 DSpark drafter (arch dflash, Markov head,
# 10.9GB) paired with the 0731 target. ~70-78% draft acceptance -> ~40 t/s tg
# (vs ~27 no-spec, +50%). n-max=3 optimal here (n-max 5 too deep for the Gen1x4 link).
# Requires PR #25784.
# WARNING: at ctx 1M this loads to ~98% VRAM (only ~1GB free on cards 0/1). It is
# STABLE for short prompts but OOM-CRASHES on long-context prefill: DSpark allocates a
# feature buffer ~ prompt_len * 3 * 4096 * 4 bytes (~13GB at 256K tokens). Verified crash
# at a 256K-token prefill. For long-context work use the plain (no-spec) config instead,
# or lower --ctx-size here to leave headroom for the DSpark prefill buffer.
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0,1,2

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf \
  --model-draft /home/m6/models/drive1/am17an/DeepseekV4-Flash-20260731-DSpark/DeepseekV4-Flash-20260731-DSpark.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  --fit off \
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
