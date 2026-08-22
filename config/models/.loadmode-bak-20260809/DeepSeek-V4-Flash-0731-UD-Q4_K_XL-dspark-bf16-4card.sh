#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: DeepSeek V4-Flash 0731 UD-Q4_K_XL +DSpark BF16 - 284B/13B, 4card, 1M ctx
# block_count: 43
# max_context: 1048576
# kv_cache_multiplier: 85
# health_timeout: 900
# === llama-studio:meta END ===
# A/B TEST: this is the plain 4-card 1M-ctx config (baseline ~28 t/s tg) with the
# unsloth BF16 DSpark speculative drafter bolted on. Every non-spec flag is IDENTICAL
# to DeepSeek-V4-Flash-0731-UD-Q4_K_XL.sh so any tg delta is attributable to the drafter.
#
# Drafter: unsloth dspark-DeepSeek-V4-Flash-0731-BF16.gguf (11.3GB, arch dflash, 81
# tensors, lossless BF16). Unsloth reports 0.764 accept @ n=3, 1.91x on a single B200.
# Optimal --spec-draft-n-max 3 (n-max 5 too deep for the Gen1x4 riser link, per prior
# am17an testing). Requires llama.cpp build >= b10269 (current rebuild is newer).
#
# VRAM: the earlier am17an drafter OOM-crashed on long-context prefill on 3 cards
# (feature buffer ~ prompt_len * 3 * 4096 * 4B, ~13GB at 256K tok). Two things change
# that here: (1) 4-card layer split leaves more headroom, (2) upstream #26605 "fit: Fix
# memory allocation for MTP layers" is in this build. If it still OOMs on a very long
# prefill, drop --ctx-size to leave room for the DSpark prefill buffer.
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=0,1,2,3

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf \
  --model-draft /home/m6/models/drive1/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  --gpu-layers-draft 999 \
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
