#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Violet Lyra Gutenberg V2 Q8 0 - Nemo 12B RP merge, ChatML
# block_count: 40
# max_context: 131072
# kv_cache_multiplier: 1101
# === llama-studio:meta END ===
# NOTE: this GGUF ships with NO tokenizer.chat_template (mradermacher quant).
# The model card specifies ChatML, so it is set explicitly below. The tokenizer
# is Mistral-Nemo's, so <|im_start|>/<|im_end|> are NOT real tokens -- if the
# model runs on past its turn, add <|im_end|> as a stop string.
LLAMA_BIN=/home/m6/projects/llama.cpp/build-cuda-avx1/bin/llama-server
export CUDA_VISIBLE_DEVICES=3

exec "$LLAMA_BIN" \
  -m /home/m6/models/drive1/mradermacher/Violet-Lyra-Gutenberg-v2-GGUF/Violet-Lyra-Gutenberg-v2.Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8400 \
  --ctx-size 131072 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --gpu-layers 999 \
  --batch-size 512 \
  --threads 12 \
  --chat-template chatml \
  --no-mmap \
  -dio
