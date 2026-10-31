# Llama Studio — notes for Claude sessions

Solo project: commit directly on main. Saving `backend/*.py` reloads uvicorn and
unloads every running model, so ask before editing backend code while models are loaded.

## Writing config scripts (`config/models/*.sh`)
The UI parses these scripts and rewrites the launch line when saved, so a script that does not follow the format shows "This script will not start as written". Rules:
1. **File:** `config/models/<Name>.sh`, `chmod +x`. Retired scripts become `.sh.old`; the UI keeps edit backups in `.backups/`.
2. **Meta fence first**, exactly `# === llama-studio:meta BEGIN ===` ... `# === llama-studio:meta END ===`, keys `# key: value`:
   `display_name`, `engine` (`vllm` or `llamacpp`, mandatory), `health_timeout`. vLLM calibration keys (optional, the UI re-measures): `kv_bytes_per_token`,
   `calibrated_gpus`, `weights_max_gb`, `runtime_overhead_gb`, `calibrated_partition` (e.g. `24,24` when `VLLM_PP_LAYER_PARTITION` is exported). llama.cpp: `block_count`, `max_context`, `kv_cache_multiplier`.
3. **GPUs:** `export CUDA_VISIBLE_DEVICES=0,1` as a plain line near the top. The UI derives the GPU count from it; it must equal `calibrated_gpus` or the measurements are discarded.
4. **Preamble:** only comments, `export VAR=value`, `VAR=value` assignments and harmless commands (`set -euo pipefail`, `ulimit`). The parser resolves `$VAR`/`${VAR}`/`${VAR:-default}` from these assignments.
5. **ONE launch line, native, no wrappers:** `exec "$VENV/bin/vllm" serve /abs/path/to/model-dir --flag value \` (continuation lines OK). The model dir is the positional right after `serve`; the binary path must end in `/vllm`.
   Never launch through another script (`exec .../run-xyz.sh`): the UI mistakes it for the engine, adds `serve` + default flags ("stray non-flag token 'serve'") and the flags are not forwarded anyway. The first line starting with `exec ` is taken as the invocation, so no earlier non-comment line may contain `llama-server`, `$LLAMA_BIN` or `vllm serve`.
6. **vLLM flags:** put every setting in the exec line (env that the engine needs goes in `export` lines above). JSON values in single quotes (`--additional-config '{...}'`, `--speculative-config '{...}'`).
   `--kv-cache-memory-bytes N` is required or the UI says "not configured" (take it from the log line `Available KV cache memory`, round down). `--max-model-len`, `--gpu-memory-utilization` etc. are then edited in the UI.
   **Flags are validated against the manifest of the engine named in the script's own `exec` line** (since 2026-10-31): the first time llama_studio sees a vLLM binary it runs `vllm serve --help=all`
   once and caches `config/vllm/manifest-<version>-<pathhash>.json` (the path hash keeps engines that report the same placeholder version, e.g. `0.0.0+precompiled`, apart). Pointing a script at a new engine
   therefore needs no setup; `app.json.vllm_binary` is only the default for new skeletons and the fallback when a binary can't be probed. Never hand-edit a manifest. An "unknown vllm option" message now means the
   flag really does not exist in that script's engine. Restart llama_studio after code changes; manifests are read at startup.
7. **Machine-specific engines** (venv paths, bundle layouts) are described in `CLAUDE.local.md`, which is not committed.
8. **Check before telling the user it works** (parses only, launches nothing):
   `venv/bin/python backend/check_script.py config/models/NAME.sh` — prints the engine, GPU count and port, and lists
   parse warnings, an unconfigured reason, or a `calibrated_gpus` mismatch. Exit status 0 means clean.

## Refactor safety net
`venv/bin/python tests/snapshot_real_scripts.py check` compares parse results, derived VRAM/GPU numbers and
no-change-save output for every active script against a recorded baseline (`record` to re-baseline, only after
reviewing the diff). Run it with the test suite before committing backend changes.
