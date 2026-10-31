# llama-studio
WebUI for managing llama-server sessions

This tool allows you to launch various llm models onto various llama-server sessions, fiddle with and save custom configurations, and launch llama-server sessions to various GPU up the the VRAM limit.  It is intended for the use case where you want fixed models on fixed ports for interaction with other toolsets, but can be used to play around as well.

Currently tested only on NVIDIA devices.  Consider this a MVP (Minimum Viable Product) release

# What's new in 0.2.0
- **Model configs converted to executable shell scripts. No more JSON for models** Each model lives at `config/models/<name>.sh`. Self-contained, portable, runnable directly from the CLI. Existing JSON configs are migrated automatically on first start (originals kept as `.json.old`).
- **Paste-in support.** The config modal has an Edit textarea: paste an Unsloth-style snippet and it parses straight into the form.
- **Multi-GPU model splitting.** Add `--tensor-split a,b,…` to a model's args and the GPU picker switches to multi-slot mode. Per-GPU VRAM share is honored on availability checks.
- **Auto-load on startup.** "Store this state to autoload" snapshots the currently-loaded set into `app.json`; toggle "Load on startup" and they come back on next launch. Runs in the background so the WebUI is responsive while models load.
- **Save-while-loaded warning.** Editing a loaded model's config prompts to unload or keep the prior load.
- **VRAM calculator fix.** Cross-model contamination bug squashed (every modal now reads fresh values).

# Gallery
## Main Page
- GPU Panel: Shows status of all GPU, including Power, Temp, VRAM Usage, and any sessions running on this GPU with URL to llama-server's mini webUI for testing
- Model Table: Shows status of all scanned models.  Click to Load, Unload, View console Log, or edit Configuration/Launch Args
<p align="left">
<img width="512" alt="1_llama_server_main" src="https://github.com/user-attachments/assets/e9b666c9-6c32-4829-8777-2dd2a731562e" />
</p>

## Simple Setup
Just point out the path of llama-server binary and model dir.  The llama-server binary will be tested with -version and --help to extract all command line arguments, and they will be stored for easy confguration later.  The model dir will be scanned for GGUF files and they will be added to the model table for config & launching.
<p align="left">
<img width="512"  alt="1a_initial_setup" src="https://github.com/user-attachments/assets/bc59114c-f251-49ff-801d-12568205b48a" />

</p>

## GPU Selection
Clicking Load on a model allows selection of GPU.  Load as many sessions as your GPU will hold.
In the example here, Qwen 3.6 is too big to fit on GPU 0 & 1, but could load to 2 or 3.  Adjust KV quant and context to pack more models per GPU, or dedicate whole GPU to maximize context.
<p align="left">
  <img width="512" alt="2_llama_server_gpu_selection" src="https://github.com/user-attachments/assets/c57a8a69-e7ee-4217-9487-e3808a88b554" />
</p> 

## Model Configuration
Since the tool is running llama-server sessions, the config is based around launch args.  But there is a rudimentary VRAM calculator for optimizing context and KV quantization in order to pack as much context into your VRAM as you can, without enlessly failing loads.
<p align="left">
<img width="512" alt="3_model_config" src="https://github.com/user-attachments/assets/9c19e4dd-07c4-4a8d-b1ad-7a2a7652890d" />
</p>

## Option Browser
Uses the real arg list parsed from **llama-server --help** to make a clickable, searchable table of options to add to launch args.  Nice if you don't like memorizing CLI args
<p align="left">
<img width="512" alt="4_option_browser" src="https://github.com/user-attachments/assets/33c6ee23-7471-41b9-b542-f58ae2a0b0d0" />
</p>


# Quickstart
## 1. Clone the repository
git clone https://github.com/m94301/llama-studio.git

cd llama-studio

## 2. Create a virtual environment (So my python imports don't pollute your main environment)
python3 -m venv venv

source venv/bin/activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Run the app (with setup guide)
./start.sh

# Requirements (For reference)
You can see it in requirements.txt, but just to note it here.  Tried to keep it as light as possible
- fastapi==0.115.5
- uvicorn[standard]==0.30.6
- pydantic==2.10.4
- pynvml>=12.0.0
- httpx==0.28.1
- python-multipart==0.0.9
- jinja2>=3.1.0
- gguf_parser>=0.0.6


## Writing config scripts (`config/models/*.sh`) — rules for Claude sessions
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
7. **Portable-bundle engine** (`~/projects/vllm-qfn-portable`, Qwen/Ornith family): `BUNDLE=/home/m6/projects/vllm-qfn-portable; VENV=$BUNDLE/engine/.venv`, then export `CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13`,
   `PATH=$VENV/bin:$CUDA_HOME/bin:$PATH`, `LD_LIBRARY_PATH=$CUDA_HOME/lib...`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `NCCL_ALGO=Ring NCCL_PROTO=Simple`, `HF_HUB_OFFLINE=1`.
   Templates that parse cleanly: `Ornith-1.5-122B-AWQ-INT4-2card.sh` (plain), `Swift-1.5-Qwen3.8-Flash-Next-2card-PLE.sh` (PLE table on SSD, MTP + n-gram, PP partition env).
8. **Check before telling the user it works:** run the parser (does not launch anything):
   `cd ~/projects/llama_studio && ./venv/bin/python -c "import sys,json,glob; sys.path.insert(0,'backend'); import launch_script as L; L.set_vllm_manifest(json.load(open(glob.glob('config/vllm/manifest-*.json')[0]))); ps=L.parse_script(open('config/models/NAME.sh').read()); print(ps.engine, ps.warnings, ps.is_configured(), ps.gpu_count, ps.calibrated_gpus)"`
   Expect no stray-token warning, `is_configured` `(True, None)` and `gpu_count == calibrated_gpus`.
