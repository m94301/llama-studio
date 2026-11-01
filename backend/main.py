"""FastAPI application for Llama Studio."""

import asyncio
import logging
import json
import re
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional
import os
import sys
import argparse
import socket

import httpx

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Update on each release. Surfaced in the page <title> and header.
APP_VERSION = "0.4.0"

from fastapi import FastAPI, BackgroundTasks, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config_manager import ConfigManager, ModelConfig
from kv_calc import llama_kv_gb
from engine_flags import flag_manifest, install_vllm_parse_manifests, validate_llama_option
from launch_script import (
    canonical_args,
    parse_pasted_script,
    patch_meta_fence,
    manifest_for_binary,
)
import chat_proxy
from gpu_manager import GpuManager, ModelState, LoadingPhase
from templates import jinja_env
from event_bus import event_bus
from llama_version import get_vllm_version

# Get project root (one level up from backend directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Parse command line arguments for verbose logging
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
args, _ = parser.parse_known_args()

# Also check environment variable (set by start.sh)
verbose_env = os.environ.get("VERBOSE", "").lower() in ("--verbose", "true", "1")
verbose_mode = args.verbose or verbose_env

# Configure logging
log_level = logging.DEBUG if verbose_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Log startup info
if verbose_mode:
    logger.debug("🔍 Verbose logging enabled")

# Suppress verbose HTTP access logs to reduce noise and keep app logs visible
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("starlette.middleware.base").setLevel(logging.WARNING)
# httpx logs every request at INFO: the chat page heartbeat (/v1/models every
# 5s per open tab) and load health checks would flood the log.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize managers
config_manager = ConfigManager(PROJECT_ROOT / "config" / "app.json", project_root=PROJECT_ROOT)
gpu_manager = GpuManager(config_manager)

# Progress state for the startup auto-load background task. Cleared when complete.
AUTO_LOAD_PROGRESS: Dict[str, int] = {"active": 0, "current_index": 0, "total": 0}
AUTO_LOAD_CURRENT_NAME: str = ""

# Progress state for the synchronous rescan operation. Polled by the UI via
# /api/rescan-progress while the rescan request is in flight.
RESCAN_STATE: Dict[str, object] = {"active": False, "current": "", "done": 0, "total": 0}

# Cached machine IPv4, resolved once on first use. The GPU panel embeds this in
# per-model llama-server URLs; it doesn't change while the server runs, so we
# avoid opening a probe socket on every panel render (polled ~1/s per client).
_SERVER_IPV4: Optional[str] = None


def get_server_ipv4() -> str:
    """Return the machine's local IPv4, resolving + caching it on first call."""
    global _SERVER_IPV4
    if _SERVER_IPV4 is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # No packets sent; just picks the outbound iface.
            _SERVER_IPV4 = s.getsockname()[0]
            s.close()
        except Exception:
            _SERVER_IPV4 = "localhost"
    return _SERVER_IPV4


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("🦙 Llama Studio Starting")
    logger.info("=" * 60)

    try:
        await config_manager.load_app_config()
        # Before scanning: a vLLM script's flags decide which tokens are values
        # and how -tp/-pp normalize, so each is parsed against its own engine's
        # manifest (captured + cached on first sight), else the default engine's.
        install_vllm_parse_manifests(_config_dir(), lambda: config_manager.app_config.vllm_binary)

        await config_manager.load_all_models()

        await gpu_manager.initialize()

        logger.info("✓ Initialization complete")
        logger.info("=" * 60)
        logger.info(f"🌐 Llama Studio WebUI started at: http://localhost:{config_manager.app_config.webui_port}")
        logger.info("=" * 60)

        # Auto-load stored models as a background task so the WebUI is reachable
        # immediately. Each load is async (health-check polling, etc.) and can
        # take 30-120s; the banner endpoint surfaces progress to the UI.
        if config_manager.app_config.auto_load_on_startup and config_manager.app_config.startup_models:
            asyncio.create_task(_auto_load_startup_models())

        # Warm the vllm version cache in the background so the first Settings
        # open doesn't pay for `vllm --version` importing torch.
        if config_manager.app_config.vllm_binary:
            asyncio.create_task(
                asyncio.to_thread(get_vllm_version, config_manager.app_config.vllm_binary)
            )
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("🦙 Llama Studio Shutting Down")
    logger.info("=" * 60)
    try:
        await gpu_manager.cleanup(keep_alive=config_manager.app_config.keep_sessions_on_shutdown)
    except Exception as e:
        logger.error(f"✗ Error during cleanup: {e}")

    logger.info("✓ Shutdown complete")


def _config_dir() -> Path:
    return config_manager.project_root / "config"


def _flags(engine: str, binary: Optional[str]) -> Optional[dict]:
    """The one flag lookup (engine_flags.flag_manifest) against this app's config dir."""
    return flag_manifest(engine, binary, _config_dir())


def _llama_version_label(manifest: Optional[dict]) -> str:
    """'11067_fe93f196b' -> '11067 (fe93f196b)'; 'unknown' when it couldn't be probed."""
    version = (manifest or {}).get("version")
    if not version:
        return "unknown"
    parts = version.split("_")
    return f"{parts[0]} ({parts[1]})" if len(parts) == 2 else version


app = FastAPI(title="Llama Studio", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = PROJECT_ROOT / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning(f"⚠ Static files directory not found: {static_dir}")


def _render(template: str, **ctx) -> str:
    return jinja_env.get_template(template).render(**ctx)


def _error_fragment(message: str, status_code: int = 500) -> HTMLResponse:
    """Inline error shown in place of a fragment that failed to render."""
    return HTMLResponse(_render("snippets/error.html", message=message), status_code=status_code)


# ============================================================================
# API Routes - GPU Panel
# ============================================================================


@app.get("/api/gpu-panel", response_class=HTMLResponse)
async def gpu_panel():
    """Return GPU visualization HTML."""
    gpu_data = gpu_manager.get_gpu_status()

    if not gpu_data:
        return _render("snippets/empty_state.html", message="GPU detection not available")

    rows = [
        (gpu_id, info, (info["allocated"] / info["memory"]) * 100 if info["memory"] > 0 else 0)
        for gpu_id, info in sorted(gpu_data.items())
    ]
    # Machine's local IPv4 (resolved once, then cached) for per-model server links.
    return _render("snippets/gpu_panel.html", rows=rows, server_host=get_server_ipv4())


# ============================================================================
# API Routes - Model List
# ============================================================================


@app.get("/api/model-list", response_class=HTMLResponse)
async def model_list():
    """Return model list HTML using templates."""
    models = config_manager.get_all_models()
    logger.info(f"📋 model_list called: {len(models)} models found")

    # Check if llama-server is configured
    llama_binary = config_manager.app_config.llama_server_binary
    if not Path(llama_binary).exists():
        return _render("snippets/empty_state.html", warn=True,
                       message="⚠ Llama-server binary not configured",
                       hint="Configure path in settings to load models")

    if not models:
        logger.warning("⚠ No models in config_manager")
        return _render("snippets/empty_state.html", message="No models found")

    # Sort: configured first, then unconfigured; within each group alphabetical
    # by the *display name* the table actually shows, case-insensitively (so
    # e.g. "gemma" and "GLM" interleave the way a human reads them).
    sorted_models = sorted(
        models.items(),
        key=lambda x: (not x[1].is_configured,
                       (x[1].display_name or x[0]).lower())
    )

    rows = []
    for model_name, model_config in sorted_models:
        port_display = f":{model_config.port}" if model_config.port else "—"
        vram = model_config.total_need_gb
        vram_display = f"{vram} GB" if vram else "—"
        rows.append((model_name, gpu_manager.get_model_state(model_name).value, model_config,
                     port_display, vram_display, str(config_manager.get_log_path(model_name))))
    try:
        html = _render("snippets/model_list.html", rows=rows)
    except Exception as e:
        logger.error(f"✗ Error rendering model rows: {type(e).__name__}: {e}", exc_info=True)
        return _error_fragment(f"Error: {e}", 200)
    logger.info(f"✓ Generated model list HTML: {len(html)} chars, {len(rows)} models")
    return html


# ============================================================================
# API Routes - Model Status Inner
# ============================================================================


@app.get("/api/get-model-status-inner", response_class=HTMLResponse)
async def get_model_status_inner(model: str):
    """Return status badge + action button HTML for inner polling."""
    if model not in config_manager.get_all_models():
        return _error_fragment("Model not found", 404)

    state = gpu_manager.get_model_state(model)
    status = state.value
    model_config = config_manager.get_model_config(model)

    # Get error message for failed state
    error_msg = None
    if status == "failed":
        snapshot = gpu_manager.state_info.get(model)
        if snapshot:
            error_msg = snapshot.error_msg

    template = jinja_env.get_template("snippets/model_status_inner.html")
    return HTMLResponse(template.render(
        model_name=model,
        status=status,
        model_config=model_config,
        error_msg=error_msg,
    ))


# ============================================================================
# HTML Routes
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page with config status."""
    # Check if configuration is complete
    needs_config, error_msg = config_manager.needs_configuration()
    config_incomplete = needs_config

    # Render template with config status
    template = jinja_env.get_template("index.html")
    return template.render(
        config_incomplete=config_incomplete,
        config_error=error_msg,
        app_version=APP_VERSION,
        store_button_html=_render_store_button(),
        auto_load_on_startup=config_manager.app_config.auto_load_on_startup,
        keep_sessions_on_shutdown=config_manager.app_config.keep_sessions_on_shutdown,
    )


# ============================================================================
# API Routes - Rescan Models
# ============================================================================


@app.get("/api/rescan-progress", response_class=HTMLResponse)
async def rescan_progress():
    """Return current rescan status — polled by the UI while a rescan is in flight."""
    if not RESCAN_STATE.get("active"):
        return HTMLResponse("")
    current = RESCAN_STATE.get("current") or "…"
    done = RESCAN_STATE.get("done", 0)
    total = RESCAN_STATE.get("total", 0)
    return HTMLResponse(_render("snippets/status_line.html", tone="info",
                                text=f"⏳ Rescanning ({done}/{total}): ", code=current))


@app.post("/api/rescan-models", response_class=HTMLResponse)
async def rescan_models():
    """Rescan models directory and regenerate skeleton configs for missing models.

    Also forces a multiplier refresh on every model so SWA-architecture models
    pick up any ctx changes made outside the modal-save path (e.g., hand-edits).
    """
    # Reset and activate progress state. Workers update `current` as they pick
    # up each script; the UI polls /api/rescan-progress to read it.
    config_models_dir = config_manager.project_root / "config" / "models"
    total = sum(1 for _ in config_models_dir.glob("*.sh")) if config_models_dir.exists() else 0
    RESCAN_STATE.update({"active": True, "current": "", "done": 0, "total": total})

    def _progress(name: str) -> None:
        RESCAN_STATE["current"] = name
        RESCAN_STATE["done"] = int(RESCAN_STATE.get("done", 0)) + 1

    try:
        await config_manager.load_all_models(
            force_recompute_multiplier=True,
            on_progress=_progress,
        )
        # Register any newly discovered models in gpu_manager so state tracking works
        gpu_manager.sync_models_from_config()
        count = len(config_manager.get_all_models())
        return HTMLResponse(
            content=_render("snippets/status_line.html", tone="ok", text=f"✓ Rescanned: {count} model(s) found"),
            status_code=200,
            headers={"HX-Trigger": "modelsRescanned"},
        )
    except Exception as e:
        logger.error(f"✗ Error rescanning models: {e}")
        return HTMLResponse(
            content=_render("snippets/status_line.html", tone="error", text=f"✗ Error: {e}"),
            status_code=500,
        )
    finally:
        RESCAN_STATE.update({"active": False, "current": "", "done": 0, "total": 0})


async def _auto_load_startup_models() -> None:
    """Sequentially load each model named in startup_models. Runs as a background
    task so the WebUI stays responsive during loading.

    Reads gpu_ids from each model's CUDA_VISIBLE_DEVICES (the .sh script is the
    source of truth for which GPUs). Skips with a logged warning on any failure:
    missing model, missing/malformed CVD, count mismatch with --tensor-split,
    port collision, insufficient VRAM, or any load_model exception. Never aborts
    the loop.

    Updates AUTO_LOAD_PROGRESS and AUTO_LOAD_CURRENT_NAME and announces each
    step over the WebSocket, so open pages refresh the banner.
    """
    global AUTO_LOAD_CURRENT_NAME
    names = sorted(config_manager.app_config.startup_models or [])
    if not names:
        return

    logger.info("=" * 60)
    logger.info(f"🚀 Auto-loading {len(names)} stored model(s) on startup")
    logger.info("=" * 60)

    AUTO_LOAD_PROGRESS["active"] = 1
    AUTO_LOAD_PROGRESS["total"] = len(names)
    AUTO_LOAD_PROGRESS["current_index"] = 0
    AUTO_LOAD_CURRENT_NAME = ""

    def announce() -> None:
        asyncio.create_task(event_bus.broadcast({"type": "auto_load"}))

    loaded = 0
    skipped = 0
    try:
        for idx, name in enumerate(names, start=1):
            AUTO_LOAD_PROGRESS["current_index"] = idx
            AUTO_LOAD_CURRENT_NAME = name
            announce()
            try:
                try:
                    mc = config_manager.get_model_config(name)
                except FileNotFoundError:
                    logger.warning(f"   ⏭  {name}: skipped — script no longer exists")
                    skipped += 1
                    continue

                if name in gpu_manager.sessions:
                    # Kept running across the restart and re-attached already.
                    logger.info(f"   ⏭  {name}: already running (re-attached)")
                    skipped += 1
                    continue

                cvd = mc.cuda_visible_devices
                if cvd is None:
                    logger.warning(
                        f"   ⏭  {name}: skipped — no CUDA_VISIBLE_DEVICES in script "
                        f"(load it through the picker once to register a GPU assignment)"
                    )
                    skipped += 1
                    continue

                if not mc.cvd_matches_split:
                    logger.warning(
                        f"   ⏭  {name}: skipped — CVD has {len(cvd)} GPU(s) but "
                        f"--tensor-split declares {mc.gpu_count}"
                    )
                    skipped += 1
                    continue

                logger.info(f"   ▶  {name}: loading on GPU(s) {cvd}")
                await gpu_manager.load_model(name, list(cvd))
                loaded += 1
            except Exception as e:
                logger.warning(f"   ⏭  {name}: skipped — {type(e).__name__}: {e}")
                skipped += 1
    finally:
        AUTO_LOAD_PROGRESS["active"] = 0
        AUTO_LOAD_CURRENT_NAME = ""
        announce()

    logger.info("=" * 60)
    logger.info(f"🚀 Auto-load complete: {loaded} loaded, {skipped} skipped")
    logger.info("=" * 60)


@app.get("/api/auto-load-banner", response_class=HTMLResponse)
async def auto_load_banner():
    """Return banner HTML when startup auto-load is in progress, empty otherwise.

    Fetched on page load and on each `auto_load` WebSocket event. When nothing
    is loading it answers 286, htmx's "stop polling", so the page stops asking.
    """
    if not AUTO_LOAD_PROGRESS.get("active"):
        return HTMLResponse("", status_code=286)
    return HTMLResponse(_render(
        "snippets/auto_load_banner.html",
        current=AUTO_LOAD_PROGRESS.get("current_index", 0),
        total=AUTO_LOAD_PROGRESS.get("total", 0),
        name=AUTO_LOAD_CURRENT_NAME or "…",
    ))


# ============================================================================
# API Routes - Startup Auto-Load
# ============================================================================


def _currently_loaded_names() -> list:
    """Sorted list of model names currently loaded (have a session)."""
    return sorted(gpu_manager.sessions.keys())


def _startup_state() -> dict:
    """Snapshot the data the UI needs to render the Store button + checkbox."""
    stored = sorted(config_manager.app_config.startup_models or [])
    loaded = _currently_loaded_names()
    return {
        "stored": stored,
        "loaded": loaded,
        "matches": stored == loaded,
        "auto_load": bool(config_manager.app_config.auto_load_on_startup),
        "loaded_count": len(loaded),
    }


def _render_store_button() -> str:
    """Render the Store-Current-State button HTML fragment.

    Hidden (empty string) when the currently-loaded set matches the stored set —
    nothing to store, so the button just adds noise. Tooltip lists stored models.
    """
    state = _startup_state()
    if state["matches"]:
        return ""
    return _render("snippets/store_button.html", **state)


@app.get("/api/startup-config-status", response_class=JSONResponse)
async def startup_config_status():
    """Return current startup-config state for UI refresh."""
    return JSONResponse(_startup_state())


@app.get("/api/startup-store-button", response_class=HTMLResponse)
async def startup_store_button():
    """Return just the Store button HTML — used by the UI to refresh color/tooltip
    after a load/unload elsewhere in the app changes the match status."""
    return HTMLResponse(_render_store_button())


@app.post("/api/store-startup-config", response_class=HTMLResponse)
async def store_startup_config():
    """Snapshot currently-loaded model names into app.json. Overwrites prior list.

    Clicking with 0 loaded stores an empty list (= clear startup config), per the
    'Store this state' philosophy.
    """
    try:
        names = _currently_loaded_names()
        config_manager.app_config.startup_models = list(names)
        await config_manager.save_app_config()
        logger.info(f"💾 Stored startup config: {names!r}")
        return HTMLResponse(_render_store_button())
    except Exception as e:
        logger.error(f"✗ Error storing startup config: {e}", exc_info=True)
        return HTMLResponse(
            _render("snippets/status_line.html", tone="error", small=True, text=f"✗ Save failed: {e}"),
            status_code=500,
        )


@app.post("/api/toggle-auto-load", response_class=JSONResponse)
async def toggle_auto_load(enabled: bool = Form(...)):
    """Update auto_load_on_startup."""
    try:
        config_manager.app_config.auto_load_on_startup = bool(enabled)
        await config_manager.save_app_config()
        logger.info(f"💾 auto_load_on_startup = {enabled}")
        return JSONResponse({"auto_load": bool(enabled)})
    except Exception as e:
        logger.error(f"✗ Error toggling auto-load: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/toggle-keep-alive", response_class=JSONResponse)
async def toggle_keep_alive(enabled: bool = Form(...)):
    """Update keep_sessions_on_shutdown."""
    try:
        config_manager.app_config.keep_sessions_on_shutdown = bool(enabled)
        await config_manager.save_app_config()
        logger.info(f"💾 keep_sessions_on_shutdown = {enabled}")
        return JSONResponse({"keep_alive": bool(enabled)})
    except Exception as e:
        logger.error(f"✗ Error toggling keep-alive: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# API Routes - Model Configuration
# ============================================================================


# ============================================================================
# API Routes - Config (Modal)
# ============================================================================


@app.get("/api/model-script-raw", response_class=PlainTextResponse)
async def model_script_raw(model: str):
    """Return raw contents of a model's launch script."""
    try:
        return PlainTextResponse(config_manager.get_script_text(model))
    except FileNotFoundError:
        return PlainTextResponse(f"# script not found for {model}", status_code=404)


@app.get("/api/model-script-download")
async def model_script_download(model: str):
    """Serve a model's launch script as a file download."""
    try:
        text = config_manager.get_script_text(model)
        return PlainTextResponse(
            content=text,
            headers={"Content-Disposition": f'attachment; filename="{model}.sh"'},
        )
    except FileNotFoundError:
        return PlainTextResponse(f"# script not found for {model}", status_code=404)


@app.post("/api/parse-script", response_class=JSONResponse)
async def parse_script_endpoint(script_text: str = Form(...)):
    """Parse a pasted launch script and return structured args for the form.

    Used by the 'Paste Script' nested modal to populate the config modal.
    Always overwrites display_name if --alias is present (resolved decision #1).
    """
    try:
        ps = parse_pasted_script(script_text)
        warnings = list(ps.warnings)

        # Resolved decision #3: if pasted binary path is invalid, do NOT change
        # llama_path — warn instead.
        suggested_llama_path = None
        if ps.llama_bin and Path(ps.llama_bin).is_absolute() and Path(ps.llama_bin).exists():
            suggested_llama_path = ps.llama_bin
        elif ps.llama_bin:
            warnings.append(
                f"llama-server path from pasted script not found: {ps.llama_bin}; "
                f"using current default."
            )

        # Split args into core fields + advanced
        args = dict(ps.args)
        port = args.pop("--port", None)
        host = args.pop("--host", None)
        ctx_size = args.pop("--ctx-size", None)
        cache_type_k = args.pop("--cache-type-k", None)
        cache_type_v = args.pop("--cache-type-v", None)

        return JSONResponse({
            "model_path": ps.model_path,
            "port": port,
            "host": host,
            "ctx_size": ctx_size,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "llama_path": suggested_llama_path,  # None if invalid/missing
            "display_name": ps.display_name,
            "health_timeout": ps.health_timeout,  # None if absent from meta fence
            "advanced": args,
            "warnings": warnings,
        })
    except Exception as e:
        logger.error(f"✗ Error parsing pasted script: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e), "warnings": []},
            status_code=400,
        )


@app.post("/api/save-raw-script", response_class=HTMLResponse)
async def save_raw_script(
    model_name: str = Form(...),
    script_text: str = Form(...),
):
    """Overwrite a model's script with raw text (used by the raw-script editor tab)."""
    logger.info(f"💾 Saving raw script for model: {model_name}")
    try:
        mc = config_manager.save_script_text(model_name, script_text)
        if not mc:
            return _error_modal(f"Model {model_name} not found after save")
        return _save_success_modal(mc)
    except Exception as e:
        logger.error(f"✗ Error saving raw script: {e}", exc_info=True)
        return _error_modal(f"Failed to save script: {str(e)}")



@app.get("/api/config-modal", response_class=HTMLResponse)
async def config_modal(model: str):
    """Return improved config modal with form + advanced table + VRAM calculator."""
    logger.info(f"📝 Edit config requested for model: {model}")
    try:
        model_config = config_manager.get_model_config(model)

        # vLLM models get their own modal, saved by /api/save-vllm-config,
        # which patches the exec line and leaves the env-var preamble intact.
        if model_config.engine == "vllm":
            from kv_calc import DEFAULT_KV_HEADROOM, DEFAULT_OVERHEAD_GB

            manifest = manifest_for_binary(model_config.parsed.llama_bin) or {}
            args = dict(model_config.launch_args)
            core_keys = {"--host", "--port", "--max-model-len", "--max-num-seqs",
                         "--kv-cache-memory-bytes", "--kv-cache-dtype"}
            mprofile = model_config.parsed.load_profile
            # KV cache dtype: the engine's own choices, each with its bytes/token,
            # so the dialog re-sizes when the dropdown changes.
            kv_dtype = model_config.parsed.kv_cache_dtype
            kv_choices = (manifest.get("flags", {}).get("--kv-cache-dtype") or {}).get("choices") or [kv_dtype]
            if kv_dtype not in kv_choices:
                kv_choices = [kv_dtype] + kv_choices
            kv_bpt = {d: model_config.kv_bytes_per_token_for(d) for d in kv_choices}
            bpt, bpt_source = kv_bpt[kv_dtype]

            gpu_caps = [info["memory"] for info in (gpu_manager.get_gpu_status() or {}).values()]
            # Utilization is a fraction of each card, so derive it against the
            # smallest one present; anything larger then has room to spare.
            gpu_capacity = round(min(gpu_caps), 2) if gpu_caps else 0.0

            # Render the opening VRAM figure server-side. The box previously
            # relied entirely on JS having run to show anything, so it appeared
            # blank on open if the script hadn't executed yet.
            from kv_calc import vram_gb_for
            initial_ctx = model_config.context_length or 8192
            initial_conc = model_config.concurrency or 1
            initial_vram, initial_tokens = ("", 0)
            if bpt:
                initial_vram, initial_tokens = vram_gb_for(
                    context_length=initial_ctx, concurrency=initial_conc,
                    bytes_per_token=bpt, weights_gb=model_config.size_gb or 0,
                    headroom=DEFAULT_KV_HEADROOM, gpu_count=model_config.gpu_count,
                    overhead_gb=(mprofile["runtime_overhead_gb"]
                                 if mprofile["runtime_overhead_gb"] is not None
                                 else DEFAULT_OVERHEAD_GB),
                    weights_per_gpu_gb=mprofile["weights_max_gb"],
                )

            template = jinja_env.get_template("modals/config_modal_vllm.html")
            return HTMLResponse(template.render(
                model_name=model,
                display_name=model_config.display_name or model,
                script_text=config_manager.get_script_text(model),
                config_error=model_config.config_error,
                # Parse warnings that predict a failed launch (e.g. a layer
                # partition whose length no longer matches --pipeline-parallel-size)
                # belong in front of the user before they spend minutes on a load.
                warnings=[w for w in model_config.parsed.warnings
                          if "meta fence missing" not in w],
                gpu_count=model_config.gpu_count,
                health_timeout=model_config.health_timeout,
                default_health_timeout=model_config.effective_health_timeout(),
                host=args.get("--host", "0.0.0.0"),
                port=args.get("--port", ""),
                context_length=model_config.context_length or "",
                concurrency=model_config.concurrency or "",
                kv_cache_memory_bytes=model_config.kv_cache_memory_bytes or "",
                max_context=model_config.max_context or 0,
                advanced_options={k: v for k, v in args.items() if k not in core_keys},
                option_schema=manifest.get("flags", {}),
                vllm_version=manifest.get("version", "unknown"),
                bytes_per_token=bpt or 0,
                bytes_per_token_source=bpt_source,
                kv_dtype=kv_dtype,
                kv_bpt=kv_bpt,
                kv_dtype_in_script="--kv-cache-dtype" in args,
                weights_gb=model_config.size_gb or 0,
                overhead_gb=(mprofile["runtime_overhead_gb"]
                             if mprofile["runtime_overhead_gb"] is not None
                             else DEFAULT_OVERHEAD_GB),
                weights_per_gpu_gb=mprofile["weights_max_gb"] or 0,
                kv_headroom=DEFAULT_KV_HEADROOM,
                env_assignments=model_config.parsed.env_assignments,
                pp_size=model_config.parsed.pipeline_parallel_size,
                gpu_capacity=gpu_capacity,
                initial_vram=initial_vram,
                initial_tokens=initial_tokens,
            ))

        # Options and version of the binary this script actually runs.
        manifest = await asyncio.to_thread(
            _flags, "llamacpp",
            model_config.llama_bin_absolute or config_manager.app_config.llama_server_binary,
        )
        option_schema = (manifest or {}).get("flags", {})
        version_str = _llama_version_label(manifest)

        # Split launch_args into individual options
        launch_args = model_config.launch_args or {}

        # Context and KV quantization for the VRAM calculator (parsed args are
        # always long-form).
        current_ctx = launch_args.get("--ctx-size") or "4096"
        current_ctk = launch_args.get("--cache-type-k") or "f16"
        current_ctv = launch_args.get("--cache-type-v") or "f16"

        # These have their own rows in the modal; keep them out of the free-form table.
        advanced_options = {k: v for k, v in launch_args.items()
                           if k not in ("--host", "--port", "--ctx-size", "--cache-type-k", "--cache-type-v")}
        host = launch_args.get("--host", "0.0.0.0")
        port = launch_args.get("--port", model_config.port or "")

        # Render template
        template = jinja_env.get_template("modals/config_modal_llama.html")
        html = template.render(
            model_name=model,
            display_name=model_config.display_name or model,
            port=port,
            host=host,
            llama_path=model_config.llama_path or "default",
            advanced_options=advanced_options,
            option_schema=option_schema,
            llama_version=version_str,
            # VRAM calculator metadata
            block_count=model_config.block_count,
            max_context=model_config.max_context,
            kv_cache_multiplier=model_config.kv_cache_multiplier,
            size_gb=model_config.size_gb,
            current_ctx=current_ctx,
            current_ctk=current_ctk,
            current_ctv=current_ctv,
            # Per-model health-check timeout (None → use default at load time)
            health_timeout=model_config.health_timeout,
            default_health_timeout=model_config.effective_health_timeout(),
        )
        return HTMLResponse(html)
    except Exception as e:
        logger.error(f"✗ Error in config_modal: {type(e).__name__}: {e}", exc_info=True)
        return _error_fragment(f"Error: {e}")


@app.post("/api/save-llama-config", response_class=HTMLResponse)
async def save_llama_config(
    model_name: str = Form(...),
    core_json: str = Form(...),
    advanced_json: str = Form(...),
    source_model: str = Form(""),
):
    """Save config with validation at each step.

    'Save As': when source_model is set and differs from model_name, the current
    edits are written to a new <model_name>.sh serving the same weights as the
    source, leaving the source untouched.
    """
    src = source_model.strip() or model_name
    is_save_as = bool(source_model.strip() and source_model.strip() != model_name)
    logger.info(
        f"💾 {'Save-As' if is_save_as else 'Saving'} llama.cpp config for model: "
        f"{src + ' → ' + model_name if is_save_as else model_name}"
    )
    try:
        if is_save_as:
            name_error = _validate_new_script_name(model_name)
            if name_error:
                return name_error
        # Each engine's form posts to its own endpoint (the field sets differ).
        try:
            if config_manager.get_model_config(src).engine == "vllm":
                return _error_modal("vLLM models are saved by the vLLM config editor.")
        except FileNotFoundError:
            return _error_modal(f"Model not found: {src}")

        # Parse core fields
        try:
            core_data = json.loads(core_json)
        except json.JSONDecodeError as e:
            logger.error(f"✗ Core config JSON parse error: {e}")
            error_msg = f"Core config JSON parse error: {str(e)}"
            return _error_modal(error_msg)

        # Validate core fields
        port = core_data.get("port")
        display_name = core_data.get("display_name", "")
        host = core_data.get("host", "0.0.0.0")
        llama_path = core_data.get("llama_path", "default")
        if not llama_path or llama_path.strip() == "":
            llama_path = "default"
        llama_bin = config_manager.app_config.llama_server_binary if llama_path == "default" else llama_path

        if not port:
            return _error_modal("Port is required")

        try:
            port = int(port)
            if not (1 <= port <= 65535):
                return _error_modal("Port must be between 1 and 65535")
        except (ValueError, TypeError):
            return _error_modal("Port must be a valid number")

        # Optional health-check timeout (seconds). Empty / missing → None (use default).
        ht_raw = core_data.get("health_timeout")
        health_timeout: Optional[int] = None
        if ht_raw not in (None, "", 0):
            try:
                ht_val = int(ht_raw)
            except (ValueError, TypeError):
                return _error_modal("Health-check timeout must be a whole number of seconds.")
            from launch_script import HEALTH_TIMEOUT_MIN, HEALTH_TIMEOUT_MAX
            if ht_val < HEALTH_TIMEOUT_MIN or ht_val > HEALTH_TIMEOUT_MAX:
                return _error_modal(
                    f"Health-check timeout must be between {HEALTH_TIMEOUT_MIN} and {HEALTH_TIMEOUT_MAX} seconds."
                )
            health_timeout = ht_val

        # Parse advanced fields
        try:
            advanced_data = json.loads(advanced_json) if advanced_json.strip() else {}
        except json.JSONDecodeError as e:
            logger.error(f"✗ Advanced config JSON parse error: {e}")
            error_msg = f"Advanced config JSON parse error: {str(e)}"
            return _error_modal(error_msg)

        # Validate each advanced option and collect invalid keys.
        # Skip core fields (--host, --port) — they're already validated as the
        # `host` / `port` core fields above and the modal duplicates them into
        # advanced for the launch_args dict. Running them through the schema
        # validator on top would re-flag any schema-type drift.
        CORE_FIELDS_SKIP = {"--host", "-h", "--port", "-p"}
        flags = ((await asyncio.to_thread(_flags, "llamacpp", llama_bin)) or {}).get("flags")
        invalid_keys = []
        for key, value in advanced_data.items():
            if key in CORE_FIELDS_SKIP:
                continue
            is_valid, error_msg = validate_llama_option(key, str(value), flags)
            if not is_valid:
                invalid_keys.append(key)
                logger.warning(f"⚠ Invalid option in config: {key} ({error_msg})")

        # If there are invalid options, log them but continue (allow saving)
        has_valid_options = len(invalid_keys) == 0
        if invalid_keys:
            logger.info(f"⚠ Model {model_name} has invalid options: {invalid_keys}")

        # Build launch_args from core + advanced. Rows the user typed may use
        # short aliases (-c); canonicalize as the parser does so everything
        # downstream sees one spelling.
        launch_args = canonical_args({"--host": host, "--port": str(port), **advanced_data})

        # Validate --tensor-split if present: must be a comma-separated list of positive numbers.
        ts_raw = launch_args.get("--tensor-split")
        new_split_count = 1
        if ts_raw is not None and str(ts_raw).strip() != "":
            parts = [p.strip() for p in str(ts_raw).split(",") if p.strip()]
            try:
                weights = [float(p) for p in parts]
            except ValueError:
                return _error_modal(
                    f"--tensor-split is malformed: {ts_raw!r}. Use comma-separated positive numbers (e.g. '1,1,1' or '2,1')."
                )
            if not weights or any(w <= 0 for w in weights):
                return _error_modal(
                    f"--tensor-split weights must all be positive: got {ts_raw!r}"
                )

        try:
            model_config = config_manager.save_script_edits(
                src, model_name,
                args=launch_args,
                display_name=display_name or config_manager.get_model_config(src).display_name,
                health_timeout=health_timeout,
                llama_bin=llama_bin,
            )
        except Exception as e:
            logger.error(f"✗ Failed to save script: {e}", exc_info=True)
            return _error_modal(f"Failed to save script: {str(e)}")

        if not model_config:
            return _error_modal(f"Model {model_name} not found after save")

        # A GPU assignment that no longer fits the saved split is dropped; the
        # picker asks again at the next load.
        if model_config.cuda_visible_devices and not model_config.cvd_matches_split:
            logger.info(
                f"⚠ Dropping stale CUDA_VISIBLE_DEVICES for {model_name}: "
                f"{len(model_config.cuda_visible_devices)} GPU(s) no longer match a "
                f"{model_config.gpu_count}-GPU split"
            )
            model_config = config_manager.save_cuda_visible_devices(model_name, None)

        logger.info(f"✓ Script saved for {model_name}")
        # Save-As and normal save both end on the confirmation popup.
        return _save_success_modal(model_config, invalid_keys=invalid_keys)
    except Exception as e:
        logger.error(f"✗ Error in save_llama_config: {type(e).__name__}: {e}", exc_info=True)
        return _error_modal(f"Error saving configuration: {str(e)}")


def _save_success_modal(model_config: ModelConfig, invalid_keys=None) -> str:
    """Save-confirmation modal.

    If the model is currently loaded (or loading), it warns that the running
    process still uses the prior args and offers to unload now or keep it.
    """
    invalid_keys = invalid_keys or []
    if model_config.is_configured:
        status_text = "✓ Model is now configured!"
    elif invalid_keys:
        status_text = f"⚠ Configuration saved but model is unconfigured (invalid options: {', '.join(invalid_keys)})"
    elif model_config.config_error:
        status_text = f"⚠ Configuration saved — {model_config.config_error}"
    else:
        status_text = "⚠ Configuration saved but model is still unconfigured"
    return _render(
        "modals/save_success.html",
        model_name=model_config.name,
        display_name=model_config.display_name,
        is_configured=model_config.is_configured,
        status_text=status_text,
        is_loaded=(
            model_config.name in gpu_manager.sessions
            or gpu_manager.get_model_state(model_config.name) in (ModelState.RUNNING, ModelState.LOADING)
        ),
    )


def _error_modal(message: str) -> str:
    return _render("modals/message.html", tone="error", title="Configuration Error", message=message)


# --- Save As helpers (shared by both structured-save endpoints) -------------

def _validate_new_script_name(name: str) -> Optional[str]:
    """Validate a Save-As target name. Returns an error-modal HTML string if the
    name is unusable as a new script filename, else None. The name becomes the
    <name>.sh filename, so it is restricted to filename-safe characters and must
    not collide with an existing script."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return _error_modal(
            "New name may contain only letters, numbers, dot, dash and "
            "underscore (it becomes the script filename)."
        )
    if config_manager.get_script_path(name).exists():
        return _error_modal(f"A script named '{name}' already exists. Pick another name.")
    return None


# ============================================================================
# API Routes - Model Control
# ============================================================================


def _stored_cvd_usability(model_config: ModelConfig, gpu_data: dict) -> tuple:
    """Evaluate whether the script's CVD is usable right now.

    Returns (slots, all_ok). `slots` is a list of dicts (gpu_id, gpu_name, total,
    available, need, ok, reason) — one per CVD entry. `all_ok` is True when every
    slot is OK and the CVD count matches the model's split count.
    """
    cvd = model_config.cuda_visible_devices or []
    # A count mismatch (or no CVD) makes the caller force the picker.
    if not model_config.cvd_matches_split:
        return [], False
    needs = model_config.need_per_gpu(
        [gpu_data[gid]["memory"] if gid in gpu_data else 0.0 for gid in cvd]
    )
    if len(needs) != len(cvd):
        return [], False

    slots = []
    all_ok = True
    for gid, need in zip(cvd, needs):
        gpu = gpu_data.get(gid)
        if gpu is None:
            slots.append({
                "gpu_id": gid, "gpu_name": "(not detected)",
                "total": 0.0, "available": 0.0, "need": need,
                "ok": False, "reason": "GPU not available on this machine",
            })
            all_ok = False
            continue
        available = gpu["memory"] - gpu["allocated"]
        ok = available >= need
        slots.append({
            "gpu_id": gid, "gpu_name": gpu["name"],
            "total": gpu["memory"], "available": available, "need": need,
            "ok": ok,
            "reason": "" if ok else f"only {available:.1f} GB free",
        })
        if not ok:
            all_ok = False
    return slots, all_ok


@app.get("/api/gpu-selector", response_class=HTMLResponse)
async def gpu_selector(model: str, force_pick: int = 0):
    """Return the right GPU modal for the model:

    - If the script has a usable CUDA_VISIBLE_DEVICES and `force_pick` is not set,
      render the confirm modal (Cases 2/3 in the load-flow matrix).
    - Else, render the picker — multi-slot for tensor-split, single-button for
      single-GPU (Cases 1/4 plus all forced re-picks).
    """
    logger.info(f"📋 GPU selector requested for model: {model} (force_pick={force_pick})")
    try:
        model_config = config_manager.get_model_config(model)
        if not model_config.is_configured:
            return _error_fragment("Model not configured", 200)

        gpu_data = gpu_manager.get_gpu_status()
        if not gpu_data:
            return _error_fragment("No GPUs detected", 200)

        for gid, info in gpu_data.items():
            info["memory_pct"] = (info["allocated"] / info["memory"]) * 100 if info["memory"] > 0 else 0

        gpu_count = model_config.gpu_count

        # Confirm modal path: CVD present, count matches split, not forced to re-pick
        if not force_pick and model_config.cuda_visible_devices is not None:
            slots, all_ok = _stored_cvd_usability(model_config, gpu_data)
            if slots:  # CVD count matched gpu_count
                # Build hx-vals for the Load button: repeated gpu_ids form fields via JSON array
                load_vals = {"model_name": model, "gpu_ids": list(model_config.cuda_visible_devices)}
                cvd_csv = ",".join(str(g) for g in model_config.cuda_visible_devices)
                template = jinja_env.get_template("modals/gpu_confirm.html")
                return HTMLResponse(template.render(
                    model_name=model,
                    display_name=model_config.display_name or model,
                    # Use the actual slot count (flexible-split models report
                    # gpu_count=1 but may span several GPUs via their stored CVD).
                    gpu_count=len(slots),
                    cvd_csv=cvd_csv,
                    slots=slots,
                    all_ok=all_ok,
                    load_hx_vals=json.dumps(load_vals),
                ))
            # CVD count mismatch → fall through to picker (blank-slate)

        # Flexible-split path: --split-mode layer/row without --tensor-split.
        # The user picks any set of GPUs (>=1); the model splits evenly across
        # them. Per-GPU need is computed live in the template as total / count.
        if model_config.is_flexible_split:
            total_need = model_config.total_need_gb
            template = jinja_env.get_template("modals/gpu_flex_picker.html")
            return HTMLResponse(template.render(
                model_name=model,
                display_name=model_config.display_name or model,
                split_mode=model_config.split_mode,
                total_need=total_need,
                gpu_data=gpu_data,
            ))

        # Multi-pick path: tensor-split present, no usable CVD (or forced)
        if gpu_count > 1:
            # The GPUs aren't picked yet; size against the largest card, so a
            # vLLM figure (which scales with capacity) errs high.
            max_mem = max((g["memory"] for g in gpu_data.values()), default=0)
            per_gpu_need = model_config.need_per_gpu([max_mem] * gpu_count)
            template = jinja_env.get_template("modals/gpu_multi_picker.html")
            return HTMLResponse(template.render(
                model_name=model,
                display_name=model_config.display_name or model,
                gpu_count=gpu_count,
                per_gpu_need=per_gpu_need,
                gpu_data=gpu_data,
            ))

        # Single-GPU picker: the need on each candidate GPU on its own.
        need_by_gpu = {
            gid: model_config.need_per_gpu([info["memory"]])[0]
            for gid, info in gpu_data.items()
        }

        template = jinja_env.get_template("modals/gpu_selector.html")
        return HTMLResponse(template.render(
            model_name=model,
            model_config=model_config,
            display_name=model_config.display_name or model,
            gpu_data=gpu_data,
            need_by_gpu=need_by_gpu,
        ))
    except FileNotFoundError as e:
        logger.error(f"✗ Model {model} not found: {e}")
        return _error_fragment("Model not found", 404)
    except Exception as e:
        logger.error(f"✗ Error in gpu_selector: {type(e).__name__}: {e}", exc_info=True)
        return _error_fragment(f"Error: {e}")


@app.post("/api/trigger-load", response_class=HTMLResponse)
async def trigger_load(
    background_tasks: BackgroundTasks,
    model_name: str = Form(),
    gpu_ids: list[int] = Form(),
):
    """Trigger a load operation.

    Accepts a list of GPU IDs (single-element for single-GPU models, N-element for
    tensor-split). Writes back CUDA_VISIBLE_DEVICES to the script only if the
    assignment differs from what the script already declares (persisting the
    last-used state for auto-bootup, without churning on confirmed defaults).
    """
    logger.info(f"🚀 Load requested: {model_name} on GPUs {gpu_ids}")
    try:
        # Persist CVD if it changed (the save helper is idempotent — no-ops if equal).
        try:
            mc = config_manager.get_model_config(model_name)
            if mc.cuda_visible_devices != gpu_ids:
                config_manager.save_cuda_visible_devices(model_name, gpu_ids)
        except Exception as e:
            logger.warning(f"⚠ Could not persist CVD for {model_name}: {e}")

        await gpu_manager._set_model_state(model_name, ModelState.LOADING, LoadingPhase.QUEUED)
        background_tasks.add_task(gpu_manager.load_model, model_name, gpu_ids)
        # Close modal and scroll to the model row
        return f'''<script>
document.getElementById("modal")?.remove();
// Scroll to the model row (if it exists)
(function() {{
    const modelRow = document.querySelector('[data-model-name="{model_name}"]');
    if (modelRow) {{
        modelRow.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
}})();
</script>'''
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return _error_fragment(f"Error: {e}")


@app.get("/api/unload-confirm", response_class=HTMLResponse)
async def unload_confirm(model: str):
    """Return unload confirmation modal."""
    try:
        model_config = config_manager.get_model_config(model)
        template = jinja_env.get_template("modals/unload_confirm.html")
        return HTMLResponse(template.render(
            model_name=model,
            display_name=model_config.display_name or model,
        ))
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return _error_fragment(f"Error: {e}")


@app.post("/api/trigger-unload", response_class=HTMLResponse)
async def trigger_unload(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
):
    """Trigger an unload operation."""
    logger.info(f"🛑 Unload requested: {model_name}")
    try:
        await gpu_manager._set_model_state(model_name, ModelState.LOADING, LoadingPhase.UNLOADING)
        background_tasks.add_task(gpu_manager.unload_model, model_name)
        # Close modal and scroll to the model row
        return f'''<script>
document.getElementById("modal")?.remove();
// Scroll to the model row (if it exists)
(function() {{
    const modelRow = document.querySelector('[data-model-name="{model_name}"]');
    if (modelRow) {{
        modelRow.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
}})();
</script>'''
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return _error_fragment(f"Error: {e}")


@app.get("/api/cancel-confirm", response_class=HTMLResponse)
async def cancel_confirm(model: str):
    """Return cancel load confirmation modal."""
    try:
        model_config = config_manager.get_model_config(model)
        template = jinja_env.get_template("modals/cancel_confirm.html")
        return HTMLResponse(template.render(
            model_name=model,
            display_name=model_config.display_name or model,
        ))
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return _error_fragment(f"Error: {e}")


@app.post("/api/trigger-cancel", response_class=HTMLResponse)
async def trigger_cancel(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
):
    """Trigger a cancel operation for a loading model."""
    logger.info(f"⏹ Cancel load requested: {model_name}")
    try:
        await gpu_manager.cancel_load(model_name)
        # Close modal and scroll to the model row
        return f'''<script>
document.getElementById("modal")?.remove();
// Scroll to the model row (if it exists)
(function() {{
    const modelRow = document.querySelector('[data-model-name="{model_name}"]');
    if (modelRow) {{
        modelRow.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
}})();
</script>'''
    except Exception as e:
        logger.error(f"✗ Error cancelling {model_name}: {e}")
        return _error_fragment(f"Error: {e}")


@app.post("/api/clear-error", response_class=HTMLResponse)
async def clear_error(model_name: str = Form(...)):
    """Clear error state for a model."""
    logger.info(f"🔄 Clearing error for model: {model_name}")
    try:
        await gpu_manager._set_model_state(model_name, ModelState.IDLE)

        # Return updated model row
        model_config = config_manager.get_model_config(model_name)
        state = gpu_manager.get_model_state(model_name)
        status = state.value

        # Get VRAM display
        vram_display = "—"
        port_display = "—"
        if model_config.port:
            port_display = f":{model_config.port}"
            if model_config.total_vram:
                vram_display = f"{model_config.total_vram} GB"
            elif model_config.size_gb:
                vram_display = f"{model_config.size_gb} GB"

        template = jinja_env.get_template("snippets/model_row.html")
        html = template.render(
            model_name=model_name,
            model_config=model_config,
            status=status,
            vram_display=vram_display,
            port_display=port_display,
        )
        return html
    except Exception as e:
        logger.error(f"✗ Error clearing error: {type(e).__name__}: {e}", exc_info=True)
        return _error_fragment(f"Error: {e}")


@app.get("/api/delete-script-confirm", response_class=HTMLResponse)
async def delete_script_confirm(model: str):
    """Confirmation modal for deleting a model's launch script."""
    try:
        model_config = config_manager.get_model_config(model)
    except FileNotFoundError:
        return _error_fragment("Model not found", 404)
    state = gpu_manager.get_model_state(model)
    running = state in (ModelState.RUNNING, ModelState.LOADING)
    siblings = config_manager.sibling_script_count(model)
    winfo = config_manager.model_weight_info(model)
    # Deleting the weights is offered only when this is the sole script for them,
    # nothing else references them (drafter/projector), and they aren't a symlink.
    can_delete_weights = (
        siblings == 0 and winfo["exists"]
        and not winfo["referenced_by"] and not winfo["is_symlink"]
    )
    template = jinja_env.get_template("modals/delete_script_confirm.html")
    return HTMLResponse(template.render(
        model_name=model,
        display_name=model_config.display_name or model,
        running=running,
        siblings=siblings,
        sole=(siblings == 0),
        weights_exist=winfo["exists"],
        weight_size_gb=winfo["size_gb"],
        weight_is_dir=winfo["is_dir"],
        weight_is_symlink=winfo["is_symlink"],
        weight_referenced_by=winfo["referenced_by"],
        weight_path=winfo["path"] or "",
        can_delete_weights=can_delete_weights,
    ))


@app.post("/api/delete-script", response_class=HTMLResponse)
async def delete_script(model_name: str = Form(...)):
    """Delete a model's launch script (weights untouched). Refuses while the
    model is loaded. If it was the only script for an on-disk model, the model
    is gone, so a rescan will not bring it back."""
    logger.info(f"🗑 Delete script requested: {model_name}")
    state = gpu_manager.get_model_state(model_name)
    if state in (ModelState.RUNNING, ModelState.LOADING):
        return _error_modal(
            f"'{model_name}' is currently loaded. Unload it before deleting its script."
        )
    try:
        result = config_manager.delete_script(model_name)
    except FileNotFoundError:
        return _error_modal(f"No script found for '{model_name}'.")
    except Exception as e:
        logger.error(f"✗ Delete failed for {model_name}: {e}", exc_info=True)
        return _error_modal(f"Failed to delete script: {e}")
    # Reflect the deletion in the loaded set.
    await config_manager.load_all_models()
    # Close the modal and refresh the table from the server.
    return HTMLResponse(
        '<script>'
        'document.getElementById("modal")?.remove();'
        'htmx.ajax("GET", "/api/model-list", {target: "#models-container", swap: "innerHTML"});'
        '</script>'
    )


@app.post("/api/delete-model", response_class=HTMLResponse)
async def delete_model(model_name: str = Form(...), confirm_name: str = Form("")):
    """Delete a model's script AND its weights from disk. Irreversible for the
    weights. Requires YES typed back, refuses while loaded, and refuses if
    another script references the weights."""
    logger.info(f"🗑💥 Delete MODEL (weights) requested: {model_name}")
    if confirm_name.strip().upper() != "YES":
        return _error_modal(
            "Deletion not confirmed, so nothing was deleted. "
            "To delete the weights, type YES in the confirmation box."
        )
    state = gpu_manager.get_model_state(model_name)
    if state in (ModelState.RUNNING, ModelState.LOADING):
        return _error_modal(
            f"'{model_name}' is currently loaded. Unload it before deleting it."
        )
    try:
        result = config_manager.delete_model(model_name)
    except FileNotFoundError:
        return _error_modal(f"No script found for '{model_name}'.")
    except ValueError as e:
        return _error_modal(f"Cannot delete: {e}.")
    except Exception as e:
        logger.error(f"✗ Delete model failed for {model_name}: {e}", exc_info=True)
        return _error_modal(f"Failed to delete model: {e}")
    await config_manager.load_all_models()
    logger.info(f"✓ Deleted {model_name}: {result['gb']} GB freed")
    return HTMLResponse(
        '<script>'
        'document.getElementById("modal")?.remove();'
        'htmx.ajax("GET", "/api/model-list", {target: "#models-container", swap: "innerHTML"});'
        '</script>'
    )


# ============================================================================
# API Routes - Configuration
# ============================================================================


@app.post("/api/calculate-vram", response_class=JSONResponse)
async def calculate_vram(
    block_count: int = Form(...),
    kv_cache_multiplier: int = Form(...),
    size_gb: float = Form(...),
    ctx_size: int = Form(4096),
    cache_type_k: str = Form("f16"),
    cache_type_v: str = Form("f16")
):
    """Calculate total VRAM needed for a model."""
    try:
        kv_cache_gb = llama_kv_gb(block_count, ctx_size, kv_cache_multiplier, cache_type_k, cache_type_v)
        return {
            "total": round(size_gb + kv_cache_gb, 2),
            "weights": round(size_gb, 2),
            "kv_cache": round(kv_cache_gb, 2),
        }
    except Exception as e:
        logger.error(f"Error calculating VRAM: {e}")
        return {"error": str(e)}


@app.post("/api/save-vllm-config", response_class=HTMLResponse)
async def save_vllm_config(
    model_name: str = Form(...),
    display_name: str = Form(""),
    health_timeout: str = Form(""),
    args_json: str = Form(...),
    env_json: str = Form("{}"),
    source_model: str = Form(""),
):
    """Structured save for a vLLM model — rewrites only the exec line (§2.2).

    The meta fence takes display_name / health_timeout; everything else is an
    engine argument and goes into the invocation. The script's preamble (env
    exports, tuning comments, variable defaults) is preserved byte-for-byte,
    and arguments the user didn't change keep their original spelling, so
    `$VAR` indirection survives.

    'Save As': when source_model is set and differs from model_name, the current
    (unsaved) form edits are applied to the SOURCE script's preamble and written
    to a brand-new <model_name>.sh, leaving the source untouched.
    """

    src = source_model.strip() or model_name
    is_save_as = bool(source_model.strip() and source_model.strip() != model_name)
    if is_save_as:
        logger.info(f"💾 Save-As vLLM: {src} → {model_name}")
        name_error = _validate_new_script_name(model_name)
        if name_error:
            return name_error
    else:
        logger.info(f"💾 Saving vLLM config for: {model_name}")

    try:
        mc = config_manager.get_model_config(src)
    except FileNotFoundError:
        return _error_modal(f"Model not found: {src}")
    if mc.engine != "vllm":
        return _error_modal(f"{src} is not a vLLM model")

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        return _error_modal(f"Could not parse launch arguments: {e}")
    if not isinstance(args, dict):
        return _error_modal("Launch arguments must be an object")

    ht: Optional[int] = None
    if health_timeout.strip():
        try:
            ht = int(health_timeout)
        except ValueError:
            return _error_modal("Health-check timeout must be a whole number of seconds.")
        from launch_script import HEALTH_TIMEOUT_MAX, HEALTH_TIMEOUT_MIN
        if not (HEALTH_TIMEOUT_MIN <= ht <= HEALTH_TIMEOUT_MAX):
            return _error_modal(
                f"Health-check timeout must be between {HEALTH_TIMEOUT_MIN} and {HEALTH_TIMEOUT_MAX} seconds."
            )

    for key in ("--max-model-len", "--max-num-seqs", "--kv-cache-memory-bytes"):
        value = args.get(key)
        if value in (None, ""):
            continue
        try:
            if int(value) <= 0:
                return _error_modal(f"{key} must be a positive whole number.")
        except (TypeError, ValueError):
            return _error_modal(f"{key} must be a whole number, got {value!r}.")

    # Refuse to write a configuration that would fill a card. Checked here
    # rather than only at load time, because the damage is done by the config
    # existing: a load that OOMs mid-GEMM can wedge the GPU entirely.
    from kv_calc import DEFAULT_OVERHEAD_GB, GIB, headroom_ok, MIN_HEADROOM_GB
    budget = args.get("--kv-cache-memory-bytes")
    caps = [i["memory"] for i in (gpu_manager.get_gpu_status() or {}).values()]
    if budget and caps:
        profile = mc.parsed.load_profile
        gpus = max(1, mc.gpu_count)
        weights = profile["weights_max_gb"] or (mc.size_gb or 0.0) / gpus
        overhead = profile["runtime_overhead_gb"]
        if overhead is None:
            overhead = DEFAULT_OVERHEAD_GB
        peak = weights + overhead + int(budget) / GIB
        fits, free = headroom_ok(peak, min(caps))
        if not fits:
            return _error_modal(
                f"This configuration would leave only {free:.2f} GB free on the "
                f"tightest GPU ({peak:.2f} of {min(caps):.1f} GB used), below the "
                f"{MIN_HEADROOM_GB} GB floor. Reduce context or concurrency. "
                f"Filling a card risks an allocation failure that can require a reboot."
            )

    # Environment assignments carry launch config with no CLI flag.
    try:
        env = json.loads(env_json) or {}
    except json.JSONDecodeError:
        env = {}
    try:
        updated = config_manager.save_script_edits(
            src, model_name,
            args=args,
            display_name=display_name.strip() or None,
            health_timeout=ht,
            env=env,
        )
    except Exception as e:
        logger.error(f"✗ vLLM save failed for {model_name}: {e}", exc_info=True)
        return _error_modal(f"Failed to save: {e}")

    if not updated:
        return _error_modal(f"{model_name} could not be re-read after saving")
    # Save-As and normal save both end on the confirmation popup; its OK button
    # refreshes the table, which surfaces the new fork.
    return _save_success_modal(updated)


@app.post("/api/calculate-vllm-vram", response_class=JSONResponse)
async def calculate_vllm_vram(
    model: str = Form(...),
    context_length: int = Form(0),
    concurrency: int = Form(1),
    vram_gb: float = Form(0.0),
    solve_for: str = Form("vram"),
    kv_cache_dtype: str = Form(""),
):
    """Both directions of the context/concurrency/VRAM relation (doc 21 §2.4).

    solve_for='vram'    : context + concurrency -> total VRAM
    solve_for='context' : VRAM budget + concurrency -> largest context that fits
    solve_for='fit'     : the largest context the emptiest GPU can take while
                          keeping the same spare room save_vllm_config demands
                          (MIN_HEADROOM_GB on the smallest card), so a fitted
                          value is always one the save accepts

    One shared helper backs both so a round-trip returns to the same slider
    position. Returns the pool token count and the KV budget to write into the
    script as --kv-cache-memory-bytes.
    """
    from kv_calc import (
        DEFAULT_KV_HEADROOM, DEFAULT_OVERHEAD_GB, GIB, MIN_HEADROOM_GB,
        context_for_vram, vram_gb_for,
    )
    try:
        mc = config_manager.get_model_config(model)
    except FileNotFoundError:
        return JSONResponse({"error": f"model not found: {model}"}, status_code=404)

    # Measured figures, or empties when the profile was taken at a different GPU count.
    profile = mc.parsed.load_profile
    # The dialog sends its (possibly unsaved) KV cache dtype; default to the script's.
    bpt, source = mc.kv_bytes_per_token_for(kv_cache_dtype or None)
    if not bpt:
        return JSONResponse({
            "error": "KV bytes/token unknown for this model — load it once to calibrate",
        }, status_code=422)

    weights = mc.size_gb or 0.0
    weights_per_gpu = profile["weights_max_gb"]
    overhead = profile["runtime_overhead_gb"]
    if overhead is None:
        overhead = DEFAULT_OVERHEAD_GB
    concurrency = max(1, concurrency)
    fit = {}
    if solve_for == "fit":
        gpus = gpu_manager.get_gpu_status() or {}
        if not gpus:
            return JSONResponse({"error": "no GPUs detected"}, status_code=422)
        gpu_id, info = max(gpus.items(), key=lambda kv: kv[1]["memory"] - kv[1]["allocated"])
        free = info["memory"] - info["allocated"]
        smallest = min(g["memory"] for g in gpus.values())
        vram_gb = min(free, smallest) - MIN_HEADROOM_GB
        fit = {"gpu_id": gpu_id, "free_gb": round(free, 1), "spare_gb": MIN_HEADROOM_GB}
        solve_for = "context"
    if solve_for == "context":
        ctx = context_for_vram(
            vram_gb=vram_gb, concurrency=concurrency, bytes_per_token=bpt,
            weights_gb=weights, max_context=mc.max_context,
            headroom=DEFAULT_KV_HEADROOM, gpu_count=mc.gpu_count,
            overhead_gb=overhead, weights_per_gpu_gb=weights_per_gpu,
        )
        total, tokens = vram_gb_for(
            context_length=ctx, concurrency=concurrency,
            bytes_per_token=bpt, weights_gb=weights,
            headroom=DEFAULT_KV_HEADROOM, gpu_count=mc.gpu_count,
            overhead_gb=overhead, weights_per_gpu_gb=weights_per_gpu,
        )
    else:
        ctx = max(0, context_length)
        total, tokens = vram_gb_for(
            context_length=ctx, concurrency=concurrency,
            bytes_per_token=bpt, weights_gb=weights,
            headroom=DEFAULT_KV_HEADROOM, gpu_count=mc.gpu_count,
            overhead_gb=overhead, weights_per_gpu_gb=weights_per_gpu,
        )

    return JSONResponse({
        "context_length": ctx,
        "concurrency": concurrency,
        "pool_tokens": tokens,
        "bytes_per_token": bpt,
        "bytes_per_token_source": source,
        "weights_gb": round(weights, 2),
        "overhead_gb": overhead,
        "weights_per_gpu_gb": weights_per_gpu,
        "measured": bool(weights_per_gpu),
        "headroom": DEFAULT_KV_HEADROOM,
        "kv_gb": round(tokens * bpt / GIB, 2),
        "per_gpu_gb": total,
        "gpu_count": mc.gpu_count,
        "total_gb": round(total * mc.gpu_count, 2),
        "kv_cache_memory_bytes": tokens * bpt,
        **fit,
    })


@app.get("/api/get-current-settings", response_class=JSONResponse)
async def get_current_settings():
    """Return current settings for settings modal population."""
    # Same label as the per-model modal ("9244 (b28a2f372)"); cached after first probe.
    version_str = _llama_version_label(
        await asyncio.to_thread(_flags, "llamacpp", config_manager.app_config.llama_server_binary)
    )
    # `vllm --version` imports torch on a cold cache (tens of seconds), so it
    # runs in a thread — never inline, or opening Settings freezes the app.
    vllm_bin = config_manager.app_config.vllm_binary
    vllm_version = ""
    if vllm_bin:
        vllm_version = await asyncio.to_thread(get_vllm_version, vllm_bin) or "unknown"
    return {
        "llama_server": config_manager.app_config.llama_server_binary,
        "llama_server_version": version_str,
        "vllm_binary": vllm_bin or "",
        "vllm_version": vllm_version,
        "models_directory": config_manager.app_config.models_directory,
        "webui_port": config_manager.app_config.webui_port,
    }


@app.get("/api/file-browser", response_class=HTMLResponse)
async def file_browser(path_type: str, current_path: str = "/"):
    """Return the directory-picker fragment for the Settings / model-config modals.

    Swapped into #file-browser-content; the caller supplies selectPath() and
    closeFileBrowser(). Args: path_type is 'llama_server', 'vllm_binary', or
    'models_directory'; current_path is where to start browsing.
    """
    try:
        logger.info(f"📂 File browser requested for {path_type} at {current_path}")
        
        # Parse current path
        current_path_obj = Path(current_path)
        
        # Ensure path is absolute and exists
        if not current_path_obj.is_absolute():
            current_path_obj = Path.home()
        if not current_path_obj.exists():
            current_path_obj = current_path_obj.parent
            while not current_path_obj.exists() and current_path_obj != current_path_obj.parent:
                current_path_obj = current_path_obj.parent
        # If path is a file (not directory), use its parent directory
        if current_path_obj.is_file():
            current_path_obj = current_path_obj.parent
        
        # Parent link first (unless at /), then visible subdirectories.
        entries = []
        if current_path_obj != current_path_obj.parent:
            entries.append(("../", str(current_path_obj.parent)))
        try:
            entries += [
                (f"{item.name}/", str(item))
                for item in sorted(current_path_obj.iterdir())
                if item.is_dir() and not item.name.startswith(".")
            ]
        except PermissionError:
            logger.warning(f"⚠ Permission denied accessing {current_path_obj}")

        return _render("snippets/file_browser.html", path_type=path_type,
                       current_path=str(current_path_obj), entries=entries)
    except Exception as e:
        logger.error(f"✗ Error in file_browser: {type(e).__name__}: {e}", exc_info=True)
        return _error_fragment(f"Error: {e}")


@app.post("/api/update-config-path", response_class=HTMLResponse)
async def update_config_path(path_type: str = Form(...), new_path: str = Form(...)):
    """Update a configuration path and save to config file."""
    logger.info(f"🔄 Updating {path_type} to: {new_path}")
    try:
        new_path_obj = Path(new_path)
        if not new_path_obj.exists():
            raise ValueError(f"Path does not exist: {new_path}")

        # Update app config
        if path_type == "llama_server":
            # Accept a directory (append binary name) or a file directly
            if new_path_obj.is_dir():
                new_path_obj = new_path_obj / "llama-server"
                if not new_path_obj.exists():
                    raise ValueError(f"No 'llama-server' binary found in {new_path}")
            if not new_path_obj.is_file():
                raise ValueError(f"Path is not a file: {new_path_obj}")
            # Probe (and cache) its options now; refuse a binary that can't be read.
            if not await asyncio.to_thread(_flags, "llamacpp", str(new_path_obj)):
                raise ValueError(f"Not a working llama-server (version/help could not be read): {new_path_obj}")
            config_manager.app_config.llama_server_binary = str(new_path_obj)
        elif path_type == "vllm_binary":
            # Accept the executable directly, or a directory (venv root or its
            # bin/) from which we locate `vllm`.
            if new_path_obj.is_dir():
                for candidate in (new_path_obj / "vllm", new_path_obj / "bin" / "vllm"):
                    if candidate.is_file():
                        new_path_obj = candidate
                        break
                else:
                    raise ValueError(f"No 'vllm' executable found in {new_path}")
            if not new_path_obj.is_file():
                raise ValueError(f"Path is not a file: {new_path_obj}")
            # Capture the engine's flag manifest now (its own `serve --help`) —
            # generated args and arg parsing are drawn only from this.
            manifest = await asyncio.to_thread(_flags, "vllm", str(new_path_obj))
            if not manifest:
                raise ValueError(
                    f"Not a working vllm executable (--version or `serve --help` failed): {new_path_obj}"
                )
            config_manager.app_config.vllm_binary = str(new_path_obj)
            logger.info(
                f"✓ vLLM executable set: {new_path_obj} "
                f"(version {manifest['version']}, {len(manifest['flags'])} flags)"
            )
        elif path_type == "models_directory":
            if not new_path_obj.is_dir():
                raise ValueError(f"Path is not a directory: {new_path}")
            config_manager.app_config.models_directory = str(new_path_obj)
            # Reload models from new directory
            logger.info(f"🔄 Reloading models from new directory: {new_path_obj}")
            await config_manager.load_all_models()
            logger.info(f"✓ Models reloaded: {len(config_manager.get_all_models())} model(s)")
        else:
            raise ValueError(f"Unknown path type: {path_type}")

        await config_manager.save_app_config()
        logger.info(f"✓ Config updated and saved: {path_type} = {new_path}")
        return _render("modals/message.html", tone="success", title="✓ Path Updated",
                       label=path_type, detail=new_path)
    except Exception as e:
        logger.error(f"✗ Error in update_config_path: {type(e).__name__}: {e}", exc_info=True)
        # The status stays 200 so htmx still swaps this into the modal; the
        # data-config-error marker is what callers test to detect failure.
        return _render("modals/message.html", tone="error", title="Error Updating Path",
                       message=str(e), marker="config-error")


@app.post("/api/update-webui-port", response_class=JSONResponse)
async def update_webui_port(port: int = Form(...)):
    """Update WebUI port and save to config."""
    logger.info(f"🔄 Updating WebUI port to: {port}")
    try:
        if not (1 <= port <= 65535):
            return {"error": "Port must be between 1 and 65535", "success": False}

        # Update app config
        config_manager.app_config.webui_port = port

        await config_manager.save_app_config()
        logger.info(f"✓ WebUI port updated to {port} and saved")
        return {
            "success": True,
            "message": f"Port updated to {port}. Restart the app to apply changes.",
            "port": port
        }
    except Exception as e:
        logger.error(f"✗ Error updating port: {type(e).__name__}: {e}", exc_info=True)
        return {"error": str(e), "success": False}


@app.post("/api/validate-binary-path", response_class=JSONResponse)
async def validate_binary_path(path: str = Form(...)):
    """Validate a llama-server binary path and pre-warm schema cache."""
    logger.info(f"🔍 Validating binary path: {path}")
    try:
        path_obj = Path(path)

        # If path is a directory, append "llama-server" binary name
        if path_obj.is_dir():
            path_obj = path_obj / "llama-server"
            if not path_obj.exists():
                return {"version": None, "error": f"No 'llama-server' binary found in {path}"}
        elif not path_obj.exists():
            return {"version": None, "error": f"Path not found: {path}"}

        if not path_obj.is_file():
            return {"version": None, "error": f"Path is not a file: {path_obj}"}

        # Validate binary and cache its option schema
        resolved_path = str(path_obj)
        version_str = ((await asyncio.to_thread(_flags, "llamacpp", resolved_path)) or {}).get("version")

        if version_str is None:
            return {"version": None, "error": f"Invalid llama-server binary or unable to detect version: {resolved_path}"}

        logger.info(f"✓ Binary validated: {resolved_path} (version {version_str})")
        return {"version": version_str, "error": None, "resolved_path": resolved_path}
    except Exception as e:
        logger.error(f"✗ Error validating binary: {type(e).__name__}: {e}", exc_info=True)
        return {"version": None, "error": str(e)}


# ============================================================================
# API Routes - Log Viewer
# ============================================================================


def _tail_lines(path: Path, n: int, block_size: int = 65536) -> list[str]:
    """Return the last `n` lines of `path`, reading only from the end.

    Reads fixed-size blocks backwards until `n` newlines are covered, so a large
    logfile (llama-server logs grow) isn't slurped into memory on every poll.
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        remaining = f.tell()
        data = b""
        # One extra newline's worth of slack so we don't clip the last line.
        while remaining > 0 and data.count(b"\n") <= n:
            read_size = min(block_size, remaining)
            remaining -= read_size
            f.seek(remaining)
            data = f.read(read_size) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


@app.get("/api/model-log-tail", response_class=PlainTextResponse)
async def model_log_tail(model: str, lines: int = 30):
    """Return last N lines from a model's logfile as plain text."""
    try:
        log_path = config_manager.get_log_path(model)

        if not log_path.exists():
            return "(no log file)"

        tail_lines = _tail_lines(log_path, lines)

        # Return as plain text, strip trailing whitespace from each line
        return '\n'.join(line.rstrip() for line in tail_lines)

    except Exception as e:
        logger.error(f"✗ Error reading log for {model}: {e}")
        return f"(error reading log: {str(e)})"


# ============================================================================
# Chat page - a minimal test chat for any OpenAI-compatible server (vLLM has no web UI)
# ============================================================================


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(model: Optional[str] = None, endpoint: Optional[str] = None):
    """Standalone test chat for any OpenAI-compatible server.

    Opened from a model row as /chat?model=<name>, it starts pointed at that
    model's port; /chat?endpoint=host:port targets anything else. The server
    box on the page can be changed and reconnected either way.
    """
    title = None
    if model and model in config_manager.get_all_models():
        mc = config_manager.get_model_config(model)
        title = mc.display_name or model
        if not endpoint and mc.port:
            endpoint = f"localhost:{mc.port}"
    template = jinja_env.get_template("chat.html")
    return template.render(endpoint=endpoint or "", studio_title=title, app_version=APP_VERSION)


@app.get("/api/chat/probe", response_class=JSONResponse)
async def chat_probe(endpoint: str):
    """Resolve a typed server:port and report the models it serves + its engine."""
    try:
        base_url, label = await chat_proxy.resolve_endpoint(endpoint)
    except chat_proxy.EndpointError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    try:
        info = await chat_proxy.probe(base_url)
    except httpx.HTTPStatusError as e:
        return JSONResponse({"ok": False, "endpoint": label,
                             "error": f"{label}/v1/models returned HTTP {e.response.status_code}"}, status_code=502)
    except httpx.HTTPError as e:
        return JSONResponse({"ok": False, "endpoint": label,
                             "error": f"No OpenAI-compatible server answering at {label} ({type(e).__name__})"}, status_code=502)
    if not info["models"]:
        return JSONResponse({"ok": False, "endpoint": label, "error": f"{label} reports no models"}, status_code=502)
    return {"ok": True, "endpoint": label, **info}


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Forward one chat turn to the connected server and stream normalized NDJSON events."""
    body = await request.json()
    try:
        base_url, _ = await chat_proxy.resolve_endpoint(body.get("endpoint", ""))
    except chat_proxy.EndpointError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    messages = body.get("messages") or []
    if not messages:
        return JSONResponse({"error": "No messages"}, status_code=400)
    payload = chat_proxy.build_payload(body.get("engine") or "openai", body.get("model"),
                                       messages, body.get("sampling") or {})
    return StreamingResponse(
        chat_proxy.stream_chat(base_url, payload),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# API Routes - Status
# ============================================================================

 
@app.get("/api/status")
async def status():
    """System status endpoint."""
    return {
        "app": {
            "port": config_manager.app_config.webui_port,
            "models_dir": str(config_manager.app_config.models_directory),
        },
        "gpu_detection": gpu_manager.get_pynvml_status(),
        "gpus": gpu_manager.get_gpu_status(),
        "models": {
            name: {
                "size_gb": m.size_gb,
                "port": m.port,
                "state": gpu_manager.get_model_state(name).value,
            }
            for name, m in config_manager.get_all_models().items()
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "app": "llama-studio"}


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates."""
    await event_bus.connect(websocket)
    try:
        while True:
            # The page pings to prove the connection is alive; it reconnects
            # when nothing at all arrives for a while.
            if await websocket.receive_text() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception:
        event_bus.disconnect(websocket)

