"""FastAPI application for Llama Studio."""

import asyncio
import logging
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional
import os
import sys
import argparse
import socket

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Update on each release. Surfaced in the page <title> and header.
APP_VERSION = "0.2.3"

from fastapi import FastAPI, BackgroundTasks, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from config_manager import ConfigManager, ModelConfig, calculate_kv_cache_gb
from launch_script import parse_pasted_script
import migrate_configs
from gpu_manager import GpuManager, ModelState, LoadingPhase
from templates import jinja_env
from event_bus import event_bus
from llama_options import get_option_schema, validate_option, set_runtime_schema
from llama_version import get_version
from schema_parser import parse_help, load_schema, save_schema

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

# Initialize managers
config_manager = ConfigManager(PROJECT_ROOT / "config" / "app.json", project_root=PROJECT_ROOT)
gpu_manager = GpuManager(config_manager)

# Set by lifespan if legacy JSON configs are present at startup; cleared after migration.
LEGACY_CONFIGS_DETECTED = False

# Progress state for the startup auto-load background task. Cleared when complete.
AUTO_LOAD_PROGRESS: Dict[str, int] = {"active": 0, "current_index": 0, "total": 0}
AUTO_LOAD_CURRENT_NAME: str = ""

# Progress state for the synchronous rescan operation. Polled by the UI via
# /api/rescan-progress while the rescan request is in flight.
RESCAN_STATE: Dict[str, object] = {"active": False, "current": "", "done": 0, "total": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("🦙 Llama Studio Starting")
    logger.info("=" * 60)

    try:
        await config_manager.load_app_config()
        # Load or detect llama-server schema BEFORE model scanning
        await ensure_schema_for_binary(config_manager.app_config.llama_server_binary, config_manager, update_global=True)

        # Hard-block on legacy JSON configs — they must be migrated to shell scripts.
        if migrate_configs.has_legacy_configs(PROJECT_ROOT):
            global LEGACY_CONFIGS_DETECTED
            LEGACY_CONFIGS_DETECTED = True
            logger.warning("⚠ Legacy JSON model configs detected — model loading is blocked until migration completes.")
            logger.warning("   Visit the WebUI and click 'Migrate now', or run: python backend/migrate_configs.py")
        else:
            await config_manager.load_all_models()

        await gpu_manager.initialize()

        logger.info("✓ Initialization complete")
        logger.info("=" * 60)
        logger.info(f"🌐 Llama Studio WebUI started at: http://localhost:{config_manager.app_config.webui_port}")
        logger.info("=" * 60)

        # Auto-load stored models as a background task so the WebUI is reachable
        # immediately. Each load is async (health-check polling, etc.) and can
        # take 30-120s; the banner endpoint surfaces progress to the UI.
        if (
            not LEGACY_CONFIGS_DETECTED
            and config_manager.app_config.auto_load_on_startup
            and config_manager.app_config.startup_models
        ):
            asyncio.create_task(_auto_load_startup_models())
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("🦙 Llama Studio Shutting Down")
    logger.info("=" * 60)
    try:
        await gpu_manager.cleanup()
        logger.info("✓ All sessions terminated")
    except Exception as e:
        logger.error(f"✗ Error during cleanup: {e}")

    logger.info("✓ Shutdown complete")


async def ensure_schema_for_binary(binary_path: str, config_mgr, update_global: bool = False) -> str | None:
    """
    Validate binary, get version, load or generate+cache schema.
    Returns version_str on success, None on failure.
    If update_global=True, updates app_config.llama_schema_version and saves app.json.
    """
    config_dir = config_mgr.project_root / "config"

    # Step 1: Always detect binary version
    version_str = get_version(binary_path)
    if not version_str:
        logger.warning(f"⚠ Could not detect llama-server version for {binary_path}")
        if update_global:
            set_runtime_schema({})
        return None

    # Step 2: For global binary, check if version changed
    if update_global:
        config = config_mgr.app_config
        if config.llama_schema_version and config.llama_schema_version == version_str:
            schema = load_schema(config.llama_schema_version, config_dir)
            if schema:
                set_runtime_schema(schema)
                logger.info(f"✓ Using cached schema: {config.llama_schema_version}")
                return version_str
        else:
            if config.llama_schema_version:
                logger.info(f"⚠ Binary version changed: {config.llama_schema_version} → {version_str}")

    # Step 3: Try to load cached schema for this version
    schema = load_schema(version_str, config_dir)
    if schema:
        if update_global:
            config_mgr.app_config.llama_schema_version = version_str
            await config_mgr.save_app_config()
            set_runtime_schema(schema)
        logger.info(f"✓ Loaded cached schema for version: {version_str}")
        return version_str

    # Step 4: Parse help text and generate new schema
    logger.info(f"⏳ Parsing help text for version: {version_str}")
    schema = parse_help(binary_path)
    if not schema:
        logger.warning(f"⚠ Could not parse llama-server help for {binary_path}")
        if update_global:
            set_runtime_schema({})
        return None

    # Step 5: Save schema
    save_schema(schema, version_str, config_dir)
    if update_global:
        config_mgr.app_config.llama_schema_version = version_str
        await config_mgr.save_app_config()
        set_runtime_schema(schema)
    logger.info(f"✓ Generated and cached schema for version: {version_str}")
    return version_str


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


# ============================================================================
# API Routes - GPU Panel
# ============================================================================


@app.get("/api/gpu-panel", response_class=HTMLResponse)
async def gpu_panel():
    """Return GPU visualization HTML."""
    gpu_data = gpu_manager.get_gpu_status()

    if not gpu_data:
        return """
        <div class="text-center py-8 text-gray-400">
            <p>GPU detection not available</p>
        </div>
        """

    # Get machine's local IPv4 address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Doesn't actually connect, just determines local IP
        server_ipv4 = s.getsockname()[0]
        s.close()
    except Exception:
        # Fallback if can't determine
        server_ipv4 = "localhost"

    html = '<div class="gpu-table">'

    # Render each GPU row with loaded models
    row_template = jinja_env.get_template("snippets/gpu_row.html")
    for gpu_id, gpu_info in sorted(gpu_data.items()):
        memory_pct = (gpu_info["allocated"] / gpu_info["memory"]) * 100 if gpu_info["memory"] > 0 else 0

        html += row_template.render(
            gpu_id=gpu_id,
            gpu=gpu_info,
            memory_pct=memory_pct,
            server_host=server_ipv4,
        )

    html += '</div>'
    return html


# ============================================================================
# API Routes - Model List
# ============================================================================


@app.get("/api/model-list", response_class=HTMLResponse)
async def model_list():
    """Return model list HTML using templates."""
    models = config_manager.get_all_models()
    logger.info(f"📋 model_list called: {len(models)} models found")

    # === NEW: Check if llama-server is configured ===
    llama_binary = config_manager.app_config.llama_server_binary
    if not Path(llama_binary).exists():
        return f"""
        <div class="text-center py-8 text-yellow-400">
            <p>⚠ Llama-server binary not configured</p>
            <p class="text-sm text-gray-400 mt-2">Configure path in settings to load models</p>
        </div>
        """

    if not models:
        logger.warning("⚠ No models in config_manager")
        return '<div class="text-center py-8 text-gray-400"><p>No models found</p></div>'

    # Sort: configured first (alphabetically), then unconfigured (alphabetically)
    sorted_models = sorted(
        models.items(),
        key=lambda x: (not x[1].is_configured, x[0])
    )

    html = '<div class="models-table">'

    try:
        # Render each model row using the model_row template
        row_template = jinja_env.get_template("snippets/model_row.html")
        for model_name, model_config in sorted_models:
            state = gpu_manager.get_model_state(model_name)
            status = state.value

            port_display = f":{model_config.port}" if model_config.port else "—"

            # Display total_vram (calculated memory needed) if available, otherwise show file size as fallback
            if model_config.total_vram:
                vram_display = f"{model_config.total_vram} GB"
            elif model_config.size_gb:
                vram_display = f"{model_config.size_gb} GB"
            else:
                vram_display = "—"

            log_path = config_manager.get_log_path(model_name)

            html += row_template.render(
                model_name=model_name,
                status=status,
                model_config=model_config,
                port_display=port_display,
                vram_display=vram_display,
                log_path=str(log_path),
            )
    except Exception as e:
        logger.error(f"✗ Error rendering model rows: {type(e).__name__}: {e}", exc_info=True)
        return f'<div class="text-red-500 p-4">Error: {str(e)}</div>'

    html += '</div>'
    logger.info(f"✓ Generated model list HTML: {len(html)} chars, {len(sorted_models)} models")
    return html


# ============================================================================
# API Routes - Model Status Inner
# ============================================================================


@app.get("/api/get-model-status-inner", response_class=HTMLResponse)
async def get_model_status_inner(model: str):
    """Return status badge + action button HTML for inner polling."""
    if model not in config_manager.get_all_models():
        return HTMLResponse('<div class="text-red-400 p-4">Model not found</div>', status_code=404)

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


def _legacy_migration_page() -> HTMLResponse:
    """Render a one-page migration UI shown when legacy JSON configs are present."""
    legacy_files = migrate_configs.find_legacy_configs(PROJECT_ROOT / "config" / "models")
    file_list_html = "".join(
        f'<li class="font-mono text-xs text-gray-300 py-1">{p.name}</li>' for p in legacy_files
    )
    html = f"""<!DOCTYPE html>
<html><head><title>Llama Studio — Migration Required</title>
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen flex items-center justify-center p-6">
  <div class="bg-gray-800 border border-yellow-600 rounded-lg p-6 max-w-2xl w-full">
    <h1 class="text-2xl font-bold text-yellow-400 mb-3">⚠ Legacy Configs Detected</h1>
    <p class="text-sm text-gray-300 mb-4">
      Llama Studio now stores model configs as executable shell scripts (<code>.sh</code>)
      instead of JSON. Found {len(legacy_files)} legacy file(s) that need to be migrated
      before models can be loaded.
    </p>
    <p class="text-sm text-gray-400 mb-4">
      Migration is non-destructive: each <code>&lt;name&gt;.json</code> will be converted to
      <code>&lt;name&gt;.sh</code> and the original renamed to <code>&lt;name&gt;.json.old</code>
      (kept as a backup for rollback).
    </p>
    <details class="mb-4">
      <summary class="cursor-pointer text-sm text-blue-400 hover:text-blue-300">
        Show files to migrate
      </summary>
      <ul class="mt-2 ml-4 list-disc">{file_list_html}</ul>
    </details>
    <div id="migration-result" class="mb-4"></div>
    <div class="flex gap-3">
      <button
        hx-post="/api/run-migration"
        hx-target="#migration-result"
        hx-swap="innerHTML"
        class="flex-1 px-4 py-2 rounded bg-green-600 hover:bg-green-500 text-white font-bold">
        Migrate now
      </button>
      <a href="/" class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold text-center">
        Reload page
      </a>
    </div>
    <p class="text-xs text-gray-500 mt-4">
      Or run from CLI: <code>python backend/migrate_configs.py</code>
    </p>
  </div>
</body></html>"""
    return HTMLResponse(html)


@app.post("/api/run-migration", response_class=HTMLResponse)
async def run_migration():
    """Trigger one-shot JSON -> SH migration and reload models."""
    global LEGACY_CONFIGS_DETECTED
    try:
        successes, failures = migrate_configs.migrate_all(PROJECT_ROOT, dry_run=False)
    except Exception as e:
        logger.error(f"✗ Migration failed: {e}", exc_info=True)
        return HTMLResponse(
            f'<div class="p-3 bg-red-900 border border-red-600 rounded text-sm">'
            f'✗ Migration failed: {str(e)}</div>'
        )

    if failures:
        body = "<div class='p-3 bg-red-900 border border-red-600 rounded text-sm'>"
        body += "<p class='font-bold mb-2'>✗ Migration completed with failures:</p><ul class='list-disc ml-4'>"
        for f in failures:
            body += f"<li>{f}</li>"
        body += "</ul></div>"
        return HTMLResponse(body)

    # All migrated cleanly — clear the flag and load models
    LEGACY_CONFIGS_DETECTED = False
    if not migrate_configs.has_legacy_configs(PROJECT_ROOT):
        try:
            await config_manager.load_all_models()
            gpu_manager.sync_models_from_config()
        except Exception as e:
            logger.error(f"✗ Error loading models after migration: {e}", exc_info=True)

    body = "<div class='p-3 bg-green-900 border border-green-600 rounded text-sm'>"
    body += f"<p class='font-bold mb-2'>✓ Migrated {len(successes)} model(s):</p>"
    body += "<ul class='list-disc ml-4 max-h-48 overflow-y-auto'>"
    for s in successes:
        body += f"<li class='text-xs'>{s}</li>"
    body += "</ul>"
    body += "<p class='mt-3'>Reload the page to continue.</p>"
    body += '<a href="/" class="inline-block mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded">Reload</a>'
    body += "</div>"
    return HTMLResponse(body)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page with config status."""
    # If legacy JSON configs exist, render the migration page instead.
    if LEGACY_CONFIGS_DETECTED:
        return _legacy_migration_page()

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
    return HTMLResponse(
        f'<span class="text-blue-300">⏳ Rescanning ({done}/{total}): '
        f'<code class="text-blue-200">{current}</code></span>'
    )


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
            content=f'<span class="text-green-400">✓ Rescanned: {count} model(s) found</span>',
            status_code=200,
            headers={"HX-Trigger": "modelsRescanned"},
        )
    except Exception as e:
        logger.error(f"✗ Error rescanning models: {e}")
        return HTMLResponse(
            content=f'<span class="text-red-400">✗ Error: {str(e)}</span>',
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

    Updates AUTO_LOAD_PROGRESS and AUTO_LOAD_CURRENT_NAME so the banner endpoint
    can surface progress to the UI.
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

    loaded = 0
    skipped = 0
    try:
        for idx, name in enumerate(names, start=1):
            AUTO_LOAD_PROGRESS["current_index"] = idx
            AUTO_LOAD_CURRENT_NAME = name
            try:
                try:
                    mc = config_manager.get_model_config(name)
                except FileNotFoundError:
                    logger.warning(f"   ⏭  {name}: skipped — script no longer exists")
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

                if len(cvd) != mc.gpu_count:
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

    logger.info("=" * 60)
    logger.info(f"🚀 Auto-load complete: {loaded} loaded, {skipped} skipped")
    logger.info("=" * 60)


@app.get("/api/auto-load-banner", response_class=HTMLResponse)
async def auto_load_banner():
    """Return banner HTML when startup auto-load is in progress, empty otherwise.

    Polled by the index page. When loading finishes the swap clears the banner.
    """
    if not AUTO_LOAD_PROGRESS.get("active"):
        return HTMLResponse("")
    cur = AUTO_LOAD_PROGRESS.get("current_index", 0)
    tot = AUTO_LOAD_PROGRESS.get("total", 0)
    name = AUTO_LOAD_CURRENT_NAME or "…"
    return HTMLResponse(
        f'<div class="bg-red-900 border border-red-600 text-red-100 px-4 py-2 text-sm flex items-center gap-3">'
        f'<span class="animate-pulse">⏳</span>'
        f'<span><b>Auto-loading stored models</b> ({cur}/{tot}): <code class="text-red-200">{name}</code></span>'
        f'</div>'
    )


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
    tooltip_lines = state["stored"] if state["stored"] else ["(nothing stored)"]
    tooltip = "Stored for startup:\n" + "\n".join(f"• {n}" for n in tooltip_lines)
    return (
        f'<button id="store-config-btn" '
        f'hx-post="/api/store-startup-config" '
        f'hx-target="#store-config-btn-wrap" hx-swap="innerHTML" '
        f'title="{tooltip}" '
        f'class="px-3 py-1 rounded border text-white text-xs font-bold transition-colors '
        f'bg-green-700 hover:bg-green-600 border-green-500">'
        f'Store this state to autoload [{state["loaded_count"]} Loaded]'
        f'</button>'
    )


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
            f'<span class="text-red-400 text-xs">✗ Save failed: {e}</span>',
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



@app.get("/api/config-modal-new", response_class=HTMLResponse)
async def config_modal_new(model: str):
    """Return improved config modal with form + advanced table + VRAM calculator."""
    logger.info(f"📝 Edit config (v2) requested for model: {model}")
    try:
        model_config = config_manager.get_model_config(model)

        # Determine which schema to use based on model's llama_path
        version_raw = None
        if model_config.llama_path and model_config.llama_path != "default":
            # Use custom binary's schema if available
            custom_version = get_version(model_config.llama_path)
            if custom_version:
                version_raw = custom_version
                option_schema = load_schema(custom_version, config_manager.project_root / "config")
                if not option_schema:
                    logger.warning(f"⚠ Schema not found for custom binary version {custom_version}, using global schema")
                    option_schema = get_option_schema()

        # Fall back to global schema if not using custom binary
        if not version_raw:
            version_raw = config_manager.app_config.llama_schema_version
            option_schema = get_option_schema()

        # Format the version string
        if version_raw:
            parts = version_raw.split("_")
            version_str = f"{parts[0]} ({parts[1]})" if len(parts) == 2 else version_raw
        else:
            version_str = "unknown"

        # Split launch_args into individual options
        launch_args = model_config.launch_args or {}

        # Extract context and KV quantization settings for VRAM calculator
        # Long form first — matches the lookup order in _calculate_total_vram()
        current_ctx = launch_args.get("--ctx-size") or launch_args.get("-c") or "4096"
        current_ctk = launch_args.get("--cache-type-k") or launch_args.get("-ctk") or "f16"
        current_ctv = launch_args.get("--cache-type-v") or launch_args.get("-ctv") or "f16"

        # Remove -c, -ctk, -ctv from advanced_options to prevent duplication
        advanced_options = {k: v for k, v in launch_args.items()
                           if k not in ["--host", "--port", "-c", "--ctx-size", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v"]}
        host = launch_args.get("--host", "0.0.0.0")
        port = launch_args.get("--port", model_config.port or "")

        # Render template
        template = jinja_env.get_template("modals/config_modal_new.html")
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
        logger.error(f"✗ Error in config_modal_new: {type(e).__name__}: {e}", exc_info=True)
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


@app.post("/api/save-model-config-new", response_class=HTMLResponse)
async def save_model_config_new(
    model_name: str = Form(...),
    core_json: str = Form(...),
    advanced_json: str = Form(...),
):
    """Save config with validation at each step."""
    logger.info(f"💾 Saving config (v2) for model: {model_name}")
    try:
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
        invalid_keys = []
        for key, value in advanced_data.items():
            if key in CORE_FIELDS_SKIP:
                continue
            is_valid, error_msg = validate_option(key, str(value))
            if not is_valid:
                invalid_keys.append(key)
                logger.warning(f"⚠ Invalid option in config: {key} ({error_msg})")

        # If there are invalid options, log them but continue (allow saving)
        has_valid_options = len(invalid_keys) == 0
        if invalid_keys:
            logger.info(f"⚠ Model {model_name} has invalid options: {invalid_keys}")

        # Build launch_args from core + advanced
        launch_args = {"--host": host, "--port": str(port), **advanced_data}

        # Validate --tensor-split if present: must be a comma-separated list of positive numbers.
        ts_raw = launch_args.get("--tensor-split") or launch_args.get("-ts")
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
            new_split_count = len(weights)

        # Get existing model config to preserve model_path and (if still valid) CVD
        existing_config = config_manager.get_model_config(model_name)
        prior_cvd = existing_config.cuda_visible_devices  # captured before save_model_script clobbers it

        # Resolve llama_bin: "default" -> use app config's binary
        if llama_path == "default":
            llama_bin = config_manager.app_config.llama_server_binary
        else:
            llama_bin = llama_path

        try:
            model_config = config_manager.save_model_script(
                model_name,
                display_name=display_name or existing_config.display_name,
                model_path=existing_config.model_path,
                llama_bin=llama_bin,
                args=launch_args,
                health_timeout=health_timeout,
            )
        except Exception as e:
            logger.error(f"✗ Failed to save script: {e}", exc_info=True)
            return _error_modal(f"Failed to save script: {str(e)}")

        if not model_config:
            return _error_modal(f"Model {model_name} not found after save")

        # Re-apply prior CVD if it survives validation against the new --tensor-split count.
        # save_model_script wipes everything below the meta fence, so we must explicitly
        # restore CVD that's still consistent with the saved split. Mismatched CVDs are
        # silently dropped (the picker will surface re-pick at next load).
        if prior_cvd is not None:
            if len(prior_cvd) == new_split_count:
                model_config = config_manager.save_cuda_visible_devices(model_name, prior_cvd)
            else:
                logger.info(
                    f"⚠ Dropping stale CUDA_VISIBLE_DEVICES for {model_name}: "
                    f"had {len(prior_cvd)} GPU(s), tensor-split now declares {new_split_count}"
                )

        logger.info(f"✓ Script saved for {model_name}")
        return _save_success_modal(model_config, invalid_keys=invalid_keys)
    except Exception as e:
        logger.error(f"✗ Error in save_model_config_new: {type(e).__name__}: {e}", exc_info=True)
        return _error_modal(f"Error saving configuration: {str(e)}")


def _save_success_modal(model_config: ModelConfig, invalid_keys=None) -> str:
    """Helper to generate the save-success modal HTML.

    If the model is currently loaded (or loading), surface a warning that the
    running process is still using the prior args, and offer to unload now or
    keep the prior load running until the user is ready to relaunch.
    """
    invalid_keys = invalid_keys or []
    is_configured = model_config.is_configured

    # Detect "currently in use" state: a session exists or model is RUNNING/LOADING.
    is_loaded = (
        model_config.name in gpu_manager.sessions
        or gpu_manager.get_model_state(model_config.name) in (ModelState.RUNNING, ModelState.LOADING)
    )

    if is_configured:
        status_text = "✓ Model is now configured!"
    elif invalid_keys:
        status_text = f"⚠ Configuration saved but model is unconfigured (invalid options: {', '.join(invalid_keys)})"
    elif model_config.config_error:
        status_text = f"⚠ Configuration saved — {model_config.config_error}"
    else:
        status_text = "⚠ Configuration saved but model is still unconfigured"
    status_color = "green" if is_configured else "yellow"

    # Mode-specific block: loaded-model warning + actions, OR plain OK button
    if is_loaded:
        body_html = f"""
            <div class="mt-2 mb-4 p-3 rounded border border-yellow-600 bg-yellow-900 bg-opacity-25">
                <p class="text-sm text-yellow-300 font-bold mb-1">⚠ This model is currently loaded.</p>
                <p class="text-xs text-gray-300">
                    The running process is still using the previous launch args. Unload now to apply
                    the new configuration on next load, or keep the prior load running.
                </p>
            </div>
            <div class="flex gap-2">
                <button onclick="document.getElementById('modal').remove(); refreshModelList();"
                        class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold transition-colors">
                    Keep prior load
                </button>
                <button hx-get="/api/unload-confirm?model={model_config.name}"
                        hx-target="#modal-container" hx-swap="innerHTML"
                        class="flex-1 px-4 py-2 rounded bg-yellow-600 hover:bg-yellow-500 text-white font-bold transition-colors">
                    Unload now
                </button>
            </div>
        """
    else:
        body_html = f"""
            <button onclick="closeModalAndRefresh();"
                    class="w-full px-4 py-2 rounded bg-{status_color}-600 hover:bg-{status_color}-500 text-white font-bold transition-colors">
                OK
            </button>
        """

    return f"""
    <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
         id="modal"
         onclick="if(event.target.id === 'modal') closeModalAndRefresh();">
        <div class="bg-gray-800 border border-{status_color}-600 rounded-lg p-6 w-full max-w-md"
             onclick="event.stopPropagation()">
            <div class="flex items-center space-x-3 mb-4">
                <span class="text-2xl">{'✓' if is_configured else '⚠'}</span>
                <h2 class="text-lg font-bold text-{status_color}-400">Configuration Saved</h2>
            </div>
            <p class="text-sm text-gray-400 mb-1">{model_config.display_name}</p>
            <p class="text-sm text-gray-300 mb-4">{status_text}</p>
            {body_html}
        </div>
    </div>
    <script>
    function refreshModelList() {{
        setTimeout(() => {{
            htmx.ajax('GET', '/api/model-list', {{ target: '#models-container', swap: 'innerHTML' }});
        }}, 100);
    }}
    function closeModalAndRefresh() {{
        document.getElementById('modal').remove();
        refreshModelList();
    }}
    refreshModelList();
    </script>
    """


def _error_modal(message: str) -> str:
    """Helper to generate error modal HTML."""
    return f"""
    <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
         id="modal"
         onclick="if(event.target.id === 'modal') document.getElementById('modal').remove();">
        <div class="bg-gray-800 border border-red-600 rounded-lg p-6 w-full max-w-md"
             onclick="event.stopPropagation()">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-lg font-bold text-red-400">Configuration Error</h2>
                <button onclick="document.getElementById('modal').remove();"
                        class="text-gray-400 hover:text-white text-2xl">
                    ×
                </button>
            </div>
            <p class="text-sm text-gray-300 mb-4">{message}</p>
            <button onclick="document.getElementById('modal').remove();"
                    class="w-full px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold transition-colors">
                Close
            </button>
        </div>
    </div>
    """


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
    needs = model_config.per_gpu_vram() or [model_config.total_vram or model_config.size_gb or 0]
    if model_config.gpu_count != len(cvd) or len(needs) != len(cvd):
        # Count mismatch (or no CVD): caller treats this as "force picker"
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
            return HTMLResponse('<div class="text-red-500 p-4">Model not configured</div>')

        gpu_data = gpu_manager.get_gpu_status()
        if not gpu_data:
            return HTMLResponse('<div class="text-red-500 p-4">No GPUs detected</div>')

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
                    gpu_count=gpu_count,
                    cvd_csv=cvd_csv,
                    slots=slots,
                    all_ok=all_ok,
                    load_hx_vals=json.dumps(load_vals),
                ))
            # CVD count mismatch → fall through to picker (blank-slate)

        # Multi-pick path: tensor-split present, no usable CVD (or forced)
        if gpu_count > 1:
            per_gpu_need = model_config.per_gpu_vram() or [
                (model_config.total_vram or model_config.size_gb or 0) / gpu_count
            ] * gpu_count
            template = jinja_env.get_template("modals/gpu_multi_picker.html")
            return HTMLResponse(template.render(
                model_name=model,
                display_name=model_config.display_name or model,
                gpu_count=gpu_count,
                per_gpu_need=per_gpu_need,
                gpu_data=gpu_data,
            ))

        # Single-GPU picker (unchanged behavior for non-split models)
        template = jinja_env.get_template("modals/gpu_selector.html")
        return HTMLResponse(template.render(
            model_name=model,
            model_config=model_config,
            display_name=model_config.display_name or model,
            gpu_data=gpu_data,
        ))
    except FileNotFoundError as e:
        logger.error(f"✗ Model {model} not found: {e}")
        return HTMLResponse('<div class="text-red-500 p-4">Model not found</div>', status_code=404)
    except Exception as e:
        logger.error(f"✗ Error in gpu_selector: {type(e).__name__}: {e}", exc_info=True)
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
        kv_cache_gb = calculate_kv_cache_gb(block_count, ctx_size, kv_cache_multiplier, cache_type_k, cache_type_v)
        return {
            "total": round(size_gb + kv_cache_gb, 2),
            "weights": round(size_gb, 2),
            "kv_cache": round(kv_cache_gb, 2),
        }
    except Exception as e:
        logger.error(f"Error calculating VRAM: {e}")
        return {"error": str(e)}


@app.get("/api/get-current-settings", response_class=JSONResponse)
async def get_current_settings():
    """Return current settings for settings modal population."""
    return {
        "llama_server": config_manager.app_config.llama_server_binary,
        "models_directory": config_manager.app_config.models_directory,
        "webui_port": config_manager.app_config.webui_port,
    }


@app.get("/api/config-paths", response_class=HTMLResponse)
async def config_paths():
    """Return configuration paths UI."""
    try:
        app_config = config_manager.app_config
        llama_server_path = app_config.llama_server_binary
        models_dir_path = app_config.models_directory
        
        html = f"""
        <div class="space-y-4">
            <!-- Llama Server Path -->
            <div class="flex items-center gap-3">
                <label class="w-32 text-sm font-bold text-gray-300">Llama Server:</label>
                <input type="text" 
                       value="{llama_server_path}"
                       readonly
                       class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300 font-mono">
                <button hx-get="/api/file-browser?path_type=llama_server&current_path={llama_server_path}"
                        hx-target="#modal-container"
                        hx-swap="innerHTML"
                        class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-bold transition-colors">
                    Edit
                </button>
            </div>
            
            <!-- Models Directory -->
            <div class="flex items-center gap-3">
                <label class="w-32 text-sm font-bold text-gray-300">Models Dir:</label>
                <input type="text" 
                       value="{models_dir_path}"
                       readonly
                       class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300 font-mono">
                <button hx-get="/api/file-browser?path_type=models_directory&current_path={models_dir_path}"
                        hx-target="#modal-container"
                        hx-swap="innerHTML"
                        class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-bold transition-colors">
                    Edit
                </button>
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error(f"✗ Error in config_paths: {type(e).__name__}: {e}")
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)



@app.get("/api/file-browser", response_class=HTMLResponse)
async def file_browser(path_type: str, current_path: str = "/", modal: str = None):
    """Return file browser modal for selecting a directory.

    Args:
        path_type: 'llama_server' or 'models_directory'
        current_path: Current directory path
        modal: If 'settings', use callback to settings modal; else use form submission
    """
    try:
        logger.info(f"📂 File browser requested for {path_type} at {current_path} (modal={modal})")
        
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
        
        # Get parent directory
        parent_path = current_path_obj.parent if current_path_obj != current_path_obj.parent else current_path_obj
        
        # List directories
        directories = []
        try:
            items = sorted(current_path_obj.iterdir())
            for item in items:
                if item.is_dir() and not item.name.startswith('.'):
                    directories.append(item)
        except PermissionError:
            logger.warning(f"⚠ Permission denied accessing {current_path_obj}")
        
        # Build directory list HTML
        dir_html = ""

        # Build modal parameter for nested calls
        modal_param = f"&modal={modal}" if modal else ""

        # Parent directory link
        if current_path_obj != current_path_obj.parent:
            dir_html += f"""
            <button hx-get="/api/file-browser?path_type={path_type}&current_path={parent_path}{modal_param}"
                    hx-target="#file-browser-content"
                    hx-swap="innerHTML"
                    class="w-full text-left px-3 py-2 hover:bg-gray-700 rounded transition-colors text-blue-400">
                📁 ../
            </button>
            """

        # Subdirectories
        for directory in directories:
            dir_html += f"""
            <button hx-get="/api/file-browser?path_type={path_type}&current_path={directory}{modal_param}"
                    hx-target="#file-browser-content"
                    hx-swap="innerHTML"
                    class="w-full text-left px-3 py-2 hover:bg-gray-700 rounded transition-colors text-gray-300">
                📁 {directory.name}/
            </button>
            """
        
        # Build action buttons based on modal type
        if modal == "settings":
            # For settings modal, use onclick callback instead of form submission
            html = f"""
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-bold">Select Directory</h2>
                <button type="button"
                        onclick="closeFileBrowser()"
                        class="text-gray-400 hover:text-white text-2xl">
                    ×
                </button>
            </div>

            <div class="mb-4 p-3 bg-gray-900 rounded border border-gray-700">
                <p class="text-xs text-gray-500 mb-1">Current Path:</p>
                <p class="text-sm text-gray-300 font-mono">{current_path_obj}</p>
            </div>

            <div class="flex-1 overflow-y-auto mb-4 border border-gray-700 rounded">
                {dir_html if dir_html else '<div class="p-4 text-gray-500">No subdirectories</div>'}
            </div>

            <div class="flex gap-3">
                <button type="button"
                        onclick="closeFileBrowser()"
                        class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold transition-colors">
                    Cancel
                </button>
                <button type="button"
                        onclick="selectPath('{current_path_obj}')"
                        class="flex-1 px-4 py-2 rounded bg-green-600 hover:bg-green-500 text-white font-bold transition-colors">
                    Select
                </button>
            </div>
            """
        else:
            # For default mode, use form submission
            html = f"""
            <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
                 id="modal"
                 onclick="if(event.target.id === 'modal') document.getElementById('modal').remove();">
                <div class="bg-gray-800 border border-gray-600 rounded-lg p-6 w-full max-w-2xl max-h-96 flex flex-col"
                     onclick="event.stopPropagation()">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-bold">Select Directory</h2>
                        <button onclick="document.getElementById('modal').remove();"
                                class="text-gray-400 hover:text-white text-2xl">
                            ×
                        </button>
                    </div>

                    <div class="mb-4 p-3 bg-gray-900 rounded border border-gray-700">
                        <p class="text-xs text-gray-500 mb-1">Current Path:</p>
                        <p class="text-sm text-gray-300 font-mono">{current_path_obj}</p>
                    </div>

                    <div class="flex-1 overflow-y-auto mb-4 border border-gray-700 rounded">
                        {dir_html if dir_html else '<div class="p-4 text-gray-500">No subdirectories</div>'}
                    </div>

                    <div class="flex gap-3">
                        <button onclick="document.getElementById('modal').remove();"
                                class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold transition-colors">
                            Cancel
                        </button>
                        <button hx-post="/api/update-config-path"
                                hx-vals='{{"path_type": "{path_type}", "new_path": "{current_path_obj}"}}'
                                hx-target="#modal-container"
                                hx-swap="innerHTML"
                                class="flex-1 px-4 py-2 rounded bg-green-600 hover:bg-green-500 text-white font-bold transition-colors">
                            Select This Path
                        </button>
                    </div>
                </div>
            </div>
            """
        return html
    except Exception as e:
        logger.error(f"✗ Error in file_browser: {type(e).__name__}: {e}", exc_info=True)
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


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
            config_manager.app_config.llama_server_binary = str(new_path_obj)

            # Re-detect schema when llama_server path changes
            await ensure_schema_for_binary(str(new_path_obj), config_manager, update_global=True)
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

        # Save updated config
        config_file = PROJECT_ROOT / "config" / "app.json"
        config_data = config_manager.app_config.model_dump()
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"✓ Config updated and saved: {path_type} = {new_path}")

        # Return success message and refresh config section
        html = f"""
        <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
             id="modal"
             onclick="if(event.target.id === 'modal') document.getElementById('modal').remove();">
            <div class="bg-gray-800 border border-green-600 rounded-lg p-6 w-full max-w-md"
                 onclick="event.stopPropagation()">
                <div class="flex items-center space-x-3 mb-4">
                    <span class="text-2xl">✓</span>
                    <h2 class="text-lg font-bold text-green-400">Path Updated</h2>
                </div>
                <p class="text-sm text-gray-300 mb-4">
                    <span class="font-bold text-gray-200">{path_type}:</span><br>
                    <span class="font-mono text-xs text-gray-400">{new_path}</span>
                </p>
                <button onclick="document.getElementById('modal').remove(); htmx.ajax('GET', '/api/config-paths', '#config-paths')"
                        class="w-full px-4 py-2 rounded bg-green-600 hover:bg-green-500 text-white font-bold transition-colors">
                    OK
                </button>
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error(f"✗ Error in update_config_path: {type(e).__name__}: {e}", exc_info=True)
        error_html = f"""
        <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
             id="modal"
             onclick="if(event.target.id === 'modal') document.getElementById('modal').remove();">
            <div class="bg-gray-800 border border-red-600 rounded-lg p-6 w-full max-w-md"
                 onclick="event.stopPropagation()">
                <h2 class="text-lg font-bold text-red-400 mb-4">Error Updating Path</h2>
                <p class="text-sm text-gray-300 mb-4">{str(e)}</p>
                <button onclick="document.getElementById('modal').remove();"
                        class="w-full px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold transition-colors">
                    Close
                </button>
            </div>
        </div>
        """
        return error_html


@app.post("/api/update-webui-port", response_class=JSONResponse)
async def update_webui_port(port: int = Form(...)):
    """Update WebUI port and save to config."""
    logger.info(f"🔄 Updating WebUI port to: {port}")
    try:
        if not (1 <= port <= 65535):
            return {"error": "Port must be between 1 and 65535", "success": False}

        # Update app config
        config_manager.app_config.webui_port = port

        # Save updated config
        config_file = PROJECT_ROOT / "config" / "app.json"
        config_data = config_manager.app_config.model_dump()
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

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

        # Validate binary and ensure schema is cached
        resolved_path = str(path_obj)
        version_str = await ensure_schema_for_binary(resolved_path, config_manager, update_global=False)

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


@app.get("/api/model-log-tail", response_class=PlainTextResponse)
async def model_log_tail(model: str, lines: int = 30):
    """Return last N lines from a model's logfile as plain text."""
    try:
        log_path = config_manager.get_log_path(model)

        if not log_path.exists():
            return "(no log file)"

        # Read last N lines from file, ignoring encoding errors
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        tail_lines = all_lines[-lines:] if all_lines else []

        # Return as plain text, strip trailing whitespace from each line
        return '\n'.join(line.rstrip() for line in tail_lines)

    except Exception as e:
        logger.error(f"✗ Error reading log for {model}: {e}")
        return f"(error reading log: {str(e)})"


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
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception:
        event_bus.disconnect(websocket)

