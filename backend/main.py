"""FastAPI application for Llama Studio."""

import logging
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List
import os
import sys
import argparse
import socket

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, BackgroundTasks, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from config_manager import ConfigManager, ModelConfig, calculate_kv_cache_gb
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
        await config_manager.load_all_models()
        await gpu_manager.initialize()

        logger.info("✓ Initialization complete")
        logger.info("=" * 60)
        logger.info(f"🌐 Llama Studio WebUI started at: http://localhost:{config_manager.app_config.webui_port}")
        logger.info("=" * 60)
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


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page with config status."""
    # Check if configuration is complete
    needs_config, error_msg = config_manager.needs_configuration()
    config_incomplete = needs_config

    # Render template with config status
    template = jinja_env.get_template("index.html")
    return template.render(config_incomplete=config_incomplete, config_error=error_msg)


# ============================================================================
# API Routes - Rescan Models
# ============================================================================


@app.post("/api/rescan-models", response_class=HTMLResponse)
async def rescan_models():
    """Rescan models directory and regenerate skeleton configs for missing models."""
    try:
        await config_manager.load_all_models()
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


# ============================================================================
# API Routes - Model Configuration
# ============================================================================


# ============================================================================
# API Routes - Config (Modal)
# ============================================================================


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

        # Parse advanced fields
        try:
            advanced_data = json.loads(advanced_json) if advanced_json.strip() else {}
        except json.JSONDecodeError as e:
            logger.error(f"✗ Advanced config JSON parse error: {e}")
            error_msg = f"Advanced config JSON parse error: {str(e)}"
            return _error_modal(error_msg)

        # === NEW: Validate each advanced option and collect invalid keys ===
        invalid_keys = []
        for key, value in advanced_data.items():
            is_valid, error_msg = validate_option(key, str(value))
            if not is_valid:
                invalid_keys.append(key)
                logger.warning(f"⚠ Invalid option in config: {key} ({error_msg})")

        # If there are invalid options, log them but continue (allow saving)
        has_valid_options = len(invalid_keys) == 0
        if invalid_keys:
            logger.info(f"⚠ Model {model_name} has invalid options: {invalid_keys}")

        # Build launch_args from core + advanced
        launch_args = {"--host": host, "--port": port, **advanced_data}

        # Get existing model config to preserve other fields
        existing_config = config_manager.get_model_config(model_name)

        # Merge all config (clean minimal schema)
        merged_config = {
            "name": existing_config.name,
            "display_name": display_name or existing_config.display_name,
            "model_path": existing_config.model_path,
            "llama_path": llama_path,
            "launch_args": launch_args,
        }

        # Create ModelConfig to validate
        try:
            model_config = ModelConfig(
                **merged_config,
                port=port,  # Include port field for consistency
            )
        except Exception as e:
            logger.error(f"✗ Invalid model config: {e}")
            return _error_modal(f"Invalid configuration: {str(e)}")

        # Save to file
        config_file = config_manager.project_root / "config" / "models" / f"{model_name}.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(merged_config, f, indent=2)

        logger.info(f"✓ Config saved for {model_name}")
        logger.debug(f"  Saved config: {json.dumps(merged_config, indent=2)}")

        # Rescan model to recompute all derived fields (size_gb, is_configured, etc.)
        model_config = config_manager.rescan_model(model_name)
        if not model_config:
            return _error_modal(f"Model {model_name} not found after save")

        is_configured = model_config.is_configured

        # Return success message
        if is_configured:
            status_text = "✓ Model is now configured!"
        elif invalid_keys:
            status_text = f"⚠ Configuration saved but model is unconfigured (invalid options: {', '.join(invalid_keys)})"
        else:
            status_text = "⚠ Configuration saved but model is still unconfigured"

        status_color = "green" if is_configured else "yellow"

        html = f"""
        <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
             id="modal"
             onclick="if(event.target.id === 'modal') closeModalAndRefresh();">
            <div class="bg-gray-800 border border-{status_color}-600 rounded-lg p-6 w-full max-w-md"
                 onclick="event.stopPropagation()">
                <div class="flex items-center space-x-3 mb-4">
                    <span class="text-2xl">{'✓' if is_configured else '⚠'}</span>
                    <h2 class="text-lg font-bold text-{status_color}-400">Configuration Saved</h2>
                </div>
                <p class="text-sm text-gray-300 mb-4">{status_text}</p>
                <button onclick="closeModalAndRefresh();"
                        class="w-full px-4 py-2 rounded bg-{status_color}-600 hover:bg-{status_color}-500 text-white font-bold transition-colors">
                    OK
                </button>
            </div>
        </div>
        <script>
        function closeModalAndRefresh() {{
            document.getElementById('modal').remove();
            // Refresh the model list table immediately
            setTimeout(() => {{
                htmx.ajax('GET', '/api/model-list', {{ target: '#models-container', swap: 'innerHTML' }});
            }}, 100);
        }}
        // Auto-refresh on load to catch changes immediately
        setTimeout(() => {{
            htmx.ajax('GET', '/api/model-list', {{ target: '#models-container', swap: 'innerHTML' }});
        }}, 50);
        </script>
        """
        return html
    except Exception as e:
        logger.error(f"✗ Error in save_model_config_new: {type(e).__name__}: {e}", exc_info=True)
        return _error_modal(f"Error saving configuration: {str(e)}")


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


@app.get("/api/gpu-selector", response_class=HTMLResponse)
async def gpu_selector(model: str):
    """Return GPU selection modal."""
    logger.info(f"📋 GPU selector requested for model: {model}")
    try:
        model_config = config_manager.get_model_config(model)
        logger.debug(f"   Model config: is_configured={model_config.is_configured}, port={model_config.port}")

        if not model_config.is_configured:
            logger.warning(f"   Model {model} is not configured")
            return f'<div class="text-red-500 p-4">Model not configured</div>'

        gpu_data = gpu_manager.get_gpu_status()
        if not gpu_data:
            logger.warning(f"   No GPUs detected")
            return '<div class="text-red-500 p-4">No GPUs detected</div>'

        # Add memory_pct to each GPU for template
        for gpu_id, gpu_info in gpu_data.items():
            gpu_info["memory_pct"] = (gpu_info["allocated"] / gpu_info["memory"]) * 100 if gpu_info["memory"] > 0 else 0

        template = jinja_env.get_template("modals/gpu_selector.html")
        html = template.render(
            model_name=model,
            model_config=model_config,
            display_name=model_config.display_name or model,
            gpu_data=gpu_data,
        )
        return HTMLResponse(html)
    except FileNotFoundError as e:
        logger.error(f"✗ Model {model} not found: {e}")
        return HTMLResponse(f'<div class="text-red-500 p-4">Model not found</div>', status_code=404)
    except Exception as e:
        logger.error(f"✗ Error in gpu_selector: {type(e).__name__}: {e}", exc_info=True)
        return HTMLResponse(f'<div class="text-red-500 p-4">Error: {str(e)}</div>', status_code=500)


@app.post("/api/trigger-load", response_class=HTMLResponse)
async def trigger_load(
    background_tasks: BackgroundTasks,
    model_name: str = Form(),
    gpu_ids: List[int] = Form(),
):
    """Trigger a load operation."""
    gpu_id = gpu_ids if len(gpu_ids) > 1 else gpu_ids[0]
    logger.info(f"🚀 Load requested: {model_name} on GPU(s) {gpu_id}")
    try:
        await gpu_manager._set_model_state(model_name, ModelState.LOADING, LoadingPhase.QUEUED)
        background_tasks.add_task(gpu_manager.load_model_to_gpu, model_name, gpu_id)
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

