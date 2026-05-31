"""Configuration management for app and models.

Models are stored as executable shell scripts at config/models/<name>.sh.
Each script has a meta fence (display_name + GGUF-derived cache) and a
launch-args fence (the llama-server invocation). See backend/launch_script.py.
"""

import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from gguf_metadata_parser import parse_gguf_metadata
from launch_script import (
    ParsedScript,
    parse_script,
    render_skeleton_script,
    patch_meta_fence,
    patch_launch_args_fence,
    patch_cuda_visible_devices,
)

logger = logging.getLogger(__name__)


KV_BYTES: Dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q6_k": 0.75,
    "q5_k": 0.625,
    "q5_0": 0.625,
    "q4_k": 0.5,
    "q4_0": 0.5,
    "q3_k": 0.375,
}


def calculate_kv_cache_gb(
    block_count: int,
    ctx_size: int,
    kv_cache_multiplier: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
) -> float:
    """Compute KV cache size in GiB."""
    k = KV_BYTES.get(cache_type_k, 2.0)
    v = KV_BYTES.get(cache_type_v, 2.0)
    return block_count * ctx_size * kv_cache_multiplier * (k + v) / (1024 ** 3)


def _sanitize_html_id(name: str) -> str:
    """Convert a model name to a safe HTML ID."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    safe = re.sub(r'^[0-9]+', '', safe)
    return safe if safe else 'model'


class ModelConfig:
    """Single model configuration backed by a shell script.

    The script is the source of truth for everything launch-related; this class
    is a thin façade that exposes the historical fields used by the rest of the
    app (name, display_name, model_path, launch_args, port, etc.) plus derived
    computed fields (size_gb, total_vram, is_configured).
    """

    def __init__(self, name: str, script_path: Path, parsed: ParsedScript, default_llama_bin: Optional[str] = None):
        self.name = name
        self.script_path = Path(script_path)
        self.parsed = parsed
        self._default_llama_bin = default_llama_bin
        self.html_id = _sanitize_html_id(name)

        # Computed at scan time
        self.size_gb: Optional[float] = None
        self.total_vram: Optional[float] = None
        self.is_configured: bool = False
        self.config_error: Optional[str] = None  # reason for unconfigured, surfaced in UI

    # ---- proxies to parsed script ----

    @property
    def display_name(self) -> str:
        return self.parsed.display_name or self.name

    @property
    def model_path(self) -> Optional[str]:
        return self.parsed.model_path

    @property
    def block_count(self) -> Optional[int]:
        return self.parsed.block_count

    @property
    def max_context(self) -> Optional[int]:
        return self.parsed.max_context

    @property
    def kv_cache_multiplier(self) -> Optional[int]:
        return self.parsed.kv_cache_multiplier

    @property
    def launch_args(self) -> Dict[str, Any]:
        """Backward-compat: launch args dict (includes --port/--host, excludes -m)."""
        return dict(self.parsed.args)

    @property
    def port(self) -> Optional[int]:
        return self.parsed.port

    @property
    def host(self) -> Optional[str]:
        return self.parsed.host

    @property
    def llama_path(self) -> str:
        """Binary path baked into the script. 'default' if it matches app default."""
        bin_path = self.parsed.llama_bin
        if bin_path and self._default_llama_bin and bin_path == self._default_llama_bin:
            return "default"
        return bin_path or "default"

    @property
    def llama_bin_absolute(self) -> Optional[str]:
        return self.parsed.llama_bin

    # ---- multi-GPU ----

    @property
    def gpu_count(self) -> int:
        return self.parsed.gpu_count

    @property
    def tensor_split_weights(self) -> Optional[list]:
        return self.parsed.tensor_split_weights

    @property
    def cuda_visible_devices(self) -> Optional[list]:
        return self.parsed.cuda_visible_devices

    def per_gpu_vram(self) -> Optional[list]:
        """Return per-GPU VRAM share (in GB) as a list, honoring tensor-split weights.

        Returns None if total_vram is unknown. For single-GPU, returns [total_vram].
        For asymmetric splits, returns total_vram * weight_i / sum(weights).
        """
        if self.total_vram is None:
            return None
        weights = self.tensor_split_weights
        if weights is None or len(weights) <= 1:
            return [self.total_vram]
        total_weight = sum(weights)
        if total_weight <= 0:
            return [self.total_vram / len(weights)] * len(weights)
        return [round(self.total_vram * w / total_weight, 2) for w in weights]


class AppConfig(BaseModel):
    """Application configuration."""

    webui_port: int = Field(default=7999, description="WebUI port")
    models_directory: str = Field(default="./models", description="Path to models directory")
    logs_directory: str = Field(default="./logs", description="Path to logs directory")
    llama_server_binary: str = Field(
        default="/home/m6/servers/llamacpp/bin/llama-server",
        description="Path to llama-server binary"
    )
    llama_schema_version: Optional[str] = Field(
        default=None,
        description="Detected llama-server version (e.g., 9030_17df5830e)"
    )
    # Startup auto-load: list of model names to load at app startup.
    # gpu_ids are read from each model's CUDA_VISIBLE_DEVICES at load time
    # (not stored here — the script is the source of truth).
    startup_models: list = Field(default_factory=list, description="Model names to auto-load on startup")
    auto_load_on_startup: bool = Field(default=False, description="If True, load startup_models when the app starts")

    @field_validator("webui_port")
    def webui_port_must_be_valid(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("WebUI port must be between 1 and 65535")
        return v

    @field_validator("startup_models")
    def startup_models_are_non_empty_strings(cls, v):
        if not isinstance(v, list):
            raise ValueError("startup_models must be a list")
        out = []
        for entry in v:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"startup_models entries must be non-empty strings: {entry!r}")
            out.append(entry.strip())
        return out


class ConfigManager:
    """Manages application and model configurations."""

    def __init__(self, app_config_path: Path = Path("config/app.json"), project_root: Path = None):
        self.app_config_path = Path(app_config_path)
        if project_root is None:
            self.project_root = self.app_config_path.parent.parent
        else:
            self.project_root = Path(project_root)
        self.app_config: Optional[AppConfig] = None
        self.models: Dict[str, ModelConfig] = {}

    async def load_app_config(self) -> AppConfig:
        """Load application configuration from JSON."""
        try:
            with open(self.app_config_path) as f:
                config_data = json.load(f)
            self.app_config = AppConfig(**config_data)
            logger.info(f"✓ Loaded app config from {self.app_config_path}")
            return self.app_config
        except FileNotFoundError:
            logger.warning(f"⚠ App config not found at {self.app_config_path}, using defaults")
            self.app_config = AppConfig()
            return self.app_config
        except Exception as e:
            logger.warning(f"⚠ Error loading app config: {e}, using defaults")
            self.app_config = AppConfig()
            return self.app_config

    async def save_app_config(self) -> None:
        """Save application configuration to JSON."""
        try:
            with open(self.app_config_path, "w") as f:
                json.dump(self.app_config.model_dump(), f, indent=2)
            logger.info(f"✓ Saved app config to {self.app_config_path}")
        except Exception as e:
            logger.error(f"✗ Error saving app config: {e}")
            raise

    async def load_all_models(self) -> Dict[str, ModelConfig]:
        """Scan config/models/*.sh; for any GGUFs without a script, generate a skeleton."""
        if not self.app_config:
            raise RuntimeError("App config not loaded yet. Call load_app_config() first.")

        models_root = Path(self.app_config.models_directory)
        if not models_root.is_absolute():
            models_root = self.project_root / models_root

        config_models_dir = self.project_root / "config" / "models"
        config_models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}

        # Step 1: find GGUFs to generate skeletons for if missing
        logger.info(f"🔍 Searching for GGUF files in: {models_root}")
        gguf_files = list(models_root.rglob("*.gguf")) if models_root.exists() else []
        logger.info(f"   Found {len(gguf_files)} GGUF file(s)")

        for gguf_path in sorted(gguf_files):
            model_name = gguf_path.stem
            if model_name.startswith("mmproj"):
                logger.debug(f"   Skipping mmproj model: {model_name}")
                continue
            script_path = config_models_dir / f"{model_name}.sh"
            if not script_path.exists():
                self._generate_skeleton(script_path, model_name, gguf_path)

        # Step 2: load every .sh in config/models
        for script_path in sorted(config_models_dir.glob("*.sh")):
            model_name = script_path.stem
            try:
                model_config = self._load_one(script_path)
                self.models[model_name] = model_config
                status = "✓ configured" if model_config.is_configured else f"⚠ {model_config.config_error}"
                size_str = f"{model_config.size_gb:.1f}GB" if model_config.size_gb else "?"
                logger.info(f"   {status}: {model_name} ({size_str})")
            except Exception as e:
                logger.error(f"✗ Error loading {script_path.name}: {e}")

        configured = sum(1 for m in self.models.values() if m.is_configured)
        skeleton = len(self.models) - configured
        logger.info(f"✓ Loaded {len(self.models)} model(s) ({configured} configured, {skeleton} unconfigured)")
        return self.models

    def _load_one(self, script_path: Path) -> ModelConfig:
        """Read + parse a script, backfill GGUF metadata if missing, compute derived fields."""
        text = script_path.read_text()
        parsed = parse_script(text)

        # Ensure script is executable
        st = script_path.stat()
        if not (st.st_mode & stat.S_IXUSR):
            script_path.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        # Backfill GGUF metadata if missing
        needs_backfill = (
            parsed.model_path
            and Path(parsed.model_path).exists()
            and any(getattr(parsed, f) is None for f in ("block_count", "max_context", "kv_cache_multiplier"))
        )
        if needs_backfill:
            try:
                meta = parse_gguf_metadata(parsed.model_path)
                new_text = patch_meta_fence(
                    text,
                    display_name=parsed.display_name,
                    block_count=meta.block_count,
                    max_context=meta.max_context,
                    kv_cache_multiplier=meta.kv_cache_multiplier,
                )
                script_path.write_text(new_text)
                parsed = parse_script(new_text)
                logger.debug(f"   Backfilled GGUF metadata for {script_path.stem}")
            except Exception as e:
                logger.warning(f"⚠ Could not backfill metadata for {script_path.stem}: {e}")

        mc = ModelConfig(
            name=script_path.stem,
            script_path=script_path,
            parsed=parsed,
            default_llama_bin=self.app_config.llama_server_binary if self.app_config else None,
        )

        # Compute size and total VRAM
        if mc.model_path and Path(mc.model_path).exists():
            mc.size_gb = self._get_gguf_size_gb(Path(mc.model_path))
            mc.total_vram = self._calculate_total_vram(mc)

        # Configured check
        ok, reason = parsed.is_configured()
        mc.is_configured = ok
        mc.config_error = reason
        return mc

    def _generate_skeleton(self, script_path: Path, model_name: str, gguf_path: Path) -> None:
        """Write a skeleton script for a newly-discovered GGUF."""
        try:
            meta = parse_gguf_metadata(str(gguf_path))
            llama_bin = self.app_config.llama_server_binary if self.app_config else "/usr/local/bin/llama-server"
            text = render_skeleton_script(
                display_name=self._format_display_name(model_name),
                model_path=str(gguf_path),
                llama_bin=llama_bin,
                block_count=meta.block_count,
                max_context=meta.max_context,
                kv_cache_multiplier=meta.kv_cache_multiplier,
            )
            script_path.write_text(text)
            script_path.chmod(0o755)
            logger.debug(f"   Generated skeleton: {script_path}")
        except Exception as e:
            logger.error(f"✗ Error generating skeleton {script_path}: {e}")

    def rescan_model(self, model_name: str) -> Optional[ModelConfig]:
        """Re-read a model's script (picks up hand-edits) and recompute derived fields."""
        try:
            script_path = self.project_root / "config" / "models" / f"{model_name}.sh"
            if not script_path.exists():
                logger.warning(f"⚠ Script not found: {script_path}")
                return None
            mc = self._load_one(script_path)
            self.models[model_name] = mc
            status = "✓ configured" if mc.is_configured else f"⚠ {mc.config_error}"
            size_str = f"{mc.size_gb:.1f}GB" if mc.size_gb else "?"
            logger.info(f"   Rescanned {status}: {model_name} ({size_str})")
            return mc
        except Exception as e:
            logger.error(f"✗ Error rescanning {model_name}: {e}")
            return None

    def save_model_script(
        self,
        model_name: str,
        *,
        display_name: str,
        model_path: str,
        llama_bin: str,
        args: Dict[str, Optional[str]],
    ) -> ModelConfig:
        """Update a model's script by patching the launch-args and meta fences in place.

        Preserves any user content outside the fences. If the script doesn't
        exist yet, creates one from scratch.
        """
        script_path = self.project_root / "config" / "models" / f"{model_name}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # Pull GGUF metadata for the meta fence (use existing parsed if present)
        existing = self.models.get(model_name)
        block_count = existing.block_count if existing else None
        max_context = existing.max_context if existing else None
        kv_cache_multiplier = existing.kv_cache_multiplier if existing else None

        if script_path.exists():
            text = script_path.read_text()
            text = patch_meta_fence(
                text,
                display_name=display_name,
                block_count=block_count,
                max_context=max_context,
                kv_cache_multiplier=kv_cache_multiplier,
            )
            text = patch_launch_args_fence(
                text,
                llama_bin=llama_bin,
                model_path=model_path,
                args=args,
            )
        else:
            from launch_script import render_script
            text = render_script(
                display_name=display_name,
                block_count=block_count,
                max_context=max_context,
                kv_cache_multiplier=kv_cache_multiplier,
                llama_bin=llama_bin,
                model_path=model_path,
                args=args,
            )

        script_path.write_text(text)
        script_path.chmod(0o755)
        return self.rescan_model(model_name)

    def save_cuda_visible_devices(self, model_name: str, ids: Optional[list]) -> Optional[ModelConfig]:
        """Update the script's CUDA_VISIBLE_DEVICES line, no-op if unchanged.

        Pass ids=None to strip CVD entirely. Returns the rescan'd ModelConfig,
        or None if the script doesn't exist.
        """
        script_path = self.get_script_path(model_name)
        if not script_path.exists():
            logger.warning(f"⚠ Cannot update CVD: script not found {script_path}")
            return None
        existing = self.models.get(model_name)
        if existing is not None:
            current = existing.cuda_visible_devices
            if (ids is None and current is None) or (
                ids is not None and current is not None and list(ids) == list(current)
            ):
                logger.debug(f"   CVD unchanged for {model_name}; no write")
                return existing
        text = script_path.read_text()
        new_text = patch_cuda_visible_devices(text, ids)
        if new_text != text:
            script_path.write_text(new_text)
            logger.info(f"✓ Updated CVD for {model_name}: {ids}")
        return self.rescan_model(model_name)

    def get_script_path(self, model_name: str) -> Path:
        return self.project_root / "config" / "models" / f"{model_name}.sh"

    def get_script_text(self, model_name: str) -> str:
        return self.get_script_path(model_name).read_text()

    def save_script_text(self, model_name: str, text: str) -> ModelConfig:
        """Overwrite a model's script with raw text (used by the 'raw script' editor tab)."""
        path = self.get_script_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)
        return self.rescan_model(model_name)

    def _get_gguf_size_gb(self, gguf_path: Path) -> float:
        try:
            size_bytes = gguf_path.stat().st_size
            return round(size_bytes / (1024**3), 1)
        except Exception as e:
            logger.warning(f"⚠ Could not get size for {gguf_path}: {e}")
            return 0.0

    def _calculate_total_vram(self, mc: ModelConfig) -> Optional[float]:
        if not mc.size_gb or not mc.block_count or not mc.kv_cache_multiplier:
            return None
        args = mc.launch_args
        if not args:
            return None
        ctx_size = args.get("--ctx-size") or args.get("-c") or 4096
        cache_type_k = args.get("--cache-type-k") or args.get("-ctk") or "f16"
        cache_type_v = args.get("--cache-type-v") or args.get("-ctv") or "f16"
        try:
            ctx_size = int(ctx_size)
        except (ValueError, TypeError):
            ctx_size = 4096
        kv = calculate_kv_cache_gb(
            mc.block_count, ctx_size, mc.kv_cache_multiplier,
            str(cache_type_k), str(cache_type_v),
        )
        return round(mc.size_gb + kv, 2)

    def _format_display_name(self, filename: str) -> str:
        name = filename.replace('.gguf', '')
        name = name.replace('-', ' ').replace('_', ' ')
        return ' '.join(word.capitalize() for word in name.split())

    def get_model_config(self, model_name: str) -> ModelConfig:
        if model_name not in self.models:
            raise FileNotFoundError(f"Model config not found: {model_name}")
        return self.models[model_name]

    def get_all_models(self) -> Dict[str, ModelConfig]:
        return self.models

    def get_log_path(self, model_name: str) -> Path:
        if not self.app_config:
            raise RuntimeError("App config not loaded yet.")
        logs_dir = Path(self.app_config.logs_directory)
        if not logs_dir.is_absolute():
            logs_dir = self.project_root / logs_dir
        return logs_dir / f"{model_name}.log"

    def needs_configuration(self) -> tuple[bool, Optional[str]]:
        """First-run / corrupted-install check."""
        if not self.app_config:
            return True, "Configuration not loaded"
        if not self.app_config.llama_server_binary:
            return True, "llama-server binary not configured"
        binary_path = Path(self.app_config.llama_server_binary)
        if not binary_path.exists():
            return True, f"llama-server binary not found: {self.app_config.llama_server_binary}"
        models_path = Path(self.app_config.models_directory)
        if not models_path.is_absolute():
            models_path = self.project_root / models_path
        if not models_path.exists():
            return True, f"Models directory not found: {self.app_config.models_directory}"
        return False, None
