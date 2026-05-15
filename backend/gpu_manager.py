"""GPU state management and session coordination."""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from config_manager import ConfigManager, ModelConfig
from llama_session import LlamaSession
from port_manager import is_port_available
from event_bus import event_bus

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    """Primary model states for UI rendering."""

    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    FAILED = "failed"


class LoadingPhase(str, Enum):
    """Detailed phase tracking during load/unload operations."""

    IDLE = "idle"
    QUEUED = "queued"
    SPAWNING = "spawning"
    HEALTH_CHECKING = "health_checking"
    RUNNING = "running"
    UNLOADING = "unloading"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ModelStateSnapshot:
    """Complete state snapshot of a model at a point in time."""

    model_name: str
    state: ModelState
    phase: LoadingPhase
    error_msg: Optional[str] = None
    pid: Optional[int] = None
    gpu_id: int = -1
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoadedModel:
    """Represents a loaded model on a GPU."""

    name: str
    size_gb: float
    state: ModelState
    gpu_id: int
    port: int
    pid: Optional[int] = None
    vram_allocated: float = 0.0  # actual GB added to gpu.allocated (total_vram or size_gb)

    def to_dict(self):
        return asdict(self)


@dataclass
class GpuInfo:
    """GPU information and state."""

    gpu_id: int
    name: str
    memory: float  # Total VRAM in GB
    allocated: float  # Currently allocated in GB
    loaded_models: list  # List of LoadedModel dicts
    power_draw: float = None  # Current power draw in watts
    temperature: float = None  # Current temperature in °C

    def to_dict(self):
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "memory": self.memory,
            "allocated": self.allocated,
            "loaded_models": self.loaded_models,
            "power_draw": self.power_draw,
            "temperature": self.temperature,
        }


class GpuManager:
    """Manages GPU state and session lifecycle."""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.gpus: Dict[int, GpuInfo] = {}
        self.models: Dict[str, LoadedModel] = {}
        self.sessions: Dict[str, LlamaSession] = {}
        self.state_info: Dict[str, ModelStateSnapshot] = {}
        self.state_lock = asyncio.Lock()
        self.gpu_detection_status: str = "unavailable"
        self.gpu_detection_error: Optional[str] = None

    async def initialize(self) -> None:
        """Detect available GPUs on startup."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                if device_count == 0:
                    self.gpu_detection_status = "failed"
                    self.gpu_detection_error = "No NVIDIA GPUs detected"
                    logger.warning("⚠ No NVIDIA GPUs detected")
                    return

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    # Handle both str and bytes (different pynvml versions)
                    if isinstance(name, bytes):
                        name = name.decode()
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory_gb = mem_info.total / (1024**3)

                    self.gpus[i] = GpuInfo(
                        gpu_id=i,
                        name=name,
                        memory=total_memory_gb,
                        allocated=0.0,
                        loaded_models=[],
                    )
                    logger.info(f"✓ Detected GPU {i}: {name} ({total_memory_gb:.1f} GB)")

                # Success - at least one GPU detected
                self.gpu_detection_status = "ok"

            except Exception as e:
                self.gpu_detection_status = "failed"
                self.gpu_detection_error = f"{type(e).__name__}: {e}"
                logger.warning(f"⚠ GPU initialization failed: {type(e).__name__}: {e}")
                logger.warning("   → Check NVIDIA drivers: nvidia-smi")
                logger.warning("   → Check pynvml: python -c 'import pynvml; pynvml.nvmlInit()'")
        else:
            self.gpu_detection_status = "unavailable"
            self.gpu_detection_error = "pynvml not installed"
            logger.warning("⚠ pynvml not installed - GPU detection disabled")

        # Register all models and initialize state snapshots
        for model_name, model_config in self.config.get_all_models().items():
            self.models[model_name] = LoadedModel(
                name=model_name,
                size_gb=model_config.size_gb,
                state=ModelState.IDLE,
                gpu_id=-1,
                port=model_config.port,
            )
            self.state_info[model_name] = ModelStateSnapshot(
                model_name=model_name,
                state=ModelState.IDLE,
                phase=LoadingPhase.IDLE,
            )

        # Start GPU monitoring task for continuous power/temp updates
        if self.gpus:
            asyncio.create_task(self._monitor_gpu_stats())

    async def cleanup(self) -> None:
        """Terminate all running llama-server sessions."""
        logger.info("⏳ Cleaning up sessions...")

        # Stop all active sessions
        tasks = []
        for model_name, session in list(self.sessions.items()):
            logger.info(f"   Stopping session: {model_name}")
            tasks.append(session.stop())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"✓ Stopped {len(tasks)} session(s)")
        else:
            logger.info("   No active sessions to stop")

        # Clear sessions dict
        self.sessions.clear()
        logger.info("✓ All sessions cleaned up")

    async def load_model_to_gpu(self, model_name: str, gpu_id: Union[int, List[int]] = 0) -> None:
        """
        Load a model to GPU and spawn llama-server session.

        Args:
            model_name: Name of model to load
            gpu_id: GPU ID to load to (default: 0)

        Raises:
            ValueError: If model not found or can't be loaded
        """
        logger.info(f"🔄 load_model_to_gpu called: model={model_name}, gpu_id={gpu_id}")
        
        if model_name not in self.config.get_all_models():
            raise ValueError(f"Model not found: {model_name}")

        # Reload model config from disk to ensure latest values
        # (in case config was edited since app startup)
        config_file = self.config.project_root / "config" / "models" / f"{model_name}.json"
        if config_file.exists():
            logger.debug(f"   Reloading config from disk: {config_file}")
            with open(config_file) as f:
                config_data = json.load(f)
            logger.debug(f"      Raw JSON from disk: {config_data}")

            # Extract port from launch_args (stored separately in JSON)
            launch_args = config_data.get("launch_args", {})
            if "--port" in launch_args:
                config_data["port"] = launch_args["--port"]

            model_config = ModelConfig(**config_data)

            # Preserve size_gb from existing config (not persisted in JSON)
            existing_config = self.config.models.get(model_name)
            if existing_config and existing_config.size_gb:
                model_config.size_gb = existing_config.size_gb

            # Restore is_configured (not persisted in JSON, computed from port + launch_args)
            model_config.is_configured = (
                model_config.launch_args is not None and model_config.port is not None
            )
            # Update the in-memory cache
            self.config.models[model_name] = model_config
        else:
            model_config = self.config.get_model_config(model_name)
            
        logger.debug(f"   Model config loaded: {model_config.name} ({model_config.size_gb} GB)")
        logger.debug(f"   Model config launch_args: {model_config.launch_args}")

        # Check if already loaded
        if model_name in self.sessions:
            raise ValueError(f"Model {model_name} already loaded")

        # Validate GPU ID(s)
        gpu_ids_list = gpu_id if isinstance(gpu_id, list) else [gpu_id]
        invalid = [g for g in gpu_ids_list if g not in self.gpus]
        if invalid and self.gpus:
            raise ValueError(f"GPU(s) {invalid} not found. Available: {list(self.gpus.keys())}")
        logger.debug(f"   GPU(s) {gpu_ids_list} validated")

        # Check that port is configured
        if model_config.port is None:
            raise ValueError(
                f"Model {model_name} is not configured (port is required). Cannot load."
            )

        # Check port availability
        if not is_port_available(model_config.port):
            raise ValueError(
                f"Port {model_config.port} is already in use. Cannot load {model_name}."
            )

        logger.debug(f"   Port {model_config.port} is available")

        # Check GPU has enough VRAM (sum across all selected GPUs for multi-GPU)
        gpu_ids_list = gpu_id if isinstance(gpu_id, list) else [gpu_id]
        available_gb = sum(
            (self.gpus[g].memory - self.gpus[g].allocated)
            for g in gpu_ids_list if g in self.gpus
        )
        size_needed = model_config.total_vram or model_config.size_gb or 0
        if available_gb < size_needed and size_needed > 0:
            raise ValueError(
                f"GPUs {gpu_ids_list} have only {available_gb:.1f} GB available combined, "
                f"but {model_config.name} needs {size_needed:.1f} GB"
            )
        logger.debug(f"   GPUs {gpu_ids_list} have {available_gb:.1f} GB available combined")

        # Ensure model is registered in self.models (models added via rescan after startup won't be)
        if model_name not in self.models:
            self.models[model_name] = LoadedModel(
                name=model_name,
                size_gb=model_config.size_gb or 0.0,
                state=ModelState.IDLE,
                gpu_id=-1,
                port=model_config.port or 0,
            )
            self.state_info[model_name] = ModelStateSnapshot(
                model_name=model_name,
                state=ModelState.IDLE,
                phase=LoadingPhase.IDLE,
            )
            logger.info(f"   Auto-registered {model_name} in gpu_manager (added after startup)")

        # Update state to loading (queued)
        await self._set_model_state(model_name, ModelState.LOADING, LoadingPhase.QUEUED)
        logger.info(f"⏳ Model state set to LOADING (queued)")

        try:
            # Determine which llama-server binary to use
            effective_binary = self.config.app_config.llama_server_binary
            if model_config.llama_path and model_config.llama_path != "default":
                custom_path = model_config.llama_path
                if not Path(custom_path).is_file():
                    raise ValueError(f"Custom llama-server binary not found: {custom_path}")
                # Import here to avoid circular imports
                from main import ensure_schema_for_binary
                version_str = await ensure_schema_for_binary(custom_path, self.config, update_global=False)
                if version_str is None:
                    raise ValueError(f"Custom llama-server binary failed version check: {custom_path}")
                effective_binary = custom_path
                logger.info(f"   Using custom binary for {model_name}: {custom_path} (version {version_str})")

            # Create session
            log_path = self.config.get_log_path(model_name)
            log_path.parent.mkdir(exist_ok=True)

            session = LlamaSession(
                model_config=model_config,
                llama_server_binary=effective_binary,
                log_file=log_path,
                gpu_id=gpu_id,
            )

            # Store session BEFORE start() so cancel_load can find it
            self.sessions[model_name] = session

            # Start session - spawning phase
            await self._set_model_state(model_name, ModelState.LOADING, LoadingPhase.SPAWNING)
            pid = await session.start()

            # Health check phase
            await self._set_model_state(model_name, ModelState.LOADING, LoadingPhase.HEALTH_CHECKING)
            self.models[model_name].pid = pid
            self.models[model_name].gpu_id = gpu_id

            # Update GPU allocation (if GPU detection available)
            gpu_ids_list = gpu_id if isinstance(gpu_id, list) else [gpu_id]
            size_tracked = model_config.total_vram or model_config.size_gb or 0
            per_gpu = size_tracked / len(gpu_ids_list) if gpu_ids_list else size_tracked
            if self.gpus:
                for gid in gpu_ids_list:
                    if gid in self.gpus:
                        self.gpus[gid].allocated += per_gpu
                        loaded_model_dict = self.models[model_name].to_dict()
                        loaded_model_dict['port'] = model_config.port
                        self.gpus[gid].loaded_models.append(loaded_model_dict)
                self.models[model_name].vram_allocated = size_tracked

            # Mark as running
            await self._set_model_state(model_name, ModelState.RUNNING, LoadingPhase.RUNNING)
            logger.info(f"✓ {model_name} loaded on GPU {gpu_id}, PID {pid}")

            # Broadcast state change for immediate UI update
            asyncio.create_task(event_bus.broadcast({
                "type": "model_state_change",
                "model_name": model_name,
                "state": ModelState.RUNNING.value,
                "html_id": model_config.html_id,
            }))
            asyncio.create_task(event_bus.broadcast({"type": "gpu_update"}))

        except asyncio.CancelledError:
            logger.info(f"✓ {model_name} load cancelled (background task exiting)")
            # Don't set FAILED — cancel_load already set IDLE
            if model_name in self.sessions:
                del self.sessions[model_name]
            # Don't re-raise — FastAPI background tasks expect clean exit
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
            await self._set_model_state(model_name, ModelState.FAILED, LoadingPhase.ERROR, str(e))
            # Clean up session if it was partially created (so retries will work)
            if model_name in self.sessions:
                del self.sessions[model_name]
            # Broadcast failure event for immediate UI update
            model_config = self.config.get_model_config(model_name)
            asyncio.create_task(event_bus.broadcast({
                "type": "model_state_change",
                "model_name": model_name,
                "state": ModelState.FAILED.value,
                "html_id": model_config.html_id,
            }))
            # Don't re-raise — FastAPI background tasks should exit cleanly

    async def cancel_load(self, model_name: str) -> None:
        """
        Cancel a model that is currently loading and kill any spawned process.

        Args:
            model_name: Name of model to cancel

        Raises:
            ValueError: If model is not in loading state
        """
        current_state = self.get_model_state(model_name)
        if current_state != ModelState.LOADING:
            raise ValueError(f"Cannot cancel: model is currently {current_state.value}")

        try:
            # Kill the process if a session was created during loading
            session = self.sessions.pop(model_name, None)
            if session:
                session.cancelled = True
                await session.stop()

            # Reset GPU allocation if it was partially updated
            model = self.models.get(model_name)
            if model and model.gpu_id >= 0 and model.gpu_id in self.gpus:
                gpu = self.gpus[model.gpu_id]
                gpu.allocated = max(0.0, gpu.allocated - model.vram_allocated)
                gpu.loaded_models = [
                    m for m in gpu.loaded_models if m["name"] != model_name
                ]
                model.gpu_id = -1
                model.pid = None

            # Reset to idle
            await self._set_model_state(model_name, ModelState.IDLE, LoadingPhase.IDLE)
            logger.info(f"✓ {model_name} load cancelled")

            # Broadcast cancellation for immediate UI update
            model_config = self.config.get_model_config(model_name)
            asyncio.create_task(event_bus.broadcast({
                "type": "model_state_change",
                "model_name": model_name,
                "state": ModelState.IDLE.value,
                "html_id": model_config.html_id,
            }))

        except Exception as e:
            logger.error(f"✗ Error cancelling {model_name}: {e}")
            await self._set_model_state(model_name, ModelState.FAILED, LoadingPhase.ERROR, str(e))
            raise

    async def unload_model(self, model_name: str) -> None:
        """
        Unload model from GPU and terminate session.

        Args:
            model_name: Name of model to unload

        Raises:
            ValueError: If model not loaded
        """
        if model_name not in self.sessions:
            raise ValueError(f"Model {model_name} is not loaded")

        try:
            # Mark as unloading
            await self._set_model_state(model_name, ModelState.LOADING, LoadingPhase.UNLOADING)

            session = self.sessions[model_name]
            await session.stop()

            # Update GPU allocation
            model = self.models[model_name]
            if model.gpu_id >= 0 and model.gpu_id in self.gpus:
                gpu = self.gpus[model.gpu_id]
                gpu.allocated = max(0.0, gpu.allocated - model.vram_allocated)
                gpu.loaded_models = [
                    m for m in gpu.loaded_models if m["name"] != model_name
                ]

            # Clean up
            del self.sessions[model_name]
            model.gpu_id = -1
            model.pid = None

            # Mark as idle
            await self._set_model_state(model_name, ModelState.IDLE, LoadingPhase.IDLE)
            logger.info(f"✓ {model_name} unloaded")

            # Broadcast state change for immediate UI update
            model_config = self.config.get_model_config(model_name)
            asyncio.create_task(event_bus.broadcast({
                "type": "model_state_change",
                "model_name": model_name,
                "state": ModelState.IDLE.value,
                "html_id": model_config.html_id,
            }))
            asyncio.create_task(event_bus.broadcast({"type": "gpu_update"}))

        except Exception as e:
            logger.error(f"✗ Error unloading {model_name}: {e}")
            await self._set_model_state(model_name, ModelState.FAILED, LoadingPhase.ERROR, str(e))
            raise

    def get_gpu_status(self) -> Dict:
        """Get current GPU status for rendering."""
        if not self.gpus:
            return {}

        result = {}
        for gpu_id, gpu in self.gpus.items():
            info = gpu.to_dict()
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info["allocated"] = mem_info.used / (1024**3)  # Convert bytes to GB
                    info["memory"] = mem_info.total / (1024**3)     # Convert bytes to GB
                    power = pynvml.nvmlDeviceGetPowerUsage(handle)
                    info["power_draw"] = power / 1000.0  # Convert mW to W
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    info["temperature"] = temp
                except Exception:
                    pass
            result[gpu_id] = info
        return result

    def get_pynvml_status(self) -> dict:
        """Get GPU detection status."""
        return {
            "status": self.gpu_detection_status,
            "error": self.gpu_detection_error,
        }

    async def _monitor_gpu_stats(self) -> None:
        """Continuously monitor and broadcast GPU power/temperature updates."""
        if not PYNVML_AVAILABLE:
            return

        while True:
            try:
                # Update GPU power and temperature every 1 second
                for gpu_id, gpu in self.gpus.items():
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        power = pynvml.nvmlDeviceGetPowerUsage(handle)
                        gpu.power_draw = power / 1000.0  # Convert mW to W
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu.temperature = temp
                    except Exception as e:
                        logger.debug(f"   [GPU Monitor] Error reading GPU {gpu_id}: {e}")
                        continue

                # Broadcast update to all connected WebSocket clients
                await event_bus.broadcast({"type": "gpu_update"})

                # Wait before next update
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"✗ GPU monitoring error: {e}")
                await asyncio.sleep(1.0)

    def sync_models_from_config(self) -> None:
        """Register models from config that aren't in self.models (added after startup)."""
        for model_name, model_config in self.config.get_all_models().items():
            if model_name not in self.models:
                self.models[model_name] = LoadedModel(
                    name=model_name,
                    size_gb=model_config.size_gb or 0.0,
                    state=ModelState.IDLE,
                    gpu_id=-1,
                    port=model_config.port or 0,
                )
                self.state_info[model_name] = ModelStateSnapshot(
                    model_name=model_name,
                    state=ModelState.IDLE,
                    phase=LoadingPhase.IDLE,
                )
                logger.info(f"   Registered new model: {model_name}")

    def get_model_state(self, model_name: str) -> ModelState:
        """Get state of a model."""
        if model_name in self.models:
            return self.models[model_name].state
        return ModelState.IDLE

    async def _set_model_state(
        self,
        model_name: str,
        state: ModelState,
        phase: LoadingPhase = None,
        error_msg: str = None,
    ) -> None:
        """Update model state (thread-safe with lock)."""
        async with self.state_lock:
            if phase is None:
                phase_map = {
                    ModelState.IDLE: LoadingPhase.IDLE,
                    ModelState.LOADING: LoadingPhase.QUEUED,
                    ModelState.RUNNING: LoadingPhase.RUNNING,
                    ModelState.FAILED: LoadingPhase.ERROR,
                }
                phase = phase_map.get(state, LoadingPhase.IDLE)

            if model_name in self.models:
                self.models[model_name].state = state

            pid = self.models[model_name].pid if model_name in self.models else None
            gpu_id = self.models[model_name].gpu_id if model_name in self.models else -1

            self.state_info[model_name] = ModelStateSnapshot(
                model_name=model_name,
                state=state,
                phase=phase,
                error_msg=error_msg,
                pid=pid,
                gpu_id=gpu_id,
            )

            logger.info(f"📊 State: {model_name} → {state.value} ({phase.value})")

            # Get html_id from config for selector matching
            html_id = None
            if model_name in self.config.models:
                html_id = self.config.models[model_name].html_id

            asyncio.create_task(event_bus.broadcast({
                "type": "model_state_change",
                "model_name": model_name,
                "html_id": html_id,
                "state": state.value,
                "phase": phase.value,
                "error_msg": error_msg,
            }))

    def get_model_state_snapshot(self, model_name: str) -> Optional[ModelStateSnapshot]:
        """Get current state snapshot for a model."""
        if model_name not in self.state_info:
            # Auto-initialize if model exists
            if model_name in self.models:
                self.state_info[model_name] = ModelStateSnapshot(
                    model_name=model_name,
                    state=self.models[model_name].state,
                    phase=LoadingPhase.IDLE,
                    pid=self.models[model_name].pid,
                    gpu_id=self.models[model_name].gpu_id,
                )
            else:
                return None
        return self.state_info[model_name]
