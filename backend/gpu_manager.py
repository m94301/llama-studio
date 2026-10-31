"""GPU state management and session coordination."""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import time
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from config_manager import ConfigManager, ModelConfig
from llama_session import (
    ENGINE_PROC_RE, LlamaSession, launch_fingerprint, port_owned_by_group, proc_start_time,
    process_group_members,
)
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
    gpu_ids: List[int] = field(default_factory=list)  # empty when not loaded; multi-entry for split
    timestamp: float = field(default_factory=time.time)

    @property
    def gpu_id(self) -> int:
        """Backward-compat scalar accessor: first GPU or -1 if none."""
        return self.gpu_ids[0] if self.gpu_ids else -1


@dataclass
class LoadedModel:
    """Represents a loaded model. May span multiple GPUs when tensor-split is configured."""

    name: str
    size_gb: float
    state: ModelState
    gpu_ids: List[int]  # empty when not loaded; multi-entry for tensor-split
    port: int
    pid: Optional[int] = None
    # Per-GPU VRAM allocations (parallel to gpu_ids). Each entry is the GB added to
    # the corresponding GPU's allocated counter at load time. Captured at load so
    # unload/cancel releases the same amounts.
    vram_per_gpu: List[float] = field(default_factory=list)

    @property
    def gpu_id(self) -> int:
        """Backward-compat scalar accessor: first GPU or -1 if none."""
        return self.gpu_ids[0] if self.gpu_ids else -1

    @property
    def vram_allocated(self) -> float:
        """Backward-compat: total VRAM across all GPUs."""
        return sum(self.vram_per_gpu)

    def to_dict(self):
        d = asdict(self)
        # Surface scalar for legacy consumers
        d["gpu_id"] = self.gpu_id
        d["vram_allocated"] = self.vram_allocated
        return d


@dataclass
class GpuInfo:
    """GPU information and state."""

    gpu_id: int
    name: str
    memory: float  # Total VRAM in GB
    allocated: float  # App-tracked allocation in GB (maintained by load/unload)
    loaded_models: list  # List of LoadedModel dicts
    power_draw: float = None  # Current power draw in watts
    temperature: float = None  # Current temperature in °C
    # Real VRAM used, as reported by NVML. Refreshed every ~1s by the monitor
    # loop so get_gpu_status() can serve it without a per-request NVML round-trip.
    # This is what the UI shows and the picker checks against — distinct from the
    # app-tracked `allocated` field used by load_model's own admission guard.
    mem_used_gb: float = 0.0

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
        # Process groups that survived their session's stop(), kept so they
        # stay in the on-disk registry and get reaped at the next start.
        self._orphans: Dict[str, dict] = {}
        # Per-model notes for the row (e.g. "kept running across a restart").
        # Registry entries the startup triage judged safe to re-attach to.
        self._adoptable: Dict[str, dict] = {}

    # ---- session registry: crash-proof cleanup across app restarts ----
    #
    # Every live session is recorded in sessions/active.json the moment it has
    # a PID. On startup, before anything else, each entry from the previous run
    # is triaged (see _triage_registry): with "keep servers alive" on, a
    # process that is provably the one we launched is re-attached; everything
    # else of ours is reaped — so no engine process ever outlives the app
    # unnoticed — and anything that is no longer ours is left alone.

    def _registry_path(self) -> Path:
        return self.config.project_root / "sessions" / "active.json"

    def _registry_sync(self) -> None:
        """Rewrite the registry from live sessions plus any known orphans."""
        try:
            data = {
                name: s.registry_record()
                for name, s in self.sessions.items()
                if s.pid
            }
            # Groups that outlived their session (survived unload). They are no
            # longer tracked as sessions but must stay reapable; the next start
            # skips any whose group is already empty.
            for name, info in self._orphans.items():
                data.setdefault(f"{name}:orphan", info)
            path = self._registry_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1))
        except Exception as e:
            logger.warning(f"⚠ Could not write session registry: {e}")

    def _reap_group(self, name: str, pgid: int) -> bool:
        """SIGTERM, wait up to 10s, SIGKILL a process group. True once it's empty."""
        logger.warning(f"♻ Reaping stale session from previous run: {name} (pgid {pgid})")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.time() + 10
        while time.time() < deadline and process_group_members(pgid):
            time.sleep(0.5)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.2)
        if process_group_members(pgid):
            logger.error(
                f"✗ Could not fully reap pgid {pgid} for {name} — it may still hold VRAM. "
                f"Keeping it in the registry to retry next start."
            )
            return False
        logger.info(f"✓ Reaped {name}")
        return True

    def _triage_registry(self, keep_alive: bool) -> None:
        """Decide, per entry left by the previous run, whether to adopt, reap, or leave it.

        - Adopt (keep_alive on): the group's leader has the same PID *and* kernel
          start time we recorded, so it is provably the process we launched,
          and its model still has a script. Queued in self._adoptable; the
          actual re-attach happens in initialize() once models are registered.
        - Reap: ours, but not adoptable (keep_alive off, script gone, a
          survivor of a failed unload, or the leader died leaving workers).
        - Leave alone: the PID now belongs to a process with a different start
          time, i.e. it was recycled for something that isn't ours, possibly a
          server the user started by hand.

        Entries from before start times were recorded are reaped as before.
        Blocking: call via asyncio.to_thread.
        """
        self._adoptable = {}
        path = self._registry_path()
        if not path.exists() or os.name == "nt":
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}

        keep: Dict[str, dict] = {}
        models = self.config.get_all_models()
        for name, info in data.items():
            pgid = info.get("pid")
            if not pgid:
                continue
            members = process_group_members(pgid)
            if not members:
                continue  # already gone
            recorded = info.get("start_time")
            leader_now = proc_start_time(pgid)
            if recorded is not None and leader_now is not None and leader_now != recorded:
                logger.warning(
                    f"⚠ Stale session {name}: PID {pgid} now belongs to a different process; not touching it"
                )
                continue
            # PID-reuse guard for entries we can't identify by start time.
            if not ENGINE_PROC_RE.search(members):
                logger.warning(
                    f"⚠ Stale session {name}: pgid {pgid} now holds unrelated processes; not touching it"
                )
                continue

            is_orphan = name.endswith(":orphan")
            adoptable = (
                keep_alive and not is_orphan
                and recorded is not None and leader_now == recorded
                and name in models
            )
            if adoptable:
                logger.info(f"↻ Found {name} still running from the previous run (pgid {pgid})")
                self._adoptable[name] = info
                keep[name] = info  # stays recorded until the re-attach replaces it
                continue
            if not self._reap_group(name, pgid):
                keep[name] = info

        try:
            path.write_text(json.dumps(keep, indent=1))
        except Exception as e:
            logger.warning(f"⚠ Could not rewrite session registry: {e}")

    def _attach_allocation(self, model_name: str, model_config: ModelConfig,
                           gpu_ids: List[int], per_gpu_need: List[float]) -> None:
        """Record a model as resident on its GPUs (app-tracked allocation + panel entry)."""
        model = self.models[model_name]
        model.gpu_ids = list(gpu_ids)
        model.vram_per_gpu = list(per_gpu_need)
        if not self.gpus:
            return
        for gid, need in zip(gpu_ids, per_gpu_need):
            if gid in self.gpus:
                self.gpus[gid].allocated += need
                loaded_model_dict = model.to_dict()
                loaded_model_dict['port'] = model_config.port
                loaded_model_dict['engine'] = model_config.engine
                # Surface per-GPU share for UI clarity
                loaded_model_dict['vram_share'] = need
                self.gpus[gid].loaded_models.append(loaded_model_dict)

    def _release_allocation(self, model_name: str) -> None:
        """Undo _attach_allocation."""
        model = self.models.get(model_name)
        if not model:
            return
        for gid, need in zip(model.gpu_ids, model.vram_per_gpu):
            if gid in self.gpus:
                gpu = self.gpus[gid]
                gpu.allocated = max(0.0, gpu.allocated - need)
                gpu.loaded_models = [m for m in gpu.loaded_models if m["name"] != model_name]
        model.gpu_ids = []
        model.vram_per_gpu = []
        model.pid = None

    def _drop_session(self, model_name: str) -> None:
        """Forget a session: its GPU allocation and registry entry.

        Stopping or reaping the process is the caller's job (it differs per
        path) and must happen first, so the registry keeps a record of the
        process until it is gone.
        """
        self.sessions.pop(model_name, None)
        self._release_allocation(model_name)
        self._registry_sync()

    async def _adopt_sessions(self) -> None:
        """Re-attach every session the startup triage found adoptable."""
        for name, record in list(self._adoptable.items()):
            try:
                await self._adopt(name, record)
            except Exception as e:
                # Never leave a process of ours running untracked: reap it.
                logger.error(f"✗ Could not re-attach {name}: {e}", exc_info=True)
                if not await asyncio.to_thread(self._reap_group, name, record["pid"]):
                    self._orphans[name] = {"pid": record["pid"], "port": record.get("port")}
                self._drop_session(name)
        self._adoptable = {}
        self._registry_sync()

    async def _adopt(self, name: str, record: dict) -> None:
        model_config = self.config.get_model_config(name)
        session = LlamaSession.adopt(
            model_config=model_config,
            log_file=self.config.get_log_path(name),
            record=record,
            term_grace=30.0 if model_config.engine == "vllm" else 5.0,
        )
        self.sessions[name] = session
        self.models[name].pid = session.pid
        self._attach_allocation(name, model_config, session.gpu_ids, session.vram_per_gpu)

        recorded = record.get("script_fingerprint")
        if recorded and launch_fingerprint(model_config.script_path) != recorded:
            logger.warning(f"⚠ {name}: script edited since launch — it keeps running the old settings until reloaded")

        owned = await asyncio.to_thread(port_owned_by_group, model_config.port, session.pid)
        if owned is not False and await session.is_healthy():
            await self._set_model_state(name, ModelState.RUNNING, LoadingPhase.RUNNING)
            logger.info(f"✓ Re-attached {name} (pgid {session.pid}, port {model_config.port}): running")
            return

        # Still loading when the previous run went away (or not answering yet):
        # wait for the remainder of its health timeout, as the original load would have.
        elapsed = time.time() - (record.get("launched_at") or time.time())
        remaining = max(60, int(session.health_timeout - elapsed))
        await self._set_model_state(name, ModelState.LOADING, LoadingPhase.HEALTH_CHECKING)
        logger.info(f"↻ Re-attached {name} mid-load; waiting up to {remaining}s for it to become ready")
        asyncio.create_task(self._finish_adopted_load(name, session, remaining))

    async def _finish_adopted_load(self, name: str, session: LlamaSession, timeout: int) -> None:
        try:
            await session.wait_ready(timeout=timeout)
        except asyncio.CancelledError:
            return  # cancel_load already cleaned up
        except Exception as e:
            if self.sessions.get(name) is not session:
                return  # the crash watchdog (or a cancel) got there first
            logger.error(f"✗ Re-attached {name} never became ready: {e}")
            await session.stop()
            self._drop_session(name)
            await self._set_model_state(name, ModelState.FAILED, LoadingPhase.ERROR,
                                        f"re-attached after restart but never became ready: {e}")
            asyncio.create_task(event_bus.broadcast({"type": "gpu_update"}))
            return
        if self.sessions.get(name) is session:
            await self._set_model_state(name, ModelState.RUNNING, LoadingPhase.RUNNING)
            logger.info(f"✓ Re-attached {name} is now ready")
            asyncio.create_task(event_bus.broadcast({"type": "gpu_update"}))

    async def initialize(self) -> None:
        """Detect available GPUs on startup."""
        # Before anything else: clean up engine processes a previous run left
        # behind. Runs in a thread — it polls with blocking sleeps for up to
        # ~10s per stale group and must not stall the event loop.
        await asyncio.to_thread(
            self._triage_registry, bool(getattr(self.config.app_config, "keep_sessions_on_shutdown", True))
        )

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
                        # Seed real usage so the first panel render (before the
                        # monitor loop's first tick) isn't reported as 0 GB used.
                        mem_used_gb=mem_info.used / (1024**3),
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
                gpu_ids=[],
                port=model_config.port,
            )
            self.state_info[model_name] = ModelStateSnapshot(
                model_name=model_name,
                state=ModelState.IDLE,
                phase=LoadingPhase.IDLE,
            )

        # Re-attach servers the previous run left running (keep-alive on).
        # After model registration (needs self.models) and before the monitor
        # starts, so the watchdog sees them as sessions from its first tick.
        await self._adopt_sessions()

        # Start the monitor task: GPU power/temp/VRAM refresh + the session
        # crash watchdog. Runs even without NVML so crash detection still works.
        asyncio.create_task(self._monitor_gpu_stats())

    async def cleanup(self, keep_alive: bool = False) -> None:
        """Terminate all running sessions — or, with keep_alive, leave them for the next start.

        Engines run in their own session (start_new_session=True) and log to a
        file, so nothing ties their lifetime to this process: with keep_alive
        the registry is simply left describing them, and the next start's
        triage re-attaches whatever is provably still ours.
        """
        if keep_alive and self.sessions:
            self._registry_sync()
            logger.info(
                f"↻ Leaving {len(self.sessions)} server(s) running (Keep servers alive is on): "
                + ", ".join(self.sessions)
            )
            return
        logger.info("⏳ Cleaning up sessions...")

        names = list(self.sessions.keys())
        if not names:
            logger.info("   No active sessions to stop")
            self._registry_sync()
            return

        for name in names:
            logger.info(f"   Stopping session: {name}")
        results = await asyncio.gather(
            *(self.sessions[n].stop() for n in names), return_exceptions=True
        )

        # Drop only the sessions that are genuinely gone. Anything still alive
        # stays in self.sessions so _registry_sync keeps its record and the next
        # start reaps it, instead of losing track of a process holding VRAM.
        stopped = 0
        for name, result in zip(names, results):
            session = self.sessions.get(name)
            if session is None:
                continue
            if isinstance(result, Exception):
                logger.error(f"✗ {name} raised during stop: {result}")
            if session.group_is_alive():
                logger.error(
                    f"✗ {name} still has live processes after stop — leaving it in the "
                    f"session registry so the next start reaps it"
                )
                continue
            self.sessions.pop(name, None)
            stopped += 1

        self._registry_sync()
        logger.info(f"✓ Stopped {stopped}/{len(names)} session(s)")

    async def load_model(self, model_name: str, gpu_ids: List[int]) -> None:
        """
        Load a model and spawn llama-server, distributing VRAM allocation across gpu_ids.

        Args:
            model_name: Name of model to load
            gpu_ids: Ordered list of GPU IDs. Length must equal model_config.gpu_count.
                     For single-GPU models, pass [gpu_id]. For tensor-split, pass all
                     selected GPUs in the order they should be visible to llama-server.

        Raises:
            ValueError: If model not found, gpu_ids count mismatches gpu_count, or VRAM insufficient.

        Note: callers run this as a FastAPI BackgroundTask *after* trigger_load
        has already set the row to LOADING, and a background task's exception is
        swallowed. Any rejection must therefore leave the row in a terminal
        state itself — see the wrapper below.
        """
        try:
            await self._load_model_inner(model_name, gpu_ids)
        except Exception as e:
            # Admission failures (already loaded, GPU mismatch, port in use,
            # not enough VRAM) are raised before the inner try block. Without
            # this the row stays LOADING forever with a CANCEL button.
            logger.error(f"✗ Load rejected for {model_name}: {e}")
            await self._set_model_state(model_name, ModelState.FAILED, LoadingPhase.ERROR, str(e))

    async def _load_model_inner(self, model_name: str, gpu_ids: List[int]) -> None:
        """Validation + spawn. See load_model for the error contract."""
        logger.info(f"🔄 load_model called: model={model_name}, gpu_ids={gpu_ids}")

        if model_name not in self.config.get_all_models():
            raise ValueError(f"Model not found: {model_name}")

        model_config = self.config.get_model_config(model_name)
        logger.debug(f"   Model config loaded: {model_config.name} ({model_config.size_gb} GB), gpu_count={model_config.gpu_count}")

        # Check if already loaded
        if model_name in self.sessions:
            raise ValueError(f"Model {model_name} already loaded")

        # Validate gpu_ids count matches model's expected split count. A flexible
        # split (--split-mode layer/row, no --tensor-split) accepts any count >= 1
        # since it divides evenly across whatever GPUs are made visible.
        if model_config.is_flexible_split:
            if len(gpu_ids) < 1:
                raise ValueError(f"Model {model_name} needs at least one GPU; got none.")
        else:
            expected = model_config.gpu_count
            if len(gpu_ids) != expected:
                raise ValueError(
                    f"Model {model_name} declares {expected} GPU(s) via --tensor-split; got {len(gpu_ids)}."
                )
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(f"Duplicate GPU IDs in selection: {gpu_ids}")

        # Validate each GPU exists
        if self.gpus:
            for gid in gpu_ids:
                if gid not in self.gpus:
                    raise ValueError(f"GPU {gid} not found. Available: {list(self.gpus.keys())}")

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

        # Compute per-GPU VRAM need. vLLM claims util × capacity on every
        # visible GPU regardless of model size (doc 21 decision 3); llama.cpp
        # honors tensor-split weights / even-splits flexible models.
        if model_config.engine == "vllm":
            if self.gpus:
                per_gpu_need = [model_config.needed_on_gpu(self.gpus[gid].memory) for gid in gpu_ids]
            else:
                per_gpu_need = [model_config.size_gb or 0] * len(gpu_ids)
        else:
            per_gpu_need = model_config.per_gpu_vram(count=len(gpu_ids)) or [model_config.total_vram or model_config.size_gb or 0]
        if len(per_gpu_need) != len(gpu_ids):
            # Defensive: pad/trim to match (shouldn't normally happen since gpu_count is the source)
            if len(gpu_ids) == 1:
                per_gpu_need = [sum(per_gpu_need)]
            else:
                per_gpu_need = [(model_config.total_vram or model_config.size_gb or 0) / len(gpu_ids)] * len(gpu_ids)

        # Check each chosen GPU has enough free VRAM for its share
        if self.gpus:
            for gid, need in zip(gpu_ids, per_gpu_need):
                gpu = self.gpus[gid]
                available_gb = gpu.memory - gpu.allocated
                if available_gb < need:
                    raise ValueError(
                        f"GPU {gid} has only {available_gb:.1f} GB available, "
                        f"but {model_config.name} needs {need:.1f} GB on that GPU"
                    )
                logger.debug(f"   GPU {gid}: needs {need:.1f} GB, {available_gb:.1f} GB available")

        # Ensure model is registered in self.models (models added via rescan after startup won't be)
        if model_name not in self.models:
            self.models[model_name] = LoadedModel(
                name=model_name,
                size_gb=model_config.size_gb or 0.0,
                state=ModelState.IDLE,
                gpu_ids=[],
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
            # Determine which llama-server binary to use. vLLM scripts carry
            # their own executable path; the custom-binary version/schema check
            # is llama.cpp-only.
            effective_binary = self.config.app_config.llama_server_binary
            if model_config.engine != "vllm" and model_config.llama_path and model_config.llama_path != "default":
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
                gpu_ids=gpu_ids,
                health_timeout=model_config.effective_health_timeout(),
                term_grace=30.0 if model_config.engine == "vllm" else 5.0,
                on_spawn=self._registry_sync,
            )

            # Baseline the target GPUs so the post-load sample measures *this*
            # model rather than whatever else is resident on the card.
            baseline = {}
            if PYNVML_AVAILABLE and self.gpus:
                for gid in gpu_ids:
                    if gid in self.gpus:
                        baseline[gid] = self.gpus[gid].mem_used_gb

            session.vram_per_gpu = list(per_gpu_need)  # persisted in the registry for re-attach
            # Store session BEFORE start() so cancel_load can find it
            self.sessions[model_name] = session

            # Start session - spawning phase
            await self._set_model_state(model_name, ModelState.LOADING, LoadingPhase.SPAWNING)
            pid = await session.start()

            # Health check phase
            await self._set_model_state(model_name, ModelState.LOADING, LoadingPhase.HEALTH_CHECKING)
            self.models[model_name].pid = pid
            # Update GPU allocation across all chosen GPUs
            self._attach_allocation(model_name, model_config, gpu_ids, per_gpu_need)

            # A successful vLLM load reports its exact KV bytes/token; capture
            # it so the calculator stops relying on the config.json estimate.
            if model_config.engine == "vllm":
                try:
                    await asyncio.to_thread(self.config.calibrate_kv_bytes_per_token, model_name)
                    # Peak on the heaviest card, net of what was already there.
                    peak = None
                    if baseline:
                        await asyncio.sleep(2)  # let allocation settle
                        deltas = []
                        for gid, before in baseline.items():
                            if gid in self.gpus:
                                deltas.append(max(0.0, self.gpus[gid].mem_used_gb - before))
                        peak = round(max(deltas), 2) if deltas else None
                    await asyncio.to_thread(
                        self.config.calibrate_load_profile, model_name, peak
                    )
                except Exception as e:
                    logger.debug(f"   Load calibration skipped for {model_name}: {e}")

            # Mark as running
            await self._set_model_state(model_name, ModelState.RUNNING, LoadingPhase.RUNNING)
            logger.info(f"✓ {model_name} loaded on GPU(s) {gpu_ids}, PID {pid}")
            asyncio.create_task(event_bus.broadcast({"type": "gpu_update"}))

        except asyncio.CancelledError:
            logger.info(f"✓ {model_name} load cancelled (background task exiting)")
            # Don't set FAILED — cancel_load already set IDLE
            self._drop_session(model_name)
            # Don't re-raise — FastAPI background tasks expect clean exit
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
            # session.start() stops its own process on failure; forget it so a
            # retry works, then report.
            self._drop_session(model_name)
            await self._set_model_state(model_name, ModelState.FAILED, LoadingPhase.ERROR, str(e))
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
            # Kill the process if a session was created during loading. Popped
            # first so the load task, on waking, sees it is no longer current.
            session = self.sessions.pop(model_name, None)
            if session:
                session.cancelled = True
                await session.stop()
            self._drop_session(model_name)

            await self._set_model_state(model_name, ModelState.IDLE, LoadingPhase.IDLE)
            logger.info(f"✓ {model_name} load cancelled")

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
            if session.group_is_alive():
                # stop() TERMed and KILLed the group; if anything survived it is
                # unkillable from here (uninterruptible sleep, or not ours).
                # Say so loudly rather than reporting a clean unload.
                logger.error(
                    f"✗ {model_name}: processes survived unload (pgid {session.pid}) — "
                    f"VRAM may not be released; recorded for reaping at next start"
                )
                self._orphans[model_name] = {
                    "pid": session.pid,
                    "port": session.model_config.port,
                }

            self._drop_session(model_name)
            await self._set_model_state(model_name, ModelState.IDLE, LoadingPhase.IDLE)
            logger.info(f"✓ {model_name} unloaded")
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
            # Serve NVML-derived metrics from the cache the monitor loop keeps
            # fresh (~1s), rather than issuing a fresh round-trip per request.
            # "allocated" here means real VRAM used (what the UI/picker want),
            # which is why we override the app-tracked field from to_dict().
            info["allocated"] = gpu.mem_used_gb
            result[gpu_id] = info
        return result

    def get_pynvml_status(self) -> dict:
        """Get GPU detection status."""
        return {
            "status": self.gpu_detection_status,
            "error": self.gpu_detection_error,
        }

    async def _monitor_gpu_stats(self) -> None:
        """Monitor loop (~1s): GPU power/temp/VRAM refresh + session crash watchdog."""
        while True:
            try:
                if PYNVML_AVAILABLE and self.gpus:
                    # Update GPU power and temperature every 1 second
                    for gpu_id, gpu in self.gpus.items():
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                            power = pynvml.nvmlDeviceGetPowerUsage(handle)
                            gpu.power_draw = power / 1000.0  # Convert mW to W
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            gpu.temperature = temp
                            # Refresh real VRAM usage so get_gpu_status() can serve it
                            # from cache (the panel and GPU picker read this).
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu.mem_used_gb = mem_info.used / (1024**3)
                        except Exception as e:
                            logger.debug(f"   [GPU Monitor] Error reading GPU {gpu_id}: {e}")
                            continue

                    # Broadcast update to all connected WebSocket clients
                    await event_bus.broadcast({"type": "gpu_update"})

                # Crash watchdog: a RUNNING session whose process has exited
                # (OOM, crash, external kill) gets its allocations released and
                # its row flipped to FAILED instead of showing RUNNING forever.
                # LOADING is included deliberately: a session whose process dies
                # while the row is mid-load (or after a rejected second load
                # request left the row stuck) would otherwise never be cleaned
                # up, leaving a stale registry entry and a frozen row.
                for name, session in list(self.sessions.items()):
                    if self.get_model_state(name) not in (ModelState.RUNNING, ModelState.LOADING):
                        continue
                    if session.process is not None and session.process.poll() is not None:
                        await self._handle_crashed_session(name, session.process.returncode)
                    elif session.adopted and session.process is None and session.pid and not session.leader_alive():
                        # Re-attached sessions have no exit code to read.
                        await self._handle_crashed_session(name, None)

                # Wait before next update
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"✗ GPU monitoring error: {e}")
                await asyncio.sleep(1.0)

    async def _handle_crashed_session(self, model_name: str, returncode) -> None:
        """Clean up after a session whose process died out from under us."""
        logger.error(f"✗ {model_name} process exited unexpectedly (code {returncode})")
        crashed = self.sessions.get(model_name)
        if crashed is not None:
            # A dead leader can leave engine workers holding VRAM — sweep the group.
            crashed.kill_process_group(signal.SIGKILL)
        self._drop_session(model_name)
        await self._set_model_state(
            model_name, ModelState.FAILED, LoadingPhase.ERROR,
            f"process exited unexpectedly (code {returncode}) — see log",
        )
        asyncio.create_task(event_bus.broadcast({"type": "gpu_update"}))

    def sync_models_from_config(self) -> None:
        """Register models from config that aren't in self.models (added after startup)."""
        for model_name, model_config in self.config.get_all_models().items():
            if model_name not in self.models:
                self.models[model_name] = LoadedModel(
                    name=model_name,
                    size_gb=model_config.size_gb or 0.0,
                    state=ModelState.IDLE,
                    gpu_ids=[],
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
            gpu_ids = list(self.models[model_name].gpu_ids) if model_name in self.models else []

            self.state_info[model_name] = ModelStateSnapshot(
                model_name=model_name,
                state=state,
                phase=phase,
                error_msg=error_msg,
                pid=pid,
                gpu_ids=gpu_ids,
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
                    gpu_ids=list(self.models[model_name].gpu_ids),
                )
            else:
                return None
        return self.state_info[model_name]
