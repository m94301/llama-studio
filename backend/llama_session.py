"""Wrapper for llama-server process management."""

import asyncio
import subprocess
import logging
import os
import re
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

import hashlib

from config_manager import ModelConfig
from launch_script import META_BEGIN, META_END

logger = logging.getLogger(__name__)

# Cmdline signature of a process we're entitled to kill. Used before signalling
# a process group whose leader we may already have reaped — a PID freed by
# wait()/poll() can in principle be recycled, and we must never signal a group
# that no longer belongs to us.
ENGINE_PROC_RE = re.compile(r"llama-server|vllm|EngineCore|\.sh\b")

# Llama Studio's own control variables. These must never reach an engine
# process: we hand the child a copy of our environment, and a name collision
# is silently fatal. VERBOSE is the live example — start.sh sets it to "" (or
# "--verbose"), while vLLM reads its own VERBOSE as bool(int(...)), so an
# inherited value crashes engine-core init ~100s into startup.
STUDIO_ONLY_ENV_VARS = ("VERBOSE",)


def process_group_members(pgid: Optional[int]) -> str:
    """Cmdlines of every *live* process in `pgid`, or '' if none remain.

    Zombies are excluded: a killed child stays in the process table until its
    parent reaps it, and counting those as alive would make a successful kill
    look like a failure.
    """
    if not pgid or os.name == "nt":
        return ""
    try:
        # -ww: never truncate the args column. Without it ps clips at the
        # terminal width, which can cut the engine name out of a long model
        # path and make a group look like it isn't ours.
        out = subprocess.run(
            ["ps", "-ww", "-eo", "pgid=,pid=,stat=,args="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return ""
    members = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        row_pgid, pid, stat = parts[0], parts[1], parts[2]
        if row_pgid != str(pgid) or stat.startswith("Z"):
            continue
        members.append(f"{pid} {parts[3] if len(parts) > 3 else ''}".strip())
    return "\n".join(members)


def proc_start_time(pid: Optional[int]) -> Optional[int]:
    """Kernel start time of `pid` (clock ticks since boot), or None if it's gone.

    Together with the PID this identifies a process uniquely: a recycled PID
    has a different start time. That is what lets a restarted Llama Studio tell
    "the server I launched" from "some other process that got the same PID".
    Zombies count as gone.
    """
    if not pid or os.name == "nt":
        return None
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    # comm (field 2) is parenthesised and may contain spaces; split after it.
    fields = stat[stat.rfind(")") + 2:].split()
    if not fields or fields[0] == "Z":
        return None
    try:
        return int(fields[19])  # field 22 overall: starttime
    except (IndexError, ValueError):
        return None


def _listen_inodes(port: int) -> set:
    """Socket inodes LISTENing on `port` (IPv4 and IPv6)."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as f:
                next(f, None)
                for line in f:
                    cols = line.split()
                    if len(cols) < 10 or cols[3] != "0A":  # 0A = LISTEN
                        continue
                    if int(cols[1].rsplit(":", 1)[1], 16) == port:
                        inodes.add(cols[9])
        except OSError:
            continue
    return inodes


def port_owned_by_group(port: Optional[int], pgid: Optional[int]) -> Optional[bool]:
    """Is the process LISTENing on `port` a member of process group `pgid`?

    True / False when it can be determined, None when it can't (no /proc).
    False also covers "nobody is listening yet", the normal state while an
    engine is still loading. This is the identity check for adopted sessions:
    /health alone can't tell our server from a different one on the same port.
    """
    if not port or not pgid or not os.path.isdir("/proc/net"):
        return None
    inodes = _listen_inodes(int(port))
    if not inodes:
        return False
    wanted = {f"socket:[{i}]" for i in inodes}
    for line in process_group_members(pgid).splitlines():
        pid = line.split(None, 1)[0]
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}") in wanted:
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def launch_fingerprint(script_path: Path) -> Optional[str]:
    """Hash of everything in a script that affects how the server launches.

    The meta fence is excluded: Llama Studio itself rewrites it after a
    successful load (KV/weights calibration) and on display-name edits, so a
    raw mtime or full-text comparison would report every calibrated model as
    "edited since launch".
    """
    try:
        text = script_path.read_text(errors="replace")
    except OSError:
        return None
    kept, inside = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == META_BEGIN:
            inside = True
        elif stripped == META_END:
            inside = False
        elif not inside:
            kept.append(line)
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()[:16]


class LlamaSession:
    """Manages a single llama-server process."""

    def __init__(
        self,
        model_config: ModelConfig,
        llama_server_binary: str,
        log_file: Path,
        gpu_ids: Optional[list] = None,
        health_timeout: int = 120,
        term_grace: float = 5.0,
        on_spawn=None,
    ):
        self.model_config = model_config
        self.llama_server_binary = llama_server_binary
        self.log_file = log_file
        self.gpu_ids = list(gpu_ids) if gpu_ids else [0]
        self.health_timeout = int(health_timeout)
        # SIGTERM → SIGKILL window. vLLM multi-GPU teardown (NCCL, spawned
        # workers) needs far longer than llama-server's near-instant exit.
        self.term_grace = float(term_grace)
        # Called as soon as the process has a PID — the session registry uses
        # this so a crash mid-health-check still leaves a reapable record.
        self.on_spawn = on_spawn
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.cancelled: bool = False
        # Identity + bookkeeping persisted in the session registry so a
        # restarted Llama Studio can re-attach to this process (see adopt()).
        self.start_time: Optional[int] = None   # proc_start_time(pid) at spawn
        self.launched_at: Optional[float] = None  # wall clock, for the remaining health timeout
        self.script_fingerprint: Optional[str] = None  # launch_fingerprint() at spawn
        self.vram_per_gpu: list = []
        # True when this session was re-attached after a restart: there is no
        # Popen handle, so liveness and termination go through the PID/pgid.
        self.adopted: bool = False

    @classmethod
    def adopt(cls, *, model_config: ModelConfig, log_file: Path, record: Dict[str, Any],
              term_grace: float = 5.0) -> "LlamaSession":
        """Re-attach to a process a previous Llama Studio run launched."""
        session = cls(
            model_config=model_config,
            llama_server_binary="",
            log_file=log_file,
            gpu_ids=record.get("gpu_ids") or [],
            health_timeout=int(record.get("health_timeout") or 120),
            term_grace=term_grace,
        )
        session.pid = record["pid"]
        session.start_time = record.get("start_time")
        session.launched_at = record.get("launched_at")
        session.script_fingerprint = record.get("script_fingerprint")
        session.vram_per_gpu = list(record.get("vram_per_gpu") or [])
        session.adopted = True
        return session

    def registry_record(self) -> Dict[str, Any]:
        """What the session registry stores to recognise and re-attach this process."""
        return {
            "pid": self.pid,
            "start_time": self.start_time,
            "port": self.model_config.port,
            "engine": self.model_config.engine,
            "gpu_ids": list(self.gpu_ids),
            "vram_per_gpu": list(self.vram_per_gpu),
            "launched_at": self.launched_at,
            "script_fingerprint": self.script_fingerprint,
            "health_timeout": self.health_timeout,
        }

    def leader_alive(self) -> bool:
        """The process we launched (or adopted) is still that same process."""
        if self.process is not None:
            return self.process.poll() is None
        if not self.pid:
            return False
        now = proc_start_time(self.pid)
        return now is not None and (self.start_time is None or now == self.start_time)

    def _build_env(self) -> Dict[str, str]:
        """Environment for the engine process.

        Sets CUDA_VISIBLE_DEVICES for GPU assignment — for tensor-split models
        gpu_ids is ordered, since --main-gpu / --tensor-split index into this
        visible-devices list — and strips Llama Studio's own control variables
        so they can't collide with the engine's.
        """
        env = os.environ.copy()
        for var in STUDIO_ONLY_ENV_VARS:
            env.pop(var, None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
        return env

    async def start(self) -> int:
        """
        Start llama-server process by executing the model's shell script.

        Returns:
            Process ID

        Raises:
            RuntimeError: If process fails to start
        """
        script_path = self.model_config.script_path
        cmd = [str(script_path)]

        logger.info(f"⏳ Starting {self.model_config.name} on port {self.model_config.port}")
        logger.debug(f"   Script: {script_path}")

        try:
            self.cancelled = False
            # Open log file for writing
            log_file_handle = open(self.log_file, "w")

            # Copy the script contents into the log for debugging visibility
            try:
                script_text = script_path.read_text()
            except Exception as e:
                script_text = f"<could not read script: {e}>"
            log_file_handle.write(f"{'='*80}\n")
            log_file_handle.write(f"Launch script: {script_path}\n")
            log_file_handle.write(f"{'='*80}\n")
            log_file_handle.write(script_text)
            log_file_handle.write(f"\n{'='*80}\n")
            log_file_handle.flush()

            env = self._build_env()
            logger.debug(f"   CUDA_VISIBLE_DEVICES set to: {env['CUDA_VISIBLE_DEVICES']}")

            # Spawn process
            self.process = subprocess.Popen(
                cmd,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Create new process group
                env=env,  # Pass modified environment with GPU selection
            )

            self.pid = self.process.pid
            self.start_time = proc_start_time(self.pid)
            self.launched_at = time.time()
            self.script_fingerprint = launch_fingerprint(script_path)
            logger.info(f"✓ Process spawned with PID {self.pid}")
            if self.on_spawn:
                try:
                    self.on_spawn()
                except Exception:
                    pass
            logger.info(f"   Polling health check at 5s intervals using {self.health_timeout}s timeout")

            # Wait for llama-server to be ready
            await self.wait_ready(timeout=self.health_timeout)

            logger.info(f"✓ {self.model_config.name} ready on port {self.model_config.port}")
            return self.pid

        except asyncio.TimeoutError:
            logger.error(f"✗ {self.model_config.name} failed to start (health check timeout)")
            await self.stop()
            raise RuntimeError(f"llama-server startup timeout for {self.model_config.name}")
        except Exception as e:
            logger.error(f"✗ Failed to start {self.model_config.name}: {e}")
            await self.stop()
            raise

    def kill_process_group(self, sig: int = signal.SIGKILL, *, verify: bool = False) -> None:
        """Signal the session's entire process group (leader may already be dead).

        start_new_session=True makes the spawned script its own group leader,
        so self.pid doubles as the pgid even after the leader exits — this is
        how orphaned engine workers (vLLM spawn-mode) get swept.

        `verify=True` checks the group still contains engine processes before
        signalling. Use it once the leader has been reaped (wait()/poll()),
        because the PID is then eligible for reuse and could name a group that
        isn't ours.
        """
        if not self.pid or os.name == "nt":
            return
        if verify:
            members = process_group_members(self.pid)
            if not members:
                return  # group already empty — nothing to do
            if not ENGINE_PROC_RE.search(members):
                logger.warning(
                    f"⚠ Not signalling pgid {self.pid}: it no longer looks like "
                    f"{self.model_config.name}'s process group"
                )
                return
        try:
            os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass  # group fully gone — the normal case

    def group_is_alive(self) -> bool:
        """True while any process in this session's group is still running."""
        return bool(process_group_members(self.pid))

    async def stop(self) -> None:
        """Terminate the session's entire process group.

        The SIGTERM must go to the *group*, not just the leader: vLLM's
        spawn-mode workers are separate processes, and a leader that exits
        cleanly can still leave workers holding VRAM. A final SIGKILL group
        sweep runs unconditionally for the same reason.
        """
        self.cancelled = True
        if self.adopted and self.process is None:
            await self._stop_adopted()
            return
        if not self.process:
            return

        try:
            logger.info(f"⏳ Stopping {self.model_config.name} (PID {self.pid})")

            # Graceful termination of the whole group (engine workers included)
            if os.name != "nt":
                self.kill_process_group(signal.SIGTERM)
            else:
                self.process.terminate()

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.process.wait),
                    timeout=self.term_grace
                )
                logger.info(f"✓ {self.model_config.name} terminated gracefully")
            except asyncio.TimeoutError:
                logger.warning(f"⚠ Force-killing {self.model_config.name} and process group")
                self.kill_process_group(signal.SIGKILL)
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
            finally:
                # Unconditional sweep: catches workers the leader left behind.
                # Verified, because the leader has now been reaped.
                self.kill_process_group(signal.SIGKILL, verify=True)

            self.process = None
            self.pid = None

        except Exception as e:
            logger.error(f"✗ Error stopping {self.model_config.name}: {e}")

    async def _stop_adopted(self) -> None:
        """stop() for a re-attached session: no Popen handle, so go by the group."""
        if not self.pid:
            return
        try:
            logger.info(f"⏳ Stopping adopted {self.model_config.name} (pgid {self.pid})")
            # The leader is still the process we launched (start time matches),
            # so the PID can't have been recycled; signal the whole group.
            if self.leader_alive():
                self.kill_process_group(signal.SIGTERM)
            deadline = time.time() + self.term_grace
            while time.time() < deadline and self.group_is_alive():
                await asyncio.sleep(0.5)
            if self.group_is_alive():
                logger.warning(f"⚠ Force-killing adopted {self.model_config.name} process group")
            # Verified sweep: once the leader is gone its PID may be reused.
            self.kill_process_group(signal.SIGKILL, verify=True)
            await asyncio.sleep(0.2)
            if not self.group_is_alive():
                logger.info(f"✓ {self.model_config.name} terminated")
                self.pid = None
        except Exception as e:
            logger.error(f"✗ Error stopping adopted {self.model_config.name}: {e}")

    async def is_running(self) -> bool:
        """Check if process is still running."""
        if self.adopted and self.process is None:
            return self.leader_alive()
        if not self.process:
            return False
        return self.process.poll() is None

    async def is_healthy(self) -> bool:
        """
        Check if llama-server is healthy via HTTP health endpoint.

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"http://localhost:{self.model_config.port}/health"
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False

    async def wait_ready(self, timeout: int = 120, check_interval: float = 5.0) -> None:
        """
        Wait for llama-server to be ready.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between health checks in seconds

        Raises:
            asyncio.TimeoutError: If process doesn't become ready in time
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check if cancelled
            if self.cancelled:
                raise asyncio.CancelledError("Load cancelled by user")

            # Check if process is still running
            if not await self.is_running():
                raise RuntimeError(f"Process exited before becoming ready")

            # Check health endpoint. An adopted session must also own the
            # port: during its load nothing of ours is listening yet, and a
            # different server answering /health there is not our model.
            if await self.is_healthy():
                if not self.adopted:
                    return
                if await asyncio.to_thread(port_owned_by_group, self.model_config.port, self.pid) is not False:
                    return

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise asyncio.TimeoutError(
                    f"llama-server didn't become ready within {timeout}s"
                )

            # Wait before next check
            await asyncio.sleep(check_interval)

    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "model_name": self.model_config.name,
            "pid": self.pid,
            "port": self.model_config.port,
            "running": self.leader_alive(),
            "adopted": self.adopted,
        }
