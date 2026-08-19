"""Wrapper for llama-server process management."""

import asyncio
import subprocess
import logging
import os
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

from config_manager import ModelConfig

logger = logging.getLogger(__name__)


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

            # Set up environment with CUDA_VISIBLE_DEVICES for GPU assignment.
            # For tensor-split models, gpu_ids contains all GPUs in the order they
            # should be visible to llama-server (--main-gpu and --tensor-split index
            # into this visible-devices list).
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ",".join(str(g) for g in self.gpu_ids)
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

    def kill_process_group(self, sig: int = signal.SIGKILL) -> None:
        """Signal the session's entire process group (leader may already be dead).

        start_new_session=True makes the spawned script its own group leader,
        so self.pid doubles as the pgid even after the leader exits — this is
        how orphaned engine workers (vLLM spawn-mode) get swept.
        """
        if not self.pid or os.name == "nt":
            return
        try:
            os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass  # group fully gone — the normal case

    async def stop(self) -> None:
        """Terminate the session's entire process group.

        The SIGTERM must go to the *group*, not just the leader: vLLM's
        spawn-mode workers are separate processes, and a leader that exits
        cleanly can still leave workers holding VRAM. A final SIGKILL group
        sweep runs unconditionally for the same reason.
        """
        self.cancelled = True
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
                self.kill_process_group(signal.SIGKILL)

            self.process = None
            self.pid = None

        except Exception as e:
            logger.error(f"✗ Error stopping {self.model_config.name}: {e}")

    async def is_running(self) -> bool:
        """Check if process is still running."""
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

            # Check health endpoint
            if await self.is_healthy():
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
            "running": self.process is not None and self.process.poll() is None,
        }
