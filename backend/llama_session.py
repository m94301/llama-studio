"""Wrapper for llama-server process management."""

import asyncio
import subprocess
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
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
        gpu_id: Union[int, List[int]] = 0,
    ):
        self.model_config = model_config
        self.llama_server_binary = llama_server_binary
        self.log_file = log_file
        self.gpu_id = gpu_id
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.cancelled: bool = False

    async def start(self) -> int:
        """
        Start llama-server process.

        Returns:
            Process ID

        Raises:
            RuntimeError: If process fails to start
        """
        cmd = self._build_command()

        logger.info(f"⏳ Starting {self.model_config.name} on port {self.model_config.port}")
        logger.debug(f"   Command: {' '.join(cmd)}")

        try:
            self.cancelled = False
            # Open log file for writing
            log_file_handle = open(self.log_file, "w")

            # Write the full command to log file for reference
            full_cmd_str = ' '.join(cmd)
            log_file_handle.write(f"{'='*80}\n")
            log_file_handle.write(f"Launch Command:\n")
            log_file_handle.write(f"{full_cmd_str}\n")
            log_file_handle.write(f"{'='*80}\n")
            log_file_handle.flush()

            # Set up environment with CUDA_VISIBLE_DEVICES for GPU assignment
            env = os.environ.copy()
            gpu_ids = self.gpu_id if isinstance(self.gpu_id, list) else [self.gpu_id]
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
            env['GGML_CUDA_ALLREDUCE'] = 'internal'
            logger.debug(f"   CUDA_VISIBLE_DEVICES set to: {self.gpu_id}")

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
            logger.info(f"   Polling health check at 5s intervals using 300s timeout")

            # Wait for llama-server to be ready
            await self.wait_ready(timeout=300)

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

    async def stop(self) -> None:
        """Terminate llama-server process and entire process group."""
        self.cancelled = True
        if not self.process:
            return

        try:
            logger.info(f"⏳ Stopping {self.model_config.name} (PID {self.pid})")

            # Graceful termination
            self.process.terminate()

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.process.wait),
                    timeout=5.0
                )
                logger.info(f"✓ {self.model_config.name} terminated gracefully")
            except asyncio.TimeoutError:
                # Force kill if graceful termination times out
                logger.warning(f"⚠ Force-killing {self.model_config.name} and process group")
                self.process.kill()
                await asyncio.to_thread(self.process.wait)

                # Also kill entire process group to catch any child processes
                if self.pid and os.name != 'nt':  # Not Windows
                    try:
                        os.killpg(os.getpgid(self.pid), 9)  # SIGKILL entire group
                    except (ProcessLookupError, OSError):
                        pass  # Process group already gone

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

    def _build_command(self) -> list:
        """
        Build llama-server command from model config.

        Returns:
            List of command arguments
        """
        cmd = [self.llama_server_binary]

        # Model path
        cmd.extend(["-m", str(self.model_config.model_path)])

        # Launch arguments from config (passed directly, config must include hyphens)
        # Note: --port and --host are included in launch_args; no need to add them separately
        logger.debug(f"   launch_args dict: {self.model_config.launch_args}")
        for key, value in self.model_config.launch_args.items():
            logger.debug(f"      Adding arg: {key} = {value} (type: {type(value).__name__})")
            # Handle flag arguments (no value, e.g., --no-mmap) vs. arguments with values
            if value is None or str(value).strip() == "":
                cmd.append(key)
            else:
                cmd.extend([key, str(value)])

        logger.info(f"   Full command: {' '.join(cmd)}")
        return cmd

    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "model_name": self.model_config.name,
            "pid": self.pid,
            "port": self.model_config.port,
            "running": self.process is not None and self.process.poll() is None,
        }
