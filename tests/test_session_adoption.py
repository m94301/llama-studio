"""Keeping model servers alive across a Llama Studio restart, and re-attaching.

Real processes, not mocks: what's under test is the OS-level identity checks
(PID + kernel start time, which process owns the port) that decide whether a
leftover server is ours to adopt, ours to reap, or someone else's to leave
alone.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from gpu_manager import GpuManager, LoadedModel, ModelState  # noqa: E402
from llama_session import (  # noqa: E402
    LlamaSession, launch_fingerprint, port_owned_by_group, proc_start_time, process_group_members,
)

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/net"), reason="needs Linux /proc")

HEALTH_SERVER = """
import http.server, sys, time
delay = float(sys.argv[2])
time.sleep(delay)   # 'loading': nothing listens until this passes
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listening(port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(port)


@pytest.fixture
def engine(tmp_path):
    """Start a fake engine: a /health server in its own process group, named like vLLM."""
    procs = []
    script = tmp_path / "health_server.py"
    script.write_text(HEALTH_SERVER)

    def _start(port, delay=0.0, name="vllm"):
        exe = tmp_path / name
        if not exe.exists():
            exe.symlink_to(sys.executable)
        p = subprocess.Popen([str(exe), str(script), str(port), str(delay)], start_new_session=True)
        procs.append(p)
        time.sleep(0.2)
        if not delay:
            _wait_listening(port)
        return p

    yield _start
    for p in procs:
        try:
            os.killpg(p.pid, 9)
        except OSError:
            pass
        p.wait()


class _StubModelConfig:
    def __init__(self, name, port, script_path):
        self.name = name
        self.port = port
        self.engine = "vllm"
        self.script_path = script_path
        self.html_id = name


class _StubAppConfig:
    keep_sessions_on_shutdown = True


class _StubConfig:
    """Just enough ConfigManager for GpuManager's registry/adoption paths."""

    def __init__(self, root, models):
        self.project_root = root
        self.app_config = _StubAppConfig()
        self.models = models

    def get_all_models(self):
        return self.models

    def get_model_config(self, name):
        return self.models[name]

    def get_log_path(self, name):
        return self.project_root / f"{name}.log"


def _manager(tmp_path, port, name="big-model"):
    script = tmp_path / f"{name}.sh"
    script.write_text("#!/bin/sh\n")
    cfg = _StubConfig(tmp_path, {name: _StubModelConfig(name, port, script)})
    gm = GpuManager(cfg)
    gm.models[name] = LoadedModel(name=name, size_gb=1.0, state=ModelState.IDLE, gpu_ids=[], port=port)
    return gm, script


def _record(p, port, script, **extra):
    rec = {"pid": p.pid, "start_time": proc_start_time(p.pid), "port": port, "engine": "vllm",
           "gpu_ids": [0, 1], "vram_per_gpu": [40.0, 40.0], "launched_at": time.time(),
           "script_fingerprint": launch_fingerprint(script), "health_timeout": 900}
    rec.update(extra)
    return rec


def _write_registry(tmp_path, entries):
    path = tmp_path / "sessions" / "active.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path


# ---------- identity helpers ----------

def test_start_time_identifies_a_process(engine):
    p = engine(_free_port())
    t = proc_start_time(p.pid)
    assert isinstance(t, int) and t == proc_start_time(p.pid)
    os.killpg(p.pid, 9)
    p.wait()
    assert proc_start_time(p.pid) is None


def test_port_ownership(engine):
    port = _free_port()
    ours = engine(port)
    other = engine(_free_port())
    assert port_owned_by_group(port, ours.pid) is True
    assert port_owned_by_group(port, other.pid) is False      # someone else's port
    assert port_owned_by_group(_free_port(), ours.pid) is False  # nobody listening


# ---------- startup triage ----------

def test_adopts_a_server_it_launched(tmp_path, engine):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})

    gm._triage_registry(keep_alive=True)
    assert "big-model" in gm._adoptable
    asyncio.run(gm._adopt_sessions())

    s = gm.sessions["big-model"]
    assert s.adopted and s.pid == p.pid
    assert gm.get_model_state("big-model") == ModelState.RUNNING
    assert gm.models["big-model"].gpu_ids == [0, 1]
    assert "restart" in gm.session_notes["big-model"]
    assert process_group_members(p.pid), "adoption must not touch the process"
    reg = json.loads((tmp_path / "sessions" / "active.json").read_text())
    assert reg["big-model"]["pid"] == p.pid and reg["big-model"]["start_time"]


def test_recycled_pid_is_left_alone_even_if_it_looks_like_an_engine(tmp_path, engine):
    """A server the user started by hand that got a recorded PID: not adopted, not killed."""
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script, start_time=proc_start_time(p.pid) - 1)})

    for keep in (True, False):
        gm._triage_registry(keep_alive=keep)
        assert gm._adoptable == {}
        time.sleep(0.2)
        assert process_group_members(p.pid), f"foreign process was killed (keep_alive={keep})"


def test_reaped_when_keep_alive_is_off(tmp_path, engine):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    gm._triage_registry(keep_alive=False)
    p.wait(timeout=12)
    assert gm._adoptable == {} and not process_group_members(p.pid)


def test_reaped_when_its_script_is_gone(tmp_path, engine):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    gm.config.models.clear()
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    gm._triage_registry(keep_alive=True)
    p.wait(timeout=12)
    assert gm._adoptable == {}


# ---------- adopted session lifecycle ----------

def test_adopted_mid_load_becomes_running(tmp_path, engine):
    port = _free_port()
    p = engine(port, delay=1.5)   # still "loading": not listening yet
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    gm._triage_registry(keep_alive=True)

    async def go():
        await gm._adopt_sessions()
        assert gm.get_model_state("big-model") == ModelState.LOADING
        for _ in range(60):
            if gm.get_model_state("big-model") == ModelState.RUNNING:
                return True
            await asyncio.sleep(0.25)
        return False

    # Speed up the 5s health poll for the test.
    orig = LlamaSession.wait_ready
    LlamaSession.wait_ready = lambda self, timeout=120, check_interval=5.0: orig(self, timeout, 0.2)
    try:
        assert asyncio.run(go())
    finally:
        LlamaSession.wait_ready = orig


def test_mid_load_does_not_trust_someone_elses_health(tmp_path, engine):
    """While ours is loading, a foreign server answering /health on the port isn't 'ready'."""
    port = _free_port()
    foreign = engine(port)                          # holds the port and answers /health
    ours = engine(_free_port(), delay=30)           # our 'engine', never ready in time
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(ours, port, script)})
    gm._triage_registry(keep_alive=True)
    asyncio.run(gm._adopt_sessions())
    assert gm.get_model_state("big-model") == ModelState.LOADING
    assert process_group_members(foreign.pid)


def test_unload_stops_an_adopted_server(tmp_path, engine):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    gm._triage_registry(keep_alive=True)

    async def go():
        await gm._adopt_sessions()
        await gm.unload_model("big-model")
    asyncio.run(go())
    p.wait(timeout=10)
    assert "big-model" not in gm.sessions
    assert gm.get_model_state("big-model") == ModelState.IDLE
    assert gm.models["big-model"].gpu_ids == []
    assert json.loads((tmp_path / "sessions" / "active.json").read_text()) == {}


def test_adopted_leader_death_is_detected(tmp_path, engine):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    gm._triage_registry(keep_alive=True)
    asyncio.run(gm._adopt_sessions())
    s = gm.sessions["big-model"]
    assert s.leader_alive()
    os.killpg(p.pid, 9)
    p.wait()
    assert not s.leader_alive()


def _note_after(tmp_path, engine, edit):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    script.write_text("#!/bin/sh\n# === llama-studio:meta BEGIN ===\n# display_name: A\n"
                      "# === llama-studio:meta END ===\nexec vllm serve /m --port 1\n")
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    script.write_text(edit(script.read_text()))
    gm._triage_registry(keep_alive=True)
    asyncio.run(gm._adopt_sessions())
    return gm.session_notes["big-model"]


def test_edited_launch_args_are_flagged(tmp_path, engine):
    assert "edited" in _note_after(tmp_path, engine, lambda t: t.replace("--port 1", "--port 2"))


def test_meta_fence_rewrite_is_not_an_edit(tmp_path, engine):
    """Studio's own post-load calibration rewrites the meta fence; that's not 'edited'."""
    note = _note_after(tmp_path, engine,
                       lambda t: t.replace("# display_name: A", "# display_name: B\n# kv_bytes_per_token: 22094"))
    assert "edited" not in note and "restart" in note


# ---------- shutdown ----------

def test_keep_alive_shutdown_leaves_servers_and_registry(tmp_path, engine):
    port = _free_port()
    p = engine(port)
    gm, script = _manager(tmp_path, port)
    _write_registry(tmp_path, {"big-model": _record(p, port, script)})
    gm._triage_registry(keep_alive=True)

    async def go():
        await gm._adopt_sessions()
        await gm.cleanup(keep_alive=True)
    asyncio.run(go())
    assert process_group_members(p.pid)
    reg = json.loads((tmp_path / "sessions" / "active.json").read_text())
    assert reg["big-model"]["pid"] == p.pid

    # ...and the next start finds it again
    gm2, _ = _manager(tmp_path, port)
    gm2._triage_registry(keep_alive=True)
    assert "big-model" in gm2._adoptable
