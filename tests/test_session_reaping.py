"""Tests for process-group cleanup and the session registry (audit #5, #6, #7).

These use real process groups (a `sleep` renamed so it matches the engine
signature) rather than mocks, because the thing under test is precisely
whether the OS-level group is gone.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from llama_session import (  # noqa: E402
    ENGINE_PROC_RE,
    LlamaSession,
    process_group_members,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")


@pytest.fixture
def fake_engine(tmp_path):
    """Spawn a process group whose cmdline looks like an engine process."""
    spawned = []

    def _spawn(name="vllm", seconds=60):
        exe = tmp_path / name
        if not exe.exists():
            exe.symlink_to("/bin/sleep")
        p = subprocess.Popen([str(exe), str(seconds)], start_new_session=True)
        spawned.append(p)
        time.sleep(0.3)  # let it appear in the process table
        return p

    yield _spawn
    for p in spawned:
        try:
            os.killpg(p.pid, 9)
        except OSError:
            pass


class TestGroupHelpers:
    def test_members_listed_then_empty_after_kill(self, fake_engine):
        p = fake_engine()
        assert str(p.pid) in process_group_members(p.pid)
        os.killpg(p.pid, 9)
        p.wait()
        time.sleep(0.2)
        assert process_group_members(p.pid) == ""

    def test_empty_for_unknown_pgid(self):
        assert process_group_members(None) == ""
        assert process_group_members(999999) == ""

    def test_engine_signature_matches_real_engines(self):
        assert ENGINE_PROC_RE.search("1 /venv/bin/vllm serve /models/x")
        assert ENGINE_PROC_RE.search("2 VLLM::EngineCore")
        assert ENGINE_PROC_RE.search("3 /usr/bin/llama-server -m x.gguf")
        assert ENGINE_PROC_RE.search("4 /bin/bash /config/models/foo.sh")
        # A recycled PID belonging to something else must not match
        assert not ENGINE_PROC_RE.search("5 /usr/bin/firefox")
        assert not ENGINE_PROC_RE.search("6 postgres: writer process")


class _StubConfig:
    def __init__(self, port=9999):
        self.name = "fake-model"
        self.port = port
        self.script_path = Path("/tmp/fake.sh")


def _session_for(pid: int) -> LlamaSession:
    s = LlamaSession(
        model_config=_StubConfig(),
        llama_server_binary="/bin/true",
        log_file=Path("/tmp/fake.log"),
    )
    s.pid = pid
    return s


class TestVerifiedKill:
    def test_kills_a_group_that_is_ours(self, fake_engine):
        p = fake_engine()
        _session_for(p.pid).kill_process_group(verify=True)
        time.sleep(0.3)
        assert process_group_members(p.pid) == ""

    def test_refuses_a_group_that_is_not_ours(self, fake_engine):
        """Guards the PID-reuse window after wait()/poll() frees the leader PID."""
        p = fake_engine(name="firefox")  # cmdline does not match the engine signature
        _session_for(p.pid).kill_process_group(verify=True)
        time.sleep(0.3)
        assert process_group_members(p.pid) != "", "unrelated process group was killed"

    def test_group_is_alive_reflects_reality(self, fake_engine):
        p = fake_engine()
        session = _session_for(p.pid)
        assert session.group_is_alive() is True
        os.killpg(p.pid, 9)
        p.wait()
        time.sleep(0.2)
        assert session.group_is_alive() is False


class TestReaper:
    def _manager(self, tmp_path):
        from config_manager import ConfigManager
        from gpu_manager import GpuManager

        return GpuManager(ConfigManager(tmp_path / "app.json", project_root=tmp_path))

    def _registry(self, tmp_path, entries: dict) -> Path:
        path = tmp_path / "sessions" / "active.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries))
        return path

    def test_reaps_a_stale_group(self, tmp_path, fake_engine):
        p = fake_engine()
        reg = self._registry(tmp_path, {"m": {"pid": p.pid, "port": 9999}})
        self._manager(tmp_path)._reap_stale_sessions()
        time.sleep(0.3)
        assert process_group_members(p.pid) == ""
        assert json.loads(reg.read_text()) == {}

    def test_spares_a_recycled_pid(self, tmp_path, fake_engine):
        p = fake_engine(name="firefox")
        self._registry(tmp_path, {"m": {"pid": p.pid, "port": 9999}})
        self._manager(tmp_path)._reap_stale_sessions()
        time.sleep(0.3)
        assert process_group_members(p.pid) != ""

    def test_forgets_entries_whose_group_is_already_gone(self, tmp_path):
        reg = self._registry(tmp_path, {"m": {"pid": 999999, "port": 9999}})
        self._manager(tmp_path)._reap_stale_sessions()
        assert json.loads(reg.read_text()) == {}

    def test_keeps_entries_it_could_not_kill(self, tmp_path, fake_engine, monkeypatch):
        """An un-killable group must stay recorded instead of being forgotten."""
        import gpu_manager

        p = fake_engine()
        reg = self._registry(tmp_path, {"m": {"pid": p.pid, "port": 9999}})
        monkeypatch.setattr(gpu_manager.os, "killpg", lambda *a: None)  # kills fail silently
        self._manager(tmp_path)._reap_stale_sessions()
        assert "m" in json.loads(reg.read_text())


class TestEngineEnvironment:
    """The engine child must not inherit Llama Studio's own control vars.

    start.sh exports VERBOSE as "" (or "--verbose"); vLLM reads its own VERBOSE
    as bool(int(...)), so inheriting ours killed engine-core init ~100s into
    startup, after weights and CUDA graphs were already loaded.
    """

    def _session(self, gpu_ids):
        return LlamaSession(
            model_config=_StubConfig(),
            llama_server_binary="/bin/true",
            log_file=Path("/tmp/fake.log"),
            gpu_ids=gpu_ids,
        )

    @pytest.mark.parametrize("value", ["", "--verbose", "1"])
    def test_verbose_never_reaches_the_engine(self, monkeypatch, value):
        monkeypatch.setenv("VERBOSE", value)
        assert "VERBOSE" not in self._session([0])._build_env()

    def test_sets_cuda_visible_devices_in_order(self):
        assert self._session([2, 0, 1])._build_env()["CUDA_VISIBLE_DEVICES"] == "2,0,1"

    def test_other_environment_is_preserved(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert self._session([0])._build_env()["HF_HUB_OFFLINE"] == "1"
