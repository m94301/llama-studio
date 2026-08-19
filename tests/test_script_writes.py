"""Tests for atomic script writes + backups (audit finding #3).

Model scripts are user-authored config that the scanner rewrites in place on
every startup and rescan. A torn write must be impossible, and a bad transform
must be recoverable.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config_manager import (  # noqa: E402
    SCRIPT_BACKUP_DIRNAME,
    SCRIPT_BACKUP_KEEP,
    write_script_atomic,
)


@pytest.fixture
def script(tmp_path) -> Path:
    p = tmp_path / "model.sh"
    p.write_text("#!/usr/bin/env bash\noriginal\n")
    p.chmod(0o755)
    return p


def _backups(script: Path):
    d = script.parent / SCRIPT_BACKUP_DIRNAME
    return sorted(d.glob(f"{script.name}.*")) if d.exists() else []


class TestAtomicWrite:
    def test_writes_new_content(self, script):
        assert write_script_atomic(script, "new\n") is True
        assert script.read_text() == "new\n"

    def test_unchanged_content_is_not_rewritten(self, script):
        before = script.stat().st_mtime_ns
        assert write_script_atomic(script, script.read_text()) is False
        assert script.stat().st_mtime_ns == before
        assert _backups(script) == []  # no churn, no snapshot

    def test_preserves_executable_bit(self, script):
        write_script_atomic(script, "changed\n")
        assert os.access(script, os.X_OK)

    def test_creates_file_that_does_not_exist(self, tmp_path):
        target = tmp_path / "sub" / "new.sh"
        assert write_script_atomic(target, "hello\n") is True
        assert target.read_text() == "hello\n"

    def test_leaves_no_temp_files_behind(self, script):
        write_script_atomic(script, "changed\n")
        assert [p.name for p in script.parent.iterdir() if p.name.startswith(".")
                and "tmp" in p.name] == []

    def test_original_survives_a_failed_write(self, script, monkeypatch):
        """A crash mid-write must not truncate the real file."""
        import config_manager

        def boom(src, dst):
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(config_manager.os, "replace", boom)
        with pytest.raises(OSError):
            write_script_atomic(script, "half-written garbage")
        assert script.read_text() == "#!/usr/bin/env bash\noriginal\n"


class TestBackups:
    def test_modification_snapshots_prior_content(self, script):
        original = script.read_text()
        write_script_atomic(script, "v2\n")
        snaps = _backups(script)
        assert len(snaps) == 1
        assert snaps[0].read_text() == original

    def test_new_file_creates_no_backup(self, tmp_path):
        target = tmp_path / "fresh.sh"
        write_script_atomic(target, "content\n", backup=False)
        assert _backups(target) == []

    def test_backup_can_be_opted_out(self, script):
        write_script_atomic(script, "v2\n", backup=False)
        assert _backups(script) == []

    def test_snapshots_are_pruned(self, script, monkeypatch):
        import config_manager

        # Distinct timestamps so each write lands in its own snapshot file.
        stamps = iter([f"2026010{i}-000000" for i in range(1, 9)])
        monkeypatch.setattr(config_manager.time, "strftime", lambda _f: next(stamps))
        for i in range(SCRIPT_BACKUP_KEEP + 2):
            write_script_atomic(script, f"v{i}\n")
        assert len(_backups(script)) == SCRIPT_BACKUP_KEEP
