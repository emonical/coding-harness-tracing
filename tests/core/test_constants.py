"""Tests for the harness runtime root (core/constants.py BASE_DIR and ARIZE_HARNESS_HOME)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._pyproject import REPO_ROOT

_REPORT_PATHS = """
import json, sys
import core.constants as c
print(json.dumps({
    "base": str(c.BASE_DIR),
    "config": str(c.CONFIG_FILE),
    "state": str(c.STATE_BASE_DIR),
    "logs": str(c.LOG_DIR),
    "run": str(c.PID_DIR),
    "claude_log": str(c.HARNESSES["claude-code"]["default_log_file"]),
}))
"""


def _paths_with_env(tmp_path: Path, **extra_env: str) -> dict[str, str]:
    """Import core.constants in a fresh interpreter and report the paths it derived."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARIZE_", "PHOENIX_", "OTEL_"))}
    env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)})
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", _REPORT_PATHS],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestHarnessHome:
    def test_default_is_under_home(self, tmp_path: Path) -> None:
        paths = _paths_with_env(tmp_path)
        base = tmp_path / ".arize" / "harness"
        assert paths["base"] == str(base)
        assert paths["config"] == str(base / "config.json")
        assert paths["state"] == str(base / "state")
        assert paths["logs"] == str(base / "logs")
        assert paths["run"] == str(base / "run")

    def test_override_relocates_every_derived_path(self, tmp_path: Path) -> None:
        """Config, state, logs, run and the per-harness log all follow ARIZE_HARNESS_HOME."""
        root = tmp_path / "elsewhere" / "harness"
        paths = _paths_with_env(tmp_path, ARIZE_HARNESS_HOME=str(root))
        assert paths["base"] == str(root)
        assert paths["config"] == str(root / "config.json")
        assert paths["state"] == str(root / "state")
        assert paths["logs"] == str(root / "logs")
        assert paths["run"] == str(root / "run")
        assert paths["claude_log"] == str(root / "logs" / "claude-code.log")
        assert not str(root).startswith(str(tmp_path / ".arize"))

    def test_override_expands_tilde(self, tmp_path: Path) -> None:
        paths = _paths_with_env(tmp_path, ARIZE_HARNESS_HOME="~/stores/side/harness")
        assert paths["base"] == str(tmp_path / "stores" / "side" / "harness")

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_override_falls_back_to_default(self, tmp_path: Path, value: str) -> None:
        paths = _paths_with_env(tmp_path, ARIZE_HARNESS_HOME=value)
        assert paths["base"] == str(tmp_path / ".arize" / "harness")

    def test_override_does_not_move_the_install_dir(self, tmp_path: Path) -> None:
        """The install tree is code, not session state; it stays under HOME."""
        script = "from core.setup import INSTALL_DIR; print(INSTALL_DIR)"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ARIZE_", "PHOENIX_", "OTEL_"))}
        env.update(
            {
                "HOME": str(tmp_path),
                "USERPROFILE": str(tmp_path),
                "ARIZE_HARNESS_HOME": str(tmp_path / "elsewhere"),
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, env=env, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(tmp_path / ".arize" / "harness")
