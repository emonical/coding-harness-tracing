"""Tests for workspace profiles (core/profiles.py) and their hook-in to ``arize-harness hook``."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from core import cli, profiles
from tests._pyproject import REPO_ROOT

SECRET = "s3cret-writer-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The shared harness root that holds config.json, pointed at by ARIZE_HARNESS_HOME."""
    root = tmp_path / "harness"
    root.mkdir()
    monkeypatch.setenv("ARIZE_HARNESS_HOME", str(root))
    return root


@pytest.fixture
def resolver(tmp_path: Path) -> dict:
    """A resolver that maps a cwd's basename through map.json and records every call."""
    script = tmp_path / "resolve.py"
    calls = tmp_path / "calls.txt"
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"repo-a": "team-a", "repo-cloud": "cloud", "repo-x": "nobody", "repo-boom": "!fail"})
    )
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os, sys
            cwd = sys.argv[1]
            with open({str(calls)!r}, "a") as fh:
                fh.write(cwd + "\\n")
            name = json.load(open({str(mapping)!r})).get(os.path.basename(cwd), "")
            if name == "!fail":
                sys.exit(3)
            print(name)
            """
        )
    )
    return {"argv": [sys.executable, str(script)], "calls": calls}


def _calls(resolver: dict) -> list[str]:
    return resolver["calls"].read_text().splitlines() if resolver["calls"].exists() else []


@pytest.fixture
def block(tmp_path: Path, resolver: dict) -> dict:
    key = tmp_path / "writer.key"
    key.write_text(SECRET + "\n")
    return {
        "resolver": resolver["argv"],
        "by_name": {
            "team-a": {
                "harness_home": str(tmp_path / "team-a-home"),
                "target": "phoenix",
                "endpoint": "http://127.0.0.1:6007",
                "api_key_file": str(key),
                "project_name": "claude-code",
                "logging": {"prompts": True, "tool_details": True, "tool_content": False},
                "attributes": {"workspace": "team-a"},
            },
            "cloud": {"target": "arize", "api_key": "ax-key", "space_id": "space-1", "project_name": "codex"},
        },
    }


def _write(root: Path, block: dict | None) -> None:
    data = {"harnesses": {}} if block is None else {"harnesses": {}, "profiles": block}
    (root / "config.json").write_text(json.dumps(data))


@pytest.fixture
def fake_handler(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Route ``claude stop``/``session-start``/``session-end`` and ``codex notify`` to a recording handler."""
    calls: list[dict] = []
    module = types.ModuleType("fake_profile_handlers")

    def record() -> str:
        buffer = getattr(sys.stdin, "buffer", None)
        raw = buffer.read().decode("utf-8") if buffer is not None else sys.stdin.read()
        calls.append({"argv": list(sys.argv), "stdin": raw, "env": dict(os.environ)})
        return "handled"

    module.record = record  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_profile_handlers", module)
    target = "fake_profile_handlers:record"
    monkeypatch.setattr(
        cli,
        "EVENT_HOOKS",
        {
            "claude": {"stop": target, "session-start": target, "session-end": target},
            "codex": {"notify": target},
        },
    )
    monkeypatch.setattr(cli, "SINGLE_HOOKS", {})
    monkeypatch.setattr(sys, "argv", ["arize-harness"])
    return calls


@pytest.fixture
def run(root: Path, monkeypatch: pytest.MonkeyPatch):
    """Run one hook event as its own process would: launcher env, payload on stdin."""

    def _run(event: str, payload: dict, harness: str = "claude") -> Any:
        # Every hook event is a fresh process that starts from the launcher's env;
        # the previous run's profile must not leak its root into this one.
        monkeypatch.setenv("ARIZE_HARNESS_HOME", str(root))
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()), encoding="utf-8"))
        return cli.main(["hook", harness, event])

    return _run


def _log(root: Path) -> str:
    path = root / "logs" / "profiles.log"
    return path.read_text() if path.exists() else ""


# ---------------------------------------------------------------------------
# No profiles: the dispatcher behaves exactly as before
# ---------------------------------------------------------------------------


class TestWithoutProfiles:
    def test_missing_config_passes_through(self, root, fake_handler, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"cwd": "/x"}'))
        assert cli.main(["hook", "claude", "stop"]) == "handled"
        assert fake_handler[0]["stdin"] == '{"cwd": "/x"}'
        assert "PHOENIX_ENDPOINT" not in fake_handler[0]["env"]

    def test_config_without_block_passes_through(self, root, fake_handler, monkeypatch) -> None:
        _write(root, None)
        monkeypatch.setenv("PHOENIX_ENDPOINT", "http://inherited:6006")
        monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
        cli.main(["hook", "claude", "stop"])
        assert fake_handler[0]["env"]["PHOENIX_ENDPOINT"] == "http://inherited:6006"
        assert not (root / "logs").exists()

    def test_malformed_config_passes_through(self, root, fake_handler, monkeypatch) -> None:
        (root / "config.json").write_text("{not json")
        monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
        assert cli.main(["hook", "claude", "stop"]) == "handled"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_phoenix_profile_sets_backend_root_logging_and_attributes(
        self, root, block, fake_handler, run, monkeypatch, tmp_path
    ):
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-a"), "hook_event_name": "SessionStart"}
        assert run("session-start", payload) == "handled"
        env = fake_handler[0]["env"]
        assert env["PHOENIX_ENDPOINT"] == "http://127.0.0.1:6007"
        assert env["PHOENIX_API_KEY"] == SECRET
        assert env["PHOENIX_PROJECT_NAME"] == "claude-code"
        assert env["ARIZE_HARNESS_HOME"] == str(tmp_path / "team-a-home")
        assert (env["ARIZE_LOG_PROMPTS"], env["ARIZE_LOG_TOOL_DETAILS"], env["ARIZE_LOG_TOOL_CONTENT"]) == (
            "true",
            "true",
            "false",
        )
        assert env["OTEL_RESOURCE_ATTRIBUTES"] == "workspace=team-a"
        assert "ARIZE_API_KEY" not in env
        assert fake_handler[0]["stdin"] == json.dumps(payload)
        assert fake_handler[0]["argv"] == ["arize-hook-session-start"]
        assert "session s1 -> profile 'team-a' (resolved)" in _log(root)
        assert SECRET not in _log(root)

    def test_arize_profile(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        _write(root, block)
        run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-cloud")})
        env = fake_handler[0]["env"]
        assert (env["ARIZE_API_KEY"], env["ARIZE_SPACE_ID"], env["ARIZE_PROJECT_NAME"]) == (
            "ax-key",
            "space-1",
            "codex",
        )
        assert "PHOENIX_ENDPOINT" not in env
        assert "ARIZE_HARNESS_HOME" in env  # the shared root from the fixture, untouched

    def test_inherited_routing_env_is_replaced(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        """Credentials an installed harness exported into the session must not survive into another profile."""
        _write(root, block)
        monkeypatch.setenv("ARIZE_API_KEY", "leaked")
        monkeypatch.setenv("ARIZE_SPACE_ID", "leaked-space")
        monkeypatch.setenv("PHOENIX_ENDPOINT", "http://other:6008")
        monkeypatch.setenv("PHOENIX_PROJECT", "other")
        monkeypatch.setenv("ARIZE_LOG_TOOL_CONTENT", "true")
        monkeypatch.setenv("ARIZE_VERBOSE", "true")
        run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")})
        env = fake_handler[0]["env"]
        assert "ARIZE_API_KEY" not in env and "ARIZE_SPACE_ID" not in env and "PHOENIX_PROJECT" not in env
        assert env["PHOENIX_ENDPOINT"] == "http://127.0.0.1:6007"
        assert env["ARIZE_LOG_TOOL_CONTENT"] == "false"
        assert env["ARIZE_VERBOSE"] == "true"

    def test_profile_attributes_append_to_existing_resource_attributes(
        self, root, block, fake_handler, run, monkeypatch, tmp_path
    ):
        _write(root, block)
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "team=payments")
        run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")})
        assert fake_handler[0]["env"]["OTEL_RESOURCE_ATTRIBUTES"] == "team=payments,workspace=team-a"

    def test_cwd_falls_back_to_the_process_directory(
        self, root, block, fake_handler, run, monkeypatch, tmp_path
    ) -> None:
        _write(root, block)
        repo = tmp_path / "repo-a"
        repo.mkdir()
        monkeypatch.chdir(repo)
        run("stop", {"session_id": "s1"})
        assert fake_handler[0]["env"]["PHOENIX_ENDPOINT"] == "http://127.0.0.1:6007"

    def test_argv_payload_is_read_when_stdin_is_empty(
        self, root, block, fake_handler, run, monkeypatch, tmp_path
    ) -> None:
        """Codex passes its notify payload as the first argument, not on stdin."""
        _write(root, block)
        payload = json.dumps({"type": "agent-turn-complete", "cwd": str(tmp_path / "repo-a")})
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b""), encoding="utf-8"))
        assert cli.main(["hook", "codex", "notify", payload]) == "handled"
        assert fake_handler[0]["argv"] == ["arize-hook-codex-notify", payload]
        assert fake_handler[0]["env"]["PHOENIX_API_KEY"] == SECRET


class TestFailClosed:
    def test_unknown_name_skips_and_logs_at_session_start(self, root, block, fake_handler, run, monkeypatch, tmp_path):
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-x"), "hook_event_name": "SessionStart"}
        assert run("session-start", payload) == 0
        assert fake_handler == []
        assert "skip session s1: no profile named 'nobody'" in _log(root)

    def test_unknown_name_on_a_later_event_skips_silently(self, root, block, fake_handler, run, monkeypatch, tmp_path):
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-x")}) == 0
        assert fake_handler == []
        assert _log(root) == ""

    def test_resolver_error_skips_and_logs(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-boom")}) == 0
        assert fake_handler == []
        assert "resolver failed" in _log(root)

    def test_missing_resolver_skips(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        block["resolver"] = str(tmp_path / "no-such-resolver")
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}) == 0
        assert fake_handler == []

    def test_empty_resolver_output_skips(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "unmapped")}) == 0
        assert fake_handler == []

    def test_unreadable_key_file_skips_and_logs(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        block["by_name"]["team-a"]["api_key_file"] = str(tmp_path / "missing.key")
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}) == 0
        assert fake_handler == []
        assert "profile 'team-a' not usable" in _log(root)

    def test_skip_imports_no_harness_module(self, root, block, tmp_path) -> None:
        """A skipped run must not load a handler: handlers write state at import."""
        _write(root, block)
        script = (
            "import atexit, json, sys\n"
            'atexit.register(lambda: sys.__stderr__.write("MODULES=" + json.dumps(sorted(sys.modules)) + "\\n"))\n'
            "from core.cli import main\n"
            "sys.exit(main(sys.argv[1:]))\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ARIZE_", "PHOENIX_", "OTEL_"))}
        env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "ARIZE_HARNESS_HOME": str(root)})
        payload = json.dumps({"session_id": "s1", "cwd": str(tmp_path / "repo-x")})
        result = subprocess.run(
            [sys.executable, "-c", script, "hook", "claude", "stop"],
            input=payload,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        line = next(line for line in result.stderr.splitlines() if line.startswith("MODULES="))
        modules = set(json.loads(line[len("MODULES=") :]))
        assert not {m for m in modules if m.startswith("tracing")}
        assert "core.constants" not in modules


class TestFallback:
    def test_unknown_name_uses_the_fallback_and_logs_it(self, root, block, fake_handler, run, monkeypatch, tmp_path):
        block["fallback"] = "cloud"
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-x"), "hook_event_name": "SessionStart"}
        assert run("session-start", payload) == "handled"
        assert fake_handler[0]["env"]["ARIZE_SPACE_ID"] == "space-1"
        assert "profile 'cloud' (fallback after: no profile named 'nobody')" in _log(root)

    def test_resolver_error_uses_the_fallback(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        block["fallback"] = "cloud"
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-boom")}) == "handled"

    def test_fallback_never_covers_an_unreadable_key(self, root, block, fake_handler, run, monkeypatch, tmp_path):
        block["fallback"] = "cloud"
        block["by_name"]["team-a"]["api_key_file"] = str(tmp_path / "missing.key")
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}) == 0
        assert fake_handler == []

    def test_fallback_naming_no_profile_still_skips(self, root, block, fake_handler, run, monkeypatch, tmp_path):
        block["fallback"] = "ghost"
        _write(root, block)
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-x")}) == 0
        assert fake_handler == []

    def test_forced_unknown_profile_ignores_the_fallback(self, root, block, fake_handler, run, monkeypatch, tmp_path):
        block["fallback"] = "cloud"
        _write(root, block)
        monkeypatch.setenv("ARIZE_PROFILE", "ghost")
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}) == 0
        assert fake_handler == []


class TestForcedProfile:
    def test_env_picks_the_profile_without_the_resolver(
        self, root, block, resolver, fake_handler, run, monkeypatch, tmp_path
    ):
        _write(root, block)
        monkeypatch.setenv("ARIZE_PROFILE", "cloud")
        run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")})
        assert fake_handler[0]["env"]["ARIZE_SPACE_ID"] == "space-1"
        assert _calls(resolver) == []

    def test_env_naming_an_unknown_profile_skips(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        _write(root, block)
        monkeypatch.setenv("ARIZE_PROFILE", "ghost")
        assert run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}) == 0
        assert fake_handler == []


class TestSessionCache:
    def test_resolver_runs_once_per_session(self, root, block, resolver, fake_handler, run, monkeypatch, tmp_path):
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}
        run("session-start", payload)
        run("stop", payload)
        run("stop", dict(payload, cwd=str(tmp_path / "repo-x")))  # cwd moved: profile is sticky
        assert len(fake_handler) == 3
        assert _calls(resolver) == [str(tmp_path / "repo-a")]
        cache = root / "run" / "profiles" / "claude" / "s1"
        assert cache.read_text().strip() == "team-a"
        assert (cache.stat().st_mode & 0o777) == 0o600

    def test_other_sessions_resolve_separately(self, root, block, resolver, fake_handler, run, monkeypatch, tmp_path):
        _write(root, block)
        run("stop", {"session_id": "s1", "cwd": str(tmp_path / "repo-a")})
        run("stop", {"session_id": "s2", "cwd": str(tmp_path / "repo-cloud")})
        assert len(_calls(resolver)) == 2
        assert fake_handler[1]["env"]["ARIZE_SPACE_ID"] == "space-1"

    def test_session_end_drops_the_cache(self, root, block, fake_handler, run, monkeypatch, tmp_path) -> None:
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}
        run("stop", payload)
        run("session-end", payload)
        assert not (root / "run" / "profiles" / "claude" / "s1").exists()
        assert len(fake_handler) == 2

    def test_skipped_names_are_cached_too(
        self, root, block, resolver, fake_handler, run, monkeypatch, tmp_path
    ) -> None:
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-x")}
        run("stop", payload)
        run("stop", payload)
        assert fake_handler == []
        assert len(_calls(resolver)) == 1

    def test_expired_cache_is_ignored(self, root, block, resolver, fake_handler, run, monkeypatch, tmp_path) -> None:
        block["cache_ttl_seconds"] = 1
        _write(root, block)
        payload = {"session_id": "s1", "cwd": str(tmp_path / "repo-a")}
        run("stop", payload)
        cache = root / "run" / "profiles" / "claude" / "s1"
        os.utime(cache, (0, 0))
        run("stop", payload)
        assert len(_calls(resolver)) == 2

    @pytest.mark.parametrize("session_id", ["../escape", "a/b", ""])
    def test_unsafe_session_ids_are_not_cached(self, root, block, fake_handler, run, monkeypatch, tmp_path, session_id):
        _write(root, block)
        run("stop", {"session_id": session_id, "cwd": str(tmp_path / "repo-a")})
        assert len(fake_handler) == 1
        assert not (root / "run").exists()


# ---------------------------------------------------------------------------
# ``arize-harness profile``
# ---------------------------------------------------------------------------


class TestProfileCommand:
    def test_prints_the_decision_without_secrets(self, root, block, capsys, tmp_path) -> None:
        _write(root, block)
        assert cli.main(["profile", str(tmp_path / "repo-a")]) == 0
        out = capsys.readouterr().out
        assert "profile: team-a" in out
        assert "endpoint: http://127.0.0.1:6007" in out
        assert "api key present: True" in out
        assert "content logging (prompts/details/content): true/true/false" in out
        assert SECRET not in out

    def test_prints_skip_reason(self, root, block, capsys, tmp_path) -> None:
        _write(root, block)
        cli.main(["profile", str(tmp_path / "repo-x")])
        assert capsys.readouterr().out.startswith("skip: no profile named 'nobody'")

    def test_without_profiles(self, root, capsys) -> None:
        cli.main(["profile"])
        assert "no profiles configured" in capsys.readouterr().out

    def test_forced_profile(self, root, block, capsys, monkeypatch, tmp_path) -> None:
        _write(root, block)
        monkeypatch.setenv("ARIZE_PROFILE", "cloud")
        cli.main(["profile", str(tmp_path / "repo-a")])
        assert "profile: cloud" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# apply_env unit behaviour
# ---------------------------------------------------------------------------


class TestApplyEnv:
    def test_target_is_inferred_from_space_id(self) -> None:
        env: dict = {}
        profiles.apply_env({"api_key": "k", "space_id": "s"}, env)
        assert env == {"ARIZE_API_KEY": "k", "ARIZE_SPACE_ID": "s"}

    def test_phoenix_without_key_sets_no_key(self) -> None:
        env = {"PHOENIX_API_KEY": "stale"}
        profiles.apply_env({"endpoint": "http://p:6006"}, env)
        assert env == {"PHOENIX_ENDPOINT": "http://p:6006"}

    def test_logging_flags_accept_strings(self) -> None:
        env: dict = {}
        profiles.apply_env({"endpoint": "http://p", "logging": {"prompts": "false", "tool_content": "TRUE"}}, env)
        assert env["ARIZE_LOG_PROMPTS"] == "false"
        assert env["ARIZE_LOG_TOOL_CONTENT"] == "true"
        assert "ARIZE_LOG_TOOL_DETAILS" not in env

    def test_attributes_with_separators_are_dropped(self) -> None:
        env: dict = {}
        profiles.apply_env({"endpoint": "http://p", "attributes": {"ok": "1", "a,b": "2", "c": "x=y"}}, env)
        assert env["OTEL_RESOURCE_ATTRIBUTES"] == "ok=1,c=x=y"

    def test_empty_key_file_raises(self, tmp_path: Path) -> None:
        key = tmp_path / "k"
        key.write_text("\n")
        with pytest.raises(ValueError):
            profiles.apply_env({"endpoint": "http://p", "api_key_file": str(key)}, {})
