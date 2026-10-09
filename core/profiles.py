"""Workspace profiles: route one hook run to one of several backends by its working directory.

A profile bundles a backend, its credential, a project name, content-logging flags
and a runtime root under one name. A resolver command maps the working directory of
the hook invocation to a profile name, and ``arize-harness hook`` applies the chosen
profile before the handler module is imported, by setting the same environment
variables an operator would set by hand. Handlers know nothing about profiles.

``config.json``::

    "profiles": {
      "resolver": "/usr/local/bin/resolve-workspace",   # or ["python3", "/path/resolve.py"]
      "cache_ttl_seconds": 86400,                        # optional, default one day
      "fallback": "",                                    # optional: profile for unresolved runs
      "by_name": {
        "team-a": {
          "harness_home": "~/stores/team-a/harness",    # -> ARIZE_HARNESS_HOME (optional)
          "target": "phoenix",                           # or "arize"; inferred when absent
          "endpoint": "http://127.0.0.1:6007",
          "api_key_file": "~/stores/team-a/writer.key",  # or "api_key" inline
          "space_id": "",                                # arize only
          "project_name": "claude-code",
          "user_id": "",                                 # optional
          "logging": {"prompts": true, "tool_details": true, "tool_content": false},
          "attributes": {"workspace": "team-a"}          # appended to OTEL_RESOURCE_ATTRIBUTES
        }
      }
    }

The resolver is run as ``<resolver> <cwd>`` and must print one profile name on
stdout. A name that is not in ``by_name`` means "do not trace this run". Resolution
is cached per ``session_id`` under ``<root>/run/profiles/<harness>/`` so the
resolver runs once per session, not once per hook event, and a session keeps the
profile it started with even if its working directory moves.

FAIL CLOSED by default. Once a ``profiles`` block exists, any failure to decide
(resolver error or timeout, unreadable key file, unknown name) skips the handler
and logs one line to ``<root>/logs/profiles.log``. Falling through to a default
backend would send one workspace's content to another workspace's instance. An
operator who wants unresolved runs traced anyway names a ``fallback`` profile;
it is used when the resolver fails or returns no usable name, never for an
unreadable key file. Without a ``profiles`` block nothing here runs and the hook
behaves exactly as before.

``ARIZE_PROFILE=<name>`` in the environment picks a profile without running the
resolver, for scratch workspaces and tests.

This module runs on every hook event and inside the Claude plugin path, so it
must stay standard-library only and must not import ``core.constants`` (whose
paths are fixed at import time, before the profile sets ``ARIZE_HARNESS_HOME``).
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_ENV_PROFILE = "ARIZE_PROFILE"
_DEFAULT_CACHE_TTL = 86400
_RESOLVER_TIMEOUT = 5.0

# Routing, credential and content keys a profile owns. Operator toggles such as
# ARIZE_TRACE_ENABLED, ARIZE_VERBOSE, ARIZE_DRY_RUN and ARIZE_LOG_FILE are kept.
_OWNED_ENV = (
    "ARIZE_API_KEY",
    "ARIZE_SPACE_ID",
    "ARIZE_OTLP_ENDPOINT",
    "ARIZE_PROJECT_NAME",
    "PHOENIX_ENDPOINT",
    "PHOENIX_API_KEY",
    "PHOENIX_PROJECT",
    "PHOENIX_PROJECT_NAME",
    "ARIZE_LOG_PROMPTS",
    "ARIZE_LOG_TOOL_DETAILS",
    "ARIZE_LOG_TOOL_CONTENT",
)

_LOG_FLAGS = (
    ("prompts", "ARIZE_LOG_PROMPTS"),
    ("tool_details", "ARIZE_LOG_TOOL_DETAILS"),
    ("tool_content", "ARIZE_LOG_TOOL_CONTENT"),
)


class Decision:
    """Outcome of :func:`apply_profile`.

    ``proceed`` says whether the handler should run. ``stdin`` carries the payload
    bytes already consumed from stdin, which the caller must hand back to the
    handler; it is None when stdin was not read (no ``profiles`` block).
    ``profile`` is the applied profile's name, or None.
    """

    __slots__ = ("proceed", "stdin", "profile", "reason")

    def __init__(self, proceed: bool, stdin: Optional[bytes] = None, profile: Optional[str] = None, reason: str = ""):
        self.proceed = proceed
        self.stdin = stdin
        self.profile = profile
        self.reason = reason


class Resolution:
    """What ``resolve`` decided for a working directory, with no secrets."""

    __slots__ = ("name", "profile", "reason", "cwd")

    def __init__(self, name: Optional[str], profile: Optional[dict], reason: str, cwd: str):
        self.name = name
        self.profile = profile
        self.reason = reason
        self.cwd = cwd


# ---------------------------------------------------------------------------
# Config root and log
# ---------------------------------------------------------------------------


def config_root() -> Path:
    """Where the shared ``config.json`` lives, before any profile moves the root."""
    override = os.environ.get("ARIZE_HARNESS_HOME", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".arize" / "harness"


def load_profiles(root: Optional[Path] = None) -> Optional[dict]:
    """Return the ``profiles`` block, or None when there is none to apply."""
    path = (root or config_root()) / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    block = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(block, dict) or not isinstance(block.get("by_name"), dict):
        return None
    return block


def _log(root: Path, harness: str, event: Optional[str], message: str) -> None:
    try:
        path = root / "logs" / "profiles.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {harness} {event or '-'}: {message}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Payload, cache, resolver
# ---------------------------------------------------------------------------


def _parse_payload(raw: bytes, rest: List[str]) -> dict:
    """The hook payload as a dict: stdin JSON first, then a JSON first argument."""
    for candidate in (raw.decode("utf-8", "replace") if raw else "", rest[0] if rest else ""):
        if not candidate.strip():
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _is_session_start(event: Optional[str], payload: dict) -> bool:
    name = str(payload.get("hook_event_name", "")).lower().replace("_", "").replace("-", "")
    return event == "session-start" or name == "sessionstart"


def _is_session_end(event: Optional[str], payload: dict) -> bool:
    name = str(payload.get("hook_event_name", "")).lower().replace("_", "").replace("-", "")
    return event == "session-end" or name == "sessionend"


def _cache_file(root: Path, harness: str, session_id: str) -> Optional[Path]:
    if not session_id or any(ch in session_id for ch in "/\\") or session_id in (".", ".."):
        return None
    return root / "run" / "profiles" / harness / session_id


def _cache_read(path: Optional[Path], ttl: float) -> Optional[str]:
    if path is None:
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cache_write(path: Optional[Path], name: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(name + "\n")
    except OSError:
        pass


def _cache_drop(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _run_resolver(resolver: Union[str, List[str]], cwd: str) -> str:
    """Return the resolver's stdout, stripped. Raises on any failure."""
    argv: List[str] = [os.path.expanduser(resolver)] if isinstance(resolver, str) else [str(a) for a in resolver]
    if not argv or not argv[0]:
        raise ValueError("profiles.resolver is empty")
    out = subprocess.run(argv + [cwd], capture_output=True, text=True, timeout=_RESOLVER_TIMEOUT, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"resolver exited {out.returncode}")
    return out.stdout.strip()


def resolve(block: dict, cwd: str) -> Resolution:
    """Run the resolver for ``cwd`` and look the name up. Never raises."""
    by_name: Dict[str, Any] = block["by_name"]
    try:
        name = _run_resolver(block.get("resolver", ""), cwd)
    except Exception as e:  # subprocess, OS and config errors alike: no profile
        return Resolution(None, None, f"resolver failed: {type(e).__name__}: {e}", cwd)
    if not name:
        return Resolution(None, None, "resolver returned no name", cwd)
    profile = by_name.get(name)
    if not isinstance(profile, dict):
        return Resolution(name, None, f"no profile named {name!r}", cwd)
    return Resolution(name, profile, "resolved", cwd)


# ---------------------------------------------------------------------------
# Applying a profile to the environment
# ---------------------------------------------------------------------------


def _target(profile: dict) -> str:
    target = str(profile.get("target", "")).strip().lower()
    if target in ("phoenix", "arize"):
        return target
    return "arize" if profile.get("space_id") else "phoenix"


def _read_key(profile: dict) -> str:
    """The backend credential. Raises when a configured key file cannot be read."""
    key_file = str(profile.get("api_key_file", "")).strip()
    if key_file:
        key = Path(os.path.expanduser(key_file)).read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"empty key file {key_file}")
        return key
    return str(profile.get("api_key", "")).strip()


def _flag(value: Any) -> str:
    return "true" if value is True or str(value).strip().lower() == "true" else "false"


def apply_env(profile: dict, environ: Optional[Dict[str, str]] = None) -> None:
    """Set the backend, project, logging and root variables for ``profile``.

    Raises on an unreadable key file so the caller can fail closed.
    """
    env = os.environ if environ is None else environ
    key = _read_key(profile)
    for name in _OWNED_ENV:
        env.pop(name, None)

    home = str(profile.get("harness_home", "")).strip()
    if home:
        env["ARIZE_HARNESS_HOME"] = os.path.expanduser(home)

    endpoint = str(profile.get("endpoint", "")).strip()
    project = str(profile.get("project_name", "")).strip()
    if _target(profile) == "phoenix":
        if endpoint:
            env["PHOENIX_ENDPOINT"] = endpoint
        if key:
            env["PHOENIX_API_KEY"] = key
        if project:
            env["PHOENIX_PROJECT_NAME"] = project
    else:
        if key:
            env["ARIZE_API_KEY"] = key
        space_id = str(profile.get("space_id", "")).strip()
        if space_id:
            env["ARIZE_SPACE_ID"] = space_id
        if endpoint:
            env["ARIZE_OTLP_ENDPOINT"] = endpoint
        if project:
            env["ARIZE_PROJECT_NAME"] = project

    logging = profile.get("logging")
    if isinstance(logging, dict):
        for config_key, env_key in _LOG_FLAGS:
            if config_key in logging:
                env[env_key] = _flag(logging[config_key])

    user_id = str(profile.get("user_id", "")).strip()
    if user_id:
        env["ARIZE_USER_ID"] = user_id

    attributes = profile.get("attributes")
    if isinstance(attributes, dict) and attributes:
        pairs = [f"{k}={v}" for k, v in attributes.items() if k and "=" not in str(k) and "," not in f"{k}{v}"]
        existing = env.get("OTEL_RESOURCE_ATTRIBUTES", "").strip()
        env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join([existing] * bool(existing) + pairs)


def describe(name: Optional[str], profile: Optional[dict], reason: str, cwd: str) -> List[str]:
    """Lines for ``arize-harness profile``: the decision and its inputs, no secret values."""
    if profile is None:
        return [f"skip: {reason}", f"cwd: {cwd}"]
    key_file = str(profile.get("api_key_file", "")).strip()
    key_present = bool(profile.get("api_key")) or (bool(key_file) and Path(os.path.expanduser(key_file)).is_file())
    logging_cfg = profile.get("logging")
    logging: dict = logging_cfg if isinstance(logging_cfg, dict) else {}
    flags = "/".join(_flag(logging.get(k, True)) for k, _ in _LOG_FLAGS)
    return [
        f"profile: {name}",
        f"cwd: {cwd}",
        f"target: {_target(profile)}",
        f"endpoint: {profile.get('endpoint', '')}",
        f"project: {profile.get('project_name', '')}",
        f"harness home: {os.path.expanduser(str(profile.get('harness_home', ''))) or '(default)'}",
        f"api key present: {key_present}",
        f"content logging (prompts/details/content): {flags}",
    ]


# ---------------------------------------------------------------------------
# Entry point for the dispatcher
# ---------------------------------------------------------------------------


def apply_profile(harness: str, event: Optional[str], rest: List[str]) -> Decision:
    """Decide and apply the profile for this hook run.

    Returns a Decision. Reads stdin only when a ``profiles`` block exists; the
    consumed bytes come back in ``Decision.stdin`` for the caller to replay.
    """
    root = config_root()
    block = load_profiles(root)
    if block is None:
        return Decision(True)

    buffer = getattr(sys.stdin, "buffer", None)
    try:
        raw = buffer.read() if buffer is not None else sys.stdin.read().encode("utf-8")
    except (OSError, ValueError):
        raw = b""
    payload = _parse_payload(raw, rest)
    session_id = str(payload.get("session_id", "") or "")
    cache = _cache_file(root, harness, session_id)
    ttl = float(block.get("cache_ttl_seconds", _DEFAULT_CACHE_TTL) or _DEFAULT_CACHE_TTL)
    announce = _is_session_start(event, payload)
    by_name: Dict[str, Any] = block["by_name"]

    forced = os.environ.get(_ENV_PROFILE, "").strip()
    if forced:
        name: Optional[str] = forced
        profile = by_name.get(forced)
        reason = (
            "forced by ARIZE_PROFILE" if isinstance(profile, dict) else f"ARIZE_PROFILE names no profile {forced!r}"
        )
    else:
        cached = _cache_read(cache, ttl)
        if cached is not None:
            name, profile, reason = cached, by_name.get(cached), "cached"
        else:
            cwd = str(payload.get("cwd", "") or "") or os.getcwd()
            res = resolve(block, cwd)
            name, profile, reason = res.name, res.profile, res.reason
            if name:
                _cache_write(cache, name)
    if _is_session_end(event, payload):
        _cache_drop(cache)

    fallback = str(block.get("fallback", "") or "").strip()
    if not isinstance(profile, dict) and fallback and not forced:
        candidate = by_name.get(fallback)
        if isinstance(candidate, dict):
            name, profile, reason = fallback, candidate, f"fallback after: {reason}"

    if not isinstance(profile, dict):
        if announce or "failed" in reason:
            _log(root, harness, event, f"skip session {session_id or '-'}: {reason}")
        return Decision(False, raw, None, reason)

    try:
        apply_env(profile)
    except (OSError, ValueError) as e:
        _log(root, harness, event, f"skip session {session_id or '-'}: profile {name!r} not usable: {e}")
        return Decision(False, raw, None, f"profile {name!r} not usable: {e}")
    if announce:
        _log(root, harness, event, f"session {session_id or '-'} -> profile {name!r} ({reason})")
    return Decision(True, raw, name, reason)
