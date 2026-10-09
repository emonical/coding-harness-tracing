"""``arize-harness`` command line entry point.

Usage:
    arize-harness hook <harness> [<event>] [args...]
    arize-harness hook --list
    arize-harness profile [<cwd>]

``hook`` runs the same handler as the matching ``arize-hook-*`` command, with
stdin and any remaining arguments passed through untouched. When ``config.json``
has a ``profiles`` block (see ``core.profiles``), the working directory of the
run picks the backend first, and a run that resolves to no profile exits 0
without loading a handler. ``profile`` prints that decision for a directory.

It is invoked on every hook event, so it must start fast: keep this module's
imports to the standard library, and import anything heavier (``core.setup``,
which pulls in ``python-dotenv``) inside the subcommand that needs it.
"""

import importlib
import io
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, TextIO

from core.hook_table import ALIASES, EVENT_HOOKS, SINGLE_HOOKS, canonical_harness, iter_hooks, legacy_entry_point
from core.profiles import apply_profile, describe, load_profiles, resolve

_USAGE = "usage: arize-harness hook <harness> [<event>] [args...] | arize-harness hook --list | arize-harness profile [<cwd>]"


def _stderr() -> TextIO:
    """The harness's stderr, even if an adapter has already redirected ``sys.stderr`` to its log."""
    common = sys.modules.get("core.common")
    log_fh = getattr(common, "_redirected_log_fh", None)
    original = getattr(common, "_original_stderr", None)
    if log_fh is not None and sys.stderr is log_fh and original is not None:
        return original
    return sys.stderr


def _warn(message: str) -> None:
    _stderr().write(f"[arize] {message}\n")


def _default_log_file(harness: str) -> Path:
    """The log file a harness's adapter would set as ``ARIZE_LOG_FILE``."""
    from core.constants import HARNESSES, LOG_DIR

    # HARNESSES is keyed by install.sh names (claude-code, not claude). Kiro and
    # Devin are not in it; their adapters use LOG_DIR/<name>.log.
    key = next((alias for alias, name in ALIASES.items() if name == harness), harness)
    metadata = HARNESSES.get(key)
    return metadata["default_log_file"] if metadata else LOG_DIR / f"{key}.log"


def _log_load_failure(harness: str, target: str) -> Optional[str]:
    """Append the current traceback to the harness log and return its path.

    The hook still exits 0, so this log is the only place a broken install
    (a missing dependency, a syntax error in a handler) shows up.
    """
    try:
        import traceback

        path = Path(os.environ.get("ARIZE_LOG_FILE") or _default_log_file(harness))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log:
            log.write(f"[arize:error] could not load hook {target}\n{traceback.format_exc()}")
        return str(path)
    except Exception:
        return None


def _list_hooks() -> int:
    for harness, event, _target in iter_hooks():
        print(harness if event is None else f"{harness} {event}")
    return 0


def _hook(args: List[str]) -> Any:
    """Dispatch ``hook`` arguments to a handler.

    Unknown names and import failures print one line to stderr and exit 0:
    a hook must never block the harness that runs it.
    """
    if not args:
        _warn(_USAGE)
        return 0
    if args[0] == "--list":
        return _list_hooks()

    harness = canonical_harness(args[0])
    event: Optional[str] = None
    if harness in SINGLE_HOOKS:
        target = SINGLE_HOOKS[harness]
        rest = args[1:]
    elif harness in EVENT_HOOKS:
        if len(args) < 2:
            _warn(f"missing event for harness '{args[0]}' (see arize-harness hook --list)")
            return 0
        event = args[1]
        events = EVENT_HOOKS[harness]
        if event not in events:
            _warn(f"unknown event '{event}' for harness '{args[0]}' (see arize-harness hook --list)")
            return 0
        target = events[event]
        rest = args[2:]
    else:
        _warn(f"unknown harness '{args[0]}' (see arize-harness hook --list)")
        return 0

    # Pick the backend for this run before the handler module loads: the handler's
    # paths and credentials are fixed at its import time from the environment.
    decision = apply_profile(harness, event, rest)
    if not decision.proceed:
        return 0
    if decision.stdin is not None:
        sys.stdin = io.TextIOWrapper(io.BytesIO(decision.stdin), encoding="utf-8")

    module_name, _, function_name = target.partition(":")
    try:
        handler = getattr(importlib.import_module(module_name), function_name)
    except (Exception, SystemExit) as e:  # a module calling sys.exit() at import must not block either
        log_path = _log_load_failure(harness, target)
        _warn(f"could not load hook {target}: {e}" + (f" (traceback in {log_path})" if log_path else ""))
        return 0

    # Handlers read their own arguments from sys.argv (Codex's notify payload is
    # sys.argv[1]), so present them exactly as the legacy command would see them.
    sys.argv = [legacy_entry_point(harness, event), *rest]
    return handler()


def _profile(args: List[str]) -> int:
    """Print how a run from ``cwd`` (default: the current directory) would be routed."""
    if len(args) > 1:
        _warn(_USAGE)
        return 0
    cwd = os.path.abspath(args[0]) if args else os.getcwd()
    block = load_profiles()
    if block is None:
        print("no profiles configured: every run uses the shared config.json and environment")
        return 0
    forced = os.environ.get("ARIZE_PROFILE", "").strip()
    if forced:
        profile = block["by_name"].get(forced)
        lines = describe(
            forced, profile if isinstance(profile, dict) else None, f"ARIZE_PROFILE names no profile {forced!r}", cwd
        )
    else:
        res = resolve(block, cwd)
        lines = describe(res.name, res.profile, res.reason, cwd)
    print("\n".join(lines))
    return 0


def main(argv: Optional[List[str]] = None) -> Any:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "hook":
        return _hook(args[1:])
    if args and args[0] == "profile":
        return _profile(args[1:])
    # Exit 0, not the conventional 2: Claude Code treats a hook's exit 2 as a
    # blocking error, so a mistyped hook command would block every tool call.
    _warn(_USAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
