"""Global (project-less) AI backend config.

Most AI features in Scribe are scoped to a *project* — the academic
coding flow loads a project first, then the per-project AI backend
config from ``project.settings["ai_backend"]``.

A handful of features are not project-bound. The transcript editor's
"Tidy speech with AI" button is the canonical example: a researcher is
editing a single transcription, with no project picked, and wants to
run the local LLM on a run of speech. There's no project to read
config from.

This module persists a *global* :class:`scribe.ai_backend.BackendConfig`
under ``$SCRIBE_HOME/ai_backend.json`` (default: ``~/.scribe/``) using
exactly the same scalar shape as the project-scoped variant. The same
``BackendConfig.from_dict`` / ``to_dict`` round-trip is reused so users
can copy a config between project and global without surprises.

Headers are stored alongside the scalars in the same file (under the
``"extra_headers"`` key) — there's no nested-settings constraint here
like there is on :class:`scribe.projects.Project`, so the simpler
single-file shape wins.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .ai_backend import BackendConfig, BackendValidationError


# Cap the file size on read — defends against an obviously-bad on-disk
# value (e.g. someone redirecting the path to a multi-gigabyte file)
# without being so small it limits real config growth.
_MAX_FILE_BYTES = 64 * 1024


def default_config_dir() -> Path:
    """Return the directory the global config file lives under.

    Honours ``SCRIBE_HOME`` if set (used by tests + power users to
    isolate state); otherwise falls back to ``~/.scribe``. The
    directory is *not* created here — :func:`save_global_config` makes
    it on demand so a no-op import never touches the filesystem.
    """
    override = os.environ.get("SCRIBE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".scribe"


def config_path(config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / "ai_backend.json"


def load_global_config(config_dir: Path | None = None) -> BackendConfig:
    """Read the on-disk config; return defaults if the file is absent.

    A missing file is normal (first run, or never configured) — we
    return :class:`BackendConfig` defaults rather than raising, so
    downstream callers get a connection target and only fail when they
    actually try to talk to a backend that isn't there.

    A *malformed* file raises :class:`BackendValidationError` so the
    user knows their saved config is broken rather than silently
    falling back. Empty / blank file is treated as "not configured".
    """
    path = config_path(config_dir)
    if not path.exists():
        return BackendConfig.from_dict({})
    try:
        size = path.stat().st_size
    except OSError as e:
        raise BackendValidationError(f"Could not stat {path}: {e}") from e
    if size > _MAX_FILE_BYTES:
        raise BackendValidationError(
            f"{path} is suspiciously large ({size} bytes); refusing to load"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise BackendValidationError(f"Could not read {path}: {e}") from e
    if not raw.strip():
        return BackendConfig.from_dict({})
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BackendValidationError(
            f"{path} is not valid JSON: {e}"
        ) from e
    if not isinstance(data, dict):
        raise BackendValidationError(
            f"{path} must contain a JSON object; got {type(data).__name__}"
        )
    headers_raw = data.pop("extra_headers", None)
    headers: Mapping[str, str] | None
    if headers_raw is None:
        headers = None
    elif isinstance(headers_raw, dict):
        headers = {str(k): str(v) for k, v in headers_raw.items()}
    else:
        raise BackendValidationError(
            f"{path}: 'extra_headers' must be an object"
        )
    return BackendConfig.from_dict(data, extra_headers=headers)


def save_global_config(
    config: BackendConfig, config_dir: Path | None = None,
) -> Path:
    """Write the config atomically.

    Validates first, so a bad config never lands on disk. Writes via a
    sibling temp file + ``os.replace`` so a half-written file can't be
    observed even if the process is killed mid-write.
    """
    config.validate()
    target = config_path(config_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = dict(config.to_dict())
    if config.extra_headers:
        body["extra_headers"] = {k: v for k, v in config.extra_headers}
    fd, tmp_path = tempfile.mkstemp(
        prefix=".ai_backend.", suffix=".tmp", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return target
