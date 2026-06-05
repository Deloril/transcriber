"""API key auth for the public ``/api/v1/`` namespace.

The local UI under ``/`` is unauthenticated by design — it's a
single-user desktop app. The ``/api/v1/`` namespace is the opposite:
it's intended for *machines* (scripted Claude clients, MCP servers,
the user's own automation) and so it requires a bearer token.

Keys live in ``$SCRIBE_HOME/api_keys.json`` (default: ``~/.scribe/``)
in this shape::

    {
      "keys": [
        {
          "id": "key-abc123def4",         # short id (12 hex chars)
          "label": "claude-mcp",          # human label
          "hash": "sha256:<hex>",         # sha256 of the secret
          "created_at": "2026-06-05T10:00:00Z",
          "last_used_at": "2026-06-05T10:42:11Z" | null
        },
        ...
      ]
    }

Only hashes touch disk. The plaintext is shown once at creation
time and never recoverable — same shape as GitHub / Stripe /
every other API-key store. Tokens travel as
``Authorization: Bearer <plaintext>``.

Two helpers users need:

* :func:`mint_api_key` — generate a new key, persist its hash,
  return the plaintext (the caller's only chance to capture it).
* :func:`verify_api_key` — verify a presented bearer token against
  the on-disk store; updates ``last_used_at`` on success.

A FastAPI dependency :func:`require_api_key` wires the verify
helper to the request handler.

There is no UI for minting / listing keys yet — issue them via
the CLI (``python -m scribe.scripts.api_keys``). That keeps the
attack surface small: a bug in the keys-list page can't leak
keys, because the page doesn't exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Token format: ``sk_scribe_<32 url-safe chars>``. The prefix lets
# the user (and tools like git-secrets / cloud DLP) recognise the
# token at a glance, and reserves the namespace cleanly even if we
# add a second token type later.
TOKEN_PREFIX = "sk_scribe_"
TOKEN_BODY_LEN = 32

# Where the JSON file lives. Honours ``SCRIBE_HOME`` so tests don't
# touch the developer's real ``~/.scribe``.
ENV_HOME = "SCRIBE_HOME"


def _scribe_home() -> Path:
    raw = os.environ.get(ENV_HOME, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".scribe"


def _keys_path(home: Path | None = None) -> Path:
    return (home or _scribe_home()) / "api_keys.json"


def _hash_token(token: str) -> str:
    """Return ``sha256:<hex>`` for a plaintext bearer token.

    sha256 is fine here — these are 256-bit random tokens, so brute
    force is hopeless and we don't need a slow KDF. (Bcrypt's value
    is for low-entropy human passwords; an API key isn't one.)
    """
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _new_key_id() -> str:
    """Short opaque id for the keys file. 12-hex same shape as job ids."""
    return f"key-{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


@dataclass
class ApiKeyRecord:
    """One on-disk key record. Plaintext token is NEVER stored."""

    id: str
    label: str
    hash: str
    created_at: str
    last_used_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "hash": self.hash,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ApiKeyRecord":
        return cls(
            id=str(d.get("id", "")),
            label=str(d.get("label", "")),
            hash=str(d.get("hash", "")),
            created_at=str(d.get("created_at", "")),
            last_used_at=d.get("last_used_at") or None,
        )


def load_keys(home: Path | None = None) -> list[ApiKeyRecord]:
    """Read the keys file. Missing file → empty list (fresh install)."""
    path = _keys_path(home)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Corrupt file shouldn't take the server down; surface as
        # "no keys configured" so the user resets the file.
        return []
    raw = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[ApiKeyRecord] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(ApiKeyRecord.from_dict(item))
    return out


def save_keys(records: list[ApiKeyRecord], home: Path | None = None) -> Path:
    """Write the keys file atomically (sibling tmp + rename).

    Sets the file mode to 0600 so the hashes aren't world-readable
    on a multi-user box. Raises if the parent directory can't be
    created.
    """
    path = _keys_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"keys": [r.to_dict() for r in records]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        # Best-effort on platforms that don't honour chmod (Windows).
        pass
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# Mint + verify
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mint_api_key(
    label: str,
    *,
    home: Path | None = None,
    now: str | None = None,
) -> tuple[ApiKeyRecord, str]:
    """Generate + persist a new key. Returns (record, plaintext).

    The plaintext is the only reason to call this function; the
    caller MUST present it to the user immediately. It cannot be
    recovered later — only the sha256 lands on disk.
    """
    plaintext = TOKEN_PREFIX + secrets.token_urlsafe(TOKEN_BODY_LEN)
    record = ApiKeyRecord(
        id=_new_key_id(),
        label=str(label or "").strip()[:200] or "unnamed",
        hash=_hash_token(plaintext),
        created_at=now or _now_iso(),
    )
    keys = load_keys(home)
    keys.append(record)
    save_keys(keys, home)
    return record, plaintext


def revoke_api_key(key_id: str, *, home: Path | None = None) -> bool:
    """Drop a key by id. Returns True if a key was removed."""
    keys = load_keys(home)
    new = [k for k in keys if k.id != key_id]
    if len(new) == len(keys):
        return False
    save_keys(new, home)
    return True


def verify_api_key(
    bearer: str,
    *,
    home: Path | None = None,
    now: str | None = None,
) -> ApiKeyRecord | None:
    """Return the key record for a presented plaintext, or ``None``.

    Constant-time comparison via :func:`secrets.compare_digest` so
    we don't leak hash equality timing. Updates ``last_used_at`` on
    a hit so the user can tell if a key is in use without auditing
    request logs.
    """
    if not bearer or not isinstance(bearer, str):
        return None
    expected = _hash_token(bearer)
    keys = load_keys(home)
    matched: ApiKeyRecord | None = None
    for k in keys:
        # Constant-time compare so the hash check doesn't leak
        # which key is closest character-wise.
        if secrets.compare_digest(expected, k.hash):
            matched = k
            break
    if matched is None:
        return None
    matched.last_used_at = now or _now_iso()
    # Persist the touch. Best-effort — a write failure shouldn't
    # block a verified request.
    try:
        save_keys(keys, home)
    except OSError:
        pass
    return matched


# --------------------------------------------------------------------------- #
# FastAPI dependency
# --------------------------------------------------------------------------- #


def require_api_key():
    """Build a FastAPI dependency that gates an endpoint behind a key.

    Usage at a route::

        @router.get("/x", dependencies=[Depends(require_api_key())])

    Imports FastAPI lazily so this module stays test-pure.
    """
    from fastapi import Header, HTTPException

    async def _dep(
        authorization: str | None = Header(default=None),
    ) -> ApiKeyRecord:
        if not authorization:
            raise HTTPException(
                401,
                "Missing Authorization header. "
                "Send 'Authorization: Bearer <api-key>'.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                401,
                "Authorization scheme must be Bearer.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        record = verify_api_key(token.strip())
        if record is None:
            raise HTTPException(
                401, "Invalid or unknown API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return record

    return _dep
