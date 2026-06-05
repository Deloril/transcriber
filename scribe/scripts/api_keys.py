"""CLI for managing API keys.

Usage::

    python -m scribe.scripts.api_keys list
    python -m scribe.scripts.api_keys mint <label>
    python -m scribe.scripts.api_keys revoke <key-id>

Why a CLI rather than a UI:

* Minting a key returns the plaintext exactly once. Putting that on
  a web page where the browser caches it / a screenshot tool grabs
  it / a screen-share viewer reads it is asking for trouble. Running
  in the user's terminal keeps it on the local box.
* Listing keys never needs to print plaintext (only labels + ids +
  last-used timestamps). So if you've forgotten which keys are out
  there, ``list`` is safe.
* Revoke is destructive but never accidentally so — you have to type
  the id.

Programmatic callers (e.g. tests) should import :mod:`scribe.api_auth`
directly instead of shelling out to this CLI.
"""

from __future__ import annotations

import sys

from scribe import api_auth


_USAGE = """\
Usage:
  python -m scribe.scripts.api_keys list
  python -m scribe.scripts.api_keys mint <label>
  python -m scribe.scripts.api_keys revoke <key-id>
"""


def _list_keys() -> int:
    keys = api_auth.load_keys()
    if not keys:
        print("(no API keys configured)")
        return 0
    print(f"{'ID':<14} {'LABEL':<24} {'CREATED':<22} LAST USED")
    for k in keys:
        last = k.last_used_at or "(never)"
        print(f"{k.id:<14} {k.label[:22]:<24} {k.created_at:<22} {last}")
    return 0


def _mint(label: str) -> int:
    if not label or not label.strip():
        print("error: label is required", file=sys.stderr)
        return 2
    record, plaintext = api_auth.mint_api_key(label)
    print(f"Minted key {record.id} ({record.label!r}).")
    print()
    print("This is the only time the plaintext will be shown:")
    print()
    print(f"  {plaintext}")
    print()
    print("Use it in the Authorization header:")
    print()
    print(f"  curl -H 'Authorization: Bearer {plaintext}' \\")
    print("       http://127.0.0.1:8765/api/v1/")
    return 0


def _revoke(key_id: str) -> int:
    if not key_id:
        print("error: key id is required", file=sys.stderr)
        return 2
    if api_auth.revoke_api_key(key_id):
        print(f"Revoked key {key_id}.")
        return 0
    print(f"No key with id {key_id!r}.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(_USAGE, file=sys.stderr)
        return 2
    cmd = args[0]
    rest = args[1:]
    if cmd in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    if cmd == "list":
        return _list_keys()
    if cmd == "mint":
        if len(rest) != 1:
            print(_USAGE, file=sys.stderr)
            return 2
        return _mint(rest[0])
    if cmd == "revoke":
        if len(rest) != 1:
            print(_USAGE, file=sys.stderr)
            return 2
        return _revoke(rest[0])
    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
