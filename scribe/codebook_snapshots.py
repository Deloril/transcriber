"""Named codebook snapshots (F9.3).

Per PLANNING.md F9.3:

  > Named codebook snapshots ("Initial coding done 2026-04-12"). Reports
  > can be regenerated against any snapshot.

What a snapshot is
------------------

A snapshot is a **named, immutable, dated bookmark** of an entire
codebook at a moment in time. Researchers take one when a methodological
phase ends ("initial coding done", "before unlocking", "submission
draft") so they can later regenerate any report — codebook export,
retrieval — *as the codebook stood at that moment*. The name and
description give the bookmark its meaning; the embedded code state and
per-code version pointer give it reproducibility.

Why this is its own module, not part of F2.2
--------------------------------------------

F2.2 keeps a per-code, append-only log of every definition edit. That
is the *fine-grained* history: "what did this one code look like on
date X?". F9.3 is the *coarse-grained* sibling: "what did the *whole*
codebook look like on date X?". A retrieval report regenerated against
a snapshot needs the latter — the closed set of codes that existed at
that time, and which version of each was current.

Two complementary lenses; one append-only history, the other a named
checkpoint set.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F9.3 ships the data model +
  writer + reader + reconstruction helper. Routes can be added by a
  later iteration. Mirrors the staged approach in F9.1 / F9.2.
* **Stand-alone, pure Python.** No FastAPI, no engine imports. Reads
  via :mod:`scribe.codes` and :mod:`scribe.code_versions`; optionally
  writes an F9.1 :class:`scribe.event_log.Event` when a snapshot is
  created so the global audit log shows the bookmark.
* **Append-only by convention.** Snapshots are saved once, then
  treated as immutable. There's no ``update_snapshot``. There's no
  ``delete_snapshot`` exposed: deleting a snapshot would cut the
  reproducibility chain. If a researcher mints one by mistake, they
  can mint a follow-up correcting one (snapshot of a snapshot is a
  later concern, not F9.3's).

On-disk layout
--------------

::

    projects/<project_id>/
      snapshots/
        <snapshot_id>.json    # one file per snapshot; never modified

Files are named by snapshot id (12-char hex), matching every other
entity store in the project. ``list_snapshots`` sorts by ``created_at``
ascending so the natural reading order is creation order.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .code_versions import (
    CODE_VERSION_ID_RE,
    latest_code_version,
)
from .codes import (
    CODE_ID_RE,
    Code,
    list_codes,
)
from .coders import CODER_ID_RE
from .event_log import (
    EVENT_ACTION_SNAPSHOT,
    EVENT_ENTITY_SNAPSHOT,
    EVENT_ID_RE,
    record_event,
)
from .projects import (
    CODEBOOK_STAGES,
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
    load_project,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Snapshot ids share Scribe's standard 12-char hex shape.
SNAPSHOT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory name. Sibling of ``codes/`` / ``events/`` /
# ``code_versions/``; the existing ``delete_project`` cascade reaches
# it for free.
SNAPSHOTS_DIRNAME = "snapshots"

# Name / description bounds. The name is the human label
# ("Initial coding done 2026-04-12") so it's short; description is a
# free-form sentence-or-two for the *why*.
MAX_NAME_LEN = 200
MAX_DESCRIPTION_LEN = 4000

# Hard ceiling on serialised snapshot size. Codebooks with large
# exemplar lists can be a few hundred KiB; this guards against runaway
# UI bugs without being so tight that real projects bump it. 2 MiB is
# generous — a typical CGT codebook with 80 codes serialises well
# under 200 KiB.
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024

# Cardinality cap. A real grounded-theory project converges to maybe
# 50–80 focused codes; line-by-line coding can balloon to several
# hundred. 4096 is well over any realistic ceiling and still well under
# the bytes cap.
MAX_CODES_PER_SNAPSHOT = 4096


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Snapshot:
    """One named codebook snapshot (F9.3).

    Captures the closed set of codes that existed in the project at
    a moment in time, plus the latest definition-version id of each so
    F9.2's ``definition-at-apply`` resolver has a fixed point of
    reference when regenerating a report against this snapshot.

    Fields
    ------
    id
        12-char hex snapshot id. Mint with :func:`new_snapshot_id`.
    project_id
        12-char hex project id this snapshot belongs to.
    name
        Required, non-empty. The user-visible bookmark label, e.g.
        ``"Initial coding done 2026-04-12"``.
    description
        Optional free-form notes (the *why*).
    codebook_stage
        The project's ``codebook_stage`` at the moment the snapshot
        was taken — captured separately from the code dicts so a
        report can render "snapshotted at *focused* stage" without
        re-loading the project.
    actor_coder_id
        Optional 12-char hex coder id of the human who created the
        snapshot. Empty for system-created snapshots.
    event_id
        Optional 12-char hex id of the F9.1 :class:`Event` that
        recorded this snapshot. Empty when the caller opted out of
        emitting an event (e.g. testing, importers). PLANNING:
        "snapshot ids reference the event that produced them".
    codes
        List of full :class:`Code` ``to_dict()`` payloads, one per
        code that existed at snapshot time. Stored embedded so a
        snapshot is self-contained — even if the live ``codes/<id>``
        files are later deleted (retire / merge / split), the
        snapshot still reconstructs.
    code_versions
        Dict ``{code_id -> code_version_id}`` of the latest version
        recorded for each code at snapshot time (per F2.2's append-
        only version log). May omit codes that have no version yet
        (legacy data, or codes saved without
        ``save_code_with_version``).
    created_at
        ISO-8601 UTC timestamp; set at construction time.
    """

    id: str
    project_id: str
    name: str
    description: str = ""
    codebook_stage: str = "initial"
    actor_coder_id: str = ""
    event_id: str = ""
    codes: list[dict[str, Any]] = field(default_factory=list)
    code_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        name: str,
        description: str = "",
        codebook_stage: str = "initial",
        actor_coder_id: str = "",
        event_id: str = "",
        codes: Iterable[Mapping[str, Any] | Code] | None = None,
        code_versions: Mapping[str, str] | None = None,
        snapshot_id: str | None = None,
        now: str | None = None,
    ) -> "Snapshot":
        """Build a fresh :class:`Snapshot` and validate it.

        ``codes`` may be supplied as ``Code`` instances or as their
        ``to_dict()`` shape. Either way they're normalised through
        :class:`Code.from_dict` / :class:`Code.to_dict` so the
        on-disk shape is canonical and validated.
        """
        normalised_codes: list[dict[str, Any]] = []
        for c in codes or []:
            if isinstance(c, Code):
                normalised_codes.append(c.to_dict())
            elif isinstance(c, Mapping):
                # Round-trip through Code so the snapshot fails fast
                # if a caller hands us garbage.
                rebuilt = Code.from_dict(dict(c))
                normalised_codes.append(rebuilt.to_dict())
            else:
                raise ProjectValidationError(
                    "Snapshot.codes entries must be Code instances or "
                    f"to_dict()-shaped mappings; got {type(c).__name__}"
                )
        ts = now or utcnow_iso()
        s = cls(
            id=snapshot_id or new_snapshot_id(),
            project_id=project_id,
            name=name,
            description=description,
            codebook_stage=codebook_stage,
            actor_coder_id=actor_coder_id or "",
            event_id=event_id or "",
            codes=normalised_codes,
            code_versions=dict(code_versions or {}),
            created_at=ts,
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        # asdict() does the right thing here: codes are already plain
        # dicts and code_versions is a plain dict.
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Snapshot":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("Snapshot payload must be an object")
        for required in ("id", "project_id", "name"):
            if required not in d:
                raise ProjectValidationError(
                    f"Snapshot payload missing required key: {required}"
                )
        raw_codes = d.get("codes") or []
        if not isinstance(raw_codes, list):
            raise ProjectValidationError("Snapshot.codes must be a list")
        # Defensive copy so a caller can't mutate the on-disk payload by
        # holding the dict reference.
        codes_copy: list[dict[str, Any]] = []
        for c in raw_codes:
            if not isinstance(c, Mapping):
                raise ProjectValidationError(
                    "Snapshot.codes entries must be objects"
                )
            codes_copy.append(json.loads(json.dumps(dict(c))))
        raw_versions = d.get("code_versions")
        if raw_versions is None:
            raw_versions = {}
        if not isinstance(raw_versions, Mapping):
            raise ProjectValidationError(
                "Snapshot.code_versions must be an object"
            )
        s = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            name=str(d.get("name", "") or ""),
            description=str(d.get("description", "") or ""),
            codebook_stage=str(d.get("codebook_stage", "initial") or "initial"),
            actor_coder_id=str(d.get("actor_coder_id", "") or ""),
            event_id=str(d.get("event_id", "") or ""),
            codes=codes_copy,
            code_versions={str(k): str(v) for k, v in raw_versions.items()},
            created_at=str(d.get("created_at", "") or ""),
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not SNAPSHOT_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid snapshot id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        name = (self.name or "").strip()
        if not name:
            raise ProjectValidationError("Snapshot.name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Snapshot.name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist trimmed so on-disk state is canonical.
        self.name = name
        if len(self.description) > MAX_DESCRIPTION_LEN:
            raise ProjectValidationError(
                f"Snapshot.description must be ≤ {MAX_DESCRIPTION_LEN} chars"
            )
        if self.codebook_stage not in CODEBOOK_STAGES:
            raise ProjectValidationError(
                f"Snapshot.codebook_stage must be one of {CODEBOOK_STAGES}; "
                f"got {self.codebook_stage!r}"
            )
        if self.actor_coder_id and not CODER_ID_RE.match(self.actor_coder_id):
            raise ProjectValidationError(
                "Snapshot.actor_coder_id must be 12-char hex or empty; "
                f"got {self.actor_coder_id!r}"
            )
        if self.event_id and not EVENT_ID_RE.match(self.event_id):
            raise ProjectValidationError(
                "Snapshot.event_id must be 12-char hex or empty; "
                f"got {self.event_id!r}"
            )
        if not isinstance(self.codes, list):
            raise ProjectValidationError("Snapshot.codes must be a list")
        if len(self.codes) > MAX_CODES_PER_SNAPSHOT:
            raise ProjectValidationError(
                f"Snapshot.codes exceeds {MAX_CODES_PER_SNAPSHOT} entries"
            )
        # Sanity-check each code dict by round-tripping through Code.
        # This rejects malformed snapshot files at load time rather
        # than at use time.
        seen_ids: set[str] = set()
        for i, c in enumerate(self.codes):
            if not isinstance(c, dict):
                raise ProjectValidationError(
                    f"Snapshot.codes[{i}] must be an object"
                )
            try:
                rebuilt = Code.from_dict(c)
            except ProjectValidationError as e:
                raise ProjectValidationError(
                    f"Snapshot.codes[{i}] is not a valid Code payload: {e}"
                ) from e
            if rebuilt.project_id != self.project_id:
                raise ProjectValidationError(
                    f"Snapshot.codes[{i}].project_id "
                    f"{rebuilt.project_id!r} does not match "
                    f"snapshot.project_id {self.project_id!r}"
                )
            if rebuilt.id in seen_ids:
                raise ProjectValidationError(
                    f"Snapshot.codes contains duplicate code id "
                    f"{rebuilt.id!r}"
                )
            seen_ids.add(rebuilt.id)
        # Validate code_versions: keys must be 12-char hex code ids that
        # appear in ``self.codes``; values must be 12-char hex version
        # ids. Unknown code ids are rejected (a snapshot pinning a
        # version to a code that isn't in the snapshot is broken).
        if not isinstance(self.code_versions, dict):
            raise ProjectValidationError(
                "Snapshot.code_versions must be an object"
            )
        for k, v in self.code_versions.items():
            if not CODE_ID_RE.match(k):
                raise ProjectValidationError(
                    f"Snapshot.code_versions key {k!r} must be 12-char hex"
                )
            if not CODE_VERSION_ID_RE.match(v):
                raise ProjectValidationError(
                    f"Snapshot.code_versions[{k!r}] = {v!r} must be 12-char hex"
                )
            if k not in seen_ids:
                raise ProjectValidationError(
                    f"Snapshot.code_versions references unknown code id {k!r} "
                    "(not present in Snapshot.codes)"
                )
        # Combined size ceiling. Computed last so the cheaper checks
        # have already rejected obvious garbage.
        encoded = json.dumps(
            {"codes": self.codes, "code_versions": self.code_versions},
            ensure_ascii=False,
        )
        if len(encoded.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise ProjectValidationError(
                f"Snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes "
                "(codes + code_versions combined)"
            )


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_snapshot_id() -> str:
    """Mint a new 12-char hex snapshot id (matches Scribe's standard shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def snapshots_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's snapshots."""
    return project_dir(projects_root, project_id) / SNAPSHOTS_DIRNAME


def snapshot_state_path(
    projects_root: Path, project_id: str, snapshot_id: str
) -> Path:
    """Return the path for a single snapshot JSON file."""
    if not SNAPSHOT_ID_RE.match(snapshot_id):
        raise ProjectValidationError(f"Invalid snapshot id: {snapshot_id!r}")
    return snapshots_dir(projects_root, project_id) / f"{snapshot_id}.json"


def save_snapshot(projects_root: Path, snapshot: Snapshot) -> Path:
    """Persist a snapshot atomically; refuses to overwrite an existing id.

    Snapshots are append-only — once written, the on-disk file *is* the
    bookmark. Re-saving an existing id raises :class:`FileExistsError`;
    callers should mint a fresh id with :func:`new_snapshot_id` (or use
    :func:`create_codebook_snapshot`, which mints automatically).
    """
    snapshot.validate()
    parent = project_dir(projects_root, snapshot.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving snapshots."
        )
    sd = snapshots_dir(projects_root, snapshot.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = snapshot_state_path(projects_root, snapshot.project_id, snapshot.id)
    if target.exists():
        raise FileExistsError(
            f"Snapshot {snapshot.id} already exists; snapshots are append-only"
        )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_snapshot(
    projects_root: Path, project_id: str, snapshot_id: str
) -> Snapshot:
    """Load a snapshot by id. Raises ``FileNotFoundError`` if missing."""
    p = snapshot_state_path(projects_root, project_id, snapshot_id)
    if not p.exists():
        raise FileNotFoundError(f"No snapshot at {p}")
    return Snapshot.from_dict(json.loads(p.read_text()))


def list_snapshots(
    projects_root: Path, project_id: str
) -> list[Snapshot]:
    """List all snapshots in a project, ordered by ``created_at`` ascending.

    Skips files that don't parse as a valid :class:`Snapshot` so a
    single corrupt file doesn't break the bookmarks view; this matches
    the rest of the F-feature stack (codes, sources, events, etc.).
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    sd = snapshots_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[Snapshot] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sid = f.stem
        if not SNAPSHOT_ID_RE.match(sid):
            continue
        try:
            out.append(Snapshot.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda s: (s.created_at, s.id))
    return out


def count_snapshots(projects_root: Path, project_id: str) -> int:
    """Return the number of well-formed snapshots in a project.

    Cheap counter — useful for the UI badge ("3 saved snapshots")
    without paying the parse cost of :func:`list_snapshots`.
    """
    sd = snapshots_dir(projects_root, project_id)
    if not sd.exists():
        return 0
    n = 0
    for f in sd.iterdir():
        if not f.is_file():
            continue
        if not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        if SNAPSHOT_ID_RE.match(f.stem):
            n += 1
    return n


def find_snapshot_by_name(
    projects_root: Path, project_id: str, name: str
) -> Snapshot | None:
    """Return the snapshot whose ``name`` matches (after trim), or ``None``.

    Matching is case-sensitive after trimming whitespace — the same
    canonicalisation :meth:`Snapshot.validate` applies on save. Among
    multiple matches, the most recently created one wins (so a user
    who replays an export a year later still gets the latest "Initial
    coding done" rather than the original).
    """
    target = (name or "").strip()
    if not target:
        return None
    matching = [s for s in list_snapshots(projects_root, project_id) if s.name == target]
    if not matching:
        return None
    matching.sort(key=lambda s: (s.created_at, s.id))
    return matching[-1]


# --------------------------------------------------------------------------- #
# High-level helper: create a snapshot from current project state
# --------------------------------------------------------------------------- #


def create_codebook_snapshot(
    projects_root: Path,
    project_id: str,
    *,
    name: str,
    description: str = "",
    actor_coder_id: str = "",
    record_audit_event: bool = True,
    snapshot_id: str | None = None,
    now: str | None = None,
) -> Snapshot:
    """Take a snapshot of the project's *current* codebook and save it.

    Reads the live codes via :func:`scribe.codes.list_codes` and the
    latest definition-version of each via
    :func:`scribe.code_versions.latest_code_version`. Stamps the
    project's ``codebook_stage`` from :func:`scribe.projects.load_project`.

    When ``record_audit_event`` is true (default), an F9.1
    :class:`scribe.event_log.Event` is recorded with
    ``action='snapshot'`` / ``entity_type='snapshot'`` /
    ``entity_id=<snapshot id>``. The event id is back-written onto the
    snapshot so the two records reference each other (PLANNING:
    "snapshot ids reference the event that produced them").

    The order is deliberate:

      1. Read the project state.
      2. Build the :class:`Snapshot` in memory and validate.
      3. Save the snapshot file (must succeed before the event is emitted —
         a broken snapshot mustn't leave a "snapshot taken" event in
         the audit log).
      4. Emit the F9.1 event referencing the snapshot id.
      5. Re-save the snapshot with the event_id back-filled.

    If step 4 fails for any reason, the snapshot file still exists with
    an empty ``event_id`` — the audit trail loses the cross-reference,
    but the bookmark is preserved. Callers who want stricter atomicity
    should run their own transaction.
    """
    project: Project = load_project(projects_root, project_id)
    codes = list_codes(projects_root, project.id)

    # Resolve latest version for each code.
    code_versions: dict[str, str] = {}
    for c in codes:
        latest = latest_code_version(projects_root, project.id, c.id)
        if latest is not None:
            code_versions[c.id] = latest.id

    snapshot = Snapshot.new(
        project_id=project.id,
        name=name,
        description=description,
        codebook_stage=project.codebook_stage,
        actor_coder_id=actor_coder_id,
        codes=codes,
        code_versions=code_versions,
        snapshot_id=snapshot_id,
        now=now,
    )
    save_snapshot(projects_root, snapshot)

    if record_audit_event:
        # Record-only-the-summary policy: the *full* code list is
        # already on disk in the snapshot file. The event records a
        # cheap summary so audit trail readers see the bookmark
        # without paying the cost of duplicating the whole codebook in
        # the event payload.
        summary = {
            "snapshot_id": snapshot.id,
            "name": snapshot.name,
            "description": snapshot.description,
            "codebook_stage": snapshot.codebook_stage,
            "code_count": len(snapshot.codes),
        }
        try:
            ev = record_event(
                projects_root,
                project_id=project.id,
                action=EVENT_ACTION_SNAPSHOT,
                entity_type=EVENT_ENTITY_SNAPSHOT,
                entity_id=snapshot.id,
                actor_coder_id=actor_coder_id,
                before=None,
                after=summary,
                notes=description,
                now=now,
            )
        except Exception:
            # Audit-event emission is best-effort: if it fails (disk
            # full, race, etc.), the snapshot file still stands. The
            # caller can replay an "audit log fix" event later if
            # they need to.
            return snapshot
        # Back-write the event id onto the snapshot. We rewrite the
        # JSON file in place because the snapshot we just wrote
        # already exists on disk and the rest of the F-feature stack
        # treats append-only files as immutable — this is the one
        # documented exception, and it only ever turns an empty
        # ``event_id`` into a populated one. Callers can detect the
        # back-write by comparing ``snapshot.event_id`` to the file.
        snapshot.event_id = ev.id
        target = snapshot_state_path(projects_root, project.id, snapshot.id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False)
        )
        tmp.replace(target)
    return snapshot


# --------------------------------------------------------------------------- #
# Reconstruction — turn a snapshot back into Code instances
# --------------------------------------------------------------------------- #


def reconstruct_codes_from_snapshot(snapshot: Snapshot) -> list[Code]:
    """Return the snapshot's embedded codes as :class:`Code` instances.

    Pure helper — no I/O. The natural input to F6.1's
    :func:`scribe.codebook_export.render_codebook` so a report can be
    *regenerated against any snapshot* (PLANNING.md F9.3) by a caller
    that already has the snapshot in hand.

    Invalid code dicts (which :meth:`Snapshot.validate` already rejects
    on save) raise :class:`ProjectValidationError` here too, in case
    a snapshot built by hand sneaks past validation upstream.
    """
    out: list[Code] = []
    for c in snapshot.codes:
        out.append(Code.from_dict(c))
    return out


def render_codebook_at_snapshot(
    snapshot: Snapshot,
    *,
    format: str,
    project: Project | None = None,
) -> str:
    """Render the codebook *as it stood at* ``snapshot`` in the given format.

    Convenience wrapper that bridges F9.3 and F6.1. The ``project``
    argument is optional — F6.1's Markdown / RTF renderers use it for
    a small header; CSV ignores it. Kept Python-only and lazy-imported
    so :mod:`scribe.codebook_snapshots` doesn't pull the whole exporter
    surface unless a caller actually wants a rendered string.
    """
    # Lazy import to avoid a hard dependency cycle at module-load
    # (codebook_export imports projects + codes too, but stays out of
    # F9 unless the renderer is used).
    from .codebook_export import render_codebook

    codes = reconstruct_codes_from_snapshot(snapshot)
    return render_codebook(format, codes, project=project)


def code_at_snapshot(
    snapshot: Snapshot, code_id: str
) -> Code | None:
    """Return the Code with ``code_id`` as captured in ``snapshot``, or None.

    Useful for "definition at snapshot" lookups — the F9.2 sibling
    asks "definition at apply"; F9.3 asks "definition at snapshot".
    Pure helper; no I/O.
    """
    if not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(f"Invalid code id: {code_id!r}")
    for c in snapshot.codes:
        if c.get("id") == code_id:
            return Code.from_dict(c)
    return None


def code_version_id_at_snapshot(
    snapshot: Snapshot, code_id: str
) -> str | None:
    """Return the code's pinned version id at snapshot time, or ``None``.

    A return of ``None`` means the snapshot didn't pin a version (the
    code existed but no version had been recorded yet) — distinct from
    "the code wasn't in the snapshot at all", which callers check via
    :func:`code_at_snapshot`.
    """
    if not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(f"Invalid code id: {code_id!r}")
    return snapshot.code_versions.get(code_id)


def snapshot_summary(snapshot: Snapshot) -> dict[str, Any]:
    """Return a small dict suitable for UI lists / API responses.

    Cheap projection: id, name, description, stage, code count,
    coder/event refs, timestamp. Doesn't embed the full codebook —
    use :func:`reconstruct_codes_from_snapshot` for that.
    """
    return {
        "id": snapshot.id,
        "project_id": snapshot.project_id,
        "name": snapshot.name,
        "description": snapshot.description,
        "codebook_stage": snapshot.codebook_stage,
        "actor_coder_id": snapshot.actor_coder_id,
        "event_id": snapshot.event_id,
        "code_count": len(snapshot.codes),
        "version_pin_count": len(snapshot.code_versions),
        "created_at": snapshot.created_at,
    }


def list_snapshot_summaries(
    projects_root: Path, project_id: str
) -> list[dict[str, Any]]:
    """List all snapshots as cheap summaries, ascending by created_at.

    Wrapper around :func:`list_snapshots` + :func:`snapshot_summary`.
    Mirrors the pattern used by the F-feature stack for "give me a
    cheap UI list" callers.
    """
    return [snapshot_summary(s) for s in list_snapshots(projects_root, project_id)]


__all__ = [
    "MAX_CODES_PER_SNAPSHOT",
    "MAX_DESCRIPTION_LEN",
    "MAX_NAME_LEN",
    "MAX_SNAPSHOT_BYTES",
    "SNAPSHOT_ID_RE",
    "SNAPSHOTS_DIRNAME",
    "Snapshot",
    "code_at_snapshot",
    "code_version_id_at_snapshot",
    "count_snapshots",
    "create_codebook_snapshot",
    "find_snapshot_by_name",
    "list_snapshot_summaries",
    "list_snapshots",
    "load_snapshot",
    "new_snapshot_id",
    "reconstruct_codes_from_snapshot",
    "render_codebook_at_snapshot",
    "save_snapshot",
    "snapshot_state_path",
    "snapshot_summary",
    "snapshots_dir",
]
