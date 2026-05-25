"""Full-project checkpoints (F9.4).

Per PLANNING.md F9.4:

  > Project checkpoints (full project state save; git-like).

What a checkpoint is
--------------------

A **checkpoint** is the full-project sibling of an F9.3 codebook
snapshot. Where F9.3 freezes only the codebook, F9.4 freezes
*everything*: the project record, sources, participants, sampling log,
codebook (codes + version history), applications, memos, speaker maps,
saved queries, audit events, and any other JSON the project stores.

Concretely, a checkpoint is a name-stamped, immutable, hash-verified
zip archive of the project directory tree (minus the ``checkpoints/``
subdirectory itself, to avoid recursion) plus a small JSON metadata
sidecar so the UI can list checkpoints without unzipping anything.

This is the "git-like" feature in the audit-trail story:

  * Take one before a risky operation (about to merge a hundred codes;
    locking the codebook; running an AI sweep).
  * Inspect and verify them later (was this exactly the project
    state I think it was? — the SHA-256 over the archive answers).
  * Restore one to a fresh location if you need to roll back work
    (read-only restore via :func:`extract_checkpoint_to_directory`,
    which delegates to :func:`scribe.project_format.import_project_archive`).

Why this is its own module, not part of F1.5
--------------------------------------------

F1.5's ``export_project_archive`` is the *transport* mechanism — give
me a zip I can mail to a collaborator, or hand to REFI-QDA later.

F9.4 is *internal versioning* — give the researcher a record of
"this is what the project looked like on Monday" *that lives inside
the project*. Same archive shape under the hood (we reuse F1.5's
exporter), different intent: the metadata, the sha256, the audit
event, the append-only contract are all F9.4-specific.

Boundaries
----------

* **No HTTP / FastAPI surface here.** F9.4 ships data model + writer +
  reader + restore-helper. Routes can be added by a later iteration.
  Mirrors the staged approach in F9.1 / F9.2 / F9.3.
* **Stand-alone, pure Python.** No FastAPI, no engine imports.
* **Append-only by convention.** Once written, a checkpoint's metadata
  + archive are immutable. There's no ``update_checkpoint`` or
  ``delete_checkpoint``: deleting a checkpoint cuts the
  reproducibility chain. If a researcher mints one by mistake they
  mint a follow-up correcting it.
* **Restore is non-destructive.** :func:`extract_checkpoint_to_directory`
  refuses to overwrite the *live* project tree by default — restore
  produces a fresh extracted directory. Callers that want
  destructive in-place restore must do it themselves with
  :mod:`scribe.project_format` after they've taken responsibility
  for backing up first.

On-disk layout
--------------

::

    projects/<project_id>/
      checkpoints/
        <checkpoint_id>.json        # metadata sidecar; never modified
        <checkpoint_id>.scribe.zip  # archive body; never modified

Files are named by checkpoint id (12-char hex), matching every other
entity store in the project. ``list_checkpoints`` sorts by
``created_at`` ascending so the natural reading order is creation order.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .coders import CODER_ID_RE
from .event_log import (
    EVENT_ACTION_CHECKPOINT,
    EVENT_ENTITY_CHECKPOINT,
    EVENT_ID_RE,
    record_event,
)
from .project_format import (
    ARCHIVE_SUFFIX,
    ProjectFormatError,
    export_project_archive,
    import_project_archive,
)
from .projects import (
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

# Checkpoint ids share Scribe's standard 12-char hex shape.
CHECKPOINT_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# On-disk subdirectory name. Sibling of ``snapshots/`` / ``events/``.
# We deliberately *exclude* this directory from the archive each
# checkpoint produces — without that exclusion the second checkpoint
# would embed the first, the third would embed the first two, and the
# project tree's checkpoints/ directory would explode on disk.
CHECKPOINTS_DIRNAME = "checkpoints"

# Suffix of the archive body next to each metadata sidecar. We use the
# same suffix as F1.5's ``export_project_archive`` so callers who
# inspect the file in a file manager recognise it.
CHECKPOINT_ARCHIVE_SUFFIX = ARCHIVE_SUFFIX  # ".scribe.zip"

# Metadata sidecar suffix. JSON; named after the checkpoint id.
CHECKPOINT_META_SUFFIX = ".json"

# Name / description bounds. Same shape as F9.3 snapshots.
MAX_NAME_LEN = 200
MAX_DESCRIPTION_LEN = 4000

# Hard ceiling on the archive size. A project with a few transcripts
# can run hundreds of MB once outputs are bundled in; without
# ``include_outputs`` it's typically sub-MB. 8 GiB is the same ceiling
# the import path enforces (``MAX_ARCHIVE_TOTAL_BYTES``); we don't
# want to be more restrictive on output than on input.
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024

# Component-counts dict: a small summary of "how many of each thing
# were in this checkpoint". Bounded so a freak count (a dict with
# thousands of keys for some reason) can't blow up the metadata.
MAX_COMPONENT_COUNT_KEYS = 64
_COMPONENT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Checkpoint:
    """One named full-project checkpoint (F9.4).

    The metadata sidecar (this dataclass) records the *bookmark*; the
    archive body (a sibling ``<id>.scribe.zip`` file) holds the data.
    The two are linked by ``archive_filename`` and verified by
    ``archive_sha256``.

    Fields
    ------
    id
        12-char hex checkpoint id. Mint with :func:`new_checkpoint_id`.
    project_id
        12-char hex project id this checkpoint belongs to.
    name
        Required, non-empty. The user-visible bookmark label, e.g.
        ``"Pre-merge of duplicate codes"``.
    description
        Optional free-form notes (the *why* — methodological reason).
    actor_coder_id
        Optional 12-char hex coder id of the human who created the
        checkpoint. Empty for system-created checkpoints.
    parent_checkpoint_id
        Optional 12-char hex id of the checkpoint this one was based
        on. Lets a project carry a chain of checkpoints (git-like
        parent pointers). Empty when there's no recorded parent.
    event_id
        Optional 12-char hex id of the F9.1 :class:`Event` that
        recorded this checkpoint. Empty when the caller opted out of
        emitting an event (e.g. testing, importers).
    codebook_stage
        The project's ``codebook_stage`` at the moment the checkpoint
        was taken — captured separately from the (much larger)
        archive so a UI can render "checkpointed at *focused* stage"
        without unzipping anything.
    archive_filename
        Filename of the archive body, relative to the checkpoints
        directory. Always ``"<id>.scribe.zip"`` for now; the field
        exists so a future migration can rename the body without
        breaking older readers.
    archive_bytes
        Size of the archive body in bytes. Cheap pre-flight for the
        UI ("3 MB checkpoint"). Refusing to overwrite an existing
        body still relies on the file system, not on this number.
    archive_sha256
        Lowercase hex SHA-256 of the archive body. The audit-trail
        verifier compares against this to detect tampering /
        corruption / silent disk-bit-rot.
    component_counts
        Small ``dict[str, int]`` summary of contents — e.g.
        ``{"sources": 3, "codes": 41, "applications": 217}``. Cheap
        to render in a list view without unzipping. Optional: zero
        is a valid count, missing keys mean "not measured".
    created_at
        ISO-8601 UTC timestamp; set at construction time.
    """

    id: str
    project_id: str
    name: str
    description: str = ""
    actor_coder_id: str = ""
    parent_checkpoint_id: str = ""
    event_id: str = ""
    codebook_stage: str = "initial"
    archive_filename: str = ""
    archive_bytes: int = 0
    archive_sha256: str = ""
    component_counts: dict[str, int] = field(default_factory=dict)
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
        actor_coder_id: str = "",
        parent_checkpoint_id: str = "",
        event_id: str = "",
        codebook_stage: str = "initial",
        archive_filename: str = "",
        archive_bytes: int = 0,
        archive_sha256: str = "",
        component_counts: Mapping[str, int] | None = None,
        checkpoint_id: str | None = None,
        now: str | None = None,
    ) -> "Checkpoint":
        """Build a fresh :class:`Checkpoint` and validate it."""
        cid = checkpoint_id or new_checkpoint_id()
        cp = cls(
            id=cid,
            project_id=project_id,
            name=name,
            description=description,
            actor_coder_id=actor_coder_id or "",
            parent_checkpoint_id=parent_checkpoint_id or "",
            event_id=event_id or "",
            codebook_stage=codebook_stage,
            archive_filename=archive_filename or f"{cid}{CHECKPOINT_ARCHIVE_SUFFIX}",
            archive_bytes=int(archive_bytes),
            archive_sha256=archive_sha256.lower() if archive_sha256 else "",
            component_counts={str(k): int(v) for k, v in (component_counts or {}).items()},
            created_at=now or utcnow_iso(),
        )
        cp.validate()
        return cp

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Checkpoint":
        if not isinstance(d, Mapping):
            raise ProjectValidationError("Checkpoint payload must be an object")
        for required in ("id", "project_id", "name"):
            if required not in d:
                raise ProjectValidationError(
                    f"Checkpoint payload missing required key: {required}"
                )
        raw_counts = d.get("component_counts") or {}
        if not isinstance(raw_counts, Mapping):
            raise ProjectValidationError(
                "Checkpoint.component_counts must be an object"
            )
        cp = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            name=str(d.get("name", "") or ""),
            description=str(d.get("description", "") or ""),
            actor_coder_id=str(d.get("actor_coder_id", "") or ""),
            parent_checkpoint_id=str(d.get("parent_checkpoint_id", "") or ""),
            event_id=str(d.get("event_id", "") or ""),
            codebook_stage=str(d.get("codebook_stage", "initial") or "initial"),
            archive_filename=str(d.get("archive_filename", "") or ""),
            archive_bytes=int(d.get("archive_bytes", 0) or 0),
            archive_sha256=str(d.get("archive_sha256", "") or "").lower(),
            component_counts={str(k): int(v) for k, v in raw_counts.items()},
            created_at=str(d.get("created_at", "") or ""),
        )
        cp.validate()
        return cp

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not CHECKPOINT_ID_RE.match(self.id):
            raise ProjectValidationError(
                f"Invalid checkpoint id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        name = (self.name or "").strip()
        if not name:
            raise ProjectValidationError("Checkpoint.name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Checkpoint.name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist trimmed so on-disk state is canonical.
        self.name = name
        if len(self.description) > MAX_DESCRIPTION_LEN:
            raise ProjectValidationError(
                f"Checkpoint.description must be ≤ {MAX_DESCRIPTION_LEN} chars"
            )
        if self.actor_coder_id and not CODER_ID_RE.match(self.actor_coder_id):
            raise ProjectValidationError(
                "Checkpoint.actor_coder_id must be 12-char hex or empty; "
                f"got {self.actor_coder_id!r}"
            )
        if self.parent_checkpoint_id and not CHECKPOINT_ID_RE.match(
            self.parent_checkpoint_id
        ):
            raise ProjectValidationError(
                "Checkpoint.parent_checkpoint_id must be 12-char hex or empty; "
                f"got {self.parent_checkpoint_id!r}"
            )
        if self.parent_checkpoint_id and self.parent_checkpoint_id == self.id:
            raise ProjectValidationError(
                "Checkpoint.parent_checkpoint_id cannot equal Checkpoint.id "
                "(checkpoints can't be their own parent)"
            )
        if self.event_id and not EVENT_ID_RE.match(self.event_id):
            raise ProjectValidationError(
                "Checkpoint.event_id must be 12-char hex or empty; "
                f"got {self.event_id!r}"
            )
        # Stage is captured for display only; we don't import the
        # CODEBOOK_STAGES tuple here because that would couple this
        # module tighter than necessary. Loose validation: must be a
        # short-ish string.
        if not isinstance(self.codebook_stage, str):
            raise ProjectValidationError(
                "Checkpoint.codebook_stage must be a string"
            )
        if len(self.codebook_stage) > 64:
            raise ProjectValidationError(
                "Checkpoint.codebook_stage must be ≤ 64 chars"
            )
        # Archive filename sanity. Always relative to the checkpoints/
        # directory; no ``..`` components, no absolute paths.
        if not isinstance(self.archive_filename, str) or not self.archive_filename:
            raise ProjectValidationError(
                "Checkpoint.archive_filename is required"
            )
        af = Path(self.archive_filename)
        if af.is_absolute() or ".." in af.parts or "/" in self.archive_filename or "\\" in self.archive_filename:
            raise ProjectValidationError(
                "Checkpoint.archive_filename must be a flat filename "
                "inside the checkpoints/ directory; "
                f"got {self.archive_filename!r}"
            )
        if not isinstance(self.archive_bytes, int) or self.archive_bytes < 0:
            raise ProjectValidationError(
                "Checkpoint.archive_bytes must be a non-negative integer"
            )
        if self.archive_bytes > MAX_ARCHIVE_BYTES:
            raise ProjectValidationError(
                f"Checkpoint.archive_bytes exceeds {MAX_ARCHIVE_BYTES} bytes"
            )
        if self.archive_sha256:
            if not re.match(r"^[a-f0-9]{64}$", self.archive_sha256):
                raise ProjectValidationError(
                    "Checkpoint.archive_sha256 must be 64-char lowercase hex"
                )
        if not isinstance(self.component_counts, dict):
            raise ProjectValidationError(
                "Checkpoint.component_counts must be an object"
            )
        if len(self.component_counts) > MAX_COMPONENT_COUNT_KEYS:
            raise ProjectValidationError(
                f"Checkpoint.component_counts exceeds "
                f"{MAX_COMPONENT_COUNT_KEYS} keys"
            )
        for k, v in self.component_counts.items():
            if not _COMPONENT_KEY_RE.match(k):
                raise ProjectValidationError(
                    f"Checkpoint.component_counts key {k!r} must be lowercase "
                    "snake_case (≤ 64 chars)"
                )
            if not isinstance(v, int) or v < 0:
                raise ProjectValidationError(
                    f"Checkpoint.component_counts[{k!r}] must be a "
                    "non-negative integer"
                )


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_checkpoint_id() -> str:
    """Mint a new 12-char hex checkpoint id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def checkpoints_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's checkpoints."""
    return project_dir(projects_root, project_id) / CHECKPOINTS_DIRNAME


def checkpoint_meta_path(
    projects_root: Path, project_id: str, checkpoint_id: str
) -> Path:
    """Return the path for a checkpoint's metadata sidecar."""
    if not CHECKPOINT_ID_RE.match(checkpoint_id):
        raise ProjectValidationError(
            f"Invalid checkpoint id: {checkpoint_id!r}"
        )
    return (
        checkpoints_dir(projects_root, project_id)
        / f"{checkpoint_id}{CHECKPOINT_META_SUFFIX}"
    )


def checkpoint_archive_path(
    projects_root: Path, project_id: str, checkpoint_id: str
) -> Path:
    """Return the path for a checkpoint's archive body."""
    if not CHECKPOINT_ID_RE.match(checkpoint_id):
        raise ProjectValidationError(
            f"Invalid checkpoint id: {checkpoint_id!r}"
        )
    return (
        checkpoints_dir(projects_root, project_id)
        / f"{checkpoint_id}{CHECKPOINT_ARCHIVE_SUFFIX}"
    )


def save_checkpoint_meta(projects_root: Path, checkpoint: Checkpoint) -> Path:
    """Persist a checkpoint's metadata sidecar atomically.

    Refuses to overwrite an existing id — checkpoints are append-only.
    Does *not* write the archive body; callers (or
    :func:`create_project_checkpoint`) are responsible for the body.
    """
    checkpoint.validate()
    parent = project_dir(projects_root, checkpoint.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving checkpoints."
        )
    cd = checkpoints_dir(projects_root, checkpoint.project_id)
    cd.mkdir(parents=True, exist_ok=True)
    target = checkpoint_meta_path(
        projects_root, checkpoint.project_id, checkpoint.id
    )
    if target.exists():
        raise FileExistsError(
            f"Checkpoint {checkpoint.id} already exists; "
            "checkpoints are append-only"
        )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_checkpoint(
    projects_root: Path, project_id: str, checkpoint_id: str
) -> Checkpoint:
    """Load a checkpoint's metadata by id. Raises ``FileNotFoundError`` if missing."""
    p = checkpoint_meta_path(projects_root, project_id, checkpoint_id)
    if not p.exists():
        raise FileNotFoundError(f"No checkpoint at {p}")
    return Checkpoint.from_dict(json.loads(p.read_text()))


def list_checkpoints(
    projects_root: Path, project_id: str
) -> list[Checkpoint]:
    """List all checkpoints in a project, ordered by ``created_at`` ascending.

    Skips files that don't parse as a valid :class:`Checkpoint` so a
    single corrupt file doesn't break the bookmarks view; matches the
    rest of the F-feature stack.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    cd = checkpoints_dir(projects_root, project_id)
    if not cd.exists():
        return []
    out: list[Checkpoint] = []
    for f in sorted(cd.iterdir()):
        if not f.is_file() or not f.name.endswith(CHECKPOINT_META_SUFFIX):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        cid = f.stem
        if not CHECKPOINT_ID_RE.match(cid):
            continue
        try:
            out.append(Checkpoint.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda c: (c.created_at, c.id))
    return out


def count_checkpoints(projects_root: Path, project_id: str) -> int:
    """Return the number of well-formed checkpoint metadata files."""
    cd = checkpoints_dir(projects_root, project_id)
    if not cd.exists():
        return 0
    n = 0
    for f in cd.iterdir():
        if not f.is_file():
            continue
        if not f.name.endswith(CHECKPOINT_META_SUFFIX):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        if CHECKPOINT_ID_RE.match(f.stem):
            n += 1
    return n


def find_checkpoint_by_name(
    projects_root: Path, project_id: str, name: str
) -> Checkpoint | None:
    """Return the checkpoint whose ``name`` matches (after trim), or ``None``.

    Case-sensitive after trimming whitespace — same canonicalisation
    :meth:`Checkpoint.validate` applies on save. Among multiple matches,
    the most recently created one wins.
    """
    target = (name or "").strip()
    if not target:
        return None
    matching = [
        c
        for c in list_checkpoints(projects_root, project_id)
        if c.name == target
    ]
    if not matching:
        return None
    matching.sort(key=lambda c: (c.created_at, c.id))
    return matching[-1]


# --------------------------------------------------------------------------- #
# Hashing + integrity verification
# --------------------------------------------------------------------------- #


def _hash_file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """Compute the SHA-256 of a file as a 64-char lowercase hex string.

    Streamed in 1 MiB chunks so a multi-GB archive doesn't sit in
    memory. Internal helper.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def verify_checkpoint_archive(
    projects_root: Path, checkpoint: Checkpoint
) -> bool:
    """Return True iff the archive body matches the metadata's sha256.

    A return of ``False`` means the archive on disk has drifted from
    the metadata (corruption, manual edit, restored backup of only one
    of the two files, etc.). ``True`` means the file exists and its
    digest matches.

    Raises ``FileNotFoundError`` if the archive body is missing, so
    callers can distinguish "archive gone" from "archive corrupted".
    Returns ``False`` if the metadata recorded an empty sha256 — we
    can't *verify* a checkpoint that wasn't hashed, so we err on the
    side of "not verified".
    """
    archive = checkpoint_archive_path(
        projects_root, checkpoint.project_id, checkpoint.id
    )
    if not archive.exists():
        raise FileNotFoundError(f"Archive missing for checkpoint: {archive}")
    if not checkpoint.archive_sha256:
        return False
    actual = _hash_file_sha256(archive)
    return actual == checkpoint.archive_sha256


# --------------------------------------------------------------------------- #
# Component counts — what's in a project right now
# --------------------------------------------------------------------------- #


def _count_jsons(directory: Path) -> int:
    """Count the JSON files immediately under ``directory`` (no recursion).

    Skips ``*.tmp`` and dotfiles. Returns 0 when the directory doesn't
    exist; doesn't raise. Internal helper for component-count summaries.
    """
    if not directory.exists():
        return 0
    n = 0
    for f in directory.iterdir():
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue
        if not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        n += 1
    return n


def compute_component_counts(
    projects_root: Path, project_id: str
) -> dict[str, int]:
    """Return a small ``{component: count}`` map of the live project.

    Cheap survey of the project's sub-directories — counts JSON files
    by directory name. Used by :func:`create_project_checkpoint` to
    populate the metadata sidecar. The keys are deliberately stable
    across releases (``sources``, ``codes``, ``applications``, …) so
    UIs / reports can rely on them.

    Unknown directories are not counted; missing directories report 0.
    The ``checkpoints/`` directory is intentionally not counted (it's
    excluded from the archive too, and including it would make the
    summary lie about "what's in the checkpoint").
    """
    base = project_dir(projects_root, project_id)
    if not base.exists():
        raise FileNotFoundError(f"No project at {base}")
    out: dict[str, int] = {}
    # Subdirectories that store one entity per JSON file.
    for component, subdir in (
        ("sources", "sources"),
        ("participants", "participants"),
        ("codes", "codes"),
        ("applications", "applications"),
        ("memos", "memos"),
        ("coders", "coders"),
        ("speaker_maps", "speaker_maps"),
        ("saved_queries", "saved_queries"),
        ("snapshots", "snapshots"),
        ("events", "events"),
        ("ai_events", "ai_events"),
        ("code_versions", "code_versions"),
    ):
        out[component] = _count_jsons(base / subdir)
    # The sampling log is one JSONL file; count its entries.
    sl = base / "sampling_log.jsonl"
    if sl.exists():
        try:
            text = sl.read_text(encoding="utf-8")
            out["sampling_log_entries"] = sum(
                1 for line in text.splitlines() if line.strip()
            )
        except OSError:
            out["sampling_log_entries"] = 0
    else:
        out["sampling_log_entries"] = 0
    return out


# --------------------------------------------------------------------------- #
# High-level helper: create a checkpoint of current project state
# --------------------------------------------------------------------------- #


def create_project_checkpoint(
    projects_root: Path,
    project_id: str,
    *,
    name: str,
    description: str = "",
    actor_coder_id: str = "",
    parent_checkpoint_id: str = "",
    record_audit_event: bool = True,
    include_outputs: bool = False,
    outputs_root: Path | None = None,
    checkpoint_id: str | None = None,
    now: str | None = None,
) -> Checkpoint:
    """Take a checkpoint of the project's *current* full state and save it.

    The order is deliberate:

      1. Read the project record (so we can stamp ``codebook_stage``).
      2. Mint a checkpoint id.
      3. Export the project tree to the checkpoint's archive path,
         excluding the ``checkpoints/`` subdirectory itself
         (otherwise a checkpoint embeds prior checkpoints — quadratic
         disk usage at minimum).
      4. Hash the archive (SHA-256 over the file).
      5. Write the metadata sidecar with the hash + component counts.
      6. Optionally emit an F9.1 :class:`Event` referencing the
         checkpoint id; back-write its event id onto the sidecar.

    ``include_outputs=True`` (with ``outputs_root``) bundles in every
    transcript directory referenced by a source's
    ``transcript_job_id`` — mirrors :func:`export_project_archive`'s
    own toggle. Default-off because outputs are huge and the
    transcripts are already on disk under ``outputs/``.

    ``parent_checkpoint_id`` is an optional pointer to the previous
    checkpoint this one supersedes — the "git-like" chain. Caller
    decides the semantics; the validator only checks shape.

    Failure semantics: if the archive write fails, no metadata is
    written. If the metadata write fails, the orphan archive is best-
    effort cleaned up. If audit-event emission fails, the metadata
    file still stands with an empty ``event_id`` — the audit trail
    loses the cross-reference but the checkpoint is preserved.
    """
    project: Project = load_project(projects_root, project_id)

    cid = checkpoint_id or new_checkpoint_id()
    if not CHECKPOINT_ID_RE.match(cid):
        raise ProjectValidationError(
            f"Invalid checkpoint id: {cid!r}"
        )

    cd = checkpoints_dir(projects_root, project.id)
    cd.mkdir(parents=True, exist_ok=True)
    archive = checkpoint_archive_path(projects_root, project.id, cid)
    if archive.exists():
        # Defensive: should never happen with a fresh uuid, but if it
        # does (e.g. a caller passed a deterministic id), refuse so
        # the append-only contract holds.
        raise FileExistsError(
            f"Checkpoint archive {archive} already exists; "
            "checkpoints are append-only"
        )

    # 1+2+3: export the archive.
    try:
        export_project_archive(
            projects_root,
            project.id,
            archive,
            outputs_root=outputs_root,
            include_outputs=include_outputs,
            exclude_relative=[CHECKPOINTS_DIRNAME],
        )
    except Exception:
        # Best-effort cleanup of any partial archive.
        if archive.exists():
            try:
                archive.unlink()
            except OSError:
                pass
        raise

    # 4: hash the archive.
    try:
        digest = _hash_file_sha256(archive)
        size = archive.stat().st_size
    except Exception:
        if archive.exists():
            try:
                archive.unlink()
            except OSError:
                pass
        raise

    # 5: build + persist the metadata sidecar.
    counts = compute_component_counts(projects_root, project.id)
    try:
        checkpoint = Checkpoint.new(
            project_id=project.id,
            name=name,
            description=description,
            actor_coder_id=actor_coder_id,
            parent_checkpoint_id=parent_checkpoint_id,
            codebook_stage=project.codebook_stage,
            archive_filename=archive.name,
            archive_bytes=size,
            archive_sha256=digest,
            component_counts=counts,
            checkpoint_id=cid,
            now=now,
        )
        save_checkpoint_meta(projects_root, checkpoint)
    except Exception:
        if archive.exists():
            try:
                archive.unlink()
            except OSError:
                pass
        raise

    # 6: optional audit event.
    if record_audit_event:
        summary = {
            "checkpoint_id": checkpoint.id,
            "name": checkpoint.name,
            "description": checkpoint.description,
            "codebook_stage": checkpoint.codebook_stage,
            "archive_filename": checkpoint.archive_filename,
            "archive_bytes": checkpoint.archive_bytes,
            "archive_sha256": checkpoint.archive_sha256,
            "component_counts": dict(checkpoint.component_counts),
            "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
        }
        try:
            ev = record_event(
                projects_root,
                project_id=project.id,
                action=EVENT_ACTION_CHECKPOINT,
                entity_type=EVENT_ENTITY_CHECKPOINT,
                entity_id=checkpoint.id,
                actor_coder_id=actor_coder_id,
                before=None,
                after=summary,
                notes=description,
                now=now,
            )
        except Exception:
            # Audit emission is best-effort; the checkpoint stands.
            return checkpoint
        # Back-write the event id. Mirrors F9.3's contract.
        checkpoint.event_id = ev.id
        target = checkpoint_meta_path(projects_root, project.id, checkpoint.id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False)
        )
        tmp.replace(target)
    return checkpoint


# --------------------------------------------------------------------------- #
# Restore (read-only / non-destructive)
# --------------------------------------------------------------------------- #


def extract_checkpoint_to_directory(
    projects_root: Path,
    checkpoint: Checkpoint,
    target_projects_root: Path,
    *,
    target_outputs_root: Path | None = None,
    overwrite: bool = False,
    verify: bool = True,
) -> Path:
    """Extract a checkpoint's archive into a fresh project tree.

    This is the **non-destructive** restore path. The archive is
    extracted under ``target_projects_root/<original_project_id>/``
    using :func:`scribe.project_format.import_project_archive`. By
    default it refuses to overwrite an existing project directory at
    the target — pass ``overwrite=True`` to replace it.

    ``target_outputs_root`` controls whether ``outputs/`` files in the
    archive are extracted (only relevant for checkpoints created with
    ``include_outputs=True``).

    When ``verify`` is true (default) the archive's SHA-256 is
    re-computed and compared against ``checkpoint.archive_sha256``
    before extraction. A mismatch raises :class:`ProjectFormatError`.
    Disable only if the metadata's sha256 is missing (unverified
    checkpoint).

    Returns the path of the extracted project directory.
    """
    archive = checkpoint_archive_path(
        projects_root, checkpoint.project_id, checkpoint.id
    )
    if not archive.exists():
        raise FileNotFoundError(f"No checkpoint archive at {archive}")

    if verify and checkpoint.archive_sha256:
        actual = _hash_file_sha256(archive)
        if actual != checkpoint.archive_sha256:
            raise ProjectFormatError(
                "Checkpoint archive sha256 mismatch; refusing to restore. "
                f"expected {checkpoint.archive_sha256!r}, got {actual!r}"
            )

    bundle = import_project_archive(
        target_projects_root,
        archive,
        outputs_root=target_outputs_root,
        overwrite=overwrite,
    )
    return project_dir(target_projects_root, bundle.project.id)


# --------------------------------------------------------------------------- #
# Cheap projections for UI lists / API responses
# --------------------------------------------------------------------------- #


def checkpoint_summary(checkpoint: Checkpoint) -> dict[str, Any]:
    """Return a small dict suitable for UI lists / API responses."""
    return {
        "id": checkpoint.id,
        "project_id": checkpoint.project_id,
        "name": checkpoint.name,
        "description": checkpoint.description,
        "codebook_stage": checkpoint.codebook_stage,
        "actor_coder_id": checkpoint.actor_coder_id,
        "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
        "event_id": checkpoint.event_id,
        "archive_filename": checkpoint.archive_filename,
        "archive_bytes": checkpoint.archive_bytes,
        "archive_sha256": checkpoint.archive_sha256,
        "component_counts": dict(checkpoint.component_counts),
        "created_at": checkpoint.created_at,
    }


def list_checkpoint_summaries(
    projects_root: Path, project_id: str
) -> list[dict[str, Any]]:
    """List all checkpoints as cheap summaries, ascending by created_at."""
    return [
        checkpoint_summary(c)
        for c in list_checkpoints(projects_root, project_id)
    ]


__all__ = [
    "CHECKPOINT_ARCHIVE_SUFFIX",
    "CHECKPOINT_ID_RE",
    "CHECKPOINT_META_SUFFIX",
    "CHECKPOINTS_DIRNAME",
    "Checkpoint",
    "MAX_ARCHIVE_BYTES",
    "MAX_COMPONENT_COUNT_KEYS",
    "MAX_DESCRIPTION_LEN",
    "MAX_NAME_LEN",
    "checkpoint_archive_path",
    "checkpoint_meta_path",
    "checkpoint_summary",
    "checkpoints_dir",
    "compute_component_counts",
    "count_checkpoints",
    "create_project_checkpoint",
    "extract_checkpoint_to_directory",
    "find_checkpoint_by_name",
    "list_checkpoint_summaries",
    "list_checkpoints",
    "load_checkpoint",
    "new_checkpoint_id",
    "save_checkpoint_meta",
    "verify_checkpoint_archive",
]
