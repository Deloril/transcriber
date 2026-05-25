"""Saved queries (F3.7).

Per PLANNING.md F3.7:

  > Saved queries (named, re-runnable).

F3.5 (the query builder) and F3.6 (the matrix views) both produce
ad-hoc queries. F3.7 lets a researcher *name* one ("Power-related
quotes from interviewees age 50+") and re-run it later, with a small
audit trail of when it was last run. That's the unit of analytic
reuse: a research question often outlives any particular session, and
re-running it after the codebook moves on is a deliberate act
(researcher needs to know whether the answer changed).

What's here
-----------

  * :class:`SavedQuery` — a thin wrapper around :class:`scribe.query.Query`
    that adds an id, project linkage, timestamps, and run-tracking
    (``last_run_at`` + ``run_count``).
  * On-disk persistence: one JSON file per saved query at
    ``projects/<pid>/saved_queries/<sqid>.json``. Same shape as
    :mod:`scribe.codes` (F2.1) and :mod:`scribe.speaker_map` (F3.4).
  * CRUD helpers (``save_saved_query`` / ``load_saved_query`` /
    ``list_saved_queries`` / ``delete_saved_query``).
  * :func:`record_run` — increment ``run_count`` and stamp
    ``last_run_at``. Persists on call so the audit trail isn't lost
    if the process dies mid-run.
  * :func:`run_saved_query` — a convenience that wires the saved
    query up with :func:`scribe.query.applications_for_query` and
    records the run.

What's deliberately *not* here:

  * No execution side-effects beyond updating the run timestamp.
    Callers (UI / CLI) own the actual presentation of the result set.
  * No name-uniqueness check across saved queries. Two researchers
    can each have a "By age" query in the same project; the on-disk
    id disambiguates. Uniqueness is a UI concern, not a data-model
    concern.

Conventions match :mod:`scribe.projects` (F1.1), :mod:`scribe.codes`
(F2.1), :mod:`scribe.speaker_map` (F3.4), and :mod:`scribe.query`
(F3.5). Stand-alone — no FastAPI, no engine imports.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .participants import Participant
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .query import (
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    Query,
    QueryValidationError,
    applications_for_query,
)
from .sources import Source
from .speaker_map import SpeakerMap


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Saved-query ids share the 12-char hex shape used by every other
# Scribe entity. Keeps URL routing and traversal guards uniform.
SAVED_QUERY_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Directory name under ``projects/<pid>/`` holding one JSON file per
# saved query. Mirrors ``codes/`` (F2.1) and ``speaker_maps/`` (F3.4).
SAVED_QUERIES_DIRNAME = "saved_queries"

# A saved query needs a non-empty display name (otherwise the "Re-run
# 'Untitled'" UI is nonsense). The Query.name length cap is reused so
# the wrapper doesn't drift from the underlying entity.
MIN_NAME_LEN = 1


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class SavedQuery:
    """A named, re-runnable query against a project's corpus.

    Wraps a :class:`scribe.query.Query` with persistence metadata:

      * ``id`` — 12-char hex; the on-disk filename.
      * ``project_id`` — must equal the wrapped ``query.project_id``.
        The redundancy is deliberate: every other Scribe entity has an
        explicit ``project_id``, so the validators / traversal checks
        keep uniform shape.
      * ``query`` — the :class:`Query` itself. Owns its own filter
        dimensions (sources, participants, speakers, codes, proximity)
        plus a display ``name`` and ``description``.
      * ``created_at`` / ``modified_at`` — ISO-8601 UTC strings;
        stamped by :meth:`new` and :meth:`apply_update`.
      * ``last_run_at`` — empty string until the query has been run;
        thereafter the most recent run timestamp.
      * ``run_count`` — bumped each time :func:`record_run` is called.
        Persists across edits so renaming a query doesn't reset the
        "this query has been run N times" counter.
    """

    id: str
    project_id: str
    query: Query
    created_at: str = ""
    modified_at: str = ""
    last_run_at: str = ""
    run_count: int = 0

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        query: Query,
        saved_query_id: str | None = None,
        now: str | None = None,
    ) -> "SavedQuery":
        """Build a fresh SavedQuery, validate, and stamp ``created_at``.

        ``query.project_id`` must match ``project_id`` — a saved query
        whose underlying query points at a different project would
        execute against the wrong corpus.
        """
        ts = now or utcnow_iso()
        sq = cls(
            id=saved_query_id or new_saved_query_id(),
            project_id=project_id,
            query=query,
            created_at=ts,
            modified_at=ts,
            last_run_at="",
            run_count=0,
        )
        sq.validate()
        return sq

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "query": self.query.to_dict(),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "last_run_at": self.last_run_at,
            "run_count": int(self.run_count),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SavedQuery":
        if not isinstance(d, Mapping):
            raise QueryValidationError(
                "SavedQuery payload must be an object"
            )
        for key in ("id", "project_id", "query"):
            if key not in d:
                raise QueryValidationError(
                    f"SavedQuery payload missing required key: {key}"
                )
        raw_query = d["query"]
        if not isinstance(raw_query, Mapping):
            raise QueryValidationError(
                "SavedQuery 'query' must be an object"
            )
        sq = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            query=Query.from_dict(raw_query),
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
            last_run_at=str(d.get("last_run_at", "") or ""),
            run_count=_coerce_run_count(d.get("run_count", 0)),
        )
        sq.validate()
        return sq

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: Mapping[str, Any], *, now: str | None = None
    ) -> None:
        """Apply a partial update in place.

        Accepts:
          * ``query`` — the full :class:`Query` payload (replaces the
            wrapped query). The new query's ``project_id`` must match
            this saved query's ``project_id``.
          * ``name`` / ``description`` — convenience shortcuts that
            update the wrapped query's metadata without re-sending the
            whole filter tree.

        ``id``, ``project_id``, ``created_at``, ``modified_at``,
        ``last_run_at``, and ``run_count`` are managed by the entity
        itself; passing them is allowed (and ignored) so a client can
        round-trip a fetched object.
        """
        if not isinstance(patch, Mapping):
            raise QueryValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS - _IGNORED_PATCH_KEYS
        if unknown:
            raise QueryValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )

        # Build a candidate Query so validation runs *before* we mutate
        # ``self.query``. Otherwise a failed update would leave the
        # SavedQuery half-updated.
        if "query" in patch:
            raw_query = patch["query"]
            if not isinstance(raw_query, Mapping):
                raise QueryValidationError(
                    "'query' must be an object payload"
                )
            new_query = Query.from_dict(raw_query)
        else:
            # Start from the existing query's dict so name / description
            # patches don't lose other fields.
            new_query = Query.from_dict(self.query.to_dict())

        if "name" in patch:
            new_query.name = str(patch["name"] or "")
        if "description" in patch:
            new_query.description = str(patch["description"] or "")

        # Re-validate the new query and the cross-entity constraint.
        new_query.validate()
        if new_query.project_id != self.project_id:
            raise QueryValidationError(
                f"Update 'query.project_id' must match saved query's "
                f"project_id ({self.project_id!r}); got "
                f"{new_query.project_id!r}"
            )
        self.query = new_query
        # Re-validate self (catches anything the per-field check
        # missed — name presence on the wrapped query, etc.).
        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock. Mirrors the convention
        # in Project.apply_update / Code.apply_update.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not SAVED_QUERY_ID_RE.match(self.id):
            raise QueryValidationError(
                f"Invalid saved query id: {self.id!r}"
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise QueryValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not isinstance(self.query, Query):
            raise QueryValidationError(
                "SavedQuery.query must be a Query instance"
            )
        # Validating the wrapped query also normalises its strings.
        self.query.validate()
        if self.query.project_id != self.project_id:
            raise QueryValidationError(
                f"SavedQuery.project_id {self.project_id!r} must match "
                f"wrapped query's project_id {self.query.project_id!r}"
            )

        # The display name is required for a saved query (otherwise
        # the UI's "Saved query: '<name>'" is nonsense). The wrapped
        # Query.name allows empty by design — we tighten that here.
        name = self.query.name.strip()
        if len(name) < MIN_NAME_LEN:
            raise QueryValidationError(
                "SavedQuery requires a non-empty name"
            )
        if len(name) > MAX_NAME_LEN:
            raise QueryValidationError(
                f"SavedQuery name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Already trimmed by Query.validate; mirror it here for safety.
        self.query.name = name

        if len(self.query.description) > MAX_DESCRIPTION_LEN:
            raise QueryValidationError(
                f"SavedQuery description must be ≤ "
                f"{MAX_DESCRIPTION_LEN} chars"
            )

        if not isinstance(self.run_count, int) or self.run_count < 0:
            raise QueryValidationError(
                f"run_count must be a non-negative int; "
                f"got {self.run_count!r}"
            )

        if self.last_run_at and not isinstance(self.last_run_at, str):
            raise QueryValidationError("last_run_at must be a string")

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Display name (mirrors the wrapped query's ``name``)."""
        return self.query.name

    @property
    def description(self) -> str:
        """Display description (mirrors the wrapped query's ``description``)."""
        return self.query.description


def _coerce_run_count(v: Any) -> int:
    """Coerce a stored ``run_count`` to a non-negative int.

    Stored as an int on disk, but JSON round-trips can sometimes turn
    it into a float (``1.0``). We accept both, then validate.
    """
    if isinstance(v, bool):
        # bool is a subclass of int; reject so True/False can't sneak in.
        raise QueryValidationError("run_count must be an int, not bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise QueryValidationError(
                f"run_count must be an integer; got {v!r}"
            )
        return int(v)
    raise QueryValidationError(f"run_count must be an int; got {v!r}")


# Fields :meth:`SavedQuery.apply_update` may set. ``id`` etc. are
# managed by the entity; passing them is allowed (and ignored).
_ALLOWED_PATCH_KEYS = {"query", "name", "description"}
_IGNORED_PATCH_KEYS = {
    "id",
    "project_id",
    "created_at",
    "modified_at",
    "last_run_at",
    "run_count",
}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_saved_query_id() -> str:
    """Mint a new 12-char hex saved-query id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def saved_queries_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's saved queries.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / SAVED_QUERIES_DIRNAME


def saved_query_state_path(
    projects_root: Path, project_id: str, saved_query_id: str
) -> Path:
    if not SAVED_QUERY_ID_RE.match(saved_query_id):
        raise QueryValidationError(
            f"Invalid saved query id: {saved_query_id!r}"
        )
    return saved_queries_dir(projects_root, project_id) / f"{saved_query_id}.json"


def save_saved_query(
    projects_root: Path, saved_query: SavedQuery
) -> Path:
    """Persist a saved query to ``<root>/<pid>/saved_queries/<sqid>.json``.

    The parent ``projects/<pid>`` directory must already exist (the
    project itself must have been saved). Mirrors ``save_code`` /
    ``save_speaker_map``.
    """
    saved_query.validate()
    parent = project_dir(projects_root, saved_query.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its saved queries."
        )
    sd = saved_queries_dir(projects_root, saved_query.project_id)
    sd.mkdir(parents=True, exist_ok=True)
    target = saved_query_state_path(
        projects_root, saved_query.project_id, saved_query.id
    )
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(saved_query.to_dict(), indent=2, ensure_ascii=False)
    )
    tmp.replace(target)
    return target


def load_saved_query(
    projects_root: Path, project_id: str, saved_query_id: str
) -> SavedQuery:
    """Load a saved query by id. Raises ``FileNotFoundError`` if missing."""
    p = saved_query_state_path(projects_root, project_id, saved_query_id)
    if not p.exists():
        raise FileNotFoundError(f"No saved query at {p}")
    return SavedQuery.from_dict(json.loads(p.read_text()))


def list_saved_queries(
    projects_root: Path, project_id: str
) -> list[SavedQuery]:
    """List all saved queries in a project.

    Skips files that don't parse as a valid SavedQuery so a single
    corrupt file doesn't break the project view (audit log will
    eventually surface this — F9.7). Sorted by ``modified_at`` desc,
    then by id for stability — the most-recently-edited query is the
    one the researcher is most likely re-running next.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise QueryValidationError(
            f"Invalid project id: {project_id!r}"
        )
    sd = saved_queries_dir(projects_root, project_id)
    if not sd.exists():
        return []
    out: list[SavedQuery] = []
    for f in sorted(sd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        sqid = f.stem
        if not SAVED_QUERY_ID_RE.match(sqid):
            continue
        try:
            out.append(SavedQuery.from_dict(json.loads(f.read_text())))
        except (
            ProjectValidationError,
            QueryValidationError,
            json.JSONDecodeError,
            OSError,
        ):
            continue
    out.sort(key=lambda sq: (sq.modified_at, sq.id), reverse=True)
    return out


def delete_saved_query(
    projects_root: Path, project_id: str, saved_query_id: str
) -> bool:
    """Remove a saved-query file. Returns False if it didn't exist."""
    p = saved_query_state_path(projects_root, project_id, saved_query_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise QueryValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True


# --------------------------------------------------------------------------- #
# Run tracking
# --------------------------------------------------------------------------- #


def record_run(
    projects_root: Path,
    project_id: str,
    saved_query_id: str,
    *,
    now: str | None = None,
) -> SavedQuery:
    """Increment ``run_count`` and stamp ``last_run_at``; persist.

    Returns the updated :class:`SavedQuery`. Persists immediately so a
    crash mid-run still records that the run started — better to
    over-count than lose the audit trail.

    A missing saved query raises ``FileNotFoundError``. The reverse
    case — running a deleted query — is the caller's responsibility.
    """
    sq = load_saved_query(projects_root, project_id, saved_query_id)
    sq.run_count = int(sq.run_count) + 1
    sq.last_run_at = now or utcnow_iso()
    # ``modified_at`` is *not* bumped — recording a run isn't an edit
    # in the methodological sense. Saved queries that haven't been
    # modified can still have a recent ``last_run_at``.
    save_saved_query(projects_root, sq)
    return sq


def run_saved_query(
    projects_root: Path,
    project_id: str,
    saved_query_id: str,
    applications: Iterable[Any],
    *,
    sources: Sequence[Source] | None = None,
    participants: Sequence[Participant] | None = None,
    speaker_maps: Mapping[str, SpeakerMap] | None = None,
    record: bool = True,
    now: str | None = None,
) -> tuple[SavedQuery, list[Any]]:
    """Load a saved query, execute it, and (by default) record the run.

    Returns ``(saved_query, matching_applications)``. The saved query
    returned reflects the post-run state (with ``run_count`` bumped
    and ``last_run_at`` stamped if ``record=True``). Pass ``record=False``
    to preview a query without affecting its audit trail (e.g. the
    UI's "Run on demand" toggle vs. "Save run").

    Wraps :func:`scribe.query.applications_for_query`; see that
    function's docstring for the application-shape contract and the
    reference-data requirements.
    """
    sq = load_saved_query(projects_root, project_id, saved_query_id)
    matches = applications_for_query(
        sq.query,
        applications,
        sources=sources,
        participants=participants,
        speaker_maps=speaker_maps,
    )
    if record:
        sq = record_run(projects_root, project_id, saved_query_id, now=now)
    return sq, matches
