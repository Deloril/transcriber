"""Code entity for the academic-coding workflow (F2.1).

A Code is the **labelled analytic concept** at the heart of qualitative
coding. Per PLANNING.md F2.1:

  > Code entity with full field set: UUID, name, definition, inclusion
  > criteria, exclusion criteria, exemplars, parent code, related codes
  > (typed), theoretical memo, stage, colour, status, provenance.

A code lives inside a Project and joins the project's codebook. Future
features bolt onto this entity:

  * F2.2 — code revision history (every edit creates a new version)
  * F2.3 — lifecycle ops (merge / split / rename / retire / promote)
  * F2.4 — locked-codebook stage marker
  * F4.1 — applications anchor to a code by ``code_id``
  * F8.x — AI suggestions populate ``provenance``

Codes live alongside their project on disk:

    projects/<project_id>/codes/<code_id>.json

so ``delete_project`` cleans them up for free, mirroring how Sources
(F1.2), Participants (F1.3), and the sampling log (F1.4) work.

This module is deliberately stand-alone — no FastAPI, no engine
imports — so the data model can be tested in pure Python and reused by
the CLI later. Conventions match ``scribe.projects`` (F1.1),
``scribe.sources`` (F1.2), and ``scribe.participants`` (F1.3).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .projects import (
    CODEBOOK_STAGES,
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Code IDs follow the same 12-char hex shape as project / source /
# participant / job IDs; keeps URL routing and path-traversal guards
# uniform. The PLANNING line says "UUID" — we keep that semantically
# (uuid4 source) while storing the truncated hex form Scribe uses
# everywhere else.
CODE_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Relation-type vocabulary for the `related_codes` field (F2.1's
# "related codes (typed)"). The small starting set covers the
# Charmaz / SKOS-style links researchers actually use:
#   - broader / narrower: hierarchical relations (orthogonal to the
#     primary `parent_code_id` link, which is one-directional)
#   - associated: generic "these belong together"
#   - contrasts_with: codes that mark the boundary of a category
#   - causes / follows: temporal / causal links picked up in axial
#     coding (Strauss/Corbin paradigm-style)
# F2.3 will let users extend this; for F2.1 we lock it down so the
# on-disk vocabulary doesn't drift.
CODE_RELATION_TYPES: tuple[str, ...] = (
    "broader",
    "narrower",
    "associated",
    "contrasts_with",
    "causes",
    "follows",
)

# Lifecycle status for a code. ``active`` is the default; ``draft``
# marks a code that's still being defined (won't appear in production
# reports until promoted); ``retired`` keeps the code in history but
# excludes it from new applications (F2.3 will make this enforceable).
CODE_STATUSES: tuple[str, ...] = (
    "active",
    "draft",
    "retired",
)

# Provenance source values describe *who or what* originally minted the
# code definition. Open-ended free-form text is hostile to reports, so
# we keep a closed set with an "other" escape hatch. F8.9 and F9.6 will
# extend the audit-trail story; F2.1 just records the immediate origin.
CODE_PROVENANCE_SOURCES: tuple[str, ...] = (
    "human",
    "ai_suggested",
    "ai_modified",
    "promoted_from_memo",
    "imported",
    "other",
)

# Colour: a CSS hex colour, either ``#RGB`` or ``#RRGGBB``. Empty
# string allowed (UI picks a default). We don't accept named colours
# or rgba() — keeps the on-disk format small and deterministic.
CODE_COLOUR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 50 MB code.json.
MAX_NAME_LEN = 200
MAX_DEFINITION_LEN = 4000
MAX_INCLUSION_CRITERIA_LEN = 4000
MAX_EXCLUSION_CRITERIA_LEN = 4000
MAX_THEORETICAL_MEMO_LEN = 8000
MAX_EXEMPLARS = 64
MAX_EXEMPLAR_LEN = 2000
MAX_RELATED_CODES = 128
MAX_PROVENANCE_KEYS = 16
MAX_PROVENANCE_VALUE_LEN = 1000

# Provenance keys: same shape as Source.custom_attributes / Participant
# .demographics, so the UI can render them as a small details table.
PROVENANCE_KEY_RE = re.compile(r"^[A-Za-z][\w \-]{0,63}$")


# --------------------------------------------------------------------------- #
# Helper data classes
# --------------------------------------------------------------------------- #


@dataclass
class CodeRelation:
    """One typed link from a Code to another Code in the same project.

    Stored on the *source* code's `related_codes` list. Symmetric
    relations (``associated``) need only be recorded once; asymmetric
    ones (``broader`` / ``narrower``, ``causes`` / ``follows``) point
    one way. F2.3 will add a "make symmetric" helper that mirrors the
    edge on the other code; F2.1 keeps it manual to avoid implicit
    writes during validation.
    """

    code_id: str
    relation_type: str

    def to_dict(self) -> dict[str, str]:
        return {"code_id": self.code_id, "relation_type": self.relation_type}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CodeRelation":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "related_codes entry must be an object {code_id, relation_type}"
            )
        if "code_id" not in d or "relation_type" not in d:
            raise ProjectValidationError(
                "related_codes entry missing code_id or relation_type"
            )
        return cls(
            code_id=str(d["code_id"]),
            relation_type=str(d["relation_type"]),
        )

    def validate(self) -> None:
        if not CODE_ID_RE.match(self.code_id):
            raise ProjectValidationError(
                f"related_codes code_id must be 12-char hex; "
                f"got {self.code_id!r}"
            )
        if self.relation_type not in CODE_RELATION_TYPES:
            raise ProjectValidationError(
                f"relation_type must be one of {CODE_RELATION_TYPES}; "
                f"got {self.relation_type!r}"
            )


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Code:
    """One code in a project's codebook.

    All free-text fields default to ``""`` so a researcher can mint a
    code with just a name and fill the rest in later — line-by-line
    coding generates lots of one-word codes that get fleshed out only
    if they survive the focused pass.

    ``stage`` mirrors the project's codebook stage vocabulary; on a
    code, it indicates *the analytic stage in which this code was
    introduced or last consolidated*, not the project-wide setting.
    Locked codebooks (F2.4) freeze edits to existing codes but the
    stage value itself is per-code.

    ``provenance`` carries minimal origin metadata: the ``source`` key
    is one of ``CODE_PROVENANCE_SOURCES``; arbitrary additional keys
    are allowed (e.g. ``model_id`` when ``source == "ai_suggested"``)
    and validated for shape only. F8.9 will extend this with full
    AI-call references.
    """

    id: str
    project_id: str
    name: str
    definition: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    exemplars: list[str] = field(default_factory=list)
    parent_code_id: str | None = None
    related_codes: list[CodeRelation] = field(default_factory=list)
    theoretical_memo: str = ""
    stage: str = "initial"
    colour: str = ""
    status: str = "active"
    provenance: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    modified_at: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        name: str,
        definition: str = "",
        inclusion_criteria: str = "",
        exclusion_criteria: str = "",
        exemplars: Iterable[str] | None = None,
        parent_code_id: str | None = None,
        related_codes: Iterable[Any] | None = None,
        theoretical_memo: str = "",
        stage: str = "initial",
        colour: str = "",
        status: str = "active",
        provenance: dict[str, Any] | None = None,
        code_id: str | None = None,
        now: str | None = None,
    ) -> "Code":
        """Build a fresh Code, validate, and stamp timestamps."""
        ts = now or utcnow_iso()
        # Normalise falsy ("", None) → None so callers can pass either.
        normalised_parent = parent_code_id if parent_code_id else None
        # related_codes may be passed as CodeRelation instances or as
        # dicts (e.g. from an HTTP payload). Coerce uniformly.
        coerced_relations: list[CodeRelation] = []
        for r in related_codes or []:
            if isinstance(r, CodeRelation):
                coerced_relations.append(r)
            else:
                coerced_relations.append(CodeRelation.from_dict(r))
        c = cls(
            id=code_id or new_code_id(),
            project_id=project_id,
            name=name,
            definition=definition,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            exemplars=list(exemplars or []),
            parent_code_id=normalised_parent,
            related_codes=coerced_relations,
            theoretical_memo=theoretical_memo,
            stage=stage,
            colour=colour,
            status=status,
            provenance=dict(provenance or {}),
            created_at=ts,
            modified_at=ts,
        )
        c.validate()
        return c

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict on a list of dataclasses produces dicts already, but
        # we explicitly normalise keys to keep the on-disk format
        # predictable (and easier for a future REFI-QDA exporter to
        # mirror).
        d["related_codes"] = [
            {"code_id": r["code_id"], "relation_type": r["relation_type"]}
            for r in d.get("related_codes", [])
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Code":
        if not isinstance(d, dict):
            raise ProjectValidationError("Code payload must be an object")
        if "id" not in d or "project_id" not in d or "name" not in d:
            raise ProjectValidationError(
                "Code payload missing required keys"
            )
        c = cls(
            id=str(d["id"]),
            project_id=str(d["project_id"]),
            name=str(d.get("name", "")),
            definition=str(d.get("definition", "") or ""),
            inclusion_criteria=str(d.get("inclusion_criteria", "") or ""),
            exclusion_criteria=str(d.get("exclusion_criteria", "") or ""),
            exemplars=[str(e) for e in (d.get("exemplars") or [])],
            parent_code_id=(
                str(d["parent_code_id"]) if d.get("parent_code_id") else None
            ),
            related_codes=[
                CodeRelation.from_dict(r)
                for r in (d.get("related_codes") or [])
            ],
            theoretical_memo=str(d.get("theoretical_memo", "") or ""),
            stage=str(d.get("stage", "initial") or "initial"),
            colour=str(d.get("colour", "") or ""),
            status=str(d.get("status", "active") or "active"),
            provenance={
                str(k): str(v)
                for k, v in (d.get("provenance") or {}).items()
            },
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        c.validate()
        return c

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(self, patch: dict[str, Any], *, now: str | None = None) -> None:
        """Apply a partial update in place. Mirrors ``Source.apply_update``.

        ``id``, ``project_id``, ``created_at``, and ``modified_at`` are
        ignored if present — they're managed by the entity, not the
        user. F2.2's revision history will read these timestamps to
        derive version snapshots.
        """
        if not isinstance(patch, dict):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS - _IGNORED_PATCH_KEYS
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "name" in patch:
            self.name = str(patch["name"] or "")
        if "definition" in patch:
            self.definition = str(patch["definition"] or "")
        if "inclusion_criteria" in patch:
            self.inclusion_criteria = str(patch["inclusion_criteria"] or "")
        if "exclusion_criteria" in patch:
            self.exclusion_criteria = str(patch["exclusion_criteria"] or "")
        if "exemplars" in patch:
            ex = patch["exemplars"] or []
            if not isinstance(ex, list):
                raise ProjectValidationError(
                    "exemplars must be a list of strings"
                )
            self.exemplars = [str(e) for e in ex]
        if "parent_code_id" in patch:
            v = patch["parent_code_id"]
            self.parent_code_id = str(v) if v else None
        if "related_codes" in patch:
            rels = patch["related_codes"] or []
            if not isinstance(rels, list):
                raise ProjectValidationError(
                    "related_codes must be a list of "
                    "{code_id, relation_type} objects"
                )
            self.related_codes = [
                r if isinstance(r, CodeRelation) else CodeRelation.from_dict(r)
                for r in rels
            ]
        if "theoretical_memo" in patch:
            self.theoretical_memo = str(patch["theoretical_memo"] or "")
        if "stage" in patch:
            self.stage = str(patch["stage"] or "")
        if "colour" in patch:
            self.colour = str(patch["colour"] or "")
        if "status" in patch:
            self.status = str(patch["status"] or "")
        if "provenance" in patch:
            prov = patch["provenance"] or {}
            if not isinstance(prov, dict):
                raise ProjectValidationError(
                    "provenance must be an object of string→string"
                )
            self.provenance = {str(k): str(v) for k, v in prov.items()}

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not CODE_ID_RE.match(self.id):
            raise ProjectValidationError(f"Invalid code id: {self.id!r}")
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        name = self.name.strip()
        if not name:
            raise ProjectValidationError("Code name is required")
        if len(name) > MAX_NAME_LEN:
            raise ProjectValidationError(
                f"Code name must be ≤ {MAX_NAME_LEN} chars"
            )
        # Persist the trimmed name so on-disk state is canonical.
        self.name = name

        if len(self.definition) > MAX_DEFINITION_LEN:
            raise ProjectValidationError(
                f"definition must be ≤ {MAX_DEFINITION_LEN} chars"
            )
        if len(self.inclusion_criteria) > MAX_INCLUSION_CRITERIA_LEN:
            raise ProjectValidationError(
                f"inclusion_criteria must be ≤ {MAX_INCLUSION_CRITERIA_LEN} chars"
            )
        if len(self.exclusion_criteria) > MAX_EXCLUSION_CRITERIA_LEN:
            raise ProjectValidationError(
                f"exclusion_criteria must be ≤ {MAX_EXCLUSION_CRITERIA_LEN} chars"
            )
        if len(self.theoretical_memo) > MAX_THEORETICAL_MEMO_LEN:
            raise ProjectValidationError(
                f"theoretical_memo must be ≤ {MAX_THEORETICAL_MEMO_LEN} chars"
            )

        # Exemplars: list of non-empty strings, bounded count and length.
        if not isinstance(self.exemplars, list):
            raise ProjectValidationError(
                "exemplars must be a list of strings"
            )
        if len(self.exemplars) > MAX_EXEMPLARS:
            raise ProjectValidationError(
                f"At most {MAX_EXEMPLARS} exemplars allowed"
            )
        cleaned_exemplars: list[str] = []
        for raw in self.exemplars:
            e = str(raw).strip()
            if not e:
                continue  # silently drop empties; less friction in UI
            if len(e) > MAX_EXEMPLAR_LEN:
                raise ProjectValidationError(
                    f"exemplar too long (>{MAX_EXEMPLAR_LEN}): {e[:40]!r}…"
                )
            cleaned_exemplars.append(e)
        self.exemplars = cleaned_exemplars

        # Parent code: optional, must be 12-hex if set, must not be self.
        if self.parent_code_id is not None:
            if not CODE_ID_RE.match(self.parent_code_id):
                raise ProjectValidationError(
                    f"parent_code_id must be 12-char hex; "
                    f"got {self.parent_code_id!r}"
                )
            if self.parent_code_id == self.id:
                raise ProjectValidationError(
                    "parent_code_id cannot reference the code itself"
                )

        # Related codes: typed links; can't reference self; de-dup on
        # (code_id, relation_type) pair.
        if not isinstance(self.related_codes, list):
            raise ProjectValidationError(
                "related_codes must be a list of "
                "{code_id, relation_type} objects"
            )
        if len(self.related_codes) > MAX_RELATED_CODES:
            raise ProjectValidationError(
                f"At most {MAX_RELATED_CODES} related codes allowed"
            )
        seen_rel: set[tuple[str, str]] = set()
        deduped_rel: list[CodeRelation] = []
        for r in self.related_codes:
            if not isinstance(r, CodeRelation):
                # Defensive: from_dict / apply_update normally coerce.
                r = CodeRelation.from_dict(r)  # type: ignore[arg-type]
            r.validate()
            if r.code_id == self.id:
                raise ProjectValidationError(
                    "related_codes cannot reference the code itself"
                )
            key = (r.code_id, r.relation_type)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            deduped_rel.append(r)
        self.related_codes = deduped_rel

        if self.stage not in CODEBOOK_STAGES:
            raise ProjectValidationError(
                f"stage must be one of {CODEBOOK_STAGES}; got {self.stage!r}"
            )

        if self.colour:
            if not CODE_COLOUR_RE.match(self.colour):
                raise ProjectValidationError(
                    f"colour must be #RGB or #RRGGBB hex; got {self.colour!r}"
                )

        if self.status not in CODE_STATUSES:
            raise ProjectValidationError(
                f"status must be one of {CODE_STATUSES}; got {self.status!r}"
            )

        # Provenance: free-form dict, but with bounded keys/values and a
        # validated `source` if present.
        if not isinstance(self.provenance, dict):
            raise ProjectValidationError(
                "provenance must be an object of string→string"
            )
        if len(self.provenance) > MAX_PROVENANCE_KEYS:
            raise ProjectValidationError(
                f"At most {MAX_PROVENANCE_KEYS} provenance keys allowed"
            )
        cleaned_prov: dict[str, str] = {}
        for raw_k, raw_v in self.provenance.items():
            k = str(raw_k).strip()
            if not k:
                continue
            if not PROVENANCE_KEY_RE.match(k):
                raise ProjectValidationError(
                    f"provenance key {k!r} invalid "
                    "(letters/digits/underscore/hyphen/space, "
                    "1–64 chars, must start with a letter)"
                )
            v = str(raw_v)
            if len(v) > MAX_PROVENANCE_VALUE_LEN:
                raise ProjectValidationError(
                    f"provenance[{k!r}] value too long "
                    f"(>{MAX_PROVENANCE_VALUE_LEN})"
                )
            cleaned_prov[k] = v
        # If a `source` key exists, it must be in the closed vocabulary.
        if "source" in cleaned_prov:
            if cleaned_prov["source"] not in CODE_PROVENANCE_SOURCES:
                raise ProjectValidationError(
                    f"provenance.source must be one of "
                    f"{CODE_PROVENANCE_SOURCES}; "
                    f"got {cleaned_prov['source']!r}"
                )
        self.provenance = cleaned_prov


# Fields a PATCH may set. id/project_id/created_at/modified_at are
# managed by the entity itself; passing them is allowed (and ignored)
# so a client can round-trip a fetched object.
_ALLOWED_PATCH_KEYS = {
    "name",
    "definition",
    "inclusion_criteria",
    "exclusion_criteria",
    "exemplars",
    "parent_code_id",
    "related_codes",
    "theoretical_memo",
    "stage",
    "colour",
    "status",
    "provenance",
}
_IGNORED_PATCH_KEYS = {"id", "project_id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #


def new_code_id() -> str:
    """Mint a new 12-char hex code id (matches project / source / job id shape)."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def codes_dir(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk directory holding a project's codes.

    Does not create it. Validates ``project_id`` to prevent traversal.
    """
    return project_dir(projects_root, project_id) / "codes"


def code_state_path(
    projects_root: Path, project_id: str, code_id: str
) -> Path:
    if not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(f"Invalid code id: {code_id!r}")
    return codes_dir(projects_root, project_id) / f"{code_id}.json"


def save_code(projects_root: Path, code: Code) -> Path:
    """Persist a code to ``<projects_root>/<pid>/codes/<cid>.json``.

    The parent ``projects/<pid>`` directory must already exist (i.e.
    the project itself must have been saved). Like ``save_source``, a
    code without a project is meaningless and we surface that early.
    """
    code.validate()
    parent = project_dir(projects_root, code.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its codes."
        )
    cd = codes_dir(projects_root, code.project_id)
    cd.mkdir(parents=True, exist_ok=True)
    target = code_state_path(projects_root, code.project_id, code.id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(code.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_code(
    projects_root: Path, project_id: str, code_id: str
) -> Code:
    """Load a code by id. Raises ``FileNotFoundError`` if missing."""
    p = code_state_path(projects_root, project_id, code_id)
    if not p.exists():
        raise FileNotFoundError(f"No code at {p}")
    return Code.from_dict(json.loads(p.read_text()))


def list_codes(projects_root: Path, project_id: str) -> list[Code]:
    """List all codes in a project.

    Skips files that don't parse as a valid Code so a single corrupt
    file doesn't break the codebook view (audit log will eventually
    surface this — F9.7). Sorted by ``created_at`` ascending so the
    natural order matches how the codebook was built up — line-by-line
    coding leaves a long trail of early codes that get consolidated
    later, and that timeline is methodologically meaningful.
    """
    cd = codes_dir(projects_root, project_id)
    if not cd.exists():
        return []
    out: list[Code] = []
    for f in sorted(cd.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if f.name.endswith(".json.tmp"):
            continue
        cid = f.stem
        if not CODE_ID_RE.match(cid):
            continue
        try:
            out.append(Code.from_dict(json.loads(f.read_text())))
        except (ProjectValidationError, json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda c: (c.created_at, c.id))
    return out


def delete_code(
    projects_root: Path, project_id: str, code_id: str
) -> bool:
    """Remove a code file. Returns False if it didn't exist.

    F2.3 will introduce a "retire" lifecycle that's preferable to a
    hard delete — retired codes preserve history and can still be
    queried in old reports. Hard delete is exposed here for tests and
    for the REFI-QDA import path (where a clean slate matters).
    """
    p = code_state_path(projects_root, project_id, code_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
