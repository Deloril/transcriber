"""Code-definition-at-apply reports (F9.2).

Per PLANNING.md F9.2:

  > Code definition versioning. Each edit creates a new version;
  > applications record version-at-apply. Reports can show "this code's
  > definition at the time this application was made."

The infrastructure for the first two clauses already exists:

* :mod:`scribe.code_versions` (F2.2) keeps the append-only per-code
  version log. Every definition-changing save records a snapshot.
* :mod:`scribe.applications` (F4.1) requires every Application to
  carry a ``definition_version_id_at_apply`` pointing at the version
  that was in force at apply time.

This module closes the loop. It resolves an application's recorded
version id back to the historical :class:`scribe.code_versions.CodeVersion`
snapshot, reconstructs a :class:`scribe.codes.Code` from that snapshot,
detects drift against the current code state, and renders a flat
audit-trail report in CSV / Markdown / RTF.

Why a dedicated module
----------------------

The retrieval report (F6.2) renders applications hydrated with the
**current** code/source/coder names — that's what a researcher wants
when writing up findings (or pasting quotes into a draft). F9.2's
question is different: *"under what definition was this application
made?"* — a reproducibility question, asked when a thesis examiner
raises an eyebrow at a coding decision two years on, or when a
co-author discovers the code's wording has shifted since the original
sweep. A separate module keeps that audit-trail surface uncluttered
and makes the data model explicit.

Pure-ish at the edge
--------------------

The two helpers that touch disk — :func:`lookup_definition_at_apply`
and :func:`build_definition_at_apply_rows` — read code-version JSONL
files via :func:`scribe.code_versions.find_code_version`. Everything
else (Code reconstruction, drift detection, formatters, dispatch) is
pure on dataclasses and strings, matching the style of
:mod:`scribe.retrieval_report` and :mod:`scribe.codebook_export`.

What's deferred
---------------

* Hooking this up as an HTTP endpoint and a CLI script is out of
  scope for F9.2 — the data-model + renderer surface is what F9 needs
  the rest of the trust-and-reproducibility stack to build on. F9.7
  (audit trail export) is the natural place to mount these renderers
  end-to-end.
* Diffing at the field level (highlighting *which* word changed in the
  inclusion criteria) is left to a future iteration; F9.2 surfaces a
  per-field equality flag so a future renderer can show which fields
  drifted without re-reading the snapshot.
* No JS surface. The eventual editor / report UI will hit the
  same Python helpers via the server layer.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .applications import Application
from .codebook_export import _rtf_escape, _rtf_para, _rtf_para_bold
from .code_versions import (
    CODE_VERSION_ID_RE,
    CodeVersion,
    DEFINITION_FIELDS,
    find_code_version,
)
from .codes import Code
from .projects import Project, ProjectValidationError


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# CSV columns. Order is part of the public contract — new columns go
# at the end so old consumer scripts keep working. Sentinels for
# missing snapshots / codes are empty cells, not literal "(missing)"
# strings, so a CSV consumer can detect them with an empty-cell test.
CSV_COLUMNS: tuple[str, ...] = (
    "application_id",
    "code_id",
    "source_id",
    "coder_id",
    "anchor_start_word_id",
    "anchor_end_word_id",
    "application_created_at",
    "version_id_at_apply",
    "version_number_at_apply",
    "version_recorded_at",
    "version_change_note",
    "name_at_apply",
    "definition_at_apply",
    "inclusion_criteria_at_apply",
    "exclusion_criteria_at_apply",
    "exemplars_at_apply",
    "theoretical_memo_at_apply",
    "current_name",
    "current_definition",
    "current_inclusion_criteria",
    "current_exclusion_criteria",
    "current_exemplars",
    "current_theoretical_memo",
    "snapshot_missing",
    "code_missing",
    "definition_drifted",
    "drifted_fields",
)

# Multi-valued cell separator. Same character used by retrieval_report
# and codebook_export so a researcher who learns the convention once
# carries it everywhere.
CSV_LIST_SEP = " | "


# --------------------------------------------------------------------------- #
# Code reconstruction from a snapshot
# --------------------------------------------------------------------------- #


def code_from_version_snapshot(version: CodeVersion) -> Code:
    """Reconstruct a :class:`Code` from the snapshot embedded in
    ``version``.

    A :class:`CodeVersion` stores the entire serialised Code state at
    a moment in time (see ``DEFINITION_FIELDS`` and the F2.2 design
    note). Reconstruction goes through :meth:`Code.from_dict`, which
    re-validates the payload — so an obviously-malformed historical
    snapshot raises a :class:`ProjectValidationError` rather than
    silently returning a half-built object.

    Tolerates missing-fields (older snapshots that predate a field
    addition) by leaning on ``Code.from_dict``'s ``.get`` defaults.
    """
    if not isinstance(version, CodeVersion):
        raise TypeError(
            "code_from_version_snapshot expects a CodeVersion; got "
            f"{type(version).__name__}"
        )
    if not isinstance(version.snapshot, dict):
        raise ProjectValidationError(
            "CodeVersion.snapshot must be a dict to reconstruct a Code"
        )
    return Code.from_dict(version.snapshot)


# --------------------------------------------------------------------------- #
# Resolver — version-at-apply for a single application
# --------------------------------------------------------------------------- #


def lookup_definition_at_apply(
    projects_root: Path, application: Application
) -> CodeVersion | None:
    """Return the :class:`CodeVersion` an application points at, or None.

    Reads the per-code version log on disk via
    :func:`scribe.code_versions.find_code_version`. ``None`` covers
    three cases legitimately:

      * The version log directory doesn't exist (the project hasn't
        been saved properly, or the code has been hard-deleted — F2.3
        prefers retiring, but the data layer doesn't enforce that).
      * The log exists but the specific ``definition_version_id_at_apply``
        is missing (corruption, partial restore from backup, or an
        application minted before the log was written).
      * The log exists and the id is present — this branch returns the
        :class:`CodeVersion` instance.

    ``definition_version_id_at_apply`` *must* match
    :data:`scribe.code_versions.CODE_VERSION_ID_RE`; if the application
    payload's value doesn't, that's a programming error and is
    surfaced as :class:`ProjectValidationError` (Application.validate
    would already have raised, but we guard defensively for callers
    that hand-build a payload).
    """
    if not isinstance(application, Application):
        raise TypeError(
            "lookup_definition_at_apply expects an Application; got "
            f"{type(application).__name__}"
        )
    vid = application.definition_version_id_at_apply
    if not CODE_VERSION_ID_RE.match(vid):
        raise ProjectValidationError(
            f"Invalid definition_version_id_at_apply on application "
            f"{application.id}: {vid!r}"
        )
    return find_code_version(
        projects_root,
        application.project_id,
        application.code_id,
        vid,
    )


# --------------------------------------------------------------------------- #
# Drift detection
# --------------------------------------------------------------------------- #


def drifted_definition_fields(
    snapshot: Mapping[str, Any] | None, current: Code | None
) -> tuple[str, ...]:
    """Return the definition fields whose value differs between
    ``snapshot`` and ``current``.

    ``DEFINITION_FIELDS`` (F2.2) drives the comparison — exactly the
    same closed set that decides whether a save records a new
    revision, so "drifted" here aligns with "would have triggered a
    new version" semantics.

    Either side may be ``None`` (snapshot missing → all fields
    "drifted" because we can't say either way; current missing → all
    fields drifted for the same reason). When both are present, list-
    typed fields (``exemplars``, ``related_codes``) are compared by
    value via JSON-normalisation through
    :func:`scribe.code_versions.definition_signature`-style projection.

    Returns a tuple in :data:`DEFINITION_FIELDS` order so the result
    is deterministic and stable across calls.
    """
    if snapshot is None or current is None:
        return tuple(DEFINITION_FIELDS)

    out: list[str] = []
    current_dict = current.to_dict()
    for f in DEFINITION_FIELDS:
        snap_val = snapshot.get(f)
        cur_val = current_dict.get(f)
        # Normalise list-typed fields so a missing key compares equal
        # to an explicit empty list (older snapshots may omit defaults).
        if f in ("exemplars", "related_codes"):
            if snap_val is None:
                snap_val = []
            if cur_val is None:
                cur_val = []
        if snap_val != cur_val:
            out.append(f)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Row data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DefinitionAtApply:
    """One application paired with the code definition that was in
    force when it was made.

    Frozen for hashability + safety — a row built once is the source
    of truth for every renderer. ``exemplars_at_apply`` and
    ``current_exemplars`` are tuples (rather than lists) so the
    dataclass is hashable and a row can land in a set / be a dict key.

    ``snapshot_missing`` is ``True`` when the
    ``definition_version_id_at_apply`` couldn't be resolved on disk
    (see :func:`lookup_definition_at_apply` for the three legitimate
    causes). When ``True``, all ``*_at_apply`` content fields are
    empty strings or empty tuples — renderers should surface a clear
    placeholder (see :func:`to_markdown` / :func:`to_rtf`).

    ``code_missing`` is ``True`` when the row was built without a
    matching current :class:`Code` in the supplied lookup table — the
    code may have been hard-deleted, or simply not passed in. Drift
    detection treats this as "all fields drifted".

    ``definition_drifted`` is a convenience flag mirroring
    ``len(drifted_fields) > 0``. Useful when a CSV consumer wants a
    boolean column without re-parsing ``drifted_fields``.
    """

    application_id: str
    code_id: str
    source_id: str
    coder_id: str
    anchor_start_word_id: str
    anchor_end_word_id: str
    application_created_at: str

    # version-at-apply pointer
    version_id_at_apply: str
    version_number_at_apply: int  # 0 when snapshot is missing
    version_recorded_at: str  # "" when missing
    version_change_note: str  # "" when missing

    # snapshot fields at apply (empty when missing)
    name_at_apply: str
    definition_at_apply: str
    inclusion_criteria_at_apply: str
    exclusion_criteria_at_apply: str
    exemplars_at_apply: tuple[str, ...]
    theoretical_memo_at_apply: str

    # current code state for drift comparison (empty when code missing)
    current_name: str
    current_definition: str
    current_inclusion_criteria: str
    current_exclusion_criteria: str
    current_exemplars: tuple[str, ...]
    current_theoretical_memo: str

    # bookkeeping
    snapshot_missing: bool
    code_missing: bool
    definition_drifted: bool
    drifted_fields: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #


def _exemplars_tuple(value: object) -> tuple[str, ...]:
    """Coerce a snapshot/code's exemplars field to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(e) for e in value)
    return ()


def _row_from(
    application: Application,
    version: CodeVersion | None,
    current: Code | None,
) -> DefinitionAtApply:
    """Pure row builder. Disk-free; the caller hands in the resolved
    :class:`CodeVersion` and the current :class:`Code` (if any)."""
    snapshot: dict[str, Any] = {}
    snapshot_missing = True
    version_number = 0
    version_recorded_at = ""
    version_change_note = ""
    if version is not None:
        snapshot = dict(version.snapshot or {})
        snapshot_missing = False
        version_number = int(version.version)
        version_recorded_at = str(version.created_at)
        version_change_note = str(version.change_note or "")

    code_missing = current is None

    drifted = drifted_definition_fields(
        snapshot if not snapshot_missing else None,
        current,
    )

    return DefinitionAtApply(
        application_id=application.id,
        code_id=application.code_id,
        source_id=application.source_id,
        coder_id=application.coder_id,
        anchor_start_word_id=application.anchor_start_word_id,
        anchor_end_word_id=application.anchor_end_word_id,
        application_created_at=application.created_at,
        version_id_at_apply=application.definition_version_id_at_apply,
        version_number_at_apply=version_number,
        version_recorded_at=version_recorded_at,
        version_change_note=version_change_note,
        name_at_apply=str(snapshot.get("name", "") or ""),
        definition_at_apply=str(snapshot.get("definition", "") or ""),
        inclusion_criteria_at_apply=str(
            snapshot.get("inclusion_criteria", "") or ""
        ),
        exclusion_criteria_at_apply=str(
            snapshot.get("exclusion_criteria", "") or ""
        ),
        exemplars_at_apply=_exemplars_tuple(snapshot.get("exemplars")),
        theoretical_memo_at_apply=str(
            snapshot.get("theoretical_memo", "") or ""
        ),
        current_name=current.name if current is not None else "",
        current_definition=current.definition if current is not None else "",
        current_inclusion_criteria=(
            current.inclusion_criteria if current is not None else ""
        ),
        current_exclusion_criteria=(
            current.exclusion_criteria if current is not None else ""
        ),
        current_exemplars=(
            tuple(current.exemplars) if current is not None else ()
        ),
        current_theoretical_memo=(
            current.theoretical_memo if current is not None else ""
        ),
        snapshot_missing=snapshot_missing,
        code_missing=code_missing,
        definition_drifted=bool(drifted) and not snapshot_missing,
        drifted_fields=drifted,
    )


def build_definition_at_apply_rows(
    projects_root: Path,
    applications: Sequence[Application],
    *,
    codes: Sequence[Code] = (),
) -> list[DefinitionAtApply]:
    """Hydrate applications into a list of :class:`DefinitionAtApply`.

    Reads the code-versions JSONL log on disk for every application's
    ``definition_version_id_at_apply``. ``codes`` is the *current*
    codebook used for drift comparison; pass an empty list to build
    snapshot-only rows (every row's ``code_missing`` will be ``True``,
    every row's ``drifted_fields`` will be the full
    ``DEFINITION_FIELDS`` tuple — that's still a useful audit view).

    Order matches the input order — callers who want a specific
    grouping should sort the input. The renderers don't impose a sort
    of their own (different from :mod:`scribe.retrieval_report`,
    which groups; F9.2's audit story prefers a flat, deterministic
    layout).

    Reads each code's version log at most once per call by caching
    :class:`CodeVersion` lookups by (code_id, version_id).
    """
    code_index: dict[str, Code] = {}
    for c in codes:
        if not isinstance(c, Code):
            raise TypeError(
                "codes must be an iterable of Code instances"
            )
        code_index[c.id] = c

    # Cache (code_id, version_id) → CodeVersion | None to avoid
    # re-reading a code's JSONL log for every application that points
    # at the same version.
    cache: dict[tuple[str, str], CodeVersion | None] = {}

    rows: list[DefinitionAtApply] = []
    for app in applications:
        if not isinstance(app, Application):
            raise TypeError(
                "applications must be an iterable of Application instances"
            )
        key = (app.code_id, app.definition_version_id_at_apply)
        if key in cache:
            version = cache[key]
        else:
            version = lookup_definition_at_apply(projects_root, app)
            cache[key] = version
        current = code_index.get(app.code_id)
        rows.append(_row_from(app, version, current))
    return rows


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _bool_cell(v: bool) -> str:
    """Render a boolean for CSV: ``"true"`` / ``"false"`` (lowercase).

    Excel accepts both ``TRUE`` / ``true`` but the lowercase form
    matches the rest of Scribe's text-based outputs (``status: active``
    in the codebook export, etc.) and reads cleaner in a diff.
    """
    return "true" if v else "false"


def to_csv(rows: Sequence[DefinitionAtApply]) -> str:
    """Serialise rows to CSV (RFC 4180, ``\\r\\n`` line endings).

    Empty input produces a header-only document — a valid "no
    applications" report, not an error. Multi-valued cells
    (``exemplars_at_apply``, ``current_exemplars``, ``drifted_fields``)
    are joined with :data:`CSV_LIST_SEP`.
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for r in rows:
        writer.writerow(
            [
                r.application_id,
                r.code_id,
                r.source_id,
                r.coder_id,
                r.anchor_start_word_id,
                r.anchor_end_word_id,
                r.application_created_at,
                r.version_id_at_apply,
                str(r.version_number_at_apply),
                r.version_recorded_at,
                r.version_change_note,
                r.name_at_apply,
                r.definition_at_apply,
                r.inclusion_criteria_at_apply,
                r.exclusion_criteria_at_apply,
                CSV_LIST_SEP.join(r.exemplars_at_apply),
                r.theoretical_memo_at_apply,
                r.current_name,
                r.current_definition,
                r.current_inclusion_criteria,
                r.current_exclusion_criteria,
                CSV_LIST_SEP.join(r.current_exemplars),
                r.current_theoretical_memo,
                _bool_cell(r.snapshot_missing),
                _bool_cell(r.code_missing),
                _bool_cell(r.definition_drifted),
                CSV_LIST_SEP.join(r.drifted_fields),
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


PLACEHOLDER_NO_SNAPSHOT = (
    "_(no version snapshot found — version log may have been "
    "deleted or this application predates the version log)_"
)
PLACEHOLDER_NO_CURRENT = (
    "_(code is no longer in the current codebook — drift comparison "
    "skipped)_"
)


def _md_definition_block(
    label: str, body: str, *, fallback: str = "_(empty)_"
) -> list[str]:
    """Render a labelled definition block as Markdown lines.

    Uses a bold label and a Markdown blockquote body so a
    multi-paragraph definition reads cleanly. Empty bodies render the
    fallback placeholder (italicised) so the absence is visible
    rather than swallowed.
    """
    out: list[str] = [f"**{label}**", ""]
    if not body:
        out.append(fallback)
        out.append("")
        return out
    for ln in body.splitlines() or [""]:
        out.append(f"> {ln}" if ln else ">")
    out.append("")
    return out


def _md_exemplars_block(
    label: str, items: Sequence[str], *, fallback: str = "_(none)_"
) -> list[str]:
    """Render a labelled exemplar list as Markdown bullets."""
    out: list[str] = [f"**{label}**", ""]
    if not items:
        out.append(fallback)
        out.append("")
        return out
    for ex in items:
        out.append(f"- {ex}")
    out.append("")
    return out


def _md_row_block(r: DefinitionAtApply) -> list[str]:
    """Render one row as a Markdown sub-section."""
    out: list[str] = []
    out.append(f"### Application `{r.application_id}`")
    out.append("")
    meta_bits: list[str] = [
        f"code: `{r.code_id}`",
        f"source: `{r.source_id}`",
        f"coder: `{r.coder_id}`",
        f"anchor: `{r.anchor_start_word_id}`–`{r.anchor_end_word_id}`",
    ]
    if r.application_created_at:
        meta_bits.append(f"applied: {r.application_created_at}")
    out.append(" · ".join(meta_bits))
    out.append("")

    # At-apply version header
    if r.snapshot_missing:
        out.append(
            f"**Version at apply**: `{r.version_id_at_apply}` "
            "(snapshot not found)"
        )
        out.append("")
        out.append(PLACEHOLDER_NO_SNAPSHOT)
        out.append("")
    else:
        recorded = (
            f", recorded {r.version_recorded_at}"
            if r.version_recorded_at
            else ""
        )
        out.append(
            f"**Version at apply**: v{r.version_number_at_apply} "
            f"(`{r.version_id_at_apply}`{recorded})"
        )
        if r.version_change_note:
            out.append("")
            out.append(f"_Change note:_ {r.version_change_note}")
        out.append("")

        out.append(f"**Name at apply:** {r.name_at_apply or '_(unnamed)_'}")
        out.append("")
        out.extend(
            _md_definition_block("Definition at apply", r.definition_at_apply)
        )
        if r.inclusion_criteria_at_apply:
            out.extend(
                _md_definition_block(
                    "Inclusion criteria at apply",
                    r.inclusion_criteria_at_apply,
                )
            )
        if r.exclusion_criteria_at_apply:
            out.extend(
                _md_definition_block(
                    "Exclusion criteria at apply",
                    r.exclusion_criteria_at_apply,
                )
            )
        if r.exemplars_at_apply:
            out.extend(
                _md_exemplars_block(
                    "Exemplars at apply", r.exemplars_at_apply
                )
            )
        if r.theoretical_memo_at_apply:
            out.extend(
                _md_definition_block(
                    "Theoretical memo at apply",
                    r.theoretical_memo_at_apply,
                )
            )

    # Drift section — only emitted when there's a meaningful comparison.
    if r.code_missing:
        out.append(PLACEHOLDER_NO_CURRENT)
        out.append("")
    elif r.definition_drifted and not r.snapshot_missing:
        out.append("**Definition drift since apply**")
        out.append("")
        out.append(
            "Fields changed: "
            + ", ".join(f"`{f}`" for f in r.drifted_fields)
        )
        out.append("")
        if "name" in r.drifted_fields:
            out.append(f"_Current name:_ {r.current_name or '_(unnamed)_'}")
            out.append("")
        if "definition" in r.drifted_fields:
            out.extend(
                _md_definition_block(
                    "Current definition", r.current_definition
                )
            )
        if "inclusion_criteria" in r.drifted_fields:
            out.extend(
                _md_definition_block(
                    "Current inclusion criteria",
                    r.current_inclusion_criteria,
                )
            )
        if "exclusion_criteria" in r.drifted_fields:
            out.extend(
                _md_definition_block(
                    "Current exclusion criteria",
                    r.current_exclusion_criteria,
                )
            )
        if "exemplars" in r.drifted_fields:
            out.extend(
                _md_exemplars_block(
                    "Current exemplars", r.current_exemplars
                )
            )
        if "theoretical_memo" in r.drifted_fields:
            out.extend(
                _md_definition_block(
                    "Current theoretical memo",
                    r.current_theoretical_memo,
                )
            )
    else:
        # Either no drift, or snapshot missing (already explained above).
        if not r.snapshot_missing:
            out.append("_Definition unchanged since apply._")
            out.append("")

    return out


def to_markdown(
    rows: Sequence[DefinitionAtApply],
    *,
    project: Project | None = None,
) -> str:
    """Serialise rows to a structured Markdown audit document.

    Layout:

      1. ``# Definition at apply`` heading (with project name when
         available).
      2. Optional project metadata bullets (methodology, codebook
         stage, row count).
      3. One ``### Application <id>`` block per row containing the
         version pointer, the at-apply snapshot, and a "drift since
         apply" section when the code's current definition differs.

    Empty input renders the heading + ``_(no applications)_`` — a
    valid empty audit, not an error.
    """
    lines: list[str] = []
    title = "Definition at apply"
    if project is not None and project.name.strip():
        title = f"Definition at apply — {project.name}"
    lines.append(f"# {title}")
    lines.append("")

    if project is not None:
        meta_rows: list[tuple[str, str]] = []
        if project.methodology:
            meta_rows.append(("Methodology", project.methodology))
        if project.codebook_stage:
            meta_rows.append(("Stage", project.codebook_stage))
        meta_rows.append(("Applications", str(len(rows))))
        drifted_count = sum(
            1 for r in rows if r.definition_drifted and not r.snapshot_missing
        )
        meta_rows.append(("Drifted", str(drifted_count)))
        for label, value in meta_rows:
            lines.append(f"- **{label}**: {value}")
        lines.append("")

    if not rows:
        lines.append("_(no applications)_")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    for r in rows:
        lines.extend(_md_row_block(r))

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# RTF
# --------------------------------------------------------------------------- #


def _rtf_label_block(label: str, body: str) -> list[str]:
    """Bold label paragraph followed by italic body paragraph(s)."""
    out: list[str] = [_rtf_para_bold(label)]
    if not body:
        out.append(r"\i (empty)\i0\par ")
        return out
    for ln in body.splitlines() or [""]:
        out.append(r"\i " + _rtf_escape(ln) + r"\i0\par ")
    return out


def _rtf_exemplars_block(label: str, items: Sequence[str]) -> list[str]:
    """Bold label followed by bullet paragraphs for each exemplar."""
    out: list[str] = [_rtf_para_bold(label)]
    if not items:
        out.append(_rtf_para("(none)"))
        return out
    for ex in items:
        out.append(_rtf_para(f"•\t{ex}"))
    return out


def _rtf_row_block(r: DefinitionAtApply) -> list[str]:
    """Render one row as RTF."""
    out: list[str] = []
    out.append(_rtf_para_bold(f"Application {r.application_id}", fs=28))

    meta_bits: list[str] = [
        f"code: {r.code_id}",
        f"source: {r.source_id}",
        f"coder: {r.coder_id}",
        f"anchor: {r.anchor_start_word_id}–{r.anchor_end_word_id}",
    ]
    if r.application_created_at:
        meta_bits.append(f"applied: {r.application_created_at}")
    out.append(_rtf_para(" · ".join(meta_bits)))

    if r.snapshot_missing:
        out.append(
            _rtf_para(
                f"Version at apply: {r.version_id_at_apply} "
                "(snapshot not found)"
            )
        )
        out.append(
            _rtf_para(
                "(no version snapshot found — version log may have been "
                "deleted or this application predates the version log)"
            )
        )
    else:
        recorded = (
            f", recorded {r.version_recorded_at}"
            if r.version_recorded_at
            else ""
        )
        out.append(
            _rtf_para(
                f"Version at apply: v{r.version_number_at_apply} "
                f"({r.version_id_at_apply}{recorded})"
            )
        )
        if r.version_change_note:
            out.append(_rtf_para(f"Change note: {r.version_change_note}"))

        out.append(
            _rtf_para_bold(f"Name at apply: {r.name_at_apply or '(unnamed)'}")
        )
        out.extend(
            _rtf_label_block("Definition at apply", r.definition_at_apply)
        )
        if r.inclusion_criteria_at_apply:
            out.extend(
                _rtf_label_block(
                    "Inclusion criteria at apply",
                    r.inclusion_criteria_at_apply,
                )
            )
        if r.exclusion_criteria_at_apply:
            out.extend(
                _rtf_label_block(
                    "Exclusion criteria at apply",
                    r.exclusion_criteria_at_apply,
                )
            )
        if r.exemplars_at_apply:
            out.extend(
                _rtf_exemplars_block(
                    "Exemplars at apply", r.exemplars_at_apply
                )
            )
        if r.theoretical_memo_at_apply:
            out.extend(
                _rtf_label_block(
                    "Theoretical memo at apply",
                    r.theoretical_memo_at_apply,
                )
            )

    if r.code_missing:
        out.append(
            _rtf_para(
                "(code is no longer in the current codebook — drift "
                "comparison skipped)"
            )
        )
    elif r.definition_drifted and not r.snapshot_missing:
        out.append(_rtf_para_bold("Definition drift since apply"))
        out.append(
            _rtf_para(
                "Fields changed: " + ", ".join(r.drifted_fields)
            )
        )
        if "name" in r.drifted_fields:
            out.append(
                _rtf_para(
                    f"Current name: {r.current_name or '(unnamed)'}"
                )
            )
        if "definition" in r.drifted_fields:
            out.extend(
                _rtf_label_block("Current definition", r.current_definition)
            )
        if "inclusion_criteria" in r.drifted_fields:
            out.extend(
                _rtf_label_block(
                    "Current inclusion criteria",
                    r.current_inclusion_criteria,
                )
            )
        if "exclusion_criteria" in r.drifted_fields:
            out.extend(
                _rtf_label_block(
                    "Current exclusion criteria",
                    r.current_exclusion_criteria,
                )
            )
        if "exemplars" in r.drifted_fields:
            out.extend(
                _rtf_exemplars_block(
                    "Current exemplars", r.current_exemplars
                )
            )
        if "theoretical_memo" in r.drifted_fields:
            out.extend(
                _rtf_label_block(
                    "Current theoretical memo",
                    r.current_theoretical_memo,
                )
            )
    elif not r.snapshot_missing:
        out.append(_rtf_para("Definition unchanged since apply."))

    out.append(r"\par ")
    return out


def to_rtf(
    rows: Sequence[DefinitionAtApply],
    *,
    project: Project | None = None,
) -> str:
    """Serialise rows to a minimal RTF 1.x document.

    Word, LibreOffice, and Pages all open RTF natively. Output is
    ASCII-encoded with Unicode characters escaped per the RTF
    ``\\uNNNN?`` rule (matches :mod:`scribe.codebook_export.to_rtf`).
    """
    parts: list[str] = []
    parts.append(r"{\rtf1\ansi\ansicpg1252\deff0")
    parts.append(r"{\fonttbl{\f0\fnil Calibri;}}")
    parts.append(r"\fs22")  # 11pt body

    title = "Definition at apply"
    if project is not None and project.name.strip():
        title = f"Definition at apply — {project.name}"
    parts.append(_rtf_para_bold(title, fs=36))

    if project is not None:
        meta: list[str] = []
        if project.methodology:
            meta.append(f"Methodology: {project.methodology}")
        if project.codebook_stage:
            meta.append(f"Stage: {project.codebook_stage}")
        meta.append(f"Applications: {len(rows)}")
        drifted_count = sum(
            1 for r in rows if r.definition_drifted and not r.snapshot_missing
        )
        meta.append(f"Drifted: {drifted_count}")
        for ml in meta:
            parts.append(_rtf_para(ml))
        parts.append(r"\par ")

    if not rows:
        parts.append(_rtf_para("(no applications)"))
        parts.append("}")
        return "".join(parts)

    for r in rows:
        parts.extend(_rtf_row_block(r))

    parts.append("}")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Format registry + dispatch
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FormatSpec:
    """Static description of a user-facing format. Mirrors the
    :class:`scribe.retrieval_report.FormatSpec` shape so the HTTP and
    CLI surface look the same regardless of which body they're
    fetching."""

    key: str
    extension: str
    media_type: str
    label: str


EXPORT_FORMAT_CSV = "csv"
EXPORT_FORMAT_MARKDOWN = "markdown"
EXPORT_FORMAT_RTF = "rtf"


EXPORT_FORMATS: dict[str, FormatSpec] = {
    EXPORT_FORMAT_CSV: FormatSpec(
        key=EXPORT_FORMAT_CSV,
        extension=".csv",
        media_type="text/csv; charset=utf-8",
        label="CSV",
    ),
    EXPORT_FORMAT_MARKDOWN: FormatSpec(
        key=EXPORT_FORMAT_MARKDOWN,
        extension=".md",
        media_type="text/markdown; charset=utf-8",
        label="Markdown",
    ),
    EXPORT_FORMAT_RTF: FormatSpec(
        key=EXPORT_FORMAT_RTF,
        extension=".rtf",
        media_type="application/rtf",
        label="RTF (Word)",
    ),
}


_FORMAT_ALIASES: dict[str, str] = {
    "md": EXPORT_FORMAT_MARKDOWN,
    "markdown": EXPORT_FORMAT_MARKDOWN,
    "csv": EXPORT_FORMAT_CSV,
    "rtf": EXPORT_FORMAT_RTF,
    "word": EXPORT_FORMAT_RTF,
    "doc": EXPORT_FORMAT_RTF,
    "docx": EXPORT_FORMAT_RTF,
}


def normalise_format(format: str | None) -> str:
    """Resolve a caller-supplied format string to a canonical key.

    Trims + lower-cases. Accepts the same alias set as
    :func:`scribe.retrieval_report.normalise_format` so a user who
    learns one CLI flag set carries it across.
    """
    if format is None:
        raise ValueError(
            "Definition-at-apply format is required; expected one of: "
            f"{sorted(EXPORT_FORMATS.keys())}"
        )
    key = str(format).strip().lower()
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported definition-at-apply format: {format!r}. "
        f"Expected one of: {sorted(EXPORT_FORMATS.keys())}"
    )


_RENDERERS: dict[str, Callable[..., str]] = {}


def render_report(
    format: str,
    rows: Sequence[DefinitionAtApply],
    *,
    project: Project | None = None,
) -> str:
    """Render rows in ``format``; dispatches to the right renderer.

    ``project`` is forwarded to Markdown / RTF only — CSV's column
    contract is the public schema and intentionally excludes a
    project header (same as :mod:`scribe.retrieval_report`).
    """
    fmt = normalise_format(format)
    return _RENDERERS[fmt](rows, project=project)


def _render_csv(
    rows: Sequence[DefinitionAtApply], *, project: Project | None
) -> str:
    del project
    return to_csv(rows)


def _render_markdown(
    rows: Sequence[DefinitionAtApply], *, project: Project | None
) -> str:
    return to_markdown(rows, project=project)


def _render_rtf(
    rows: Sequence[DefinitionAtApply], *, project: Project | None
) -> str:
    return to_rtf(rows, project=project)


_RENDERERS[EXPORT_FORMAT_CSV] = _render_csv
_RENDERERS[EXPORT_FORMAT_MARKDOWN] = _render_markdown
_RENDERERS[EXPORT_FORMAT_RTF] = _render_rtf


# --------------------------------------------------------------------------- #
# Filename / disk-write helpers
# --------------------------------------------------------------------------- #


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FILENAME_SLUG_MAX = 80


def slugify_report_filename(
    project: Project | None, format: str
) -> str:
    """Build a download-friendly filename for a definition-at-apply
    export.

    Pattern: ``<project-slug>-definition-at-apply<ext>`` if a project
    name is available; ``definition-at-apply<ext>`` otherwise. ASCII-
    only, lowercased, dash-separated, capped at
    :data:`_FILENAME_SLUG_MAX` characters before the suffix.
    """
    fmt = normalise_format(format)
    spec = EXPORT_FORMATS[fmt]
    slug = ""
    if project is not None and project.name and project.name.strip():
        ascii_name = (
            unicodedata.normalize("NFKD", project.name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        slug = _SLUG_RE.sub("-", ascii_name.lower()).strip("-")
        if len(slug) > _FILENAME_SLUG_MAX:
            slug = slug[:_FILENAME_SLUG_MAX].rstrip("-")
    if slug:
        return f"{slug}-definition-at-apply{spec.extension}"
    return f"definition-at-apply{spec.extension}"


def write_report(
    path: Path,
    format: str,
    rows: Sequence[DefinitionAtApply],
    *,
    project: Project | None = None,
) -> Path:
    """Render the report to ``format`` and atomically write to ``path``.

    Atomic via a ``.tmp`` swap so an interrupted write never leaves a
    half-finished export visible. Creates ``path.parent`` if missing.
    Returns the resolved target path.
    """
    fmt = normalise_format(format)
    text = render_report(fmt, rows, project=project)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(target)
    return target


__all__ = [
    "CSV_COLUMNS",
    "CSV_LIST_SEP",
    "DefinitionAtApply",
    "EXPORT_FORMAT_CSV",
    "EXPORT_FORMAT_MARKDOWN",
    "EXPORT_FORMAT_RTF",
    "EXPORT_FORMATS",
    "FormatSpec",
    "PLACEHOLDER_NO_CURRENT",
    "PLACEHOLDER_NO_SNAPSHOT",
    "build_definition_at_apply_rows",
    "code_from_version_snapshot",
    "drifted_definition_fields",
    "lookup_definition_at_apply",
    "normalise_format",
    "render_report",
    "slugify_report_filename",
    "to_csv",
    "to_markdown",
    "to_rtf",
    "write_report",
]
