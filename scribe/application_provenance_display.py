"""Per-application provenance display on hover (F9.9).

Per PLANNING.md F9.9:

  > Per-application provenance display on hover.

The data model has been carrying everything needed to answer "who
made this coded segment, when, under which definition, and was an
AI involved?" since F4.1 / F2.2 / F8.9 / F9.2. F9.9 closes the loop:
a stable, deterministic *display surface* that the editor can hand
to a tooltip / popover when a researcher rests their cursor on a
coded segment in the gutter (F4.3) or inline highlight.

Why a dedicated module
----------------------

The retrieval report (F6.2) and definition-at-apply audit (F9.2)
already render application provenance — but in the *long-form,
written-up* mode researchers paste into a thesis. F9.9's question
is different: *"in two seconds, while skimming, who applied this
code and on what authority?"* That's a UX / formatting concern, not
an audit concern. A separate module keeps both surfaces clean.

Pure, deterministic
-------------------

Like :mod:`scribe.application_gutter` and
:mod:`scribe.application_playback`, F9.9 is stand-alone: no FastAPI,
no engine, no disk I/O. Callers fetch the related entities (code,
code-version-at-apply, coder, optionally the source's display name)
via the existing per-entity loaders and hand them to the builder.
The renderers consume the resulting :class:`ProvenanceDisplay`
without further lookups, so the same data round-trips through HTTP,
the CLI, and the in-browser tooltip.

A JS mirror lives in ``scribe/static/js/helpers.mjs``
(``buildProvenanceDisplay``, ``formatProvenanceText``,
``formatProvenanceHtml``, ``provenanceSummaryLabel``) and must
agree with the Python output for any shared input. The tests
(``tests/test_application_provenance_display.py`` and
``tests/js/application-provenance-display.test.mjs``) pin both
sides to the same fixtures.

What this module is **not**
---------------------------

* Not a renderer of HTML structure — it produces structured display
  data + a few formatters (text and HTML-safe). The tooltip's CSS,
  positioning, and trigger logic live in the editor template.
* Not a writer. Read-only over its inputs; no mutation, no I/O.
* Not a drift report — that's F9.2 :mod:`scribe.definition_at_apply`.
  But we *do* surface a one-line "definition has changed since
  apply" hint (and reuse F9.2's :func:`drifted_definition_fields`
  when a current code is supplied) so the hover surfaces audit
  drift inline rather than hiding it behind a separate report.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .applications import (
    APPLICATION_PROVENANCE_SOURCES,
    Application,
)
from .coders import Coder
from .codes import Code
from .code_versions import CodeVersion
from .definition_at_apply import drifted_definition_fields


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


# Human-readable labels for the closed Application provenance vocabulary.
# Keep these short enough to fit in a hover tooltip without wrapping.
PROVENANCE_SOURCE_LABELS: dict[str, str] = {
    "human": "Human-coded",
    "ai_accepted": "AI-suggested · accepted",
    "ai_modified": "AI-suggested · accepted with edits",
    "imported": "Imported",
    "other": "Other",
}

# Default label when ``provenance.source`` is empty (early F4.1
# applications don't always carry one). "Human-coded" is the right
# default per PLANNING §"AI suggests, the human applies".
DEFAULT_PROVENANCE_SOURCE_LABEL = PROVENANCE_SOURCE_LABELS["human"]

# Human-readable feature labels for AI-touched applications. Mirrors
# :data:`scribe.ai_provenance.AI_FEATURES`. ``other`` is included so
# future features can ride the same vocabulary without a UI change.
AI_FEATURE_LABELS: dict[str, str] = {
    "code_suggestion": "Code suggestion",
    "new_code_suggestion": "New code suggestion",
    "quote_similarity": "Quote similarity",
    "transcript_review": "Transcript review",
    "second_coder": "AI second coder",
    "memo_draft": "Memo draft",
    "other": "Other AI",
}

# Human-readable AI decision labels. Mirrors
# :data:`scribe.ai_provenance.AI_DECISIONS`.
AI_DECISION_LABELS: dict[str, str] = {
    "pending": "Pending",
    "accepted": "Accepted",
    "modified": "Accepted with edits",
    "rejected": "Rejected",
}

# Free-form provenance keys we interpret specially in the legacy
# :class:`Application.provenance` dict (see F4.1 docstring). All
# other keys are surfaced verbatim in a "details" block.
_RESERVED_PROVENANCE_KEYS: tuple[str, ...] = (
    "source",
    "model_id",
    "embedding_model",
    "suggestion_id",
    "accepted_at",
    "feature",
    "backend",
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProvenanceDisplay:
    """Flat structured summary of an application's provenance.

    Designed for a tooltip / popover: every field is a string (or
    tuple of strings) that the renderer can display without further
    lookups. Empty strings / empty tuples mean "no value" — the
    renderer skips them rather than showing a placeholder, except
    where explicitly useful (e.g. ``"(unnamed)"`` for a missing
    code name).

    Two flags carry interpretation hints:

    * ``definition_drifted`` — the code's current definition differs
      from the snapshot at apply. Displays a small "definition has
      changed since apply" line so the reviewer sees the drift
      without opening the F9.2 report.
    * ``snapshot_missing`` — the version-at-apply pointer didn't
      resolve to an on-disk snapshot. The hover should still display
      the application's other fields so the user has *something*
      to act on.
    """

    application_id: str
    anchor_label: str  # e.g. "s0w0–s0w12"
    created_at: str
    modified_at: str
    confidence: str  # rendered string ("0.82") or "" when unset
    note: str

    # Code (current) — rendered when caller supplies a Code instance.
    code_id: str
    code_name: str  # "(unknown)" if no Code supplied
    code_colour: str  # "" or "#RRGGBB"
    code_stage: str

    # Code version at apply
    version_id_at_apply: str
    version_number_at_apply: str  # "v3" or "" when missing
    version_recorded_at: str
    version_change_note: str
    snapshot_missing: bool
    name_at_apply: str

    # Coder
    coder_id: str
    coder_name: str  # "(unknown)" if no Coder supplied
    coder_role: str

    # Source (optional) — only the display name; full Source not needed.
    source_id: str
    source_name: str  # "" when caller didn't supply it

    # Provenance source vocabulary (closed set).
    provenance_source: str  # raw value, e.g. "human" / "ai_accepted"
    provenance_source_label: str  # localised label for display

    # Structured AI provenance (F8.9), rendered when present.
    ai_present: bool
    ai_feature: str
    ai_feature_label: str
    ai_backend: str
    ai_generation_model: str
    ai_embedding_model: str
    ai_suggestion_id: str
    ai_decision: str
    ai_decision_label: str
    ai_decided_by_coder_id: str
    ai_decided_by_coder_name: str  # "(unknown)" / "" when no coder ref
    ai_decided_at: str
    ai_confidence: str
    ai_prompt_hash: str
    ai_notes: str

    # Drift bookkeeping (relative to caller-supplied current Code).
    code_missing: bool  # caller didn't supply a current Code
    definition_drifted: bool
    drifted_fields: tuple[str, ...]

    # Free-form extra keys from Application.provenance (after the
    # reserved ones are extracted). Each entry is ``"key: value"``.
    extra_provenance: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def _format_anchor(application: Application) -> str:
    """Compact anchor label like ``"s0w0–s0w12"`` (en-dash).

    Single-word anchors collapse to one id (no dash) so a one-word
    code reads naturally on hover.
    """
    s = application.anchor_start_word_id
    e = application.anchor_end_word_id
    if s == e and application.start_char_offset is None and application.end_char_offset is None:
        return s
    return f"{s}–{e}"


def _format_confidence(value: float | None) -> str:
    """Render a [0,1] confidence as a 2-decimal string, or ``""``."""
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if f != f or f == float("inf") or f == float("-inf"):
        return ""
    return f"{f:.2f}"


def _coder_label(coder: Coder | None) -> tuple[str, str]:
    """Return ``(name, role)`` for a coder, with sensible fallbacks."""
    if coder is None:
        return ("", "")
    name = (coder.name or "").strip() or "(unnamed)"
    role = (coder.role or "").strip()
    return (name, role)


def build_provenance_display(
    application: Application,
    *,
    code: Code | None = None,
    code_version: CodeVersion | None = None,
    coder: Coder | None = None,
    decided_by_coder: Coder | None = None,
    source_name: str = "",
) -> ProvenanceDisplay:
    """Build a :class:`ProvenanceDisplay` from an application + relations.

    All ``code`` / ``code_version`` / ``coder`` / ``decided_by_coder``
    arguments are optional — a hover surface that doesn't have a
    Coder loaded yet still gets a useful display.

    ``source_name`` is the display name to show next to the source id.
    Pass an empty string (the default) to suppress the source-name row.

    Drift detection (F9.2) runs only when both ``code`` and
    ``code_version`` are supplied; otherwise ``definition_drifted`` is
    ``False`` and ``drifted_fields`` is empty. ``snapshot_missing`` and
    ``code_missing`` carry the "why no drift comparison" hint so the
    renderer can word the empty case correctly.
    """
    if not isinstance(application, Application):
        raise TypeError(
            "application must be an Application; got "
            f"{type(application).__name__}"
        )
    if code is not None and not isinstance(code, Code):
        raise TypeError(
            f"code must be a Code or None; got {type(code).__name__}"
        )
    if code_version is not None and not isinstance(code_version, CodeVersion):
        raise TypeError(
            "code_version must be a CodeVersion or None; got "
            f"{type(code_version).__name__}"
        )
    if coder is not None and not isinstance(coder, Coder):
        raise TypeError(
            f"coder must be a Coder or None; got {type(coder).__name__}"
        )
    if decided_by_coder is not None and not isinstance(decided_by_coder, Coder):
        raise TypeError(
            "decided_by_coder must be a Coder or None; got "
            f"{type(decided_by_coder).__name__}"
        )

    # ---------------- Anchor + base bits ---------------- #
    anchor_label = _format_anchor(application)

    # ---------------- Code (current) ---------------- #
    if code is None:
        code_name = "(unknown)"
        code_colour = ""
        code_stage = ""
    else:
        code_name = code.name.strip() or "(unnamed)"
        code_colour = code.colour or ""
        code_stage = code.stage or ""

    # ---------------- Code version at apply ---------------- #
    snapshot_missing = code_version is None
    if code_version is None:
        version_number = ""
        version_recorded_at = ""
        version_change_note = ""
        snapshot_dict: dict[str, Any] = {}
        name_at_apply = ""
    else:
        version_number = f"v{int(code_version.version)}"
        version_recorded_at = str(code_version.created_at or "")
        version_change_note = str(code_version.change_note or "")
        snapshot_dict = dict(code_version.snapshot or {})
        name_at_apply = str(snapshot_dict.get("name", "") or "")

    # ---------------- Coder ---------------- #
    coder_name, coder_role = _coder_label(coder)
    if coder is None:
        coder_name = "(unknown)"

    # ---------------- Provenance source ---------------- #
    raw_source = (application.provenance.get("source", "") or "").strip()
    if raw_source not in APPLICATION_PROVENANCE_SOURCES:
        # Legacy / blank rows default to "human" so the hover never
        # leaves the source ambiguous. PLANNING §"AI suggests, the
        # human applies" — a code with no provenance is human-coded.
        provenance_source = ""
        provenance_source_label = DEFAULT_PROVENANCE_SOURCE_LABEL
    else:
        provenance_source = raw_source
        provenance_source_label = PROVENANCE_SOURCE_LABELS.get(
            raw_source, DEFAULT_PROVENANCE_SOURCE_LABEL
        )

    # ---------------- AI provenance (F8.9) ---------------- #
    aip = application.ai_provenance
    if aip is None:
        ai_present = False
        ai_feature = ""
        ai_feature_label = ""
        ai_backend = ""
        ai_generation_model = ""
        ai_embedding_model = ""
        ai_suggestion_id = ""
        ai_decision = ""
        ai_decision_label = ""
        ai_decided_by_coder_id = ""
        ai_decided_by_coder_name = ""
        ai_decided_at = ""
        ai_confidence = ""
        ai_prompt_hash = ""
        ai_notes = ""
    else:
        ai_present = True
        ai_feature = aip.feature
        ai_feature_label = AI_FEATURE_LABELS.get(
            aip.feature, AI_FEATURE_LABELS["other"]
        )
        ai_backend = aip.backend
        ai_generation_model = aip.generation_model
        ai_embedding_model = aip.embedding_model
        ai_suggestion_id = aip.suggestion_id
        ai_decision = aip.decision
        ai_decision_label = AI_DECISION_LABELS.get(
            aip.decision, aip.decision or ""
        )
        ai_decided_by_coder_id = aip.decided_by_coder_id
        if decided_by_coder is None:
            ai_decided_by_coder_name = (
                "(unknown)" if aip.decided_by_coder_id else ""
            )
        else:
            n, _ = _coder_label(decided_by_coder)
            ai_decided_by_coder_name = n or "(unnamed)"
        ai_decided_at = aip.decided_at
        ai_confidence = _format_confidence(aip.confidence)
        ai_prompt_hash = aip.prompt_hash
        ai_notes = aip.notes

    # ---------------- Drift (relative to current Code) ---------------- #
    code_missing = code is None
    if code is None or code_version is None:
        drifted: tuple[str, ...] = ()
        definition_drifted = False
    else:
        drifted = drifted_definition_fields(snapshot_dict, code)
        definition_drifted = bool(drifted)

    # ---------------- Free-form extra provenance ---------------- #
    extra_lines: list[str] = []
    for k in sorted(application.provenance.keys()):
        if k in _RESERVED_PROVENANCE_KEYS:
            continue
        v = application.provenance.get(k, "")
        # Defensive: provenance values are validated as strings on
        # entry, but renderers shouldn't trust that.
        extra_lines.append(f"{k}: {v}")

    return ProvenanceDisplay(
        application_id=application.id,
        anchor_label=anchor_label,
        created_at=str(application.created_at or ""),
        modified_at=str(application.modified_at or ""),
        confidence=_format_confidence(application.confidence),
        note=str(application.note or ""),
        code_id=application.code_id,
        code_name=code_name,
        code_colour=code_colour,
        code_stage=code_stage,
        version_id_at_apply=application.definition_version_id_at_apply,
        version_number_at_apply=version_number,
        version_recorded_at=version_recorded_at,
        version_change_note=version_change_note,
        snapshot_missing=snapshot_missing,
        name_at_apply=name_at_apply,
        coder_id=application.coder_id,
        coder_name=coder_name,
        coder_role=coder_role,
        source_id=application.source_id,
        source_name=str(source_name or ""),
        provenance_source=provenance_source,
        provenance_source_label=provenance_source_label,
        ai_present=ai_present,
        ai_feature=ai_feature,
        ai_feature_label=ai_feature_label,
        ai_backend=ai_backend,
        ai_generation_model=ai_generation_model,
        ai_embedding_model=ai_embedding_model,
        ai_suggestion_id=ai_suggestion_id,
        ai_decision=ai_decision,
        ai_decision_label=ai_decision_label,
        ai_decided_by_coder_id=ai_decided_by_coder_id,
        ai_decided_by_coder_name=ai_decided_by_coder_name,
        ai_decided_at=ai_decided_at,
        ai_confidence=ai_confidence,
        ai_prompt_hash=ai_prompt_hash,
        ai_notes=ai_notes,
        code_missing=code_missing,
        definition_drifted=definition_drifted,
        drifted_fields=tuple(drifted),
        extra_provenance=tuple(extra_lines),
    )


# --------------------------------------------------------------------------- #
# Single-line summary (for inline labels)
# --------------------------------------------------------------------------- #


def provenance_summary_label(display: ProvenanceDisplay) -> str:
    """Compact one-line summary, e.g. ``"Alex · Human-coded · 2026-04-15"``.

    Designed for an inline badge or status row. Skips empty fields
    so the order is stable regardless of which entities the caller
    hydrated. Always produces *something* — at minimum the
    provenance source label.
    """
    parts: list[str] = []
    name = (display.coder_name or "").strip()
    if name and name not in {"(unknown)", "(unnamed)"}:
        parts.append(name)
    parts.append(display.provenance_source_label)
    # Date only (no time), so the badge stays compact. We expect ISO
    # timestamps; if the value isn't ISO, fall back to the raw string.
    date = (display.created_at or "")[:10]
    if date:
        parts.append(date)
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Plain-text formatter (suitable for an HTML title= attribute)
# --------------------------------------------------------------------------- #


def format_provenance_text(display: ProvenanceDisplay) -> str:
    """Multi-line plain-text rendering of the display.

    Suitable for a ``title=`` attribute (browsers render newlines in
    title text), or any tooltip surface that doesn't render markup.
    Sections are separated by blank lines so the result reads like
    a small structured note.
    """
    lines: list[str] = []
    # Heading: code name (id) — version
    head = f"{display.code_name} ({display.code_id})"
    if display.version_number_at_apply:
        head += f" · {display.version_number_at_apply}"
    lines.append(head)

    # Provenance source + creation date
    meta_bits = [display.provenance_source_label]
    if display.created_at:
        meta_bits.append(display.created_at)
    if display.confidence:
        meta_bits.append(f"confidence {display.confidence}")
    lines.append(" · ".join(meta_bits))

    # Anchor + source
    anchor_bits = [f"anchor {display.anchor_label}"]
    if display.source_name:
        anchor_bits.append(f"source {display.source_name}")
    elif display.source_id:
        anchor_bits.append(f"source {display.source_id}")
    lines.append(" · ".join(anchor_bits))

    # Coder
    coder_bits: list[str] = []
    if display.coder_name and display.coder_name != "(unknown)":
        coder_bits.append(f"by {display.coder_name}")
    else:
        coder_bits.append(f"by {display.coder_name or '(unknown)'}")
    if display.coder_role:
        coder_bits.append(display.coder_role)
    lines.append(" · ".join(coder_bits))

    # Drift hint
    if display.snapshot_missing and display.version_id_at_apply:
        lines.append("")
        lines.append("Definition snapshot at apply not found.")
    elif display.definition_drifted and not display.code_missing:
        lines.append("")
        lines.append(
            "Definition has changed since apply ("
            + ", ".join(display.drifted_fields)
            + ")."
        )

    # AI provenance
    if display.ai_present:
        lines.append("")
        lines.append(
            "AI: "
            + " · ".join(
                p
                for p in [
                    display.ai_feature_label,
                    display.ai_backend,
                    display.ai_generation_model,
                    display.ai_decision_label,
                ]
                if p
            )
        )
        if display.ai_decided_by_coder_name and display.ai_decided_by_coder_name not in {
            "(unknown)"
        }:
            extra: list[str] = [f"decided by {display.ai_decided_by_coder_name}"]
            if display.ai_decided_at:
                extra.append(display.ai_decided_at)
            lines.append(" · ".join(extra))
        elif display.ai_decided_at:
            lines.append(f"decided at {display.ai_decided_at}")
        if display.ai_confidence:
            lines.append(f"AI confidence {display.ai_confidence}")
        if display.ai_prompt_hash:
            lines.append(f"prompt {display.ai_prompt_hash}")

    # Extra free-form provenance keys
    if display.extra_provenance:
        lines.append("")
        lines.extend(display.extra_provenance)

    # Note
    if display.note:
        lines.append("")
        lines.append("Note:")
        lines.extend(display.note.splitlines() or [display.note])

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML formatter (escaped, ready for innerHTML)
# --------------------------------------------------------------------------- #


def _esc(s: str) -> str:
    """HTML-escape, including quote characters, for safe innerHTML."""
    return _html.escape(s or "", quote=True)


def format_provenance_html(display: ProvenanceDisplay) -> str:
    """Compact HTML rendering for a hover popover.

    Produces a single ``<div class="provenance-display">…</div>`` with
    a title row, a metadata grid, and optional drift / AI / note
    sections. Output is fully HTML-escaped (no user-supplied value
    is concatenated raw) so the caller can safely write it into
    ``innerHTML``. CSS for ``.provenance-*`` classes lives in the
    editor template.
    """
    parts: list[str] = []
    parts.append('<div class="provenance-display">')

    # Title
    title = _esc(display.code_name)
    if display.version_number_at_apply:
        title += (
            ' <span class="provenance-version">'
            + _esc(display.version_number_at_apply)
            + "</span>"
        )
    if display.code_colour:
        # Colour swatch — a small inline span. The colour is validated
        # upstream by Code (#RGB / #RRGGBB), but we still escape it as a
        # belt-and-braces against an old corrupt row.
        title = (
            '<span class="provenance-swatch" style="background:'
            + _esc(display.code_colour)
            + '"></span>'
            + title
        )
    parts.append('<div class="provenance-title">' + title + "</div>")

    # Provenance source row
    parts.append(
        '<div class="provenance-source">'
        + _esc(display.provenance_source_label)
        + "</div>"
    )

    # Meta grid (anchor / source / by / when / confidence)
    rows: list[tuple[str, str]] = []
    rows.append(("Anchor", _esc(display.anchor_label)))
    if display.source_name:
        rows.append(("Source", _esc(display.source_name)))
    elif display.source_id:
        rows.append(("Source", _esc(display.source_id)))
    if display.coder_name:
        coder_html = _esc(display.coder_name)
        if display.coder_role:
            coder_html += (
                ' <span class="provenance-role">'
                + _esc(display.coder_role)
                + "</span>"
            )
        rows.append(("By", coder_html))
    if display.created_at:
        rows.append(("Applied", _esc(display.created_at)))
    if display.confidence:
        rows.append(("Confidence", _esc(display.confidence)))

    if rows:
        parts.append('<dl class="provenance-meta">')
        for k, v in rows:
            parts.append(
                "<dt>" + _esc(k) + "</dt><dd>" + v + "</dd>"
            )
        parts.append("</dl>")

    # Drift / snapshot hint
    if display.snapshot_missing and display.version_id_at_apply:
        parts.append(
            '<div class="provenance-warn">'
            + _esc("Definition snapshot at apply not found.")
            + "</div>"
        )
    elif display.definition_drifted and not display.code_missing:
        parts.append(
            '<div class="provenance-drift">'
            + _esc(
                "Definition has changed since apply: "
                + ", ".join(display.drifted_fields)
            )
            + "</div>"
        )

    # AI provenance section
    if display.ai_present:
        parts.append('<div class="provenance-ai">')
        parts.append(
            '<div class="provenance-ai-head">'
            + _esc("AI: " + display.ai_feature_label)
            + "</div>"
        )
        ai_rows: list[tuple[str, str]] = []
        if display.ai_backend:
            ai_rows.append(("Backend", _esc(display.ai_backend)))
        if display.ai_generation_model:
            ai_rows.append(("Model", _esc(display.ai_generation_model)))
        if display.ai_embedding_model:
            ai_rows.append(("Embeddings", _esc(display.ai_embedding_model)))
        if display.ai_decision_label:
            ai_rows.append(("Decision", _esc(display.ai_decision_label)))
        if display.ai_decided_by_coder_name and display.ai_decided_by_coder_name not in {
            "(unknown)"
        }:
            ai_rows.append(("Decided by", _esc(display.ai_decided_by_coder_name)))
        if display.ai_decided_at:
            ai_rows.append(("Decided at", _esc(display.ai_decided_at)))
        if display.ai_confidence:
            ai_rows.append(("AI confidence", _esc(display.ai_confidence)))
        if display.ai_prompt_hash:
            ai_rows.append(("Prompt", _esc(display.ai_prompt_hash)))
        if ai_rows:
            parts.append('<dl class="provenance-meta">')
            for k, v in ai_rows:
                parts.append("<dt>" + _esc(k) + "</dt><dd>" + v + "</dd>")
            parts.append("</dl>")
        if display.ai_notes:
            parts.append(
                '<div class="provenance-ai-notes">'
                + _esc(display.ai_notes)
                + "</div>"
            )
        parts.append("</div>")

    # Extra provenance keys
    if display.extra_provenance:
        parts.append('<dl class="provenance-extra">')
        for line in display.extra_provenance:
            # ``"key: value"`` — split on the first colon for a clean
            # <dt>/<dd> pair. If there is no colon, fall back to a
            # single-cell row.
            if ":" in line:
                k, v = line.split(":", 1)
                parts.append(
                    "<dt>"
                    + _esc(k.strip())
                    + "</dt><dd>"
                    + _esc(v.strip())
                    + "</dd>"
                )
            else:
                parts.append(
                    '<dt></dt><dd>' + _esc(line) + "</dd>"
                )
        parts.append("</dl>")

    # Note
    if display.note:
        parts.append('<div class="provenance-note">')
        parts.append('<div class="provenance-note-head">' + _esc("Note") + "</div>")
        # Preserve newlines with <br>; HTML-escape every line.
        body_lines = [_esc(ln) for ln in display.note.splitlines()] or [_esc(display.note)]
        parts.append(
            '<div class="provenance-note-body">'
            + "<br>".join(body_lines)
            + "</div>"
        )
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


__all__ = [
    "AI_DECISION_LABELS",
    "AI_FEATURE_LABELS",
    "DEFAULT_PROVENANCE_SOURCE_LABEL",
    "PROVENANCE_SOURCE_LABELS",
    "ProvenanceDisplay",
    "build_provenance_display",
    "format_provenance_html",
    "format_provenance_text",
    "provenance_summary_label",
]
