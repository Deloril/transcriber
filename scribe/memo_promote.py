"""Promote a memo into a code definition (F5.5).

Per PLANNING.md F5.5:

  > Promote a memo into a code definition (one click).

A *memo* (F5.1) is the analytic-thinking layer of grounded theory:
the running notebook in which a researcher works out what a category
*means* before there's a code for it. When the meaning has matured,
the analytic move is to promote the memo into a first-class
:class:`scribe.codes.Code` — the labelled concept that subsequent
:class:`scribe.applications.Application` rows will anchor to.

This module is the pure-Python core of that one-click action. Three
layers, in increasing convenience:

1. :func:`derive_code_name_from_memo` — pick a sensible code name from
   the memo's title (preferred) or the first non-empty body line
   (fallback). Trims trivial Markdown prefixes (``#``, ``-``, ``*``,
   ``>``) so a memo whose body opens with ``# Managing the project``
   produces ``Managing the project``, not the literal heading.
2. :func:`build_code_from_memo` — pure builder. Given a :class:`Memo`
   and optional overrides, returns a fully-validated :class:`Code`
   whose ``provenance`` records ``source='promoted_from_memo'`` and
   ``memo_id=<source memo id>``. No filesystem I/O.
3. :func:`promote_memo_to_code` — high-level persistence helper.
   Loads the memo, builds the code, calls
   :func:`scribe.code_versions.save_code_with_version` so v1 of the
   code is recorded immediately (every later application gets a
   real ``definition_version_id_at_apply`` to point at — F4.1's
   audit trail starts on day one), and (by default) appends a
   back-link from the memo to the new code with role
   ``promoted_to`` so the lineage is visible from either side.

Field mapping
-------------

The defaults are deliberately conservative — the researcher only
clicked once, and the resulting code is meant to be a starting point
they will refine, not a finished artefact:

* ``code.name`` ← :func:`derive_code_name_from_memo` unless the caller
  supplied ``name=`` explicitly.
* ``code.definition`` ← ``memo.body`` (full) unless overridden. If
  the body exceeds ``Code``'s 4 000-char definition cap,
  :class:`Code.validate` raises and the caller is expected to pass a
  shorter ``definition=``. Truncating silently would corrupt the
  audit trail.
* ``code.theoretical_memo`` ← ``memo.body`` *only* when the source
  memo is a ``theoretical`` memo and the caller didn't supply one;
  otherwise empty. Researchers expect a memo classified as
  ``theoretical`` to seed the ``theoretical_memo`` field
  automatically; other memo types stand by themselves and the field
  starts blank.
* ``code.exemplars``, ``code.inclusion_criteria``,
  ``code.exclusion_criteria``, ``code.parent_code_id``,
  ``code.related_codes`` — all default to empty / ``None``. The user
  fills these in as part of the F2.3 lifecycle ops or while applying
  the code in coding sessions.
* ``code.stage`` ← ``"initial"`` (Code's own default; the memo's stage
  doesn't exist as a concept). Caller can override.
* ``code.colour`` ← ``""`` (UI picks a default).
* ``code.status`` ← ``"active"`` (PLANNING calls this "one-click", not
  "draft for further refinement"; the user can demote to ``draft``
  via :func:`scribe.codes.Code.apply_update` if they want a soft
  start). Caller can override.
* ``code.provenance['source']`` ← ``"promoted_from_memo"`` (closed-
  vocabulary value already reserved in
  :data:`scribe.codes.CODE_PROVENANCE_SOURCES`).
* ``code.provenance['memo_id']`` ← ``memo.id``.
* Caller can pass ``extra_provenance`` to add origin keys (e.g.
  ``promoted_by`` = a coder id). Reserved keys (``source``,
  ``memo_id``) are *not* overridable; attempts raise
  :class:`ProjectValidationError` so the audit trail is unforgeable.

Codebook lock
-------------

This helper is **lock-unaware**, mirroring :mod:`scribe.code_lifecycle`
(F2.3). Lock enforcement (F2.4) is the caller's job — the HTTP layer
calls :func:`scribe.codebook_lock.assert_codebook_unlocked` first.
That keeps importers and migration scripts free to seed a codebook
through the same builder.

This module is stand-alone — no FastAPI, no engine imports — matching
every other F-feature's conventions (F1.* / F2.* / F4.* / F5.1–5.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .codes import (
    CODE_PROVENANCE_SOURCES,
    CODE_STATUSES,
    Code,
    CodeRelation,
    MAX_NAME_LEN,
)
from .code_versions import (
    CodeVersion,
    save_code_with_version,
)
from .memos import (
    MEMO_ID_RE,
    Memo,
    MemoLink,
    load_memo,
    save_memo,
)
from .projects import (
    CODEBOOK_STAGES,
    PROJECT_ID_RE,
    ProjectValidationError,
    utcnow_iso,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Provenance keys this module owns. Callers cannot override them via
# ``extra_provenance`` — that would let a noisy import overwrite the
# very lineage record this module exists to leave.
_RESERVED_PROVENANCE_KEYS = frozenset({"source", "memo_id"})

# Default link role used when back-linking the memo to the new code.
# Slug-shape so it satisfies :data:`scribe.memos.LINK_ROLE_RE`.
DEFAULT_BACK_LINK_ROLE = "promoted_to"

# Markdown-prefix regex used by ``derive_code_name_from_memo`` to strip
# leading ``# / ## / * / - / >`` (and surrounding whitespace) from the
# first usable line. We do *not* strip emphasis (``*foo*``) or links
# (``[foo](url)``) — researchers occasionally name codes in those forms
# and we shouldn't second-guess. Just the line-leading sigils.
_MD_LINE_PREFIX_RE = re.compile(r"^[#>\-*\s]+")


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class CodePromotionResult:
    """Bundle returned by :func:`promote_memo_to_code`.

    * ``code`` — the persisted :class:`Code` (already saved).
    * ``version`` — the recorded :class:`CodeVersion` (v1 by
      construction; subsequent edits will append later versions).
    * ``memo`` — the :class:`Memo` post back-link (or the original if
      ``record_back_link=False``).
    """

    code: Code
    version: CodeVersion
    memo: Memo


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def derive_code_name_from_memo(memo: Memo) -> str:
    """Pick a sensible code name from a memo.

    Preference order:

    1. ``memo.title.strip()`` if non-empty.
    2. The first non-empty line of ``memo.body`` with leading
       Markdown sigils (``#``, ``-``, ``*``, ``>``, whitespace)
       stripped.

    The result is truncated to :data:`scribe.codes.MAX_NAME_LEN`
    characters so the name always passes :class:`Code.validate`. If
    *both* title and body are empty / whitespace-only,
    :class:`ProjectValidationError` is raised — there is no usable
    name and silently inventing one would corrupt the codebook.

    >>> from scribe.memos import Memo
    >>> m = Memo.new(project_id='0'*12, title=' Managing  ', body='ignored')
    >>> derive_code_name_from_memo(m)
    'Managing'
    """
    if not isinstance(memo, Memo):
        raise TypeError("derive_code_name_from_memo expects a Memo")

    title = (memo.title or "").strip()
    if title:
        return title[:MAX_NAME_LEN]

    for raw in (memo.body or "").splitlines():
        line = _MD_LINE_PREFIX_RE.sub("", raw or "").strip()
        if line:
            return line[:MAX_NAME_LEN]

    raise ProjectValidationError(
        "memo has no usable text for a code name "
        "(title and body are both empty)"
    )


def _normalise_extra_provenance(
    extra: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Coerce ``extra_provenance`` to ``str→str`` and reject reserved keys.

    The reserved keys are ``source`` (we always set it to
    ``"promoted_from_memo"``) and ``memo_id`` (we always set it to the
    source memo's id). Letting a caller override these would let a
    forger relabel the lineage; refusing is the safer default.
    """
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise ProjectValidationError(
            "extra_provenance must be a mapping of string→string"
        )
    out: dict[str, str] = {}
    for raw_k, raw_v in extra.items():
        k = str(raw_k).strip()
        if not k:
            continue
        if k in _RESERVED_PROVENANCE_KEYS:
            raise ProjectValidationError(
                f"extra_provenance cannot override reserved key {k!r}"
            )
        out[k] = str(raw_v)
    return out


def build_code_from_memo(
    *,
    memo: Memo,
    name: str | None = None,
    definition: str | None = None,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    exemplars: Iterable[str] | None = None,
    parent_code_id: str | None = None,
    related_codes: Iterable[Any] | None = None,
    theoretical_memo: str | None = None,
    stage: str = "initial",
    colour: str = "",
    status: str = "active",
    extra_provenance: Mapping[str, Any] | None = None,
    code_id: str | None = None,
    now: str | None = None,
) -> Code:
    """Build a fresh :class:`Code` from a :class:`Memo`.

    The pure builder. Returns a fully-validated Code with provenance
    set to ``source='promoted_from_memo'`` and ``memo_id=<memo.id>``;
    no I/O. Defaults are documented in the module docstring; every
    field can be overridden by the caller.

    Raises :class:`ProjectValidationError` for any invariant violation
    — including the body being too long for ``definition`` /
    ``theoretical_memo``. Callers that anticipate long bodies should
    pass an explicit ``definition=`` (and ``theoretical_memo=`` for
    theoretical memos).
    """
    if not isinstance(memo, Memo):
        raise TypeError("build_code_from_memo expects a Memo")

    # Stamp validation runs through the regular project / memo regexes
    # (Memo.validate already enforced these on construction, but we
    # re-check defensively in case the caller hand-built a Memo).
    if not PROJECT_ID_RE.match(memo.project_id):
        raise ProjectValidationError(
            f"Source memo has invalid project_id: {memo.project_id!r}"
        )
    if not MEMO_ID_RE.match(memo.id):
        raise ProjectValidationError(
            f"Source memo has invalid id: {memo.id!r}"
        )

    chosen_name = (
        str(name).strip() if name is not None and str(name).strip()
        else derive_code_name_from_memo(memo)
    )

    chosen_definition = (
        memo.body if definition is None else str(definition or "")
    )

    # Theoretical memo defaulting: only fill from memo.body when the
    # memo is itself a ``theoretical`` memo *and* the caller didn't
    # specify. Other memo types start with an empty theoretical_memo.
    if theoretical_memo is None:
        chosen_tm = memo.body if memo.type == "theoretical" else ""
    else:
        chosen_tm = str(theoretical_memo or "")

    if stage not in CODEBOOK_STAGES:
        raise ProjectValidationError(
            f"stage must be one of {CODEBOOK_STAGES}; got {stage!r}"
        )
    if status not in CODE_STATUSES:
        raise ProjectValidationError(
            f"status must be one of {CODE_STATUSES}; got {status!r}"
        )

    # Provenance: ``source`` and ``memo_id`` are always set by us;
    # extra_provenance fills additional keys. Reserved keys raise.
    provenance: dict[str, str] = {
        "source": "promoted_from_memo",
        "memo_id": memo.id,
    }
    provenance.update(_normalise_extra_provenance(extra_provenance))
    # Defensive — if the closed-vocabulary changes, fail loudly.
    if provenance["source"] not in CODE_PROVENANCE_SOURCES:  # pragma: no cover
        raise ProjectValidationError(
            "promoted_from_memo is not in CODE_PROVENANCE_SOURCES — "
            "module constants drifted"
        )

    # Coerce related_codes into CodeRelation instances; the Code
    # constructor accepts dicts but we want errors to surface here
    # with a build-time message (clearer for the F5.5 endpoint).
    coerced_relations: list[CodeRelation] = []
    for r in related_codes or ():
        if isinstance(r, CodeRelation):
            coerced_relations.append(r)
        elif isinstance(r, dict):
            coerced_relations.append(CodeRelation.from_dict(r))
        else:
            raise ProjectValidationError(
                "related_codes entries must be CodeRelation or dict"
            )

    return Code.new(
        project_id=memo.project_id,
        name=chosen_name,
        definition=chosen_definition,
        inclusion_criteria=str(inclusion_criteria or ""),
        exclusion_criteria=str(exclusion_criteria or ""),
        exemplars=list(exemplars or ()),
        parent_code_id=parent_code_id if parent_code_id else None,
        related_codes=coerced_relations,
        theoretical_memo=chosen_tm,
        stage=stage,
        colour=str(colour or ""),
        status=status,
        provenance=provenance,
        code_id=code_id,
        now=now,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _attach_back_link(
    memo: Memo, code: Code, *, role: str, now: str | None
) -> Memo:
    """Append a memo→code link with the given role; return the (mutated) memo.

    Idempotent on (target_type, target_id, role) — :class:`Memo`'s
    own validate-time dedupe collapses repeats, but we also short-
    circuit *before* calling ``apply_update`` so we don't bump
    ``modified_at`` for a no-op.
    """
    if memo.has_link_to("code", code.id):
        for link in memo.links:
            if (
                link.target_type == "code"
                and link.target_id == code.id
                and link.role == role
            ):
                return memo
    new_links = [link.to_dict() for link in memo.links]
    new_links.append({
        "target_type": "code",
        "target_id": code.id,
        "role": role,
    })
    memo.apply_update({"links": new_links}, now=now)
    return memo


def promote_memo_to_code(
    projects_root: Path,
    project_id: str,
    memo_id: str,
    *,
    name: str | None = None,
    definition: str | None = None,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    exemplars: Iterable[str] | None = None,
    parent_code_id: str | None = None,
    related_codes: Iterable[Any] | None = None,
    theoretical_memo: str | None = None,
    stage: str = "initial",
    colour: str = "",
    status: str = "active",
    extra_provenance: Mapping[str, Any] | None = None,
    code_id: str | None = None,
    change_note: str = "",
    record_back_link: bool = True,
    back_link_role: str = DEFAULT_BACK_LINK_ROLE,
    now: str | None = None,
) -> CodePromotionResult:
    """Load a memo, build a code from it, persist with v1, and back-link.

    The high-level "one click" entry point. Steps:

    1. Validate ids (12-char hex shape).
    2. :func:`scribe.memos.load_memo` — raises ``FileNotFoundError``
       if the memo doesn't exist.
    3. :func:`build_code_from_memo` — pure builder (any field-shape
       errors land here).
    4. :func:`scribe.code_versions.save_code_with_version` — writes
       ``codes/<cid>.json`` and appends v1 to
       ``code_versions/<cid>.jsonl``.
    5. (Default) append a memo→code link with role
       ``promoted_to`` and re-save the memo.

    The codebook lock (F2.4) is **not** checked here — pass through
    :func:`scribe.codebook_lock.assert_codebook_unlocked` at the
    boundary if you want enforcement (the HTTP endpoint does).

    Returns a :class:`CodePromotionResult` with the persisted
    :class:`Code`, the recorded v1 :class:`CodeVersion`, and the
    (possibly back-linked) :class:`Memo`.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")
    if not MEMO_ID_RE.match(memo_id):
        raise ProjectValidationError(f"Invalid memo id: {memo_id!r}")

    memo = load_memo(projects_root, project_id, memo_id)
    if memo.project_id != project_id:
        # Defensive: a hand-edited memo file on disk could have a
        # mismatched project_id even though it lives under the right
        # directory. Refuse rather than silently leak codes across
        # projects.
        raise ProjectValidationError(
            f"Memo {memo_id!r} project_id {memo.project_id!r} does "
            f"not match path project {project_id!r}"
        )

    code = build_code_from_memo(
        memo=memo,
        name=name,
        definition=definition,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        exemplars=exemplars,
        parent_code_id=parent_code_id,
        related_codes=related_codes,
        theoretical_memo=theoretical_memo,
        stage=stage,
        colour=colour,
        status=status,
        extra_provenance=extra_provenance,
        code_id=code_id,
        now=now,
    )

    note = change_note or f"Promoted from memo {memo.id}"
    _, version = save_code_with_version(
        projects_root, code, change_note=note, now=now
    )
    if version is None:  # pragma: no cover — first save always records
        raise RuntimeError(
            "promote_memo_to_code: expected v1 to be recorded"
        )

    if record_back_link:
        # Use a fresh ``now`` if the caller didn't pin one — the
        # back-link write deserves the current clock so the memo's
        # ``modified_at`` is later than the code's ``created_at``.
        link_now = now or utcnow_iso()
        _attach_back_link(memo, code, role=back_link_role, now=link_now)
        save_memo(projects_root, memo)

    return CodePromotionResult(code=code, version=version, memo=memo)
