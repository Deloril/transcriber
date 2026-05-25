"""Right-click memo creation from any context (F5.2).

Per PLANNING.md F5.2:

  > Right-click memo creation from any context with link pre-populated.

F5.1 shipped the :class:`scribe.memos.Memo` entity, with multi-target
links to codes, sources, applications, participants, coders, projects,
and other memos. F5.2 closes the loop with the **right-click flow**:
the user invokes a context menu from anywhere in the UI (a code in the
codebook, a quote in a transcript, a participant card, a memo card in
F5.3's canvas), Scribe opens a "new memo" composer with the link to
that target *already populated*, and saves the memo straight to the
project on submit.

The pure-Python core of that flow lives here. Two pieces:

* :func:`default_memo_type_for_target` — a closed-vocabulary mapping
  from the entity the user clicked to the most-likely-appropriate
  :data:`scribe.memos.MEMO_TYPES` value. Researchers reclassify all
  the time, so this is only the default; the composer UI shows it
  pre-selected and lets the user change it.
* :func:`build_memo_draft` — given a target context plus any composer
  fields (title / body / tags / etc.), returns a fully-validated
  :class:`Memo` ready for ``save_memo``. The primary link to the
  target is always added first; a caller can pass ``extra_links`` to
  pre-populate additional links (e.g. a memo opened from a coded
  segment can also link the underlying code).

The same ``defaultMemoTypeForTarget`` / ``buildMemoDraftPayload``
shapes live on the JS side in :file:`scribe/static/js/helpers.mjs` so
the editor can build the same payload without a round-trip to the
server. The server's ``POST /api/projects/{pid}/memos`` endpoint
accepts either a flat memo payload *or* a ``context`` block; when a
context block is present it routes through :func:`build_memo_draft` so
the type-defaulting and link-prepopulation rules are identical
regardless of who built the payload.

Why a separate module from ``scribe/memos.py``?

* ``memos.py`` is the entity definition + persistence; this is the
  *interaction model* that says "click on a code → memo about a code".
  Mixing them muddies the F5.1 API surface that other features import.
* F5.5 ("promote a memo into a code definition") is the inverse
  transition; if and when we add a similar helper there, it will sit
  beside this file rather than bloating ``memos.py``.
* Tests are easier to read when the right-click logic isn't
  interleaved with eight hundred lines of validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .memos import (
    MEMO_LINK_TARGET_TYPES,
    MEMO_TYPES,
    TARGET_ID_RE,
    Memo,
    MemoLink,
)
from .projects import ProjectValidationError


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Default memo ``type`` for a right-click invocation, keyed by the
# target entity. The user can change this in the composer; this is
# only the "what's most likely to be useful" pre-selection.
#
#   * ``code`` → ``code``
#       A memo opened from a code is almost always *about* that code's
#       definition / criteria / exemplars — the operational-memo
#       pattern from Charmaz §5.
#   * ``application`` → ``quote``
#       Right-clicking a quote in the transcript is the "marginal
#       note" affordance: PLANNING calls this exact pattern out as
#       the ``quote`` memo type.
#   * ``source`` → ``source``
#       Per-interview reflection ("P3 was guarded today").
#   * ``project`` → ``project``
#       Corpus-level observation that doesn't fit a single code or
#       source.
#   * ``coder`` → ``methodological``
#       Notes on a coder are typically about coding decisions /
#       reconciliation — methodological territory.
#   * ``memo`` → ``theoretical``
#       Memo→memo edges in F5.3's sorting canvas almost always carry
#       theoretical synthesis ("memo A and memo B together suggest
#       category X"). Defaulting to ``theoretical`` reflects that.
#   * ``participant`` → ``free``
#       Notes on a participant span everything from logistical
#       ("re-contact for follow-up") to reflexive ("she reminded me
#       of…") to source-flavoured. ``free`` defers the choice rather
#       than presuming.
#
# Any future target type without an explicit mapping defaults to
# ``free`` via :func:`default_memo_type_for_target`.
DEFAULT_MEMO_TYPE_BY_TARGET: dict[str, str] = {
    "code": "code",
    "application": "quote",
    "source": "source",
    "project": "project",
    "coder": "methodological",
    "memo": "theoretical",
    "participant": "free",
}


# --------------------------------------------------------------------------- #
# Right-click context payload
# --------------------------------------------------------------------------- #


@dataclass
class MemoContext:
    """The bare-bones right-click payload: *what was clicked*.

    The UI emits one of these from any entity-bearing surface — a
    code chip, a quote highlight, a participant card. The composer
    converts it to a draft Memo via :func:`build_memo_draft` and
    presents that for the researcher to fill in.

    ``role`` is the optional :class:`MemoLink.role` for the resulting
    primary link ("exemplifies", "contradicts", …); empty by default
    because the right-click flow doesn't ask for it up front.
    """

    target_type: str
    target_id: str
    role: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoContext":
        if not isinstance(d, dict):
            raise ProjectValidationError("MemoContext payload must be an object")
        for required in ("target_type", "target_id"):
            if required not in d:
                raise ProjectValidationError(
                    f"MemoContext payload missing required key: {required}"
                )
        return cls(
            target_type=str(d["target_type"]),
            target_id=str(d["target_id"]),
            role=str(d.get("role", "") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        out = {"target_type": self.target_type, "target_id": self.target_id}
        if self.role:
            out["role"] = self.role
        return out

    def validate(self) -> None:
        # We piggy-back on MemoLink's validator since the on-disk shape
        # is identical — keeping the rules in one place avoids drift.
        MemoLink(
            target_type=self.target_type,
            target_id=self.target_id,
            role=self.role,
        ).validate()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def default_memo_type_for_target(target_type: str) -> str:
    """Return the default memo ``type`` for a context-menu invocation.

    For unknown ``target_type`` strings this returns ``"free"`` rather
    than raising: the right-click flow shouldn't fail closed when
    a future entity type appears in the UI before the mapping is
    updated. Callers that do want strict validation should check
    ``target_type in MEMO_LINK_TARGET_TYPES`` themselves.

    >>> default_memo_type_for_target("code")
    'code'
    >>> default_memo_type_for_target("application")
    'quote'
    >>> default_memo_type_for_target("not-a-real-target")
    'free'
    """
    if not isinstance(target_type, str):
        raise ProjectValidationError("target_type must be a string")
    return DEFAULT_MEMO_TYPE_BY_TARGET.get(target_type, "free")


def _coerce_extra_link(value: Any) -> MemoLink:
    """Accept a MemoLink or a dict; normalise to MemoLink."""
    if isinstance(value, MemoLink):
        return MemoLink(
            target_type=value.target_type,
            target_id=value.target_id,
            role=value.role,
        )
    if isinstance(value, dict):
        return MemoLink.from_dict(value)
    raise ProjectValidationError(
        "extra_links entries must be MemoLink or dict"
    )


def build_memo_draft(
    *,
    project_id: str,
    target_type: str,
    target_id: str,
    role: str = "",
    type: str | None = None,
    title: str = "",
    body: str = "",
    body_format: str = "markdown",
    author_coder_id: str | None = None,
    extra_links: Iterable[Any] | None = None,
    tags: Iterable[str] | None = None,
    provenance: dict[str, str] | None = None,
    memo_id: str | None = None,
    now: str | None = None,
) -> Memo:
    """Build a draft :class:`Memo` pre-populated with a link to a target.

    The right-click flow's pure core. Validates the target shape
    (must be one of :data:`MEMO_LINK_TARGET_TYPES` and a 12-char hex
    id), picks a default :class:`Memo.type` via
    :func:`default_memo_type_for_target` unless ``type`` was supplied
    explicitly, prepends a primary link ``(target_type, target_id,
    role)`` to ``extra_links`` (deduping if the caller already
    included the same triple), and hands the result to
    :func:`Memo.new` for full validation.

    The returned memo is a fully-formed entity (id, timestamps, all
    invariants satisfied) but has not been persisted. Callers
    typically save it with ``save_memo`` immediately, since the user
    has already invoked the right-click action; previewing without
    persistence is also fine.

    Parameters mirror :func:`Memo.new` with two additions:

    * ``target_type`` / ``target_id`` / ``role`` describe the *primary*
      link populated from the right-click context. They are always
      present (this is what F5.2 *means*).
    * ``extra_links`` is for compound contexts — e.g. right-clicking
      a coded segment that's tied to a specific code can pre-populate
      both the application link and the code link in one go.

    Raises :class:`ProjectValidationError` for any invariant
    violation; callers in the server endpoint translate to HTTP 400.
    """
    if not isinstance(target_type, str) or target_type not in MEMO_LINK_TARGET_TYPES:
        raise ProjectValidationError(
            f"target_type must be one of {MEMO_LINK_TARGET_TYPES}; "
            f"got {target_type!r}"
        )
    if not isinstance(target_id, str) or not TARGET_ID_RE.match(target_id):
        raise ProjectValidationError(
            f"target_id must be 12-char hex; got {target_id!r}"
        )

    chosen_type: str
    if type is None:
        chosen_type = default_memo_type_for_target(target_type)
    else:
        if type not in MEMO_TYPES:
            raise ProjectValidationError(
                f"type must be one of {MEMO_TYPES}; got {type!r}"
            )
        chosen_type = type

    primary = MemoLink(
        target_type=target_type,
        target_id=target_id,
        role=role,
    )

    extras: list[MemoLink] = []
    for raw in (extra_links or ()):
        link = _coerce_extra_link(raw)
        # Skip duplicates of the primary; Memo.validate would otherwise
        # silently dedupe them, but stripping early keeps the on-disk
        # link order ``[primary, …extras…]`` legible.
        if (
            link.target_type == primary.target_type
            and link.target_id == primary.target_id
            and link.role == primary.role
        ):
            continue
        extras.append(link)

    return Memo.new(
        project_id=project_id,
        type=chosen_type,
        title=title,
        body=body,
        body_format=body_format,
        author_coder_id=author_coder_id,
        links=[primary, *extras],
        tags=tags,
        provenance=provenance,
        memo_id=memo_id,
        now=now,
    )


def build_memo_draft_from_context(
    *,
    project_id: str,
    context: MemoContext | dict[str, Any],
    **fields: Any,
) -> Memo:
    """Convenience wrapper: accept a :class:`MemoContext` (or its dict
    form) plus the rest of :func:`build_memo_draft`'s kwargs.

    The server endpoint receives a JSON body that often contains a
    nested ``"context": {...}`` block; this lets the route handler
    forward it without unpacking the keys by hand. Unknown extra
    fields raise via :func:`build_memo_draft` / :func:`Memo.new`.
    """
    if isinstance(context, dict):
        ctx = MemoContext.from_dict(context)
    elif isinstance(context, MemoContext):
        ctx = context
    else:
        raise ProjectValidationError(
            "context must be a MemoContext or dict"
        )
    ctx.validate()
    return build_memo_draft(
        project_id=project_id,
        target_type=ctx.target_type,
        target_id=ctx.target_id,
        role=ctx.role,
        **fields,
    )
