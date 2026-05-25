"""Query builder (F3.5).

Per PLANNING.md F3.5:

  > Query builder: code filter + source filter + participant attribute
  > filter + speaker filter + boolean code combinator + proximity
  > (within span / paragraph / source).

This module provides the **data model** + **pure executor** for
running researcher-defined queries across a project's corpus. It is
deliberately stand-alone — no FastAPI, no engine imports — so the
filter algebra can be tested in pure Python and reused by the CLI,
the upcoming web UI, and (via F3.7) the saved-queries store.

What's here
-----------

  * :class:`AttributePredicate` — a tiny ``(key, op, value)`` triple
    that evaluates against a string-keyed attribute dict. Used by both
    :class:`SourceFilter` (over :attr:`Source.custom_attributes`) and
    :class:`ParticipantFilter` (over :attr:`Participant.demographics`).
  * :class:`CodeExpr` — a recursive boolean tree over code IDs:
    ``code(<id>)``, ``and(...)``, ``or(...)``, ``not(...)``. Powers
    the "boolean code combinator" requirement.
  * :class:`SourceFilter`, :class:`ParticipantFilter`,
    :class:`SpeakerFilter`, :class:`CodeFilter`,
    :class:`ProximityFilter` — the five filter dimensions PLANNING
    calls out.
  * :class:`Query` — the top-level entity bundling all five plus
    project metadata. Designed to round-trip JSON cleanly so F3.7's
    saved-queries store doesn't need a translation layer.
  * Pure executor functions: :func:`evaluate_code_expr`,
    :func:`predicate_matches`, :func:`filter_sources`,
    :func:`filter_participants`, :func:`applications_for_query`.

Application shape
-----------------

F4.1 (code applications anchored to word IDs) is **not yet
implemented**. To unblock F3.5 today, the executor accepts a generic
"application-like" dict shape:

  {
    "code_id":  "<12-hex>",            # required
    "source_id": "<12-hex>",           # required
    "speaker":  "<raw label>",         # optional, drives speaker filter
    "start":    <number>,              # optional, drives proximity
    "end":      <number>,              # optional, drives proximity
  }

When F4.1 lands, the on-disk Application entity is expected to expose
a ``to_query_dict()`` (or be constructible from one of these dicts);
this module's executor doesn't need to change.

Conventions match :mod:`scribe.projects` (F1.1), :mod:`scribe.sources`
(F1.2), :mod:`scribe.participants` (F1.3),
:mod:`scribe.source_schema` (F3.2), :mod:`scribe.participant_sources`
(F3.3), and :mod:`scribe.speaker_map` (F3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .codes import CODE_ID_RE
from .participants import Participant, PARTICIPANT_ID_RE
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
)
from .sources import (
    RECORDING_DATE_RE,
    SOURCE_ID_RE,
    Source,
)
from .speaker_map import SPEAKER_ROLES, SpeakerMap


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Operators a predicate may use. Closed set: rejecting unknown ops at
# the boundary keeps round-tripped queries deterministic (a saved
# query from another tool can't sneak a server-side "exec" through).
PREDICATE_OPERATORS: tuple[str, ...] = (
    "eq",
    "ne",
    "in",
    "not_in",
    "contains",
    "starts_with",
    "ends_with",
    "lt",
    "le",
    "gt",
    "ge",
    "exists",
    "missing",
)

# Operators whose value is a list of strings (membership tests).
_LIST_VALUE_OPS: frozenset[str] = frozenset({"in", "not_in"})

# Operators that ignore the predicate's ``value`` entirely.
_NO_VALUE_OPS: frozenset[str] = frozenset({"exists", "missing"})

# Operators whose value must coerce to a number.
_NUMERIC_OPS: frozenset[str] = frozenset({"lt", "le", "gt", "ge"})

# Tags for :class:`CodeExpr`. ``code`` is a leaf naming a single code
# id; the others are combinators. ``not`` takes exactly one child;
# ``and`` / ``or`` take one or more.
CODE_EXPR_OPS: tuple[str, ...] = (
    "code",
    "and",
    "or",
    "not",
)

# Proximity scopes. ``source`` is the loosest constraint ("co-occur in
# the same source"); ``paragraph`` is a numeric-distance window across
# the same source (interpreted by the caller — words or seconds, this
# module just compares numerically); ``segment`` requires anchor
# overlap.
PROXIMITY_SCOPES: tuple[str, ...] = (
    "segment",
    "paragraph",
    "source",
)

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 50 MB query.json.
MAX_NAME_LEN = 200
MAX_DESCRIPTION_LEN = 4000
MAX_LIST_ITEMS = 1024
MAX_PREDICATES = 64
MAX_CODE_EXPR_DEPTH = 16
MAX_CODE_EXPR_NODES = 256
MAX_PROXIMITY_GAP = 1_000_000


# --------------------------------------------------------------------------- #
# Public error type
# --------------------------------------------------------------------------- #


class QueryValidationError(ProjectValidationError):
    """Raised when a query, predicate, or code expression is malformed.

    Subclass of :class:`ProjectValidationError` so existing handlers
    (including the bundle-level validators) treat query problems the
    same way they treat any other entity-shape problem.
    """


# --------------------------------------------------------------------------- #
# AttributePredicate
# --------------------------------------------------------------------------- #


@dataclass
class AttributePredicate:
    """One filter clause against a key/value dict.

    ``key`` is the attribute name (e.g. ``"role"`` for a participant's
    demographics, ``"site"`` for a source's custom attributes).
    ``op`` is one of :data:`PREDICATE_OPERATORS`. ``value`` shape
    depends on ``op``:

      * ``in`` / ``not_in`` → list of strings
      * ``exists`` / ``missing`` → ignored (use ``None`` or ``""``)
      * ``lt`` / ``le`` / ``gt`` / ``ge`` → numeric (int / float / numeric string)
      * everything else → a single string
    """

    key: str
    op: str = "eq"
    value: Any = None

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AttributePredicate":
        if not isinstance(d, Mapping):
            raise QueryValidationError(
                "AttributePredicate payload must be an object"
            )
        if "key" not in d:
            raise QueryValidationError(
                "AttributePredicate payload missing 'key'"
            )
        p = cls(
            key=str(d["key"]),
            op=str(d.get("op", "eq") or "eq"),
            value=d.get("value"),
        )
        p.validate()
        return p

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        key = self.key.strip()
        if not key:
            raise QueryValidationError("AttributePredicate key is required")
        self.key = key

        if self.op not in PREDICATE_OPERATORS:
            raise QueryValidationError(
                f"AttributePredicate op must be one of {PREDICATE_OPERATORS}; "
                f"got {self.op!r}"
            )

        if self.op in _NO_VALUE_OPS:
            # Value ignored; canonicalise to None to keep round-trip stable.
            self.value = None
            return

        if self.op in _LIST_VALUE_OPS:
            if not isinstance(self.value, (list, tuple)):
                raise QueryValidationError(
                    f"AttributePredicate op {self.op!r} requires a list value; "
                    f"got {type(self.value).__name__}"
                )
            if len(self.value) > MAX_LIST_ITEMS:
                raise QueryValidationError(
                    f"AttributePredicate value list too long "
                    f"(>{MAX_LIST_ITEMS} items)"
                )
            self.value = [str(x) for x in self.value]
            return

        if self.op in _NUMERIC_OPS:
            try:
                _ = float(self.value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as e:
                raise QueryValidationError(
                    f"AttributePredicate op {self.op!r} requires a numeric "
                    f"value; got {self.value!r}"
                ) from e
            # Keep as the original string / number — coercion to float
            # at match time. This preserves "5" vs 5 round-tripping.
            return

        # Default: scalar string.
        if self.value is None:
            self.value = ""
        else:
            self.value = str(self.value)

    # ------------------------------------------------------------------ #
    # Match
    # ------------------------------------------------------------------ #

    def matches(self, attrs: Mapping[str, Any]) -> bool:
        """Return True if ``attrs[self.key]`` satisfies this predicate.

        Missing keys evaluate to:

          * ``exists`` → False
          * ``missing`` → True
          * everything else → False (the value isn't there to compare).
        """
        if not isinstance(attrs, Mapping):
            return False
        present = self.key in attrs
        raw = attrs.get(self.key)

        if self.op == "exists":
            # "exists" is *truthy* — empty string counts as missing,
            # mirroring how Source.custom_attributes treats empties.
            return present and raw not in (None, "")
        if self.op == "missing":
            return (not present) or raw in (None, "")

        if not present or raw is None:
            return False
        s = raw if isinstance(raw, str) else str(raw)

        if self.op == "eq":
            return s == self.value
        if self.op == "ne":
            return s != self.value
        if self.op == "in":
            return s in (self.value or [])
        if self.op == "not_in":
            return s not in (self.value or [])
        if self.op == "contains":
            return str(self.value) in s
        if self.op == "starts_with":
            return s.startswith(str(self.value))
        if self.op == "ends_with":
            return s.endswith(str(self.value))
        if self.op in _NUMERIC_OPS:
            try:
                lhs = float(s)
                rhs = float(self.value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False
            if self.op == "lt":
                return lhs < rhs
            if self.op == "le":
                return lhs <= rhs
            if self.op == "gt":
                return lhs > rhs
            if self.op == "ge":
                return lhs >= rhs

        # Unknown op slipped through validate() — defensively False.
        return False  # pragma: no cover


def predicate_matches(
    predicate: AttributePredicate, attrs: Mapping[str, Any]
) -> bool:
    """Module-level helper mirroring :meth:`AttributePredicate.matches`.

    Useful when the caller has a freshly-constructed predicate and
    prefers a free function (matches ``filter_segments_by_role`` style
    in :mod:`scribe.speaker_map`).
    """
    return predicate.matches(attrs)


def predicates_all_match(
    predicates: Sequence[AttributePredicate],
    attrs: Mapping[str, Any],
) -> bool:
    """All predicates must match (AND semantics). Empty list → True."""
    for p in predicates:
        if not p.matches(attrs):
            return False
    return True


# --------------------------------------------------------------------------- #
# CodeExpr
# --------------------------------------------------------------------------- #


@dataclass
class CodeExpr:
    """Boolean expression over code IDs.

    Four shapes:

      * ``op="code"`` — leaf: matches when ``code_id`` is in the
        applied set. ``children`` is empty.
      * ``op="and"``, ``op="or"`` — combinators with one or more
        ``children``.
      * ``op="not"`` — exactly one child.

    The expression is evaluated by :func:`evaluate_code_expr` against
    a set of code IDs (typically the codes applied to a particular
    span / source).
    """

    op: str = "code"
    code_id: str | None = None
    children: list["CodeExpr"] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def code(cls, code_id: str) -> "CodeExpr":
        e = cls(op="code", code_id=code_id)
        e.validate()
        return e

    @classmethod
    def all_of(cls, *children: "CodeExpr") -> "CodeExpr":
        e = cls(op="and", children=list(children))
        e.validate()
        return e

    @classmethod
    def any_of(cls, *children: "CodeExpr") -> "CodeExpr":
        e = cls(op="or", children=list(children))
        e.validate()
        return e

    @classmethod
    def negate(cls, child: "CodeExpr") -> "CodeExpr":
        e = cls(op="not", children=[child])
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.op == "code":
            d["code_id"] = self.code_id
        else:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CodeExpr":
        if not isinstance(d, Mapping):
            raise QueryValidationError("CodeExpr payload must be an object")
        op = str(d.get("op", "") or "")
        if op == "code":
            cid = d.get("code_id")
            e = cls(op=op, code_id=str(cid) if cid else None)
            e.validate()
            return e
        raw_children = d.get("children") or []
        if not isinstance(raw_children, list):
            raise QueryValidationError("CodeExpr children must be a list")
        children = [cls.from_dict(c) for c in raw_children]
        e = cls(op=op, children=children)
        e.validate()
        return e

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self, *, _depth: int = 0) -> None:
        if _depth > MAX_CODE_EXPR_DEPTH:
            raise QueryValidationError(
                f"CodeExpr nested deeper than {MAX_CODE_EXPR_DEPTH}"
            )
        if self.op not in CODE_EXPR_OPS:
            raise QueryValidationError(
                f"CodeExpr op must be one of {CODE_EXPR_OPS}; "
                f"got {self.op!r}"
            )

        if self.op == "code":
            if not self.code_id:
                raise QueryValidationError(
                    "CodeExpr op='code' requires a code_id"
                )
            if not CODE_ID_RE.match(self.code_id):
                raise QueryValidationError(
                    f"CodeExpr code_id must be 12-char hex; "
                    f"got {self.code_id!r}"
                )
            if self.children:
                raise QueryValidationError(
                    "CodeExpr op='code' must not have children"
                )
            return

        # Combinators.
        if not isinstance(self.children, list) or not self.children:
            raise QueryValidationError(
                f"CodeExpr op={self.op!r} requires at least one child"
            )
        if self.op == "not" and len(self.children) != 1:
            raise QueryValidationError(
                "CodeExpr op='not' requires exactly one child"
            )
        # The leaf vs. combinator counts both feed into the global cap.
        if len(self.children) > MAX_CODE_EXPR_NODES:
            raise QueryValidationError(
                f"CodeExpr has too many children at one level "
                f"(>{MAX_CODE_EXPR_NODES})"
            )
        # code_id must not be set on combinator nodes (would round-trip
        # to a meaningless field on disk).
        if self.code_id is not None:
            raise QueryValidationError(
                f"CodeExpr op={self.op!r} must not set code_id"
            )
        for c in self.children:
            c.validate(_depth=_depth + 1)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def referenced_code_ids(self) -> set[str]:
        """Every code id referenced anywhere in this expression."""
        out: set[str] = set()
        if self.op == "code" and self.code_id:
            out.add(self.code_id)
        for c in self.children:
            out |= c.referenced_code_ids()
        return out

    def node_count(self) -> int:
        """Total number of nodes (for cardinality checks)."""
        return 1 + sum(c.node_count() for c in self.children)


def evaluate_code_expr(expr: CodeExpr, applied_code_ids: Iterable[str]) -> bool:
    """Evaluate ``expr`` against the set of applied code IDs.

    ``applied_code_ids`` is anything iterable; we materialise once into
    a set for fast membership tests.
    """
    code_set = set(applied_code_ids)
    return _eval(expr, code_set)


def _eval(expr: CodeExpr, code_set: set[str]) -> bool:
    if expr.op == "code":
        return bool(expr.code_id and expr.code_id in code_set)
    if expr.op == "and":
        return all(_eval(c, code_set) for c in expr.children)
    if expr.op == "or":
        return any(_eval(c, code_set) for c in expr.children)
    if expr.op == "not":
        return not _eval(expr.children[0], code_set)
    return False  # pragma: no cover  # validate() prevents this


# --------------------------------------------------------------------------- #
# SourceFilter
# --------------------------------------------------------------------------- #


@dataclass
class SourceFilter:
    """Filter sources by id, language, recording date, custom attributes.

    All fields are conjunctive (AND): a source must satisfy every
    populated criterion to pass. An empty filter matches everything.

    ``recording_date_from`` / ``_to`` are inclusive on both ends and
    use the same ``YYYY-MM-DD`` shape as :class:`Source.recording_date`.
    """

    source_ids: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    recording_date_from: str = ""
    recording_date_to: str = ""
    attributes: list[AttributePredicate] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ids": list(self.source_ids),
            "languages": list(self.languages),
            "recording_date_from": self.recording_date_from,
            "recording_date_to": self.recording_date_to,
            "attributes": [a.to_dict() for a in self.attributes],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SourceFilter":
        if d is None:
            return cls()
        if not isinstance(d, Mapping):
            raise QueryValidationError("SourceFilter payload must be an object")
        f = cls(
            source_ids=[str(x) for x in (d.get("source_ids") or [])],
            languages=[str(x) for x in (d.get("languages") or [])],
            recording_date_from=str(d.get("recording_date_from", "") or ""),
            recording_date_to=str(d.get("recording_date_to", "") or ""),
            attributes=[
                AttributePredicate.from_dict(a)
                for a in (d.get("attributes") or [])
            ],
        )
        f.validate()
        return f

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if len(self.source_ids) > MAX_LIST_ITEMS:
            raise QueryValidationError(
                f"SourceFilter source_ids too long (>{MAX_LIST_ITEMS})"
            )
        for sid in self.source_ids:
            if not SOURCE_ID_RE.match(sid):
                raise QueryValidationError(
                    f"SourceFilter source_id must be 12-char hex; got {sid!r}"
                )
        if len(self.languages) > MAX_LIST_ITEMS:
            raise QueryValidationError(
                f"SourceFilter languages too long (>{MAX_LIST_ITEMS})"
            )
        for date_field in ("recording_date_from", "recording_date_to"):
            v = getattr(self, date_field)
            if v and not RECORDING_DATE_RE.match(v):
                raise QueryValidationError(
                    f"SourceFilter {date_field} must be YYYY-MM-DD; got {v!r}"
                )
        if (
            self.recording_date_from
            and self.recording_date_to
            and self.recording_date_from > self.recording_date_to
        ):
            raise QueryValidationError(
                "SourceFilter recording_date_from must be ≤ recording_date_to"
            )
        if len(self.attributes) > MAX_PREDICATES:
            raise QueryValidationError(
                f"SourceFilter has too many attribute predicates "
                f"(>{MAX_PREDICATES})"
            )
        for a in self.attributes:
            a.validate()

    # ------------------------------------------------------------------ #
    # Match
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        return (
            not self.source_ids
            and not self.languages
            and not self.recording_date_from
            and not self.recording_date_to
            and not self.attributes
        )

    def matches(self, source: Source) -> bool:
        if self.source_ids and source.id not in self.source_ids:
            return False
        if self.languages and source.language not in self.languages:
            return False
        if self.recording_date_from:
            if not source.recording_date:
                return False
            if source.recording_date < self.recording_date_from:
                return False
        if self.recording_date_to:
            if not source.recording_date:
                return False
            if source.recording_date > self.recording_date_to:
                return False
        if not predicates_all_match(self.attributes, source.custom_attributes):
            return False
        return True


def filter_sources(
    sources: Iterable[Source], f: SourceFilter
) -> list[Source]:
    """Return sources that pass the filter, in input order."""
    f.validate()
    return [s for s in sources if f.matches(s)]


# --------------------------------------------------------------------------- #
# ParticipantFilter
# --------------------------------------------------------------------------- #


@dataclass
class ParticipantFilter:
    """Filter participants by id and demographic predicates.

    Mirrors :class:`SourceFilter` but operates on
    :class:`scribe.participants.Participant`. Empty filter matches all.
    """

    participant_ids: list[str] = field(default_factory=list)
    demographics: list[AttributePredicate] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_ids": list(self.participant_ids),
            "demographics": [a.to_dict() for a in self.demographics],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "ParticipantFilter":
        if d is None:
            return cls()
        if not isinstance(d, Mapping):
            raise QueryValidationError(
                "ParticipantFilter payload must be an object"
            )
        f = cls(
            participant_ids=[
                str(x) for x in (d.get("participant_ids") or [])
            ],
            demographics=[
                AttributePredicate.from_dict(a)
                for a in (d.get("demographics") or [])
            ],
        )
        f.validate()
        return f

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if len(self.participant_ids) > MAX_LIST_ITEMS:
            raise QueryValidationError(
                f"ParticipantFilter participant_ids too long "
                f"(>{MAX_LIST_ITEMS})"
            )
        for pid in self.participant_ids:
            if not PARTICIPANT_ID_RE.match(pid):
                raise QueryValidationError(
                    f"ParticipantFilter participant_id must be 12-char hex; "
                    f"got {pid!r}"
                )
        if len(self.demographics) > MAX_PREDICATES:
            raise QueryValidationError(
                f"ParticipantFilter has too many demographic predicates "
                f"(>{MAX_PREDICATES})"
            )
        for a in self.demographics:
            a.validate()

    # ------------------------------------------------------------------ #
    # Match
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        return not self.participant_ids and not self.demographics

    def matches(self, participant: Participant) -> bool:
        if self.participant_ids and participant.id not in self.participant_ids:
            return False
        if not predicates_all_match(
            self.demographics, participant.demographics
        ):
            return False
        return True


def filter_participants(
    participants: Iterable[Participant], f: ParticipantFilter
) -> list[Participant]:
    """Return participants that pass the filter, in input order."""
    f.validate()
    return [p for p in participants if f.matches(p)]


# --------------------------------------------------------------------------- #
# SpeakerFilter
# --------------------------------------------------------------------------- #


@dataclass
class SpeakerFilter:
    """Filter applications / segments by speaker.

    Three independent dimensions:

      * ``labels`` — raw transcript-side labels (``"SPEAKER_00"`` /
        ``"Luke"`` / etc.).
      * ``roles`` — semantic roles from :data:`scribe.speaker_map.SPEAKER_ROLES`.
      * ``participant_ids`` — participants resolved through the
        :class:`scribe.speaker_map.SpeakerMap` for the source.

    The dimensions are **disjunctive** (OR): a speaker passes if any
    populated dimension matches. An empty filter matches everything.

    ``include_unmapped`` controls how labels not present in the
    speaker map are treated:
      * False (default) — they fail the role / participant tests but
        still pass the label test if the label is in ``labels``.
      * True — they pass any populated dimension (audit-the-corpus
        flow: "show me everything that's not yet bucketed").
    """

    labels: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    participant_ids: list[str] = field(default_factory=list)
    include_unmapped: bool = False

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "roles": list(self.roles),
            "participant_ids": list(self.participant_ids),
            "include_unmapped": bool(self.include_unmapped),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SpeakerFilter":
        if d is None:
            return cls()
        if not isinstance(d, Mapping):
            raise QueryValidationError(
                "SpeakerFilter payload must be an object"
            )
        f = cls(
            labels=[str(x) for x in (d.get("labels") or [])],
            roles=[str(x) for x in (d.get("roles") or [])],
            participant_ids=[
                str(x) for x in (d.get("participant_ids") or [])
            ],
            include_unmapped=bool(d.get("include_unmapped", False)),
        )
        f.validate()
        return f

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        for r in self.roles:
            if r not in SPEAKER_ROLES:
                raise QueryValidationError(
                    f"SpeakerFilter role must be one of {SPEAKER_ROLES}; "
                    f"got {r!r}"
                )
        for pid in self.participant_ids:
            if not PARTICIPANT_ID_RE.match(pid):
                raise QueryValidationError(
                    f"SpeakerFilter participant_id must be 12-char hex; "
                    f"got {pid!r}"
                )
        if (
            len(self.labels) > MAX_LIST_ITEMS
            or len(self.roles) > MAX_LIST_ITEMS
            or len(self.participant_ids) > MAX_LIST_ITEMS
        ):
            raise QueryValidationError(
                f"SpeakerFilter list too long (>{MAX_LIST_ITEMS})"
            )

    # ------------------------------------------------------------------ #
    # Match
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        return (
            not self.labels
            and not self.roles
            and not self.participant_ids
        )

    def matches(
        self, label: str, speaker_map: SpeakerMap | None
    ) -> bool:
        """Return True if ``label`` (as resolved through ``speaker_map``)
        passes any populated dimension.

        ``label == ""`` is always rejected (an unlabelled segment has no
        speaker information for any dimension to match).
        """
        if self.is_empty():
            return True
        if not label:
            return False

        entry = speaker_map.get(label) if speaker_map is not None else None
        unmapped = entry is None

        # Label dimension — direct match.
        if self.labels and label in self.labels:
            return True
        if self.roles:
            role = entry.role if entry is not None else "unknown"
            if role in self.roles:
                return True
        if self.participant_ids:
            pid = entry.participant_id if entry is not None else None
            if pid and pid in self.participant_ids:
                return True

        # Audit pass: include speakers not in the map regardless of
        # which dimension was asked for. Lets a researcher say "show me
        # the focus group plus any orphans I haven't bucketed yet".
        if unmapped and self.include_unmapped:
            return True
        return False


# --------------------------------------------------------------------------- #
# CodeFilter
# --------------------------------------------------------------------------- #


@dataclass
class CodeFilter:
    """Filter applications by a boolean expression over code IDs.

    A null ``expr`` (the default) matches every application — useful
    for queries that only constrain on source / participant / speaker.

    The expression is evaluated *per-scope* by the executor:
      * default — evaluated against the single application's own code id
        (so a leaf ``code(<id>)`` filter is "applications of <id>");
      * with proximity — evaluated against the set of code ids
        co-occurring within the proximity scope.
    """

    expr: CodeExpr | None = None

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {"expr": self.expr.to_dict() if self.expr is not None else None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "CodeFilter":
        if d is None:
            return cls()
        if not isinstance(d, Mapping):
            raise QueryValidationError("CodeFilter payload must be an object")
        raw = d.get("expr")
        expr = None
        if raw is not None:
            expr = CodeExpr.from_dict(raw)
        f = cls(expr=expr)
        f.validate()
        return f

    # ------------------------------------------------------------------ #
    # Validation / introspection
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if self.expr is not None:
            self.expr.validate()
            if self.expr.node_count() > MAX_CODE_EXPR_NODES:
                raise QueryValidationError(
                    f"CodeFilter expression too large "
                    f"(>{MAX_CODE_EXPR_NODES} nodes)"
                )

    def is_empty(self) -> bool:
        return self.expr is None

    def referenced_code_ids(self) -> set[str]:
        return self.expr.referenced_code_ids() if self.expr is not None else set()


# --------------------------------------------------------------------------- #
# ProximityFilter
# --------------------------------------------------------------------------- #


@dataclass
class ProximityFilter:
    """Co-occurrence constraint across required code ids.

    The executor keeps applications whose code id is in
    ``required_code_ids`` *and* every other id in ``required_code_ids``
    has at least one application within the same scope.

    ``scope`` semantics:

      * ``"source"`` — same source.
      * ``"segment"`` — anchor ranges overlap. An application's anchor
        is read from ``start`` / ``end`` in the application dict
        (numeric: words or seconds — caller decides; this module just
        compares numerically).
      * ``"paragraph"`` — within ``max_gap`` numeric distance in the
        same source. ``max_gap=0`` means "anchors overlap or touch";
        any positive value widens the window.
    """

    scope: str = "source"
    required_code_ids: list[str] = field(default_factory=list)
    max_gap: float = 0.0

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "required_code_ids": list(self.required_code_ids),
            "max_gap": self.max_gap,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "ProximityFilter":
        if d is None:
            return cls()
        if not isinstance(d, Mapping):
            raise QueryValidationError(
                "ProximityFilter payload must be an object"
            )
        f = cls(
            scope=str(d.get("scope", "source") or "source"),
            required_code_ids=[
                str(x) for x in (d.get("required_code_ids") or [])
            ],
            max_gap=float(d.get("max_gap", 0.0) or 0.0),
        )
        f.validate()
        return f

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if self.scope not in PROXIMITY_SCOPES:
            raise QueryValidationError(
                f"ProximityFilter scope must be one of {PROXIMITY_SCOPES}; "
                f"got {self.scope!r}"
            )
        if len(self.required_code_ids) > MAX_LIST_ITEMS:
            raise QueryValidationError(
                f"ProximityFilter required_code_ids too long "
                f"(>{MAX_LIST_ITEMS})"
            )
        for cid in self.required_code_ids:
            if not CODE_ID_RE.match(cid):
                raise QueryValidationError(
                    f"ProximityFilter required_code_id must be 12-char hex; "
                    f"got {cid!r}"
                )
        if self.max_gap < 0:
            raise QueryValidationError(
                "ProximityFilter max_gap must be ≥ 0"
            )
        if self.max_gap > MAX_PROXIMITY_GAP:
            raise QueryValidationError(
                f"ProximityFilter max_gap too large (>{MAX_PROXIMITY_GAP})"
            )

    def is_empty(self) -> bool:
        return not self.required_code_ids


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #


@dataclass
class Query:
    """A composite query bundling all filter dimensions.

    Used directly by the executor and persisted as-is by F3.7's
    saved-queries store. ``project_id`` ties the query to a project;
    callers must verify the on-disk project matches before applying
    the query (a saved query referencing codes from project A should
    not silently match project B).
    """

    project_id: str
    name: str = ""
    description: str = ""
    sources: SourceFilter = field(default_factory=SourceFilter)
    participants: ParticipantFilter = field(default_factory=ParticipantFilter)
    speakers: SpeakerFilter = field(default_factory=SpeakerFilter)
    codes: CodeFilter = field(default_factory=CodeFilter)
    proximity: ProximityFilter | None = None

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "sources": self.sources.to_dict(),
            "participants": self.participants.to_dict(),
            "speakers": self.speakers.to_dict(),
            "codes": self.codes.to_dict(),
            "proximity": (
                self.proximity.to_dict() if self.proximity is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Query":
        if not isinstance(d, Mapping):
            raise QueryValidationError("Query payload must be an object")
        if "project_id" not in d:
            raise QueryValidationError("Query payload missing 'project_id'")
        proximity_raw = d.get("proximity")
        proximity = (
            ProximityFilter.from_dict(proximity_raw)
            if proximity_raw is not None
            else None
        )
        q = cls(
            project_id=str(d["project_id"]),
            name=str(d.get("name", "") or ""),
            description=str(d.get("description", "") or ""),
            sources=SourceFilter.from_dict(d.get("sources")),
            participants=ParticipantFilter.from_dict(d.get("participants")),
            speakers=SpeakerFilter.from_dict(d.get("speakers")),
            codes=CodeFilter.from_dict(d.get("codes")),
            proximity=proximity,
        )
        q.validate()
        return q

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not PROJECT_ID_RE.match(self.project_id):
            raise QueryValidationError(
                f"Invalid project id: {self.project_id!r}"
            )

        name = self.name.strip()
        if len(name) > MAX_NAME_LEN:
            raise QueryValidationError(
                f"Query name must be ≤ {MAX_NAME_LEN} chars"
            )
        self.name = name

        desc = self.description if self.description else ""
        if len(desc) > MAX_DESCRIPTION_LEN:
            raise QueryValidationError(
                f"Query description must be ≤ {MAX_DESCRIPTION_LEN} chars"
            )
        self.description = desc

        self.sources.validate()
        self.participants.validate()
        self.speakers.validate()
        self.codes.validate()
        if self.proximity is not None:
            self.proximity.validate()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def referenced_code_ids(self) -> set[str]:
        """Every code id mentioned by this query (codes filter + proximity)."""
        out = self.codes.referenced_code_ids()
        if self.proximity is not None:
            out |= set(self.proximity.required_code_ids)
        return out


# --------------------------------------------------------------------------- #
# Application-shape helpers
# --------------------------------------------------------------------------- #


def _app_get(app: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    """Read a field off an application-shaped value (dict-like or attr-like)."""
    if isinstance(app, Mapping):
        return app.get(key, default)
    return getattr(app, key, default)


def _app_required_field(app: Any, key: str) -> str:
    v = _app_get(app, key)
    if v is None or v == "":
        raise QueryValidationError(
            f"application missing required field {key!r}"
        )
    return str(v)


def _coerce_optional_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# applications_for_query
# --------------------------------------------------------------------------- #


def applications_for_query(
    query: Query,
    applications: Iterable[Any],
    *,
    sources: Sequence[Source] | None = None,
    participants: Sequence[Participant] | None = None,
    speaker_maps: Mapping[str, SpeakerMap] | None = None,
) -> list[Any]:
    """Return the applications that satisfy ``query``.

    ``applications`` is an iterable of application-shaped dicts (see
    the module docstring). ``sources``, ``participants``,
    ``speaker_maps`` are reference data needed to resolve filters:

      * ``sources`` is required if the query has a non-empty
        :class:`SourceFilter` (we need the full :class:`Source` to
        check language / date / custom attributes — the application
        dict only carries the source id).
      * ``participants`` is required if the query has a non-empty
        :class:`ParticipantFilter`.
      * ``speaker_maps`` (keyed by source id) is required for any
        non-empty :class:`SpeakerFilter` that uses roles or
        participant_ids; label-only speaker filters work without it.

    Returns the applications in input order. Does not mutate inputs.
    """
    query.validate()

    # Resolve which sources pass the source filter (set of ids).
    allowed_sids: set[str] | None
    if query.sources.is_empty():
        allowed_sids = None
    else:
        if sources is None:
            raise QueryValidationError(
                "applications_for_query: sources is required when the "
                "query has a non-empty SourceFilter"
            )
        allowed_sids = {s.id for s in filter_sources(sources, query.sources)}

    # Resolve which participants pass the participant filter.
    allowed_pids: set[str] | None
    if query.participants.is_empty():
        allowed_pids = None
    else:
        if participants is None:
            raise QueryValidationError(
                "applications_for_query: participants is required when "
                "the query has a non-empty ParticipantFilter"
            )
        allowed_pids = {
            p.id
            for p in filter_participants(participants, query.participants)
        }

    smap_lookup: dict[str, SpeakerMap] = (
        dict(speaker_maps) if speaker_maps else {}
    )

    # First sweep: keep only applications that pass source / speaker /
    # participant filters and the (per-application) code filter.
    candidates: list[Any] = []
    for app in applications:
        sid = _app_required_field(app, "source_id")
        cid = _app_required_field(app, "code_id")
        if not SOURCE_ID_RE.match(sid):
            raise QueryValidationError(
                f"application source_id must be 12-char hex; got {sid!r}"
            )
        if not CODE_ID_RE.match(cid):
            raise QueryValidationError(
                f"application code_id must be 12-char hex; got {cid!r}"
            )
        if allowed_sids is not None and sid not in allowed_sids:
            continue

        # Speaker filter — uses the source's speaker map (if any).
        if not query.speakers.is_empty():
            label = _app_get(app, "speaker", "")
            if label is None:
                label = ""
            label = str(label)
            smap = smap_lookup.get(sid)
            if not query.speakers.matches(label, smap):
                continue

        # Participant filter — derive participant id from the speaker
        # map if we have one; otherwise fall back to an explicit
        # ``participant_id`` field on the application (forward-compat
        # for F4.1 if it stores the id directly).
        if allowed_pids is not None:
            pid: str | None = None
            label = str(_app_get(app, "speaker", "") or "")
            if label and sid in smap_lookup:
                pid = smap_lookup[sid].participant_for(label)
            if pid is None:
                pid = _app_get(app, "participant_id")
            if pid is None or pid not in allowed_pids:
                continue

        # Code filter — single-application semantics: the leaf ``code``
        # operator matches when this application's code_id is the leaf,
        # combinators evaluate against the singleton set {cid}.
        if not query.codes.is_empty():
            if not evaluate_code_expr(query.codes.expr, {cid}):  # type: ignore[arg-type]
                continue

        candidates.append(app)

    # Optional second sweep: proximity. Only applications whose code id
    # is in ``required_code_ids`` are kept; for each such application,
    # every *other* required id must have at least one application
    # within scope.
    if query.proximity is not None and not query.proximity.is_empty():
        candidates = _apply_proximity(query.proximity, candidates)

    return candidates


def _apply_proximity(
    pf: ProximityFilter, apps: Sequence[Any]
) -> list[Any]:
    required = list(pf.required_code_ids)
    required_set = set(required)
    if not required:
        return list(apps)

    # Bucket per source for cheap lookups.
    by_source: dict[str, list[Any]] = {}
    for a in apps:
        sid = _app_get(a, "source_id", "")
        by_source.setdefault(str(sid), []).append(a)

    out: list[Any] = []
    for a in apps:
        cid = _app_get(a, "code_id", "")
        if cid not in required_set:
            continue
        sid = str(_app_get(a, "source_id", ""))
        siblings = by_source.get(sid, [])
        a_start = _coerce_optional_float(_app_get(a, "start"))
        a_end = _coerce_optional_float(_app_get(a, "end"))

        # Need at least one application of every other required code id
        # somewhere in scope.
        ok = True
        for need in required:
            if need == cid:
                continue
            found = False
            for other in siblings:
                if _app_get(other, "code_id") != need:
                    continue
                if pf.scope == "source":
                    found = True
                    break
                # Need numeric anchors to compare.
                o_start = _coerce_optional_float(_app_get(other, "start"))
                o_end = _coerce_optional_float(_app_get(other, "end"))
                if a_start is None or a_end is None:
                    continue
                if o_start is None or o_end is None:
                    continue
                if pf.scope == "segment":
                    # Anchors overlap (closed intervals).
                    if o_start <= a_end and a_start <= o_end:
                        found = True
                        break
                elif pf.scope == "paragraph":
                    # Distance between the two anchor ranges within
                    # max_gap. If the ranges overlap, distance is 0;
                    # otherwise it's the gap between the nearer edges.
                    if o_start <= a_end and a_start <= o_end:
                        gap = 0.0
                    elif o_end < a_start:
                        gap = a_start - o_end
                    else:
                        gap = o_start - a_end
                    if gap <= pf.max_gap:
                        found = True
                        break
            if not found:
                ok = False
                break
        if ok:
            out.append(a)
    return out
