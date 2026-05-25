"""Source attribute schema (F3.2).

Per PLANNING.md F3.2:

  > Source attribute schema (user-defined columns per source).

F1.2 already gave each :class:`scribe.sources.Source` a free-form
``custom_attributes: dict[str, str]`` slot — enough for storage but
zero help for the UI. F3.2 layers a **project-level schema** on top so
the researcher can declare *up front* what columns the corpus has,
what shape their values take, and which ones are required. The schema
drives:

  * a consistent column header set in the eventual UI source-table view;
  * a uniform export shape (CSV / REFI-QDA exports of sources);
  * cross-source validation (a "site" column with select options
    {Hospital A, Hospital B, Clinic C} catches a typo'd ``"Hospital A "``
    in source #14);
  * future query-builder filters (F3.5) that need to know the type of
    each attribute to render the right control.

The schema is **per project**, lives as a single file at
``projects/<project_id>/source_schema.json``, and is owned by the
:class:`scribe.project_format.ProjectBundle`.

Like the rest of the F1.x / F2.x / F3.x foundation modules this is
stand-alone — no FastAPI, no engine imports — so it's testable in pure
Python and reusable from the CLI later. Conventions match
``scribe.projects`` (F1.1) and ``scribe.sources`` (F1.2).

Design notes:

* **Storage shape vs. display shape.** Source values stay as strings on
  disk (JSON's natural shape; matches F1.2's ``custom_attributes``).
  The schema documents the *intended* type — number, date, select,
  boolean — so the UI can render the right control, validation can
  catch typos, and exports can emit the right column type. Coercion
  helpers (``coerce_value`` / ``coerce_attributes``) round-trip a
  string to its canonical on-disk form.
* **Forward-compat.** Unknown attribute keys on a source are *allowed*
  — researchers commonly add an ad-hoc column, and the schema is a
  guide rather than a wall. ``strict=True`` flips that behaviour for
  callers who want a hard contract (e.g. import).
* **Append-only by spirit.** The schema isn't versioned (yet — F9.x
  territory), but the attribute-definition list keeps insertion order
  so the column order in tables is stable. Renaming a key is treated
  as a separate add+remove by callers; no auto-migration of source
  values here.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
    project_dir,
    utcnow_iso,
)
from .sources import (
    CUSTOM_ATTR_KEY_RE,
    MAX_CUSTOM_ATTR_VALUE_LEN,
    RECORDING_DATE_RE,
)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Attribute-value types we recognise. ``text`` is the catch-all (any
# UTF-8 string ≤ MAX_CUSTOM_ATTR_VALUE_LEN). The others are *intent*
# markers — values are still stored as strings on disk, but the schema
# tells the UI what control to render and the validator what shape to
# enforce.
ATTRIBUTE_TYPES: tuple[str, ...] = (
    "text",
    "number",
    "date",
    "select",
    "boolean",
)

# Legal canonical values for a boolean attribute. The validator
# accepts a few common spellings on the way in (true/false, yes/no,
# 1/0 — case-insensitive) and ``coerce_value`` normalises to "true" /
# "false" so on-disk values are uniform.
_BOOLEAN_TRUE_TOKENS = frozenset({"true", "yes", "1", "y", "t"})
_BOOLEAN_FALSE_TOKENS = frozenset({"false", "no", "0", "n", "f"})

# Field length / cardinality limits. Generous, but bounded so a typo
# can't write a 10 MB schema file.
MAX_LABEL_LEN = 200
MAX_DESCRIPTION_LEN = 1000
MAX_OPTION_LEN = MAX_CUSTOM_ATTR_VALUE_LEN  # an option is a value
MAX_OPTIONS = 256
MAX_ATTRIBUTES = 64

# Schema filename relative to ``projects/<project_id>/``.
SCHEMA_FILENAME = "source_schema.json"


# --------------------------------------------------------------------------- #
# AttributeDefinition
# --------------------------------------------------------------------------- #


@dataclass
class AttributeDefinition:
    """One column in the source attribute schema.

    ``key`` is the machine identifier (matches the keys used in
    :attr:`scribe.sources.Source.custom_attributes`); ``label`` is the
    human-readable header. ``type`` selects the value shape; ``options``
    is only meaningful when ``type == "select"`` (and required at that
    point — a select with no options is a UI dead end). ``required``
    means a source's ``custom_attributes`` must include this key with a
    non-empty value to pass strict validation.
    """

    key: str
    label: str = ""
    type: str = "text"
    required: bool = False
    options: list[str] = field(default_factory=list)
    description: str = ""

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttributeDefinition":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "AttributeDefinition payload must be an object"
            )
        if "key" not in d:
            raise ProjectValidationError(
                "AttributeDefinition payload missing 'key'"
            )
        a = cls(
            key=str(d["key"]),
            label=str(d.get("label", "") or ""),
            type=str(d.get("type", "text") or "text"),
            required=bool(d.get("required", False)),
            options=[str(x) for x in (d.get("options") or [])],
            description=str(d.get("description", "") or ""),
        )
        a.validate()
        return a

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        # Key shape mirrors Source.custom_attributes' key constraint
        # exactly — the two have to agree to be meaningful.
        key = self.key.strip()
        if not key:
            raise ProjectValidationError("AttributeDefinition key is required")
        if not CUSTOM_ATTR_KEY_RE.match(key):
            raise ProjectValidationError(
                f"AttributeDefinition key {key!r} invalid "
                "(letters/digits/underscore/hyphen/space, "
                "1–64 chars, must start with a letter)"
            )
        self.key = key

        label = self.label.strip()
        if len(label) > MAX_LABEL_LEN:
            raise ProjectValidationError(
                f"AttributeDefinition label must be ≤ {MAX_LABEL_LEN} chars"
            )
        # Empty label → fall back to the key in display contexts; we
        # don't substitute here so callers can tell user-set vs. default.
        self.label = label

        if self.type not in ATTRIBUTE_TYPES:
            raise ProjectValidationError(
                f"AttributeDefinition type must be one of {ATTRIBUTE_TYPES}; "
                f"got {self.type!r}"
            )

        if not isinstance(self.options, list):
            raise ProjectValidationError(
                "AttributeDefinition options must be a list of strings"
            )
        if self.type == "select":
            if not self.options:
                raise ProjectValidationError(
                    f"AttributeDefinition {self.key!r}: "
                    "type='select' requires at least one option"
                )
        else:
            # options are only meaningful for select; reject if set,
            # so the schema can't accidentally claim choices that the
            # validator will then ignore.
            if self.options:
                raise ProjectValidationError(
                    f"AttributeDefinition {self.key!r}: "
                    f"options only valid for type='select', "
                    f"not {self.type!r}"
                )
        if len(self.options) > MAX_OPTIONS:
            raise ProjectValidationError(
                f"AttributeDefinition {self.key!r}: "
                f"at most {MAX_OPTIONS} options allowed"
            )
        cleaned_options: list[str] = []
        seen: set[str] = set()
        for raw in self.options:
            opt = str(raw).strip()
            if not opt:
                continue  # silently drop empty options
            if len(opt) > MAX_OPTION_LEN:
                raise ProjectValidationError(
                    f"AttributeDefinition {self.key!r}: option too long "
                    f"(>{MAX_OPTION_LEN}): {opt[:40]!r}…"
                )
            if opt in seen:
                continue  # de-dupe, preserving first-seen order
            seen.add(opt)
            cleaned_options.append(opt)
        self.options = cleaned_options

        if len(self.description) > MAX_DESCRIPTION_LEN:
            raise ProjectValidationError(
                f"AttributeDefinition {self.key!r}: description too long "
                f"(>{MAX_DESCRIPTION_LEN})"
            )
        self.description = self.description.strip()


# --------------------------------------------------------------------------- #
# SourceAttributeSchema
# --------------------------------------------------------------------------- #


@dataclass
class SourceAttributeSchema:
    """Project-level schema declaring the source-attribute columns.

    A project may have at most :data:`MAX_ATTRIBUTES` attribute
    definitions. The list preserves insertion order so the UI's column
    order is stable across saves. Keys are unique (case-insensitive
    comparison? No — case-sensitive, because the keys are also dict
    keys on each source's ``custom_attributes``; differing case would
    silently produce two columns that look identical in the UI).
    """

    project_id: str
    attributes: list[AttributeDefinition] = field(default_factory=list)
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
        attributes: Iterable[AttributeDefinition | dict[str, Any]] | None = None,
        now: str | None = None,
    ) -> "SourceAttributeSchema":
        ts = now or utcnow_iso()
        attrs: list[AttributeDefinition] = []
        for a in attributes or []:
            if isinstance(a, AttributeDefinition):
                attrs.append(a)
            elif isinstance(a, dict):
                attrs.append(AttributeDefinition.from_dict(a))
            else:
                raise ProjectValidationError(
                    "attributes entries must be AttributeDefinition or dict"
                )
        s = cls(
            project_id=project_id,
            attributes=attrs,
            created_at=ts,
            modified_at=ts,
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "attributes": [a.to_dict() for a in self.attributes],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceAttributeSchema":
        if not isinstance(d, dict):
            raise ProjectValidationError(
                "SourceAttributeSchema payload must be an object"
            )
        if "project_id" not in d:
            raise ProjectValidationError(
                "SourceAttributeSchema payload missing 'project_id'"
            )
        raw_attrs = d.get("attributes")
        if raw_attrs is None:
            raw_attrs = []
        if not isinstance(raw_attrs, list):
            raise ProjectValidationError(
                "SourceAttributeSchema attributes must be a list"
            )
        attrs = [AttributeDefinition.from_dict(a) for a in raw_attrs]
        s = cls(
            project_id=str(d["project_id"]),
            attributes=attrs,
            created_at=str(d.get("created_at", "") or ""),
            modified_at=str(d.get("modified_at", "") or ""),
        )
        s.validate()
        return s

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def apply_update(
        self, patch: dict[str, Any], *, now: str | None = None
    ) -> None:
        """Replace the attribute list (and only the attribute list).

        ``project_id``, ``created_at``, ``modified_at`` are managed by
        the entity itself; passing them is allowed (and ignored) so a
        client can round-trip a fetched object.
        """
        if not isinstance(patch, dict):
            raise ProjectValidationError("Update must be an object")
        unknown = set(patch.keys()) - _ALLOWED_PATCH_KEYS - _IGNORED_PATCH_KEYS
        if unknown:
            raise ProjectValidationError(
                f"Unknown fields: {', '.join(sorted(unknown))}"
            )
        if "attributes" in patch:
            raw = patch["attributes"] or []
            if not isinstance(raw, list):
                raise ProjectValidationError(
                    "attributes must be a list of AttributeDefinitions"
                )
            new_attrs: list[AttributeDefinition] = []
            for a in raw:
                if isinstance(a, AttributeDefinition):
                    new_attrs.append(a)
                elif isinstance(a, dict):
                    new_attrs.append(AttributeDefinition.from_dict(a))
                else:
                    raise ProjectValidationError(
                        "attributes entries must be AttributeDefinition or dict"
                    )
            self.attributes = new_attrs

        self.validate()
        # Only stamp modified_at after validation succeeds — a failed
        # update should not advance the clock.
        self.modified_at = now or utcnow_iso()

    def add_attribute(
        self, attr: AttributeDefinition | dict[str, Any], *, now: str | None = None
    ) -> AttributeDefinition:
        """Append a new attribute. Returns the canonicalised definition.

        Raises if a definition with the same ``key`` already exists.
        """
        if isinstance(attr, dict):
            attr = AttributeDefinition.from_dict(attr)
        elif not isinstance(attr, AttributeDefinition):
            raise ProjectValidationError(
                "add_attribute requires an AttributeDefinition or dict"
            )
        attr.validate()
        if any(a.key == attr.key for a in self.attributes):
            raise ProjectValidationError(
                f"Attribute with key {attr.key!r} already exists"
            )
        self.attributes.append(attr)
        self.validate()
        self.modified_at = now or utcnow_iso()
        return attr

    def remove_attribute(self, key: str, *, now: str | None = None) -> bool:
        """Remove an attribute by key. Returns True if removed."""
        before = len(self.attributes)
        self.attributes = [a for a in self.attributes if a.key != key]
        if len(self.attributes) == before:
            return False
        self.modified_at = now or utcnow_iso()
        return True

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def by_key(self, key: str) -> AttributeDefinition | None:
        for a in self.attributes:
            if a.key == key:
                return a
        return None

    def keys(self) -> list[str]:
        return [a.key for a in self.attributes]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectValidationError(
                f"Invalid project id: {self.project_id!r}"
            )
        if not isinstance(self.attributes, list):
            raise ProjectValidationError(
                "attributes must be a list of AttributeDefinitions"
            )
        if len(self.attributes) > MAX_ATTRIBUTES:
            raise ProjectValidationError(
                f"At most {MAX_ATTRIBUTES} attributes allowed"
            )
        seen_keys: set[str] = set()
        for a in self.attributes:
            a.validate()
            if a.key in seen_keys:
                raise ProjectValidationError(
                    f"Duplicate attribute key in schema: {a.key!r}"
                )
            seen_keys.add(a.key)


_ALLOWED_PATCH_KEYS = {"attributes"}
_IGNORED_PATCH_KEYS = {"project_id", "created_at", "modified_at"}


# --------------------------------------------------------------------------- #
# Value coercion + validation
# --------------------------------------------------------------------------- #


def coerce_value(raw: Any, attr_type: str) -> str:
    """Coerce a raw value into its canonical on-disk string for ``attr_type``.

    Raises :class:`ProjectValidationError` if the value can't be coerced.
    Empty string is always returned as-is (caller decides whether the
    field is required); this lets a UI clear a value without tripping
    type checks.
    """
    if raw is None:
        return ""
    s = raw if isinstance(raw, str) else str(raw)
    s = s.strip()
    if not s:
        return ""

    if attr_type not in ATTRIBUTE_TYPES:
        raise ProjectValidationError(f"Unknown attribute type: {attr_type!r}")

    if attr_type == "text":
        return s
    if attr_type == "number":
        # Accept ints + floats; canonical form is whatever ``str(float())``
        # produces, but preserve integer literals (no trailing .0 for "5").
        try:
            f = float(s)
        except ValueError as e:
            raise ProjectValidationError(
                f"Value {s!r} is not a valid number"
            ) from e
        if f != f:  # NaN
            raise ProjectValidationError("Number must be finite")
        if f in (float("inf"), float("-inf")):
            raise ProjectValidationError("Number must be finite")
        # If the input looks like a clean integer, preserve that shape;
        # otherwise round-trip through float.
        if re.fullmatch(r"[+-]?\d+", s):
            return str(int(s))
        return s
    if attr_type == "date":
        if not RECORDING_DATE_RE.match(s):
            raise ProjectValidationError(
                f"Date {s!r} must be in YYYY-MM-DD form"
            )
        y, m, d = (int(x) for x in s.split("-"))
        if not (1 <= m <= 12 and 1 <= d <= 31):
            raise ProjectValidationError(
                f"Date {s!r} components out of range"
            )
        return s
    if attr_type == "boolean":
        low = s.lower()
        if low in _BOOLEAN_TRUE_TOKENS:
            return "true"
        if low in _BOOLEAN_FALSE_TOKENS:
            return "false"
        raise ProjectValidationError(
            f"Boolean value {s!r} not recognised "
            "(use true/false, yes/no, 1/0)"
        )
    if attr_type == "select":
        # Coerce just normalises whitespace; option-membership is a
        # schema-level check done in ``validate_attributes``.
        return s
    raise ProjectValidationError(  # pragma: no cover - unreachable
        f"Unhandled attribute type: {attr_type!r}"
    )


def validate_attributes(
    custom_attributes: dict[str, Any],
    schema: SourceAttributeSchema,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate a source's ``custom_attributes`` against ``schema``.

    Returns a list of human-readable error strings. Empty list means the
    attributes pass. Behaviour:

      * Each attribute's value is coerced to its canonical string form
        and checked against the type's rules (``coerce_value``).
      * Required attributes must be present *and* non-empty.
      * For ``select`` attributes the value (after coercion) must equal
        one of the schema's declared options (case-sensitive).
      * Keys not in the schema:
          - ``strict=False`` (default): allowed (forward-compat —
            researchers add ad-hoc columns mid-corpus).
          - ``strict=True``: rejected.

    The schema itself is not re-validated here; callers should pass a
    schema that already passed :meth:`SourceAttributeSchema.validate`.
    """
    if not isinstance(custom_attributes, dict):
        return ["custom_attributes must be a dict"]
    errors: list[str] = []
    by_key = {a.key: a for a in schema.attributes}

    # Forward sweep: every value in ``custom_attributes``.
    for key, raw in custom_attributes.items():
        if key not in by_key:
            if strict:
                errors.append(f"Unknown attribute key: {key!r}")
            continue
        attr = by_key[key]
        try:
            coerced = coerce_value(raw, attr.type)
        except ProjectValidationError as e:
            errors.append(f"{key!r}: {e}")
            continue
        if coerced == "":
            # Empty values are handled by the "required" sweep below.
            continue
        if attr.type == "select":
            if coerced not in attr.options:
                errors.append(
                    f"{key!r}: value {coerced!r} not in options "
                    f"({', '.join(repr(o) for o in attr.options)})"
                )

    # Required sweep: every required attribute must be present + non-empty.
    for attr in schema.attributes:
        if not attr.required:
            continue
        raw = custom_attributes.get(attr.key)
        if raw is None:
            errors.append(f"{attr.key!r} is required")
            continue
        try:
            coerced = coerce_value(raw, attr.type)
        except ProjectValidationError:
            # Already reported above.
            continue
        if coerced == "":
            errors.append(f"{attr.key!r} is required (value is empty)")

    return errors


def coerce_attributes(
    custom_attributes: dict[str, Any],
    schema: SourceAttributeSchema,
    *,
    strict: bool = False,
) -> dict[str, str]:
    """Validate + return the canonicalised attributes for a source.

    Raises :class:`ProjectValidationError` on any validation failure;
    on success returns a fresh dict with values coerced into their
    canonical on-disk strings. Keys not in the schema are passed
    through unchanged when ``strict=False``.
    """
    errors = validate_attributes(custom_attributes, schema, strict=strict)
    if errors:
        raise ProjectValidationError(
            "Source attributes failed schema validation: "
            + "; ".join(errors)
        )
    by_key = {a.key: a for a in schema.attributes}
    out: dict[str, str] = {}
    for key, raw in custom_attributes.items():
        if key in by_key:
            out[key] = coerce_value(raw, by_key[key].type)
        else:
            # Pass through unknown keys (only reachable when
            # strict=False) as plain strings; matches the Source-side
            # storage shape.
            if raw is None:
                out[key] = ""
            else:
                out[key] = str(raw).strip()
    return out


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #


def source_schema_path(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk path of a project's source-attribute schema."""
    return project_dir(projects_root, project_id) / SCHEMA_FILENAME


def save_source_schema(
    projects_root: Path, schema: SourceAttributeSchema
) -> Path:
    """Persist a schema to ``<projects_root>/<pid>/source_schema.json``.

    The parent project directory must already exist, mirroring how
    ``save_source`` works.
    """
    schema.validate()
    parent = project_dir(projects_root, schema.project_id)
    if not parent.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {parent}. "
            "Save the project before saving its source schema."
        )
    target = source_schema_path(projects_root, schema.project_id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(schema.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def load_source_schema(
    projects_root: Path, project_id: str
) -> SourceAttributeSchema:
    """Load a source schema by project id.

    Raises :class:`FileNotFoundError` if the schema file is missing.
    Callers that want a soft fallback should use
    :func:`load_or_empty_source_schema`.
    """
    p = source_schema_path(projects_root, project_id)
    if not p.exists():
        raise FileNotFoundError(f"No source schema at {p}")
    return SourceAttributeSchema.from_dict(json.loads(p.read_text()))


def load_or_empty_source_schema(
    projects_root: Path, project_id: str
) -> SourceAttributeSchema:
    """Load a source schema if present; otherwise return an empty one.

    Useful in code paths (UI list views, exports) that want to render
    the column set without forcing the user to create the schema first.
    """
    try:
        return load_source_schema(projects_root, project_id)
    except FileNotFoundError:
        return SourceAttributeSchema.new(project_id=project_id)


def delete_source_schema(projects_root: Path, project_id: str) -> bool:
    """Remove a project's source schema file. Returns False if missing."""
    p = source_schema_path(projects_root, project_id)
    if not p.exists():
        return False
    real_root = projects_root.resolve()
    real_p = p.resolve()
    if not str(real_p).startswith(str(real_root)):
        raise ProjectValidationError(f"Refusing to delete outside root: {p}")
    p.unlink()
    return True
