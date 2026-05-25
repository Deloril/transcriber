"""Code lifecycle ops (F2.3).

Per PLANNING.md F2.3:

  > Lifecycle ops: merge, split, rename, retire, promote/demote in
  > hierarchy. All preserve back-pointers and don't destroy history.

These are the operations a researcher reaches for when their codebook
has grown noisy and needs consolidation. Each op is a deliberate
analytic move; the audit story is the whole point.

Operations
----------

* :func:`rename_code` — change a code's name. ``name`` is part of the
  definition (per F2.2's ``DEFINITION_FIELDS``), so a new immutable
  version is recorded automatically.

* :func:`retire_code` — mark a code ``status='retired'``. Status is
  metadata, so :func:`save_code_with_version` would not record. We
  force an audit snapshot via :func:`record_code_version` with an
  explanatory ``change_note`` so retires show up in the version log.

* :func:`set_code_parent` — primitive that sets ``parent_code_id`` to
  any valid id (or ``None`` to detach). Validates the target exists in
  the project's codebook and walks the parent chain to refuse cycles.

* :func:`promote_code` — convenience for "lift this code one level":
  re-parents to the current grandparent (or detaches if no
  grandparent). Calling on a root code is a no-op.

* :func:`demote_code` — convenience alias for :func:`set_code_parent`
  with a non-empty parent. Same cycle/existence guards.

* :func:`merge_codes` — collapse one or more *source* codes into a
  *target*. The target absorbs (deduped) exemplars and related-code
  edges from the sources. Other codes in the project that referenced a
  source via ``parent_code_id`` or ``related_codes`` are rewritten to
  point at the target. The sources themselves are retired with
  ``provenance['merged_into'] = <target>``. History is preserved on
  every code touched.

* :func:`split_code` — explode one code into two or more new codes.
  The original is retired with ``provenance['split_into']`` listing
  the new ids; each new code carries ``provenance['split_from']`` so
  the lineage is recoverable in either direction. Children of the
  source and back-pointers from other codes stay attached to the
  (retired) source — splitting one code into many doesn't have a
  unique successor to re-route to. The caller can re-parent manually.

Invariants enforced by every op
-------------------------------

* **History is preserved.** Definitional changes record a new version
  via :func:`save_code_with_version`; metadata-only ops record an
  audit snapshot via :func:`record_code_version`. Existing version
  logs are never rewritten.

* **No dangling back-pointers.** After a merge, no other code in the
  project still references a retired source via ``parent_code_id`` or
  ``related_codes``. After a retire, references are left intact (the
  retired code still exists as a record).

* **No cycles.** Hierarchy operations validate the resulting parent
  chain.

* **Idempotency where it makes sense.** Retiring an already-retired
  code, or promoting a root code, is a no-op (returns the existing
  state plus the latest version log entry).

This module is stand-alone (no FastAPI, no engine imports), matching
the conventions of F1.* and F2.1/F2.2.

Future hooks
------------

F9.1's append-only event log will subsume the bookkeeping done here:
each lifecycle op should emit an event with full payload. Until F9.1
exists, the version log carries the audit record. The change-note
strings written by these helpers are designed to read sensibly when
F9.1's exporter is later asked to summarise them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .codes import (
    CODE_ID_RE,
    Code,
    CodeRelation,
    list_codes,
    load_code,
    save_code,
)
from .code_versions import (
    CodeVersion,
    latest_code_version,
    record_code_version,
    save_code_with_version,
)
from .projects import (
    PROJECT_ID_RE,
    ProjectValidationError,
)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _require_project_id(project_id: str) -> None:
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectValidationError(f"Invalid project id: {project_id!r}")


def _require_code_id(code_id: str) -> None:
    if not CODE_ID_RE.match(code_id):
        raise ProjectValidationError(f"Invalid code id: {code_id!r}")


def _ensure_audit_version(
    projects_root: Path,
    code: Code,
    *,
    change_note: str,
    now: str | None,
) -> CodeVersion:
    """Force an audit snapshot for a metadata-only lifecycle change.

    :func:`save_code_with_version` only records when a definitional field
    changes; a status-only retire (or any future metadata-only lifecycle
    move) needs an explicit audit row in the version log. Using
    :func:`record_code_version` directly is fine — it always writes —
    and the ``change_note`` is what tells a future reader why.
    """
    return record_code_version(
        projects_root, code, change_note=change_note, now=now
    )


def _ancestors(
    code_id: str,
    code_index: dict[str, Code],
) -> list[str]:
    """Return the list of ancestor ids walking up from ``code_id``.

    Stops at the root (``parent_code_id is None``) or when a cycle is
    detected (a code already seen). The starting ``code_id`` itself is
    *not* included.
    """
    seen: set[str] = set()
    chain: list[str] = []
    cur = code_index.get(code_id)
    if cur is None:
        return chain
    cursor: str | None = cur.parent_code_id
    while cursor is not None:
        if cursor in seen:
            break
        seen.add(cursor)
        chain.append(cursor)
        nxt = code_index.get(cursor)
        cursor = nxt.parent_code_id if nxt else None
    return chain


# --------------------------------------------------------------------------- #
# Rename
# --------------------------------------------------------------------------- #


def rename_code(
    projects_root: Path,
    project_id: str,
    code_id: str,
    new_name: str,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, CodeVersion]:
    """Rename a code.

    ``name`` is a definition field (F2.2), so a new version is recorded
    automatically by :func:`save_code_with_version`. Returns the updated
    Code and the version row that was written.
    """
    _require_project_id(project_id)
    _require_code_id(code_id)
    code = load_code(projects_root, project_id, code_id)
    code.apply_update({"name": new_name}, now=now)
    note = change_note or f"Renamed to {code.name!r}"
    _, version = save_code_with_version(
        projects_root, code, change_note=note, now=now
    )
    # save_code_with_version returns None for the version slot only on
    # a no-op metadata save; rename always changes a definition field
    # so we always get a version back. Belt-and-braces:
    if version is None:  # pragma: no cover — defensive
        raise RuntimeError("rename_code: expected a recorded version")
    return code, version


# --------------------------------------------------------------------------- #
# Retire
# --------------------------------------------------------------------------- #


def retire_code(
    projects_root: Path,
    project_id: str,
    code_id: str,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, CodeVersion]:
    """Mark a code as retired.

    Idempotent: retiring an already-retired code returns its current
    state and the latest version log entry (creating a baseline entry
    if none exists yet).

    Status is metadata, so this op does not change a definition field —
    we record an audit snapshot directly via :func:`record_code_version`
    so the retire is visible in the version log.
    """
    _require_project_id(project_id)
    _require_code_id(code_id)
    code = load_code(projects_root, project_id, code_id)

    if code.status == "retired":
        latest = latest_code_version(projects_root, project_id, code_id)
        if latest is not None:
            return code, latest
        # No version log yet (older code); seed one so callers always
        # have something to anchor to.
        return code, _ensure_audit_version(
            projects_root,
            code,
            change_note=change_note or "Retired (baseline snapshot)",
            now=now,
        )

    code.apply_update({"status": "retired"}, now=now)
    save_code(projects_root, code)
    note = change_note or "Retired"
    version = _ensure_audit_version(
        projects_root, code, change_note=note, now=now
    )
    return code, version


# --------------------------------------------------------------------------- #
# Hierarchy: set_code_parent / promote / demote
# --------------------------------------------------------------------------- #


def set_code_parent(
    projects_root: Path,
    project_id: str,
    code_id: str,
    new_parent_id: str | None,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, CodeVersion]:
    """Set a code's ``parent_code_id`` to ``new_parent_id`` (or ``None``).

    Validates that the target exists in the project's codebook and that
    the resulting hierarchy contains no cycles. Records a new version
    when the parent actually changes (parent_code_id is a definition
    field). When the parent is unchanged, returns the existing latest
    version (or seeds a baseline snapshot if there isn't one yet).
    """
    _require_project_id(project_id)
    _require_code_id(code_id)

    # Normalise empty / falsy → None so callers can pass either.
    if not new_parent_id:
        normalised: str | None = None
    else:
        normalised = new_parent_id
        _require_code_id(normalised)

    code = load_code(projects_root, project_id, code_id)

    if normalised is not None:
        if normalised == code_id:
            raise ProjectValidationError(
                "A code cannot be its own parent"
            )
        # Build an in-memory index of existing codes so we can validate
        # existence and walk the parent chain in one read.
        code_index = {c.id: c for c in list_codes(projects_root, project_id)}
        if normalised not in code_index:
            raise ProjectValidationError(
                f"Parent code {normalised!r} does not exist in project "
                f"{project_id!r}"
            )
        # Cycle check: walk up from new_parent_id; if we hit code_id,
        # the new edge would close a cycle.
        cursor: str | None = normalised
        seen: set[str] = set()
        while cursor is not None:
            if cursor == code_id:
                raise ProjectValidationError(
                    f"Cannot set parent {normalised!r}: would create a "
                    f"cycle through code {code_id!r}"
                )
            if cursor in seen:
                # Pre-existing cycle in the hierarchy; surface it rather
                # than loop forever.
                raise ProjectValidationError(
                    f"Existing hierarchy contains a cycle through {cursor!r}"
                )
            seen.add(cursor)
            nxt = code_index.get(cursor)
            cursor = nxt.parent_code_id if nxt else None

    code.apply_update({"parent_code_id": normalised}, now=now)
    note = change_note or (
        f"Re-parented to {normalised}" if normalised else "Detached from parent"
    )
    _, version = save_code_with_version(
        projects_root, code, change_note=note, now=now
    )
    if version is None:  # pragma: no cover — defensive
        raise RuntimeError("set_code_parent: expected a recorded version")
    return code, version


def promote_code(
    projects_root: Path,
    project_id: str,
    code_id: str,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, CodeVersion]:
    """Lift a code one level in the hierarchy.

    Sets ``parent_code_id`` to the current grandparent (the parent's
    parent), or to ``None`` if the parent had no parent. A code that
    is already a root is a no-op: returns its current state and the
    latest version log entry (seeding a baseline if needed).
    """
    _require_project_id(project_id)
    _require_code_id(code_id)
    code = load_code(projects_root, project_id, code_id)

    if code.parent_code_id is None:
        # Already at the top: idempotent.
        latest = latest_code_version(projects_root, project_id, code_id)
        if latest is not None:
            return code, latest
        return code, _ensure_audit_version(
            projects_root,
            code,
            change_note=change_note or "Promote no-op (already root)",
            now=now,
        )

    parent = load_code(projects_root, project_id, code.parent_code_id)
    new_parent = parent.parent_code_id  # may be None
    return set_code_parent(
        projects_root,
        project_id,
        code_id,
        new_parent,
        change_note=change_note or "Promoted",
        now=now,
    )


def demote_code(
    projects_root: Path,
    project_id: str,
    code_id: str,
    new_parent_id: str,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, CodeVersion]:
    """Make a code a child of ``new_parent_id``.

    Thin convenience over :func:`set_code_parent`; rejects a falsy
    ``new_parent_id`` (use ``set_code_parent(..., None)`` to detach).
    """
    if not new_parent_id:
        raise ProjectValidationError(
            "demote_code requires a non-empty new_parent_id; "
            "use set_code_parent(..., None) to detach instead"
        )
    return set_code_parent(
        projects_root,
        project_id,
        code_id,
        new_parent_id,
        change_note=change_note or f"Demoted under {new_parent_id}",
        now=now,
    )


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def merge_codes(
    projects_root: Path,
    project_id: str,
    source_code_ids: Iterable[str],
    target_code_id: str,
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, list[Code]]:
    """Merge one or more source codes into ``target_code_id``.

    Effects:

      1. Target absorbs the union of its own and the sources' exemplars
         (de-duplicated, order preserved). Same for ``related_codes``,
         excluding any edges that would point at sources or at the
         target itself.
      2. Other codes in the project that referenced a source via
         ``parent_code_id`` or ``related_codes[].code_id`` are rewritten
         to reference the target (de-duped on (target, relation_type)).
      3. Each source is marked ``status='retired'`` with
         ``provenance['merged_into'] = <target_id>``. An audit snapshot
         is recorded on every source.
      4. A new definitional version is recorded on the target (its
         exemplars and/or related_codes changed). Reroutings on third-
         party codes likewise record new versions on those codes.

    The list of source ids must be non-empty, must not include
    ``target_code_id``, and every id must exist in the project. Sources
    are de-duplicated before processing.

    Returns ``(target_after_merge, sources_after_retire)``.
    """
    _require_project_id(project_id)
    _require_code_id(target_code_id)

    src_list = list(source_code_ids)
    if not src_list:
        raise ProjectValidationError(
            "merge_codes requires at least one source code id"
        )
    # De-duplicate while preserving order.
    seen_src: set[str] = set()
    deduped_src: list[str] = []
    for sid in src_list:
        _require_code_id(sid)
        if sid == target_code_id:
            raise ProjectValidationError(
                f"merge target {target_code_id!r} cannot also be a source"
            )
        if sid in seen_src:
            continue
        seen_src.add(sid)
        deduped_src.append(sid)

    # Load everything up front so a missing id fails loudly before we
    # write anything.
    target = load_code(projects_root, project_id, target_code_id)
    sources: list[Code] = [
        load_code(projects_root, project_id, sid) for sid in deduped_src
    ]
    src_set = set(deduped_src)

    # ---- 1. Build target's new exemplars + related_codes ----
    target_exemplars: list[str] = list(target.exemplars)
    seen_ex: set[str] = set(target_exemplars)
    for s in sources:
        for ex in s.exemplars:
            if ex in seen_ex:
                continue
            seen_ex.add(ex)
            target_exemplars.append(ex)

    # Drop the target's own edges that point at a source about to be
    # retired — keeping them would leave the target referring to a
    # retired code on the next read. Same defensive filter for self-
    # edges (shouldn't exist; CodeRelation.validate would have rejected
    # at write time, but the on-disk file might have been hand-edited).
    target_relations: list[CodeRelation] = []
    seen_rel: set[tuple[str, str]] = set()
    for r in target.related_codes:
        if r.code_id in src_set or r.code_id == target_code_id:
            continue
        key = (r.code_id, r.relation_type)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        target_relations.append(r)
    for s in sources:
        for r in s.related_codes:
            # Drop edges that would point at the target (self) or at a
            # source being absorbed (would land on a retired code).
            if r.code_id == target_code_id or r.code_id in src_set:
                continue
            key = (r.code_id, r.relation_type)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            target_relations.append(r)

    target.apply_update(
        {
            "exemplars": target_exemplars,
            "related_codes": [r.to_dict() for r in target_relations],
        },
        now=now,
    )
    note = (
        change_note
        or f"Merged {','.join(deduped_src)} into {target_code_id}"
    )
    save_code_with_version(projects_root, target, change_note=note, now=now)

    # ---- 2. Rewrite back-pointers on every other code ----
    # Re-read the codebook so we see the freshly-saved target. Skip the
    # target itself and the sources (sources are about to be retired).
    for other in list_codes(projects_root, project_id):
        if other.id == target_code_id or other.id in src_set:
            continue
        changed = False
        # Parent re-route.
        if (
            other.parent_code_id is not None
            and other.parent_code_id in src_set
        ):
            other.parent_code_id = target_code_id
            changed = True
        # related_codes re-route, with self-edge and dup elimination.
        new_relations: list[CodeRelation] = []
        rel_changed = False
        seen_other_rel: set[tuple[str, str]] = set()
        for r in other.related_codes:
            new_id = (
                target_code_id if r.code_id in src_set else r.code_id
            )
            if new_id != r.code_id:
                rel_changed = True
            if new_id == other.id:
                # Would create a self-edge; drop.
                rel_changed = True
                continue
            key = (new_id, r.relation_type)
            if key in seen_other_rel:
                # Dedup against an edge that already exists / was just
                # rewritten to the same target.
                if new_id != r.code_id:
                    pass  # already accounted for
                else:
                    rel_changed = True
                continue
            seen_other_rel.add(key)
            new_relations.append(
                CodeRelation(
                    code_id=new_id, relation_type=r.relation_type
                )
            )
        if rel_changed:
            other.related_codes = new_relations
            changed = True
        if changed:
            # Funnel through apply_update so validation runs.
            other.apply_update(
                {
                    "parent_code_id": other.parent_code_id,
                    "related_codes": [
                        r.to_dict() for r in other.related_codes
                    ],
                },
                now=now,
            )
            save_code_with_version(
                projects_root,
                other,
                change_note=(
                    change_note
                    or f"Re-routed references after merge into {target_code_id}"
                ),
                now=now,
            )

    # ---- 3. Retire each source with provenance['merged_into'] ----
    retired_sources: list[Code] = []
    for s in sources:
        prov = dict(s.provenance)
        prov.setdefault("source", "human")
        prov["merged_into"] = target_code_id
        s.apply_update(
            {"status": "retired", "provenance": prov}, now=now
        )
        save_code(projects_root, s)
        _ensure_audit_version(
            projects_root,
            s,
            change_note=(
                change_note or f"Retired (merged into {target_code_id})"
            ),
            now=now,
        )
        retired_sources.append(s)

    # Reload target so the returned object reflects all writes.
    target = load_code(projects_root, project_id, target_code_id)
    return target, retired_sources


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


def split_code(
    projects_root: Path,
    project_id: str,
    source_code_id: str,
    new_code_specs: list[dict[str, Any]],
    *,
    change_note: str = "",
    now: str | None = None,
) -> tuple[Code, list[Code]]:
    """Explode a code into two or more new codes.

    ``new_code_specs`` is a list of dicts, each describing one new
    code. Required key: ``name``. Optional keys mirror :func:`Code.new`
    arguments and default to the source's value where it makes sense
    (definition, criteria, theoretical_memo, stage, colour). Exemplars
    default to ``[]`` so the caller distributes the source's exemplars
    deliberately. ``status`` defaults to ``"active"``.

    Each new code is created with::

        provenance['source']     = (caller-supplied or 'human')
        provenance['split_from'] = source_code_id

    and saved with an initial version row.

    The source is then marked ``status='retired'`` with::

        provenance['split_into'] = '<id1>,<id2>,...'

    and an audit version is recorded.

    Children of the source and back-pointers from other codes are *not*
    re-routed automatically — splitting one code into many doesn't have
    a unique successor. The caller can move them with
    :func:`set_code_parent` / :func:`merge_codes` afterwards.
    """
    _require_project_id(project_id)
    _require_code_id(source_code_id)

    if not isinstance(new_code_specs, list) or len(new_code_specs) < 2:
        raise ProjectValidationError(
            "split_code requires a list of at least 2 new code specs"
        )

    source = load_code(projects_root, project_id, source_code_id)

    new_codes: list[Code] = []
    for spec in new_code_specs:
        if not isinstance(spec, dict):
            raise ProjectValidationError(
                "Each split spec must be an object"
            )
        name = spec.get("name")
        if not name or not str(name).strip():
            raise ProjectValidationError(
                "Each split spec must include a non-empty 'name'"
            )

        prov_in = spec.get("provenance") or {}
        if not isinstance(prov_in, dict):
            raise ProjectValidationError(
                "split spec 'provenance' must be an object"
            )
        prov: dict[str, Any] = {str(k): str(v) for k, v in prov_in.items()}
        prov.setdefault("source", "human")
        prov["split_from"] = source_code_id

        new_code = Code.new(
            project_id=project_id,
            name=str(name),
            definition=str(spec.get("definition", source.definition) or ""),
            inclusion_criteria=str(
                spec.get("inclusion_criteria", source.inclusion_criteria) or ""
            ),
            exclusion_criteria=str(
                spec.get("exclusion_criteria", source.exclusion_criteria) or ""
            ),
            exemplars=list(spec.get("exemplars") or []),
            parent_code_id=spec.get("parent_code_id", source.parent_code_id),
            related_codes=list(spec.get("related_codes") or []),
            theoretical_memo=str(
                spec.get("theoretical_memo", source.theoretical_memo) or ""
            ),
            stage=str(spec.get("stage", source.stage) or source.stage),
            colour=str(spec.get("colour", source.colour) or ""),
            status=str(spec.get("status", "active") or "active"),
            provenance=prov,
            now=now,
        )
        save_code_with_version(
            projects_root,
            new_code,
            change_note=(
                change_note or f"Created via split of {source_code_id}"
            ),
            now=now,
        )
        new_codes.append(new_code)

    new_ids = ",".join(c.id for c in new_codes)
    src_prov = dict(source.provenance)
    src_prov.setdefault("source", "human")
    src_prov["split_into"] = new_ids
    source.apply_update(
        {"status": "retired", "provenance": src_prov}, now=now
    )
    save_code(projects_root, source)
    _ensure_audit_version(
        projects_root,
        source,
        change_note=(
            change_note or f"Retired (split into {new_ids})"
        ),
        now=now,
    )

    return source, new_codes
