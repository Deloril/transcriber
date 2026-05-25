"""Project file format on disk (F1.5).

Per PLANNING.md F1.5:

  > Project file format on disk: JSON manifests + the existing
  > ``outputs/<job>/`` artefacts. Designed to round-trip through
  > REFI-QDA.

F1.1–F1.4 organically built up a ``projects/<id>/`` tree:

  projects/<project_id>/
    project.json           # F1.1
    sources/<sid>.json     # F1.2
    participants/<pid>.json  # F1.3
    sampling_log.jsonl     # F1.4

This module formalises that layout into a **versioned, round-trip-able
project format**. It adds:

  * ``manifest.json`` at the project root — a small, stable index that
    names the format, its version, the components present, and any
    external job assets (``outputs/<job_id>/`` directories) the
    project's sources reference. Future readers can rely on this file
    instead of probing for sibling files; future writers can extend
    the schema by bumping ``format_version``.

  * A ``ProjectBundle`` aggregate that gathers project + sources +
    participants + sampling log into one in-memory value, with
    ``load`` and ``save`` round-trips.

  * Archive (zip) ``export_project_archive`` / ``import_project_archive``
    helpers, optionally including the ``outputs/<job_id>/`` artefacts
    referenced by the project's sources. The internal layout matches
    the on-disk tree exactly, so the archive format is the same as
    the on-disk format (a directory rooted at ``<project_id>/``).
    REFI-QDA's QDPX is also a zip; F6.4 will add a sibling exporter
    that produces a REFI-QDA-shaped manifest pointing at the same
    component trees.

The core principle is: **this layer doesn't add new data, it just
gives the existing data a manifest, an in-memory aggregate, and a
portable archive form.** Callers can keep using the F1.1–F1.4 helpers
directly for incremental edits; ``ProjectBundle`` is for whole-project
operations (export, import, clone, snapshot — F9.4 territory).

Like the rest of the F1.* modules this is stand-alone — no FastAPI,
no engine imports — so it's testable in pure Python and reusable
from the CLI later.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .projects import (
    PROJECT_ID_RE,
    Project,
    ProjectValidationError,
    project_dir,
    project_state_path,
    save_project,
    utcnow_iso,
)
from .sources import (
    JOB_ID_RE,
    SOURCE_ID_RE,
    Source,
    list_sources,
    save_source,
    sources_dir,
)
from .participants import (
    PARTICIPANT_ID_RE,
    Participant,
    list_participants,
    participants_dir,
    save_participant,
)
from .sampling_log import (
    SamplingEntry,
    read_sampling_log,
    sampling_log_path,
)
from .codes import (
    CODE_ID_RE,
    Code,
    codes_dir,
    list_codes,
    save_code,
)
from .source_schema import (
    SCHEMA_FILENAME,
    SourceAttributeSchema,
    load_source_schema,
    save_source_schema,
)


# --------------------------------------------------------------------------- #
# Format identifiers
# --------------------------------------------------------------------------- #

# The on-disk format string. Stable forever (don't rename).
FORMAT_NAME = "scribe-project"

# Schema version of ``manifest.json``. Bump on any breaking change to
# the manifest layout or to the on-disk component structure. The
# default reader rejects versions it doesn't understand so an old
# Scribe build can't silently misread a newer project.
FORMAT_VERSION = 1

# Manifest filename. Lives at the project root alongside ``project.json``.
MANIFEST_FILENAME = "manifest.json"


# --------------------------------------------------------------------------- #
# Component layout — the canonical names of the F1.1–F1.4 sub-trees.
# Keeping these in one place means readers/writers don't drift.
# --------------------------------------------------------------------------- #

COMPONENT_PROJECT = "project"
COMPONENT_SOURCES_DIR = "sources_dir"
COMPONENT_PARTICIPANTS_DIR = "participants_dir"
COMPONENT_SAMPLING_LOG = "sampling_log"
# F3.1: codebook directory ships under the project root alongside
# sources/ and participants/. Each code is one JSON file (see F2.1).
COMPONENT_CODES_DIR = "codes_dir"
# F3.2: project-level source attribute schema. Single JSON file at
# the project root; declares the user-defined columns that source
# ``custom_attributes`` use.
COMPONENT_SOURCE_SCHEMA = "source_schema"

# Default values for the "components" dict in the manifest. These are
# relative paths inside the project directory.
DEFAULT_COMPONENT_PATHS: dict[str, str] = {
    COMPONENT_PROJECT: "project.json",
    COMPONENT_SOURCES_DIR: "sources",
    COMPONENT_PARTICIPANTS_DIR: "participants",
    COMPONENT_SAMPLING_LOG: "sampling_log.jsonl",
    COMPONENT_CODES_DIR: "codes",
    COMPONENT_SOURCE_SCHEMA: SCHEMA_FILENAME,
}


# Kinds of external assets we know how to track. Today only transcripts
# (``outputs/<job_id>/`` produced by the existing ASR pipeline). The
# manifest's ``external_assets`` list is forward-compatible: future
# kinds (annotated PDFs, codebook XML produced by another tool) can be
# added without bumping the format version, because unknown kinds are
# simply ignored by the reader.
ASSET_KIND_TRANSCRIPT = "transcript"

# Maximum size of any single file we'll extract from an imported
# archive — defends against zip-bomb attacks. Real projects are tiny;
# a single 1 GB file inside the manifest is never legitimate.
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# Maximum total uncompressed size of an archive. Same rationale.
MAX_ARCHIVE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB


class ProjectFormatError(ProjectValidationError):
    """Raised on manifest / bundle / archive format errors.

    Subclasses ``ProjectValidationError`` so callers that already
    handle the F1.1 validation type catch this too.
    """


# --------------------------------------------------------------------------- #
# Manifest data model
# --------------------------------------------------------------------------- #


@dataclass
class ProjectManifest:
    """The ``manifest.json`` file at a project root.

    Read it to discover the format version and what components are
    present without scanning the directory tree. Write it whenever
    the project layout changes (new source, new participant, new
    sampling entry) so external readers see a current view.

    ``external_assets`` lists job-id references the project's sources
    point at (transcripts in ``outputs/<job_id>/``). It's an index:
    callers building an archive use it to decide what extra files to
    pull in. Stale entries (jobs that have been deleted from
    ``outputs/``) are tolerated — callers handle missing files.
    """

    project_id: str
    name: str
    created_at: str
    modified_at: str
    format: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    components: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COMPONENT_PATHS))
    external_assets: list[dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_project(
        cls,
        project: Project,
        *,
        external_assets: Iterable[dict[str, str]] | None = None,
    ) -> "ProjectManifest":
        """Build a manifest from a Project + an asset index."""
        m = cls(
            project_id=project.id,
            name=project.name,
            created_at=project.created_at,
            modified_at=project.modified_at,
            external_assets=[dict(a) for a in (external_assets or [])],
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectManifest":
        if not isinstance(d, dict):
            raise ProjectFormatError("Manifest payload must be an object")
        for key in ("project_id", "name", "created_at", "modified_at"):
            if key not in d:
                raise ProjectFormatError(f"Manifest missing required key: {key}")
        m = cls(
            project_id=str(d["project_id"]),
            name=str(d["name"]),
            created_at=str(d["created_at"]),
            modified_at=str(d["modified_at"]),
            format=str(d.get("format", FORMAT_NAME) or FORMAT_NAME),
            format_version=int(d.get("format_version", FORMAT_VERSION)),
            components={
                str(k): str(v)
                for k, v in (d.get("components") or {}).items()
            },
            external_assets=[
                {str(k): str(v) for k, v in entry.items()}
                for entry in (d.get("external_assets") or [])
                if isinstance(entry, dict)
            ],
        )
        m.validate()
        return m

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if self.format != FORMAT_NAME:
            raise ProjectFormatError(
                f"Unknown manifest format: {self.format!r} "
                f"(expected {FORMAT_NAME!r})"
            )
        if not isinstance(self.format_version, int) or self.format_version < 1:
            raise ProjectFormatError(
                f"Invalid format_version: {self.format_version!r}"
            )
        if self.format_version > FORMAT_VERSION:
            raise ProjectFormatError(
                f"Manifest format_version {self.format_version} is newer than "
                f"this Scribe build supports ({FORMAT_VERSION}). Upgrade Scribe."
            )
        if not PROJECT_ID_RE.match(self.project_id):
            raise ProjectFormatError(
                f"Invalid project id in manifest: {self.project_id!r}"
            )
        if not self.name.strip():
            raise ProjectFormatError("Manifest name is required")
        if not self.created_at or not self.modified_at:
            raise ProjectFormatError(
                "Manifest created_at/modified_at are required"
            )

        if not isinstance(self.components, dict):
            raise ProjectFormatError("Manifest components must be an object")
        # All component paths must be strict relatives — no absolute,
        # no `..`. We treat the project directory as the root.
        for key, rel in self.components.items():
            if not isinstance(rel, str) or not rel:
                raise ProjectFormatError(
                    f"Manifest component {key!r} must be a non-empty string"
                )
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ProjectFormatError(
                    f"Manifest component {key!r} path must be relative "
                    f"and within project: {rel!r}"
                )

        if not isinstance(self.external_assets, list):
            raise ProjectFormatError(
                "Manifest external_assets must be a list of objects"
            )
        for entry in self.external_assets:
            if not isinstance(entry, dict):
                raise ProjectFormatError(
                    "Each external_assets entry must be an object"
                )
            kind = entry.get("kind")
            ref = entry.get("ref")
            if not kind or not ref:
                raise ProjectFormatError(
                    "external_assets entries need 'kind' and 'ref'"
                )
            # We only validate the shape of transcript refs; other
            # kinds are forward-compat (unknown kind = ignored by the
            # reader, but must still parse).
            if kind == ASSET_KIND_TRANSCRIPT:
                if not _looks_like_transcript_ref(ref):
                    raise ProjectFormatError(
                        f"transcript external_assets ref must be "
                        f"'outputs/<12-hex>'; got {ref!r}"
                    )


_TRANSCRIPT_REF_RE = re.compile(r"^outputs/[a-f0-9]{12}$")


def _looks_like_transcript_ref(ref: str) -> bool:
    return bool(_TRANSCRIPT_REF_RE.match(ref))


# --------------------------------------------------------------------------- #
# Manifest persistence
# --------------------------------------------------------------------------- #


def manifest_path(projects_root: Path, project_id: str) -> Path:
    """Return the on-disk path of a project's ``manifest.json``."""
    return project_dir(projects_root, project_id) / MANIFEST_FILENAME


def write_manifest(
    projects_root: Path, project_id: str
) -> ProjectManifest:
    """Recompute and write a project's manifest.

    Reads the project + sources from disk, derives the
    ``external_assets`` list from each source's ``transcript_job_id``,
    and writes the manifest atomically. Returns the manifest written.
    """
    p = project_state_path(projects_root, project_id)
    if not p.exists():
        raise FileNotFoundError(f"No project at {p}")
    project = Project.from_dict(json.loads(p.read_text()))
    sources = list_sources(projects_root, project_id)
    assets = derive_external_assets(sources)
    manifest = ProjectManifest.from_project(project, external_assets=assets)
    _write_manifest_obj(projects_root, manifest)
    return manifest


def read_manifest(
    projects_root: Path, project_id: str
) -> ProjectManifest:
    """Load and validate the on-disk manifest.

    Raises ``FileNotFoundError`` if the manifest is missing — callers
    that want to tolerate older projects (those created before F1.5
    landed) should use ``read_or_build_manifest`` instead.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectFormatError(f"Invalid project id: {project_id!r}")
    mp = manifest_path(projects_root, project_id)
    if not mp.exists():
        raise FileNotFoundError(f"No manifest at {mp}")
    try:
        payload = json.loads(mp.read_text())
    except json.JSONDecodeError as e:
        raise ProjectFormatError(f"Manifest is not valid JSON: {e}") from e
    return ProjectManifest.from_dict(payload)


def read_or_build_manifest(
    projects_root: Path, project_id: str
) -> ProjectManifest:
    """Read the manifest, or derive one from disk if missing.

    Useful as a forward/back-compat bridge for projects created before
    F1.5 introduced ``manifest.json``: the on-disk tree is enough to
    reconstruct the manifest from the project + sources.
    """
    mp = manifest_path(projects_root, project_id)
    if mp.exists():
        return read_manifest(projects_root, project_id)
    # Build from project.json + sources.
    project_path = project_state_path(projects_root, project_id)
    if not project_path.exists():
        raise FileNotFoundError(f"No project at {project_path}")
    project = Project.from_dict(json.loads(project_path.read_text()))
    sources = list_sources(projects_root, project_id)
    return ProjectManifest.from_project(
        project, external_assets=derive_external_assets(sources)
    )


def _write_manifest_obj(
    projects_root: Path, manifest: ProjectManifest
) -> Path:
    manifest.validate()
    d = project_dir(projects_root, manifest.project_id)
    d.mkdir(parents=True, exist_ok=True)
    target = manifest_path(projects_root, manifest.project_id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------- #
# External-asset index
# --------------------------------------------------------------------------- #


def derive_external_assets(sources: list[Source]) -> list[dict[str, str]]:
    """Derive the manifest's ``external_assets`` list from sources.

    De-dupes (the same job can be referenced from multiple sources in
    principle, though we don't expect it in practice) and sorts by
    ``ref`` so the manifest is stable across writes.
    """
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for s in sources:
        if not s.transcript_job_id:
            continue
        ref = f"outputs/{s.transcript_job_id}"
        if ref in seen:
            continue
        seen.add(ref)
        out.append({"kind": ASSET_KIND_TRANSCRIPT, "ref": ref})
    out.sort(key=lambda a: a["ref"])
    return out


# --------------------------------------------------------------------------- #
# ProjectBundle — in-memory aggregate
# --------------------------------------------------------------------------- #


@dataclass
class ProjectBundle:
    """A project + all its sub-entities, in memory.

    Use this for whole-project operations (export, import, clone,
    snapshot). For incremental edits (saving one source, appending
    one sampling-log entry, persisting a code) keep using the
    F1.1–F1.4 / F2.x helpers directly.

    F3.1: the bundle is the integration point for the project shell
    — sources, participants, sampling log, *codebook* (the F2.x
    codes), and (in the future) memos. Project-level settings live on
    ``project.settings`` rather than as a separate file because they
    semantically belong to the project entity.
    """

    project: Project
    sources: list[Source] = field(default_factory=list)
    participants: list[Participant] = field(default_factory=list)
    sampling_log: list[SamplingEntry] = field(default_factory=list)
    codes: list[Code] = field(default_factory=list)
    # F3.2: per-project schema declaring user-defined source columns.
    # Optional — projects without explicit columns still work.
    source_schema: SourceAttributeSchema | None = None
    manifest: ProjectManifest | None = None

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Validate cross-entity invariants.

        Each F1.1–F1.4 entity self-validates on construction; this
        layer adds the cross-entity checks that only make sense when
        you can see the whole bundle.
        """
        self.project.validate()
        pid = self.project.id

        for s in self.sources:
            s.validate()
            if s.project_id != pid:
                raise ProjectFormatError(
                    f"Source {s.id} project_id {s.project_id!r} "
                    f"does not match bundle project {pid!r}"
                )
        # Source IDs unique within the bundle.
        sids = [s.id for s in self.sources]
        if len(set(sids)) != len(sids):
            raise ProjectFormatError("Bundle has duplicate source ids")

        for p in self.participants:
            p.validate()
            if p.project_id != pid:
                raise ProjectFormatError(
                    f"Participant {p.id} project_id {p.project_id!r} "
                    f"does not match bundle project {pid!r}"
                )
        pids = [p.id for p in self.participants]
        if len(set(pids)) != len(pids):
            raise ProjectFormatError("Bundle has duplicate participant ids")

        for e in self.sampling_log:
            e.validate()
            if e.project_id != pid:
                raise ProjectFormatError(
                    f"Sampling entry {e.id} project_id {e.project_id!r} "
                    f"does not match bundle project {pid!r}"
                )
        eids = [e.id for e in self.sampling_log]
        if len(set(eids)) != len(eids):
            raise ProjectFormatError("Bundle has duplicate sampling-log entry ids")

        # F3.1: codebook (F2.1 codes) ride along inside the bundle.
        for c in self.codes:
            c.validate()
            if c.project_id != pid:
                raise ProjectFormatError(
                    f"Code {c.id} project_id {c.project_id!r} "
                    f"does not match bundle project {pid!r}"
                )
        cids = [c.id for c in self.codes]
        if len(set(cids)) != len(cids):
            raise ProjectFormatError("Bundle has duplicate code ids")

        # F3.2: optional source attribute schema rides along too.
        if self.source_schema is not None:
            self.source_schema.validate()
            if self.source_schema.project_id != pid:
                raise ProjectFormatError(
                    f"SourceAttributeSchema project_id "
                    f"{self.source_schema.project_id!r} does not match "
                    f"bundle project {pid!r}"
                )

        if self.manifest is not None:
            self.manifest.validate()
            if self.manifest.project_id != pid:
                raise ProjectFormatError(
                    f"Manifest project_id {self.manifest.project_id!r} "
                    f"does not match bundle project {pid!r}"
                )


# --------------------------------------------------------------------------- #
# Bundle persistence
# --------------------------------------------------------------------------- #


def load_project_bundle(
    projects_root: Path, project_id: str
) -> ProjectBundle:
    """Load a complete project from disk into a single bundle.

    Reads ``project.json``, the sources directory, the participants
    directory, and the sampling log. The manifest is read if present
    (or derived if not, for backwards compatibility with projects
    created before F1.5 landed).
    """
    project_path = project_state_path(projects_root, project_id)
    if not project_path.exists():
        raise FileNotFoundError(f"No project at {project_path}")
    project = Project.from_dict(json.loads(project_path.read_text()))
    sources = list_sources(projects_root, project_id)
    participants = list_participants(projects_root, project_id)
    log = read_sampling_log(projects_root, project_id)
    codes = list_codes(projects_root, project_id)
    try:
        source_schema: SourceAttributeSchema | None = load_source_schema(
            projects_root, project_id
        )
    except FileNotFoundError:
        source_schema = None
    try:
        manifest = read_or_build_manifest(projects_root, project_id)
    except FileNotFoundError:
        manifest = None
    bundle = ProjectBundle(
        project=project,
        sources=sources,
        participants=participants,
        sampling_log=log,
        codes=codes,
        source_schema=source_schema,
        manifest=manifest,
    )
    bundle.validate()
    return bundle


def save_project_bundle(
    projects_root: Path,
    bundle: ProjectBundle,
    *,
    replace_sampling_log: bool = False,
    write_manifest_file: bool = True,
) -> ProjectManifest:
    """Persist a bundle to disk.

    Writes ``project.json``, every source file, every participant
    file, and the manifest. The sampling log is **append-only by
    default**: if the on-disk log already has entries the bundle
    doesn't, we leave it alone (so callers don't accidentally erase
    audit history). Pass ``replace_sampling_log=True`` to atomically
    rewrite the log to match the bundle exactly — that's the import
    path, not the typical edit path.

    Returns the manifest written (the in-memory bundle's manifest is
    refreshed too).
    """
    bundle.validate()
    pid = bundle.project.id

    # 1. Project.
    save_project(projects_root, bundle.project)

    # 2. Sources.
    for s in bundle.sources:
        save_source(projects_root, s)

    # 3. Participants.
    for p in bundle.participants:
        save_participant(projects_root, p)

    # 3b. Codebook (F3.1). Persist every code in the bundle. Existing
    # codes on disk that aren't in the bundle are *not* deleted —
    # mirrors the sampling-log "append-only by default" stance, so an
    # incomplete bundle can't accidentally erase the codebook.
    for c in bundle.codes:
        save_code(projects_root, c)

    # 3c. Source attribute schema (F3.2). Optional; written only when
    # the bundle carries one. Like codes, *not* deleted when omitted —
    # callers that want to drop the schema explicitly use
    # ``delete_source_schema``.
    if bundle.source_schema is not None:
        save_source_schema(projects_root, bundle.source_schema)

    # 4. Sampling log.
    if replace_sampling_log:
        _replace_sampling_log_file(projects_root, pid, bundle.sampling_log)
    else:
        # Append-only edit-path: if the log on disk is empty/missing,
        # write the bundle's log; otherwise leave it alone (caller
        # should be using append_sampling_entry for incremental edits).
        log_path = sampling_log_path(projects_root, pid)
        if not log_path.exists() or log_path.stat().st_size == 0:
            _replace_sampling_log_file(projects_root, pid, bundle.sampling_log)

    # 5. Manifest.
    manifest = ProjectManifest.from_project(
        bundle.project,
        external_assets=derive_external_assets(bundle.sources),
    )
    if write_manifest_file:
        _write_manifest_obj(projects_root, manifest)
    bundle.manifest = manifest
    return manifest


def _replace_sampling_log_file(
    projects_root: Path,
    project_id: str,
    entries: list[SamplingEntry],
) -> Path:
    """Atomically rewrite the sampling log to match `entries`.

    Used by the import path (full project restore) and by
    ``save_project_bundle`` when ``replace_sampling_log=True``. Normal
    edit traffic should use ``append_sampling_entry`` instead.
    """
    target = sampling_log_path(projects_root, project_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".jsonl.tmp")
    lines = []
    for e in entries:
        e.validate()
        lines.append(json.dumps(e.to_dict(), ensure_ascii=False))
    tmp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------- #
# Archive (zip) export / import
# --------------------------------------------------------------------------- #

# Conventional file extension used by the export. ``.zip`` is also fine
# — readers don't care about the suffix, only the contents — but
# ``.scribe`` (or ``.scribe.zip``) hints at the format in file managers.
ARCHIVE_SUFFIX = ".scribe.zip"


def export_project_archive(
    projects_root: Path,
    project_id: str,
    archive_path: Path,
    *,
    outputs_root: Path | None = None,
    include_outputs: bool = False,
) -> Path:
    """Bundle a project into a single zip file.

    The archive's internal layout matches the on-disk layout, rooted at
    a top-level ``<project_id>/`` directory:

      <project_id>/manifest.json
      <project_id>/project.json
      <project_id>/sources/<sid>.json
      <project_id>/participants/<pid>.json
      <project_id>/sampling_log.jsonl
      <project_id>/outputs/<job_id>/...   (only if include_outputs=True)

    Set ``include_outputs=True`` and pass ``outputs_root`` to bundle in
    every transcript directory referenced by a source's
    ``transcript_job_id``. Missing directories are skipped (with no
    error) so a partial corpus still exports.
    """
    if not PROJECT_ID_RE.match(project_id):
        raise ProjectFormatError(f"Invalid project id: {project_id!r}")
    pdir = project_dir(projects_root, project_id)
    if not pdir.exists():
        raise FileNotFoundError(f"No project directory: {pdir}")

    # Make sure the manifest is current before we snapshot the tree.
    write_manifest(projects_root, project_id)

    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        # Walk the project tree and add every regular file.
        for f in _iter_project_files(pdir):
            arc = f"{project_id}/" + str(f.relative_to(pdir)).replace("\\", "/")
            zf.write(f, arcname=arc)

        if include_outputs:
            if outputs_root is None:
                raise ProjectFormatError(
                    "include_outputs=True requires an outputs_root path"
                )
            sources = list_sources(projects_root, project_id)
            for s in sources:
                if not s.transcript_job_id:
                    continue
                job_dir = outputs_root / s.transcript_job_id
                if not job_dir.exists():
                    continue
                for f in _iter_project_files(job_dir):
                    rel = str(f.relative_to(outputs_root)).replace("\\", "/")
                    arc = f"{project_id}/outputs/{rel}"
                    zf.write(f, arcname=arc)

    return archive_path


def _iter_project_files(root: Path):
    """Yield every regular file under ``root`` (sorted, depth-first).

    Sorted so archives are deterministic across runs (helps git diff
    of archives, and reproducible builds).
    """
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            yield p


def import_project_archive(
    projects_root: Path,
    archive_path: Path,
    *,
    outputs_root: Path | None = None,
    overwrite: bool = False,
) -> ProjectBundle:
    """Restore a project from a zip produced by ``export_project_archive``.

    Validates the archive's manifest, refuses to overwrite an existing
    project unless ``overwrite=True``, and (if the archive contains
    ``outputs/<job_id>/`` trees) extracts them into ``outputs_root``
    when one is provided. Returns the restored bundle.

    Path-traversal safe: every member is resolved relative to a
    fixed staging directory and rejected if it escapes.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"No archive at {archive_path}")

    with zipfile.ZipFile(archive_path, mode="r") as zf:
        members = zf.infolist()
        if not members:
            raise ProjectFormatError("Archive is empty")

        # Defend against zip bombs.
        total = sum(m.file_size for m in members)
        if total > MAX_ARCHIVE_TOTAL_BYTES:
            raise ProjectFormatError(
                f"Archive uncompressed size {total} exceeds limit"
            )
        for m in members:
            if m.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ProjectFormatError(
                    f"Archive member {m.filename!r} too large"
                )

        # Discover the single top-level project_id directory in the
        # archive. We accept exactly one — refusing multi-project
        # archives keeps the API tight.
        top_levels: set[str] = set()
        for m in members:
            head = m.filename.split("/", 1)[0]
            if head:
                top_levels.add(head)
        if len(top_levels) != 1:
            raise ProjectFormatError(
                f"Archive must contain exactly one top-level project "
                f"directory; found {sorted(top_levels)!r}"
            )
        project_id = next(iter(top_levels))
        if not PROJECT_ID_RE.match(project_id):
            raise ProjectFormatError(
                f"Archive top-level directory is not a valid project id: "
                f"{project_id!r}"
            )

        # Read the manifest from the archive first, validate it before
        # writing anything to disk. This is the gatekeeper.
        manifest_member = f"{project_id}/{MANIFEST_FILENAME}"
        try:
            manifest_bytes = zf.read(manifest_member)
        except KeyError as e:
            raise ProjectFormatError(
                f"Archive missing {manifest_member}"
            ) from e
        try:
            manifest = ProjectManifest.from_dict(json.loads(manifest_bytes))
        except json.JSONDecodeError as e:
            raise ProjectFormatError(
                f"Archive manifest is not valid JSON: {e}"
            ) from e
        if manifest.project_id != project_id:
            raise ProjectFormatError(
                f"Archive manifest project_id {manifest.project_id!r} "
                f"doesn't match top-level directory {project_id!r}"
            )

        # Refuse to overwrite an existing project unless asked to.
        target_project_dir = projects_root / project_id
        if target_project_dir.exists() and not overwrite:
            raise ProjectFormatError(
                f"Project {project_id} already exists; pass overwrite=True"
            )

        # Stage extraction into a temp directory next to the target.
        # Atomic-ish: write to staging, then swap into place.
        projects_root.mkdir(parents=True, exist_ok=True)
        staging = projects_root / f".{project_id}.import.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()

        try:
            real_staging = staging.resolve()
            real_outputs = outputs_root.resolve() if outputs_root else None
            for m in members:
                if m.is_dir():
                    continue
                # Extract project files; treat outputs/ specially.
                rel = m.filename
                if not rel.startswith(project_id + "/"):
                    raise ProjectFormatError(
                        f"Archive member outside project root: {rel!r}"
                    )
                inner = rel[len(project_id) + 1 :]
                if inner.startswith("outputs/"):
                    if outputs_root is None:
                        # Ignore outputs/ if caller didn't ask to
                        # restore them. The manifest still references
                        # them by job-id, just like a project that was
                        # exported without include_outputs.
                        continue
                    job_rel = inner[len("outputs/") :]
                    target = outputs_root / job_rel
                    real_target = (real_outputs / job_rel).resolve() if real_outputs else None
                    if real_outputs is not None and not _is_within(
                        real_target, real_outputs
                    ):
                        raise ProjectFormatError(
                            f"Archive member escapes outputs root: {rel!r}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    target = staging / inner
                    real_target = target.resolve()
                    if not _is_within(real_target, real_staging):
                        raise ProjectFormatError(
                            f"Archive member escapes project root: {rel!r}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

            # Swap staging into place.
            if target_project_dir.exists():
                shutil.rmtree(target_project_dir)
            staging.rename(target_project_dir)
        except Exception:
            # Best-effort cleanup of staging on any failure.
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    # Done — load the freshly written project as a bundle so callers
    # have the in-memory aggregate ready to use.
    return load_project_bundle(projects_root, project_id)


def _is_within(child: Path, parent: Path) -> bool:
    """True iff ``child`` is at or under ``parent`` (resolved paths).

    Both arguments must already be resolved (``Path.resolve()``).
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
