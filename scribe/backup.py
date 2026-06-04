"""Server-wide backup + restore.

A full Scribe install is just a handful of directories under
``ROOT``:

* ``outputs/<job_id>/`` — transcripts (engine JSON, edited JSON,
  .txt/.srt/.vtt sidecars, ``job.json`` metadata, waveform cache).
* ``projects/<project_id>/`` — projects, sources, codes, code
  versions, applications, memos, suggestions, embeddings, chats,
  participants, coders, snapshots, lock audits — everything the
  academic-coding stack writes.
* ``profiles.json`` — the user's saved transcription profile presets.
* ``uploads/<job_id>/`` — original source media. Optional in a backup
  because (a) it's the bulky bit, often gigabytes, and (b) the user
  may have explicitly ``Discard Media``'d it to reclaim disk.

What this module does NOT pack:

* ``.env`` — holds the HF token. Backups get emailed around, dropped
  in Drive, etc.; embedding a credential in a portable archive is a
  trap. The README documents how to re-set the token after a restore.
* ``.venv``, ``node_modules``, model caches, ``logs/`` — these are
  reconstructible by re-running ``./setup.sh``.

The two public entry points are :func:`create_backup` (writes a zip
to a destination path or returns the bytes for streaming) and
:func:`restore_backup` (reads the zip and rewrites the target dirs
in-place, atomically when possible).
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

# What the manifest schema records. Bump this when the layout
# changes so a future Scribe can refuse incompatible backups.
BACKUP_FORMAT_VERSION = 1

# Top-level directory names *inside the zip*. We deliberately don't
# preserve absolute paths — a backup made on one user's machine has
# to restore cleanly on another's.
ARCHIVE_OUTPUTS_DIR = "outputs"
ARCHIVE_PROJECTS_DIR = "projects"
ARCHIVE_UPLOADS_DIR = "uploads"
ARCHIVE_PROFILES_FILE = "profiles.json"
ARCHIVE_MANIFEST_FILE = "scribe-backup.json"


@dataclass(frozen=True)
class BackupPaths:
    """Where backup helpers read from / write to.

    Decoupling these from the ``server.py`` module-level globals lets
    tests run against tmp dirs without monkeypatching, and lets the
    restore helper accept a "preview into a sandbox first" path.
    """

    outputs_dir: Path
    projects_dir: Path
    uploads_dir: Path
    profiles_path: Path

    def all_present(self) -> bool:
        # We don't *require* all four to exist on disk — a fresh
        # install has none of them — but the manifest records which
        # were captured.
        return any(p.exists() for p in (
            self.outputs_dir,
            self.projects_dir,
            self.uploads_dir,
            self.profiles_path,
        ))


@dataclass(frozen=True)
class BackupSummary:
    """Per-section file counts so the UI can show "203 files,
    1.4 GB" without having to walk the archive itself."""

    outputs_files: int
    projects_files: int
    uploads_files: int
    profiles_present: bool
    total_bytes: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs_files": self.outputs_files,
            "projects_files": self.projects_files,
            "uploads_files": self.uploads_files,
            "profiles_present": self.profiles_present,
            "total_bytes": self.total_bytes,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _build_manifest(
    *, include_uploads: bool, summary: BackupSummary,
) -> dict[str, Any]:
    return {
        "format": "scribe-backup",
        "format_version": BACKUP_FORMAT_VERSION,
        "created_at": summary.created_at,
        "include_uploads": include_uploads,
        "summary": summary.to_dict(),
        # Helpful for a human inspecting the zip without unpacking:
        # the directory names that appear at the archive root.
        "layout": {
            "outputs": ARCHIVE_OUTPUTS_DIR,
            "projects": ARCHIVE_PROJECTS_DIR,
            "uploads": ARCHIVE_UPLOADS_DIR,
            "profiles": ARCHIVE_PROFILES_FILE,
        },
    }


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    if ARCHIVE_MANIFEST_FILE not in zf.namelist():
        raise RestoreError(
            "Backup is missing scribe-backup.json — not a Scribe backup?"
        )
    with zf.open(ARCHIVE_MANIFEST_FILE) as f:
        try:
            data = json.loads(f.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RestoreError(f"Manifest is not valid JSON: {e}") from e
    if not isinstance(data, dict) or data.get("format") != "scribe-backup":
        raise RestoreError("Manifest does not identify a Scribe backup.")
    version = int(data.get("format_version") or 0)
    if version > BACKUP_FORMAT_VERSION:
        raise RestoreError(
            f"Backup format_version={version} is newer than this Scribe "
            f"install supports ({BACKUP_FORMAT_VERSION}). Upgrade Scribe "
            "before restoring."
        )
    return data


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #


class BackupError(RuntimeError):
    """Raised when create_backup / restore_backup hit a fatal problem."""


class RestoreError(BackupError):
    """Distinct subclass so HTTP layers can return 400 vs 500."""


def _safe_archive_name(rel: Path) -> str:
    """Return a forward-slash POSIX path for a zip arcname.

    Zips store paths with ``/`` regardless of platform; we convert
    explicitly so a backup made on Windows extracts cleanly on macOS
    (and vice versa). Also rejects entries with ``..`` segments —
    we never write those, but a defensive check guards against a
    rogue caller.
    """
    parts = rel.parts
    if any(p == ".." for p in parts):
        raise BackupError(f"Refusing to pack path with '..': {rel}")
    return "/".join(parts)


def _walk_for_archive(root: Path) -> Iterable[tuple[Path, Path]]:
    """Yield (absolute_path, path_relative_to_root) for every file
    under ``root``. Symlinks are followed for files (so a reattached
    media link gets its bytes archived) but not for directories (so
    a circular link can't trap us). Empty directories are not
    archived because zip + most extractors cope fine without them."""
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        # Stable ordering helps reproducible backups for tests.
        dirnames.sort()
        for fname in sorted(filenames):
            full = dp / fname
            rel = full.relative_to(root)
            yield full, rel


def create_backup(
    paths: BackupPaths,
    out_path: Path | None = None,
    *,
    include_uploads: bool = True,
    out_stream: BinaryIO | None = None,
    now: str | None = None,
) -> BackupSummary:
    """Write a zip backup of ``paths`` to ``out_path`` *or* ``out_stream``.

    Exactly one of the two destinations must be supplied. ``out_path``
    is the file-on-disk variant (used by the CLI / direct tests);
    ``out_stream`` lets the FastAPI endpoint stream the archive
    straight into the HTTP response without writing it to disk first.

    ``include_uploads=True`` packs ``uploads/<job_id>/`` directories.
    Set False to skip — the resulting backup is much smaller but
    won't restore source-media playback for the affected jobs.
    """
    if (out_path is None) == (out_stream is None):
        raise BackupError(
            "Exactly one of out_path / out_stream must be supplied."
        )

    created_at = now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    outputs_files = 0
    projects_files = 0
    uploads_files = 0
    profiles_present = False
    total_bytes = 0

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file then rename so an interrupted
        # run leaves the target untouched. zipfile doesn't have a
        # native atomic-write helper.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".scribe-backup.", suffix=".zip.tmp",
            dir=str(out_path.parent),
        )
        os.close(fd)
        target_for_zip: Path = Path(tmp_path)
        zf_handle = zipfile.ZipFile(
            target_for_zip, "w", compression=zipfile.ZIP_DEFLATED,
        )
    else:
        target_for_zip = None  # type: ignore[assignment]
        zf_handle = zipfile.ZipFile(
            out_stream, "w", compression=zipfile.ZIP_DEFLATED,  # type: ignore[arg-type]
        )

    try:
        with zf_handle as zf:
            # outputs/
            for full, rel in _walk_for_archive(paths.outputs_dir):
                arcname = f"{ARCHIVE_OUTPUTS_DIR}/{_safe_archive_name(rel)}"
                zf.write(full, arcname=arcname)
                outputs_files += 1
                try:
                    total_bytes += full.stat().st_size
                except OSError:
                    pass

            # projects/
            for full, rel in _walk_for_archive(paths.projects_dir):
                arcname = f"{ARCHIVE_PROJECTS_DIR}/{_safe_archive_name(rel)}"
                zf.write(full, arcname=arcname)
                projects_files += 1
                try:
                    total_bytes += full.stat().st_size
                except OSError:
                    pass

            # uploads/  (optional)
            if include_uploads:
                for full, rel in _walk_for_archive(paths.uploads_dir):
                    arcname = f"{ARCHIVE_UPLOADS_DIR}/{_safe_archive_name(rel)}"
                    zf.write(full, arcname=arcname)
                    uploads_files += 1
                    try:
                        total_bytes += full.stat().st_size
                    except OSError:
                        pass

            # profiles.json
            if paths.profiles_path.is_file():
                zf.write(paths.profiles_path, arcname=ARCHIVE_PROFILES_FILE)
                profiles_present = True
                try:
                    total_bytes += paths.profiles_path.stat().st_size
                except OSError:
                    pass

            summary = BackupSummary(
                outputs_files=outputs_files,
                projects_files=projects_files,
                uploads_files=uploads_files,
                profiles_present=profiles_present,
                total_bytes=total_bytes,
                created_at=created_at,
            )
            manifest = _build_manifest(
                include_uploads=include_uploads, summary=summary,
            )
            zf.writestr(
                ARCHIVE_MANIFEST_FILE,
                json.dumps(manifest, indent=2, sort_keys=True),
            )

        # Atomic rename for the file path; the stream variant is
        # already flushed by the ``with`` block.
        if out_path is not None and target_for_zip is not None:
            os.replace(target_for_zip, out_path)
        return summary
    except Exception:
        # Clean up the temp file on failure.
        if out_path is not None and target_for_zip is not None:
            try:
                target_for_zip.unlink()
            except OSError:
                pass
        raise


# --------------------------------------------------------------------------- #
# Restore
# --------------------------------------------------------------------------- #


def inspect_backup(zip_path_or_stream: Path | BinaryIO) -> dict[str, Any]:
    """Read the manifest from a backup file without unpacking it.

    Surfaced as a dry-run for the UI: shows the user "you're about to
    restore 203 outputs, 14 projects, 1.4 GB, made 2026-06-04" before
    they confirm the destructive write.
    """
    try:
        src = zipfile.ZipFile(zip_path_or_stream, "r")
    except zipfile.BadZipFile as e:
        raise RestoreError(f"Not a valid zip file: {e}") from e
    with src as zf:
        return _read_manifest(zf)


def restore_backup(
    zip_path_or_stream: Path | BinaryIO,
    paths: BackupPaths,
    *,
    force: bool = False,
    include_uploads: bool = True,
) -> BackupSummary:
    """Restore a backup over the directories in ``paths``.

    Refuses to clobber non-empty target directories unless
    ``force=True`` is set. The caller (the FastAPI endpoint) is
    responsible for surfacing a confirmation prompt to the user
    before passing the flag.

    ``include_uploads`` defaults to True; pass False to skip the
    ``uploads/`` block even when the backup contains it (useful when
    the user has reclaimed disk via Discard Media and doesn't want to
    re-fill it).
    """
    try:
        src = zipfile.ZipFile(zip_path_or_stream, "r")
    except zipfile.BadZipFile as e:
        raise RestoreError(f"Not a valid zip file: {e}") from e

    with src as zf:
        manifest = _read_manifest(zf)
        # Check destination is empty (or the user has consented to overwrite).
        if not force:
            for d in (paths.outputs_dir, paths.projects_dir):
                if d.exists() and any(d.iterdir()):
                    raise RestoreError(
                        f"{d} is not empty. Pass force=True to overwrite."
                    )
            if paths.profiles_path.exists():
                raise RestoreError(
                    f"{paths.profiles_path} exists. Pass force=True to overwrite."
                )

        # Restore. We extract entries individually rather than calling
        # ``zf.extractall`` because we route each top-level prefix to
        # a different destination directory (the user might have
        # configured non-standard paths, or set include_uploads=False
        # at restore time).
        outputs_files = 0
        projects_files = 0
        uploads_files = 0
        profiles_present = False
        total_bytes = 0

        # Wipe targets when force=True. We do this for outputs +
        # projects only — never for uploads, because the user may
        # have other reattach-media symlinks pointing at files
        # outside the upload dir that we shouldn't break.
        if force:
            for d in (paths.outputs_dir, paths.projects_dir):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)

        for member in zf.infolist():
            name = member.filename
            if name == ARCHIVE_MANIFEST_FILE:
                continue
            if member.is_dir():
                continue
            # Defensive: zipfile rejects ``..`` paths via ``Path.resolve``
            # but the explicit guard makes the intent obvious.
            norm = name.replace("\\", "/")
            if ".." in norm.split("/"):
                raise RestoreError(
                    f"Backup contains an unsafe path: {name!r}"
                )
            if norm == ARCHIVE_PROFILES_FILE:
                paths.profiles_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src_f, paths.profiles_path.open("wb") as dst:
                    shutil.copyfileobj(src_f, dst)
                profiles_present = True
                total_bytes += member.file_size
                continue
            head, _, rest = norm.partition("/")
            if head == ARCHIVE_OUTPUTS_DIR:
                target_root = paths.outputs_dir
                outputs_files += 1
            elif head == ARCHIVE_PROJECTS_DIR:
                target_root = paths.projects_dir
                projects_files += 1
            elif head == ARCHIVE_UPLOADS_DIR:
                if not include_uploads:
                    continue
                target_root = paths.uploads_dir
                uploads_files += 1
            else:
                # Unknown top-level entries are ignored rather than
                # erroring — keeps forward-compat for additions.
                continue
            target = target_root / rest
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src_f, target.open("wb") as dst:
                shutil.copyfileobj(src_f, dst)
            total_bytes += member.file_size

        return BackupSummary(
            outputs_files=outputs_files,
            projects_files=projects_files,
            uploads_files=uploads_files,
            profiles_present=profiles_present,
            total_bytes=total_bytes,
            created_at=manifest.get("created_at", ""),
        )


# --------------------------------------------------------------------------- #
# Filename helper for the download endpoint.
# --------------------------------------------------------------------------- #


_BACKUP_FILENAME_TIMESTAMP_RE = re.compile(r"[^A-Za-z0-9._-]")


def suggested_backup_filename(now: str | None = None) -> str:
    """Pick a friendly filename for a freshly-created backup.

    Format: ``scribe-backup-YYYYMMDD-HHMMSS.zip`` in UTC. Stable
    across calls for the same ``now`` so tests can pin it.
    """
    ts = now or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    safe = _BACKUP_FILENAME_TIMESTAMP_RE.sub("-", ts)
    return f"scribe-backup-{safe}.zip"
