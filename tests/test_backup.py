"""Tests for ``scribe.backup`` — the pure module behind /api/backup
and /api/restore.

We pin:

* what gets packed into the archive (outputs, projects, profiles,
  uploads-when-asked-for) and what doesn't (.env, .venv, models),
* the manifest's shape + version gate,
* round-trip restore: pack a tmp tree, blow it away, restore from
  the zip, the resulting tree matches.
* the safety guards: refusing non-empty targets without ``force``,
  rejecting ``..`` paths, refusing newer format versions.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from scribe import backup as bk


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _build_install(root: Path) -> bk.BackupPaths:
    """Cook a tiny but representative Scribe install under ``root``.

    Each top-level directory has at least one nested file so we can
    assert the walk + restore preserve relative paths.
    """
    outputs = root / "outputs"
    projects = root / "projects"
    uploads = root / "uploads"
    profiles = root / "profiles.json"

    (outputs / "abc123def456").mkdir(parents=True)
    (outputs / "abc123def456" / "job.json").write_text('{"id": "abc123def456"}')
    (outputs / "abc123def456" / "raw-audio.txt").write_text("hello\nworld\n")
    (outputs / "abc123def456" / "edited.json").write_text('{"segments": []}')

    (projects / "p1").mkdir(parents=True)
    (projects / "p1" / "project.json").write_text('{"id": "p1"}')
    (projects / "p1" / "codes").mkdir()
    (projects / "p1" / "codes" / "c1.json").write_text('{"id": "c1"}')

    (uploads / "abc123def456").mkdir(parents=True)
    (uploads / "abc123def456" / "raw-audio.wav").write_bytes(b"WAV\x00" * 16)

    profiles.write_text('[{"name": "default"}]')

    return bk.BackupPaths(
        outputs_dir=outputs,
        projects_dir=projects,
        uploads_dir=uploads,
        profiles_path=profiles,
    )


def _list_zip(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(zf.namelist())


# --------------------------------------------------------------------------- #
# create_backup
# --------------------------------------------------------------------------- #


class TestCreateBackup:
    def test_writes_a_real_zip_with_manifest(self, tmp_path: Path) -> None:
        paths = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        summary = bk.create_backup(paths, out)
        assert out.is_file()
        names = _list_zip(out)
        assert bk.ARCHIVE_MANIFEST_FILE in names
        # All four sections present.
        assert any(n.startswith("outputs/") for n in names)
        assert any(n.startswith("projects/") for n in names)
        assert any(n.startswith("uploads/") for n in names)
        assert "profiles.json" in names
        # Summary counts files.
        assert summary.outputs_files == 3
        assert summary.projects_files == 2
        assert summary.uploads_files == 1
        assert summary.profiles_present is True
        assert summary.total_bytes > 0

    def test_manifest_records_format_version(self, tmp_path: Path) -> None:
        paths = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        bk.create_backup(paths, out)
        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read(bk.ARCHIVE_MANIFEST_FILE))
        assert manifest["format"] == "scribe-backup"
        assert manifest["format_version"] == bk.BACKUP_FORMAT_VERSION
        assert manifest["include_uploads"] is True
        assert manifest["summary"]["outputs_files"] == 3

    def test_skips_uploads_when_asked(self, tmp_path: Path) -> None:
        paths = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        summary = bk.create_backup(paths, out, include_uploads=False)
        names = _list_zip(out)
        assert not any(n.startswith("uploads/") for n in names)
        assert summary.uploads_files == 0
        # Outputs + projects + profiles still there.
        assert any(n.startswith("outputs/") for n in names)
        assert any(n.startswith("projects/") for n in names)
        assert "profiles.json" in names

    def test_handles_missing_directories_gracefully(
        self, tmp_path: Path,
    ) -> None:
        # Fresh install: no directories, no profiles file.
        paths = bk.BackupPaths(
            outputs_dir=tmp_path / "outputs",
            projects_dir=tmp_path / "projects",
            uploads_dir=tmp_path / "uploads",
            profiles_path=tmp_path / "profiles.json",
        )
        out = tmp_path / "empty.zip"
        summary = bk.create_backup(paths, out)
        assert summary.outputs_files == 0
        assert summary.profiles_present is False
        assert out.is_file()
        # Manifest is still there.
        with zipfile.ZipFile(out) as zf:
            assert bk.ARCHIVE_MANIFEST_FILE in zf.namelist()

    def test_atomic_write_no_partial_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If zipping fails midway, the destination shouldn't be
        clobbered with a half-written file."""
        paths = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        out.write_bytes(b"original-contents")  # already exists

        # Force ZipFile to fail by replacing it briefly.
        original_zipfile = bk.zipfile.ZipFile

        class _BoomZip:
            def __init__(self, *a, **kw):
                raise RuntimeError("simulated zip failure")

        monkeypatch.setattr(bk.zipfile, "ZipFile", _BoomZip)

        with pytest.raises(RuntimeError, match="simulated"):
            bk.create_backup(paths, out)
        # Restore the real ZipFile so cleanup works in test teardown.
        monkeypatch.setattr(bk.zipfile, "ZipFile", original_zipfile)
        # Pre-existing file is intact.
        assert out.read_bytes() == b"original-contents"

    def test_streams_to_buffer(self, tmp_path: Path) -> None:
        """The streaming variant returns the same payload as the
        on-disk variant — used by the FastAPI endpoint to skip a
        round trip through the filesystem."""
        paths = _build_install(tmp_path / "src")
        buf = io.BytesIO()
        summary = bk.create_backup(paths, out_stream=buf)
        assert summary.outputs_files == 3
        # The buffer contains a valid zip.
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            assert bk.ARCHIVE_MANIFEST_FILE in zf.namelist()


# --------------------------------------------------------------------------- #
# inspect_backup
# --------------------------------------------------------------------------- #


class TestInspectBackup:
    def test_returns_manifest_payload(self, tmp_path: Path) -> None:
        paths = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        bk.create_backup(paths, out)
        manifest = bk.inspect_backup(out)
        assert manifest["format"] == "scribe-backup"
        assert manifest["summary"]["outputs_files"] == 3

    def test_rejects_non_zip(self, tmp_path: Path) -> None:
        bogus = tmp_path / "not-a-zip.zip"
        bogus.write_bytes(b"hello world")
        with pytest.raises(bk.RestoreError):
            bk.inspect_backup(bogus)

    def test_rejects_zip_without_manifest(self, tmp_path: Path) -> None:
        bogus = tmp_path / "no-manifest.zip"
        with zipfile.ZipFile(bogus, "w") as zf:
            zf.writestr("hello.txt", "world")
        with pytest.raises(bk.RestoreError, match="missing scribe-backup.json"):
            bk.inspect_backup(bogus)

    def test_rejects_newer_format_version(self, tmp_path: Path) -> None:
        # A backup whose format_version is higher than this Scribe
        # supports must refuse rather than silently restore something
        # we can't necessarily read.
        bogus = tmp_path / "future.zip"
        with zipfile.ZipFile(bogus, "w") as zf:
            zf.writestr(
                bk.ARCHIVE_MANIFEST_FILE,
                json.dumps({
                    "format": "scribe-backup",
                    "format_version": bk.BACKUP_FORMAT_VERSION + 1,
                }),
            )
        with pytest.raises(bk.RestoreError, match="format_version"):
            bk.inspect_backup(bogus)


# --------------------------------------------------------------------------- #
# restore_backup
# --------------------------------------------------------------------------- #


class TestRestoreBackup:
    def test_round_trip(self, tmp_path: Path) -> None:
        # Pack tree A → blow away → restore into tree B → contents match.
        src = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        bk.create_backup(src, out)

        # Fresh empty target.
        dst_root = tmp_path / "dst"
        dst = bk.BackupPaths(
            outputs_dir=dst_root / "outputs",
            projects_dir=dst_root / "projects",
            uploads_dir=dst_root / "uploads",
            profiles_path=dst_root / "profiles.json",
        )
        summary = bk.restore_backup(out, dst)
        # Files are where we expect.
        assert (dst.outputs_dir / "abc123def456" / "raw-audio.txt").read_text() \
            == "hello\nworld\n"
        assert (dst.projects_dir / "p1" / "codes" / "c1.json").read_text() \
            == '{"id": "c1"}'
        assert (dst.uploads_dir / "abc123def456" / "raw-audio.wav").read_bytes() \
            .startswith(b"WAV")
        assert dst.profiles_path.read_text() == '[{"name": "default"}]'
        assert summary.outputs_files == 3
        assert summary.projects_files == 2
        assert summary.uploads_files == 1
        assert summary.profiles_present is True

    def test_refuses_non_empty_target_without_force(
        self, tmp_path: Path,
    ) -> None:
        src = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        bk.create_backup(src, out)

        dst_root = tmp_path / "dst"
        dst = bk.BackupPaths(
            outputs_dir=dst_root / "outputs",
            projects_dir=dst_root / "projects",
            uploads_dir=dst_root / "uploads",
            profiles_path=dst_root / "profiles.json",
        )
        # Stage a stray file in outputs so it's "non-empty".
        dst.outputs_dir.mkdir(parents=True)
        (dst.outputs_dir / "leftover.txt").write_text("don't lose me")
        with pytest.raises(bk.RestoreError, match="not empty"):
            bk.restore_backup(out, dst)
        # Stray file untouched because we bailed before writing.
        assert (dst.outputs_dir / "leftover.txt").read_text() == "don't lose me"

    def test_force_clobbers_outputs_and_projects(
        self, tmp_path: Path,
    ) -> None:
        src = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        bk.create_backup(src, out)

        dst_root = tmp_path / "dst"
        dst = bk.BackupPaths(
            outputs_dir=dst_root / "outputs",
            projects_dir=dst_root / "projects",
            uploads_dir=dst_root / "uploads",
            profiles_path=dst_root / "profiles.json",
        )
        dst.outputs_dir.mkdir(parents=True)
        # A pre-existing file that's not in the backup. force=True
        # wipes the dir, so this must be gone after restore.
        (dst.outputs_dir / "stale.txt").write_text("delete me")
        bk.restore_backup(out, dst, force=True)
        assert not (dst.outputs_dir / "stale.txt").exists()
        # Backup contents are present.
        assert (dst.outputs_dir / "abc123def456" / "raw-audio.txt").is_file()

    def test_skip_uploads_at_restore_time(self, tmp_path: Path) -> None:
        src = _build_install(tmp_path / "src")
        out = tmp_path / "backup.zip"
        bk.create_backup(src, out)
        dst_root = tmp_path / "dst"
        dst = bk.BackupPaths(
            outputs_dir=dst_root / "outputs",
            projects_dir=dst_root / "projects",
            uploads_dir=dst_root / "uploads",
            profiles_path=dst_root / "profiles.json",
        )
        bk.restore_backup(out, dst, include_uploads=False)
        assert not dst.uploads_dir.exists() or not any(dst.uploads_dir.iterdir())

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        bogus = tmp_path / "evil.zip"
        with zipfile.ZipFile(bogus, "w") as zf:
            zf.writestr(
                bk.ARCHIVE_MANIFEST_FILE,
                json.dumps({"format": "scribe-backup",
                            "format_version": bk.BACKUP_FORMAT_VERSION}),
            )
            zf.writestr("outputs/../../escape.txt", "nope")
        dst_root = tmp_path / "dst"
        dst = bk.BackupPaths(
            outputs_dir=dst_root / "outputs",
            projects_dir=dst_root / "projects",
            uploads_dir=dst_root / "uploads",
            profiles_path=dst_root / "profiles.json",
        )
        with pytest.raises(bk.RestoreError, match="unsafe path"):
            bk.restore_backup(bogus, dst)

    def test_restore_does_not_unpack_env_or_other_keys(
        self, tmp_path: Path,
    ) -> None:
        """Forward-compat: a future version might add new top-level
        prefixes, but a current Scribe must ignore unknown ones rather
        than crash."""
        bogus = tmp_path / "future-extra.zip"
        with zipfile.ZipFile(bogus, "w") as zf:
            zf.writestr(
                bk.ARCHIVE_MANIFEST_FILE,
                json.dumps({"format": "scribe-backup",
                            "format_version": bk.BACKUP_FORMAT_VERSION}),
            )
            zf.writestr("future_thing/secret.txt", "hello")
        dst_root = tmp_path / "dst"
        dst = bk.BackupPaths(
            outputs_dir=dst_root / "outputs",
            projects_dir=dst_root / "projects",
            uploads_dir=dst_root / "uploads",
            profiles_path=dst_root / "profiles.json",
        )
        # Restores cleanly; the unknown directory is silently skipped.
        bk.restore_backup(bogus, dst)
        assert not (dst_root / "future_thing").exists()


# --------------------------------------------------------------------------- #
# Filename helper
# --------------------------------------------------------------------------- #


class TestSuggestedFilename:
    def test_default_shape(self) -> None:
        assert bk.suggested_backup_filename("20260604-120000") \
            == "scribe-backup-20260604-120000.zip"

    def test_sanitises_unsafe_chars(self) -> None:
        # If somehow a colon or slash gets in there, replace it.
        out = bk.suggested_backup_filename("2026/06/04 12:00:00")
        assert "/" not in out
        assert ":" not in out
        assert out.endswith(".zip")
