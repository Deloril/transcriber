"""Tests for scribe.project_format (F1.5).

Covers the manifest data model, the ProjectBundle aggregate, and the
zip-archive export/import round-trip — including path-traversal and
zip-bomb defences. Pure pytest; no FastAPI, no engine, no I/O outside
``tmp_path``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scribe.projects import (
    Project,
    save_project,
)
from scribe.sources import (
    Source,
    save_source,
)
from scribe.participants import (
    Participant,
    save_participant,
)
from scribe.sampling_log import (
    SamplingEntry,
    append_sampling_entry,
)
from scribe.codes import (
    Code,
    save_code,
    list_codes,
)
from scribe.project_format import (
    ARCHIVE_SUFFIX,
    ASSET_KIND_TRANSCRIPT,
    COMPONENT_CODES_DIR,
    COMPONENT_SOURCE_SCHEMA,
    DEFAULT_COMPONENT_PATHS,
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    ProjectBundle,
    ProjectFormatError,
    ProjectManifest,
    derive_external_assets,
    export_project_archive,
    import_project_archive,
    load_project_bundle,
    manifest_path,
    read_manifest,
    read_or_build_manifest,
    save_project_bundle,
    write_manifest,
)
from scribe.source_schema import (
    AttributeDefinition,
    SCHEMA_FILENAME,
    SourceAttributeSchema,
    load_source_schema,
    save_source_schema,
    source_schema_path,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _saved_project(
    tmp_path: Path,
    *,
    name: str = "Demo project",
    research_question: str = "",
) -> Project:
    p = Project.new(name=name, research_question=research_question)
    save_project(tmp_path, p)
    return p


def _saved_source(
    tmp_path: Path,
    project: Project,
    *,
    name: str = "Interview 01",
    transcript_job_id: str | None = None,
    language: str = "en",
) -> Source:
    s = Source.new(
        project_id=project.id,
        name=name,
        transcript_job_id=transcript_job_id,
        language=language,
    )
    save_source(tmp_path, s)
    return s


def _saved_participant(
    tmp_path: Path,
    project: Project,
    *,
    name: str = "P01",
) -> Participant:
    p = Participant.new(project_id=project.id, name=name)
    save_participant(tmp_path, p)
    return p


def _appended_sampling_entry(
    tmp_path: Path,
    project: Project,
    *,
    rationale: str = "Initial",
    action: str = "added",
) -> SamplingEntry:
    e = SamplingEntry.new(
        project_id=project.id,
        rationale=rationale,
        action=action,
    )
    append_sampling_entry(tmp_path, e)
    return e


# --------------------------------------------------------------------------- #
# Format constants
# --------------------------------------------------------------------------- #


class TestFormatConstants:
    def test_format_name_is_stable(self) -> None:
        # A different value here is a breaking change to every existing
        # archive — make sure renames are intentional.
        assert FORMAT_NAME == "scribe-project"

    def test_format_version_is_positive_int(self) -> None:
        assert isinstance(FORMAT_VERSION, int)
        assert FORMAT_VERSION >= 1

    def test_default_component_paths_have_expected_keys(self) -> None:
        for k in (
            "project",
            "sources_dir",
            "participants_dir",
            "sampling_log",
            "codes_dir",  # F3.1
        ):
            assert k in DEFAULT_COMPONENT_PATHS

    def test_default_component_paths_includes_codes_dir(self) -> None:
        # F3.1: the codebook directory rides in the manifest's
        # components dict so external readers know where to look.
        assert COMPONENT_CODES_DIR == "codes_dir"
        assert DEFAULT_COMPONENT_PATHS[COMPONENT_CODES_DIR] == "codes"

    def test_archive_suffix_ends_zip(self) -> None:
        assert ARCHIVE_SUFFIX.endswith(".zip")


# --------------------------------------------------------------------------- #
# ProjectManifest construction & validation
# --------------------------------------------------------------------------- #


class TestProjectManifestFromProject:
    def test_minimal(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        m = ProjectManifest.from_project(project)
        assert m.project_id == project.id
        assert m.name == "X"
        assert m.format == FORMAT_NAME
        assert m.format_version == FORMAT_VERSION
        assert m.created_at == project.created_at
        assert m.modified_at == project.modified_at
        assert m.components == DEFAULT_COMPONENT_PATHS
        assert m.external_assets == []

    def test_external_assets_passed_through(self) -> None:
        project = Project.new(name="X")
        assets = [{"kind": ASSET_KIND_TRANSCRIPT, "ref": "outputs/abcdef012345"}]
        m = ProjectManifest.from_project(project, external_assets=assets)
        assert m.external_assets == assets


class TestProjectManifestRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        project = Project.new(name="X")
        m1 = ProjectManifest.from_project(
            project,
            external_assets=[
                {"kind": ASSET_KIND_TRANSCRIPT, "ref": "outputs/abcdef012345"}
            ],
        )
        d = m1.to_dict()
        m2 = ProjectManifest.from_dict(d)
        assert m2.project_id == m1.project_id
        assert m2.format_version == m1.format_version
        assert m2.external_assets == m1.external_assets

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ProjectFormatError):
            ProjectManifest.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_rejects_missing_keys(self) -> None:
        with pytest.raises(ProjectFormatError):
            ProjectManifest.from_dict({"project_id": "abcdef012345"})


class TestProjectManifestValidate:
    def _make(self, **overrides) -> ProjectManifest:
        defaults = dict(
            project_id="abcdef012345",
            name="N",
            created_at="2026-01-01T00:00:00Z",
            modified_at="2026-01-01T00:00:00Z",
        )
        defaults.update(overrides)
        return ProjectManifest(**defaults)

    def test_rejects_bad_format_name(self) -> None:
        m = self._make(format="other-tool")
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_negative_version(self) -> None:
        m = self._make(format_version=0)
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_future_version(self) -> None:
        m = self._make(format_version=FORMAT_VERSION + 99)
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_bad_project_id(self) -> None:
        m = self._make(project_id="not-hex!!")
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_blank_name(self) -> None:
        m = self._make(name="   ")
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_missing_timestamps(self) -> None:
        m = self._make(created_at="")
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_absolute_component_path(self) -> None:
        m = self._make(components={"project": "/etc/passwd"})
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_traversal_component_path(self) -> None:
        m = self._make(components={"project": "../escape.json"})
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_empty_component_path(self) -> None:
        m = self._make(components={"project": ""})
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_bad_external_assets_shape(self) -> None:
        m = self._make(external_assets=[{"kind": "transcript"}])  # missing ref
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_rejects_bad_transcript_ref(self) -> None:
        m = self._make(
            external_assets=[{"kind": "transcript", "ref": "outputs/whatever"}]
        )
        with pytest.raises(ProjectFormatError):
            m.validate()

    def test_unknown_asset_kind_is_tolerated(self) -> None:
        # Forward compat: future asset kinds (e.g. "memo_index") should
        # not break older readers.
        m = self._make(
            external_assets=[{"kind": "memo_index", "ref": "memos/index.bin"}]
        )
        m.validate()  # no raise

    def test_components_default_is_valid(self) -> None:
        m = self._make()
        m.validate()  # default components dict passes


# --------------------------------------------------------------------------- #
# derive_external_assets
# --------------------------------------------------------------------------- #


class TestDeriveExternalAssets:
    def test_empty_when_no_sources(self) -> None:
        assert derive_external_assets([]) == []

    def test_skips_sources_without_job(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s1 = Source.new(project_id=project.id, name="A")
        s2 = Source.new(
            project_id=project.id, name="B", transcript_job_id="abcdef012345"
        )
        out = derive_external_assets([s1, s2])
        assert out == [{"kind": "transcript", "ref": "outputs/abcdef012345"}]

    def test_dedupes_repeated_jobs(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s1 = Source.new(
            project_id=project.id, name="A", transcript_job_id="abcdef012345"
        )
        s2 = Source.new(
            project_id=project.id, name="B", transcript_job_id="abcdef012345"
        )
        out = derive_external_assets([s1, s2])
        assert len(out) == 1

    def test_sorted_by_ref(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s1 = Source.new(
            project_id=project.id, name="A", transcript_job_id="ffffffffffff"
        )
        s2 = Source.new(
            project_id=project.id, name="B", transcript_job_id="000000000000"
        )
        out = derive_external_assets([s1, s2])
        assert out[0]["ref"] == "outputs/000000000000"
        assert out[1]["ref"] == "outputs/ffffffffffff"


# --------------------------------------------------------------------------- #
# Manifest persistence
# --------------------------------------------------------------------------- #


class TestWriteAndReadManifest:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        m = write_manifest(tmp_path, project.id)
        assert manifest_path(tmp_path, project.id).exists()
        assert m.project_id == project.id

    def test_round_trip(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        _saved_source(
            tmp_path, project, transcript_job_id="abcdef012345"
        )
        write_manifest(tmp_path, project.id)
        m = read_manifest(tmp_path, project.id)
        assert m.project_id == project.id
        assert any(
            a["ref"] == "outputs/abcdef012345" for a in m.external_assets
        )

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        # Don't write a manifest — reading it should fail.
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path, project.id)

    def test_read_invalid_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectFormatError):
            read_manifest(tmp_path, "not-hex!!")

    def test_read_invalid_json_raises(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        manifest_path(tmp_path, project.id).write_text("this is not json{{{")
        with pytest.raises(ProjectFormatError):
            read_manifest(tmp_path, project.id)

    def test_write_no_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            write_manifest(tmp_path, "abcdef012345")


class TestReadOrBuildManifest:
    def test_uses_existing_when_present(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        original = write_manifest(tmp_path, project.id)
        result = read_or_build_manifest(tmp_path, project.id)
        assert result.created_at == original.created_at

    def test_builds_when_absent(self, tmp_path: Path) -> None:
        # Pre-F1.5 project: no manifest.json on disk.
        project = _saved_project(tmp_path)
        _saved_source(
            tmp_path, project, transcript_job_id="abcdef012345"
        )
        assert not manifest_path(tmp_path, project.id).exists()
        m = read_or_build_manifest(tmp_path, project.id)
        assert m.project_id == project.id
        assert any(a["ref"] == "outputs/abcdef012345" for a in m.external_assets)

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_or_build_manifest(tmp_path, "abcdef012345")


# --------------------------------------------------------------------------- #
# ProjectBundle validation
# --------------------------------------------------------------------------- #


class TestProjectBundleValidate:
    def test_minimal_passes(self) -> None:
        project = Project.new(name="X")
        ProjectBundle(project=project).validate()  # no raise

    def test_full_passes(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        s = Source.new(project_id=project.id, name="A")
        p = Participant.new(project_id=project.id, name="P01")
        e = SamplingEntry.new(project_id=project.id)
        b = ProjectBundle(
            project=project, sources=[s], participants=[p], sampling_log=[e]
        )
        b.validate()

    def test_source_project_id_mismatch(self) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        rogue = Source.new(
            project_id="bbbbbbbbbbbb", name="Mismatched"
        )
        b = ProjectBundle(project=project, sources=[rogue])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_participant_project_id_mismatch(self) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        rogue = Participant.new(
            project_id="bbbbbbbbbbbb", name="P01"
        )
        b = ProjectBundle(project=project, participants=[rogue])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_sampling_entry_project_id_mismatch(self) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        rogue = SamplingEntry.new(project_id="bbbbbbbbbbbb")
        b = ProjectBundle(project=project, sampling_log=[rogue])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_duplicate_source_ids_rejected(self) -> None:
        project = Project.new(name="X")
        s1 = Source.new(project_id=project.id, name="A", source_id="abcdef012345")
        s2 = Source.new(project_id=project.id, name="B", source_id="abcdef012345")
        b = ProjectBundle(project=project, sources=[s1, s2])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_duplicate_participant_ids_rejected(self) -> None:
        project = Project.new(name="X")
        p1 = Participant.new(
            project_id=project.id, name="A", participant_id="abcdef012345"
        )
        p2 = Participant.new(
            project_id=project.id, name="B", participant_id="abcdef012345"
        )
        b = ProjectBundle(project=project, participants=[p1, p2])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_duplicate_sampling_entry_ids_rejected(self) -> None:
        project = Project.new(name="X")
        e1 = SamplingEntry.new(project_id=project.id, entry_id="abcdef012345")
        e2 = SamplingEntry.new(project_id=project.id, entry_id="abcdef012345")
        b = ProjectBundle(project=project, sampling_log=[e1, e2])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_manifest_project_id_mismatch(self) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        manifest = ProjectManifest.from_project(
            Project.new(name="X", project_id="bbbbbbbbbbbb")
        )
        b = ProjectBundle(project=project, manifest=manifest)
        with pytest.raises(ProjectFormatError):
            b.validate()


# --------------------------------------------------------------------------- #
# load_project_bundle / save_project_bundle
# --------------------------------------------------------------------------- #


class TestLoadProjectBundle:
    def test_full_load(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        s = _saved_source(tmp_path, project, transcript_job_id="abcdef012345")
        p = _saved_participant(tmp_path, project)
        e = _appended_sampling_entry(tmp_path, project)
        write_manifest(tmp_path, project.id)
        bundle = load_project_bundle(tmp_path, project.id)
        assert bundle.project.id == project.id
        assert [src.id for src in bundle.sources] == [s.id]
        assert [pa.id for pa in bundle.participants] == [p.id]
        assert [se.id for se in bundle.sampling_log] == [e.id]
        assert bundle.manifest is not None
        assert bundle.manifest.project_id == project.id

    def test_pre_f15_project_loads_with_derived_manifest(
        self, tmp_path: Path
    ) -> None:
        # No manifest.json on disk — load should still succeed.
        project = _saved_project(tmp_path)
        bundle = load_project_bundle(tmp_path, project.id)
        assert bundle.manifest is not None  # derived
        assert bundle.manifest.project_id == project.id

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_project_bundle(tmp_path, "abcdef012345")


class TestSaveProjectBundle:
    def test_creates_full_tree(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        s = Source.new(project_id=project.id, name="A")
        p = Participant.new(project_id=project.id, name="P01")
        e = SamplingEntry.new(project_id=project.id, rationale="seed")
        bundle = ProjectBundle(
            project=project, sources=[s], participants=[p], sampling_log=[e]
        )
        save_project_bundle(tmp_path, bundle)
        loaded = load_project_bundle(tmp_path, project.id)
        assert loaded.project.id == project.id
        assert [src.id for src in loaded.sources] == [s.id]
        assert [pa.id for pa in loaded.participants] == [p.id]
        assert [se.id for se in loaded.sampling_log] == [e.id]
        assert manifest_path(tmp_path, project.id).exists()

    def test_returns_manifest(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        bundle = ProjectBundle(project=project)
        m = save_project_bundle(tmp_path, bundle)
        assert m.project_id == project.id

    def test_does_not_clobber_existing_log_by_default(
        self, tmp_path: Path
    ) -> None:
        # Set up: project + 2 entries on disk.
        project = _saved_project(tmp_path)
        e1 = _appended_sampling_entry(tmp_path, project, rationale="first")
        e2 = _appended_sampling_entry(tmp_path, project, rationale="second")
        # Save a bundle with an empty sampling_log: existing log stays.
        bundle = ProjectBundle(project=project, sampling_log=[])
        save_project_bundle(tmp_path, bundle)
        loaded = load_project_bundle(tmp_path, project.id)
        assert [s.id for s in loaded.sampling_log] == [e1.id, e2.id]

    def test_replace_sampling_log_rewrites(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        _appended_sampling_entry(tmp_path, project)
        _appended_sampling_entry(tmp_path, project)
        # Replace with a fresh single-entry log.
        new_entry = SamplingEntry.new(project_id=project.id, rationale="reset")
        bundle = ProjectBundle(project=project, sampling_log=[new_entry])
        save_project_bundle(tmp_path, bundle, replace_sampling_log=True)
        loaded = load_project_bundle(tmp_path, project.id)
        assert [s.id for s in loaded.sampling_log] == [new_entry.id]

    def test_skip_manifest(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        bundle = ProjectBundle(project=project)
        save_project_bundle(tmp_path, bundle, write_manifest_file=False)
        assert not manifest_path(tmp_path, project.id).exists()

    def test_invalid_bundle_raises(self, tmp_path: Path) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        rogue_source = Source.new(project_id="bbbbbbbbbbbb", name="X")
        bundle = ProjectBundle(project=project, sources=[rogue_source])
        with pytest.raises(ProjectFormatError):
            save_project_bundle(tmp_path, bundle)


# --------------------------------------------------------------------------- #
# Archive export / import
# --------------------------------------------------------------------------- #


def _build_demo_project_on_disk(tmp_path: Path) -> tuple[Path, Project]:
    """Create a project tree we can export."""
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    project = Project.new(name="Demo")
    save_project(projects_root, project)
    save_source(
        projects_root,
        Source.new(
            project_id=project.id,
            name="Interview 01",
            transcript_job_id="abcdef012345",
        ),
    )
    save_participant(
        projects_root, Participant.new(project_id=project.id, name="P01")
    )
    append_sampling_entry(
        projects_root,
        SamplingEntry.new(project_id=project.id, rationale="seed"),
    )
    return projects_root, project


def _make_outputs_with_job(tmp_path: Path, job_id: str) -> Path:
    """Create a fake outputs/<job_id>/ tree with a couple of files."""
    outputs = tmp_path / "outputs"
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "transcript.json").write_text('{"segments": []}')
    (job_dir / "transcript.txt").write_text("[00:00] LUKE: hi\n")
    return outputs


class TestExportProjectArchive:
    def test_creates_archive_with_project_files(
        self, tmp_path: Path
    ) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        archive = tmp_path / f"export{ARCHIVE_SUFFIX}"
        export_project_archive(projects_root, project.id, archive)
        assert archive.exists()
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert f"{project.id}/{MANIFEST_FILENAME}" in names
        assert f"{project.id}/project.json" in names
        # Sources/participants/sampling-log files present.
        assert any(
            n.startswith(f"{project.id}/sources/") and n.endswith(".json")
            for n in names
        )
        assert any(
            n.startswith(f"{project.id}/participants/") and n.endswith(".json")
            for n in names
        )
        assert f"{project.id}/sampling_log.jsonl" in names

    def test_excludes_outputs_by_default(self, tmp_path: Path) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        _make_outputs_with_job(tmp_path, "abcdef012345")
        archive = tmp_path / "export.zip"
        export_project_archive(projects_root, project.id, archive)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert not any("/outputs/" in n for n in names)

    def test_includes_outputs_when_requested(self, tmp_path: Path) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        outputs = _make_outputs_with_job(tmp_path, "abcdef012345")
        archive = tmp_path / "export.zip"
        export_project_archive(
            projects_root,
            project.id,
            archive,
            outputs_root=outputs,
            include_outputs=True,
        )
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert f"{project.id}/outputs/abcdef012345/transcript.json" in names
        assert f"{project.id}/outputs/abcdef012345/transcript.txt" in names

    def test_include_outputs_requires_outputs_root(
        self, tmp_path: Path
    ) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        archive = tmp_path / "export.zip"
        with pytest.raises(ProjectFormatError):
            export_project_archive(
                projects_root, project.id, archive, include_outputs=True
            )

    def test_missing_outputs_dir_silently_skipped(self, tmp_path: Path) -> None:
        # Source references a job that doesn't exist on disk — export
        # should still produce a valid archive of the project itself.
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        empty_outputs = tmp_path / "outputs"
        empty_outputs.mkdir()
        archive = tmp_path / "export.zip"
        export_project_archive(
            projects_root,
            project.id,
            archive,
            outputs_root=empty_outputs,
            include_outputs=True,
        )
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert not any(n.startswith(f"{project.id}/outputs/") for n in names)
        assert f"{project.id}/project.json" in names

    def test_refuses_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectFormatError):
            export_project_archive(tmp_path, "not-hex!!", tmp_path / "x.zip")

    def test_refuses_missing_project(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            export_project_archive(
                tmp_path, "abcdef012345", tmp_path / "x.zip"
            )

    def test_writes_current_manifest(self, tmp_path: Path) -> None:
        # Even if no manifest existed before export, the archive must
        # contain one (so import has something to validate against).
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        # Wipe any manifest written by other tests just in case.
        mp = manifest_path(projects_root, project.id)
        if mp.exists():
            mp.unlink()
        archive = tmp_path / "export.zip"
        export_project_archive(projects_root, project.id, archive)
        # Manifest now exists on disk.
        assert mp.exists()


class TestImportProjectArchive:
    def test_round_trip(self, tmp_path: Path) -> None:
        src_root, project = _build_demo_project_on_disk(tmp_path)
        archive = tmp_path / "export.zip"
        export_project_archive(src_root, project.id, archive)

        # Import into a fresh projects_root.
        dest_root = tmp_path / "dest"
        bundle = import_project_archive(dest_root, archive)
        assert bundle.project.id == project.id
        assert (dest_root / project.id / "project.json").exists()
        assert (dest_root / project.id / MANIFEST_FILENAME).exists()
        # Source + participant + log all round-tripped.
        assert len(bundle.sources) == 1
        assert len(bundle.participants) == 1
        assert len(bundle.sampling_log) == 1

    def test_round_trip_with_outputs(self, tmp_path: Path) -> None:
        src_root, project = _build_demo_project_on_disk(tmp_path)
        outputs = _make_outputs_with_job(tmp_path, "abcdef012345")
        archive = tmp_path / "export.zip"
        export_project_archive(
            src_root, project.id, archive,
            outputs_root=outputs, include_outputs=True,
        )
        dest_root = tmp_path / "dest"
        dest_outputs = tmp_path / "dest_outputs"
        bundle = import_project_archive(
            dest_root, archive, outputs_root=dest_outputs
        )
        assert (dest_outputs / "abcdef012345" / "transcript.json").exists()
        assert (dest_outputs / "abcdef012345" / "transcript.txt").exists()

    def test_outputs_in_archive_ignored_without_outputs_root(
        self, tmp_path: Path
    ) -> None:
        # If the archive has outputs but the importer doesn't pass
        # outputs_root, the project still imports cleanly (just no
        # transcript files restored).
        src_root, project = _build_demo_project_on_disk(tmp_path)
        outputs = _make_outputs_with_job(tmp_path, "abcdef012345")
        archive = tmp_path / "export.zip"
        export_project_archive(
            src_root, project.id, archive,
            outputs_root=outputs, include_outputs=True,
        )
        dest_root = tmp_path / "dest"
        bundle = import_project_archive(dest_root, archive)
        assert bundle.project.id == project.id

    def test_refuses_overwrite_by_default(self, tmp_path: Path) -> None:
        src_root, project = _build_demo_project_on_disk(tmp_path)
        archive = tmp_path / "export.zip"
        export_project_archive(src_root, project.id, archive)
        dest_root = tmp_path / "dest"
        import_project_archive(dest_root, archive)  # first time fine
        with pytest.raises(ProjectFormatError):
            import_project_archive(dest_root, archive)  # second time fails

    def test_overwrite_true_replaces(self, tmp_path: Path) -> None:
        src_root, project = _build_demo_project_on_disk(tmp_path)
        archive = tmp_path / "export.zip"
        export_project_archive(src_root, project.id, archive)
        dest_root = tmp_path / "dest"
        import_project_archive(dest_root, archive)
        # Mutate the destination (rename file) and re-import.
        (dest_root / project.id / "project.json").write_text("{}")
        import_project_archive(dest_root, archive, overwrite=True)
        # Project now valid again.
        loaded = load_project_bundle(dest_root, project.id)
        assert loaded.project.id == project.id

    def test_missing_archive_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_project_archive(tmp_path, tmp_path / "nope.zip")

    def test_empty_archive_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "empty.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            pass
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)

    def test_multiple_project_dirs_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "two.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("aaaaaaaaaaaa/project.json", "{}")
            zf.writestr("bbbbbbbbbbbb/project.json", "{}")
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)

    def test_top_level_must_be_valid_project_id(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("not-hex!!/project.json", "{}")
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)

    def test_missing_manifest_in_archive_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "no_manifest.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("abcdef012345/project.json", "{}")
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)

    def test_manifest_id_mismatch_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "mismatch.zip"
        manifest_payload = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "project_id": "bbbbbbbbbbbb",  # different from top-level dir
            "name": "X",
            "created_at": "2026-01-01T00:00:00Z",
            "modified_at": "2026-01-01T00:00:00Z",
            "components": dict(DEFAULT_COMPONENT_PATHS),
            "external_assets": [],
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("aaaaaaaaaaaa/manifest.json", json.dumps(manifest_payload))
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)

    def test_zip_slip_blocked(self, tmp_path: Path) -> None:
        # Member with .. should be rejected by the path-within check.
        bad = tmp_path / "slip.zip"
        manifest_payload = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "project_id": "abcdef012345",
            "name": "Slip",
            "created_at": "2026-01-01T00:00:00Z",
            "modified_at": "2026-01-01T00:00:00Z",
            "components": dict(DEFAULT_COMPONENT_PATHS),
            "external_assets": [],
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(
                "abcdef012345/manifest.json", json.dumps(manifest_payload)
            )
            zf.writestr("abcdef012345/../escape.txt", "pwned")
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)

    def test_unrelated_top_level_member_rejected(self, tmp_path: Path) -> None:
        # A member at the archive root with no project-id prefix should
        # be rejected by the top-level discovery (it counts as a second
        # top-level entry).
        bad = tmp_path / "leak.zip"
        manifest_payload = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "project_id": "abcdef012345",
            "name": "Leak",
            "created_at": "2026-01-01T00:00:00Z",
            "modified_at": "2026-01-01T00:00:00Z",
            "components": dict(DEFAULT_COMPONENT_PATHS),
            "external_assets": [],
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(
                "abcdef012345/manifest.json", json.dumps(manifest_payload)
            )
            # Stray sibling at the archive root.
            zf.writestr("README.txt", "stray")
        with pytest.raises(ProjectFormatError):
            import_project_archive(tmp_path, bad)


# --------------------------------------------------------------------------- #
# Manifest deterministic-output sanity
# --------------------------------------------------------------------------- #


class TestManifestDeterminism:
    def test_repeated_writes_produce_same_bytes_when_state_unchanged(
        self, tmp_path: Path
    ) -> None:
        # The manifest is sorted (external_assets) so writing it twice
        # without mutating the project should yield identical bytes.
        project = _saved_project(tmp_path)
        _saved_source(tmp_path, project, transcript_job_id="abcdef012345")
        write_manifest(tmp_path, project.id)
        bytes1 = manifest_path(tmp_path, project.id).read_bytes()
        write_manifest(tmp_path, project.id)
        bytes2 = manifest_path(tmp_path, project.id).read_bytes()
        assert bytes1 == bytes2


# --------------------------------------------------------------------------- #
# F3.1: codebook (F2.1 codes) rides inside the project shell.
# --------------------------------------------------------------------------- #


def _new_code(project: Project, *, name: str = "Initial code") -> Code:
    """Build a fresh F2.1 Code attached to the given project."""
    return Code.new(project_id=project.id, name=name)


class TestProjectBundleCodes:
    def test_validate_accepts_codes(self) -> None:
        project = Project.new(name="X")
        c = _new_code(project, name="Drifting")
        b = ProjectBundle(project=project, codes=[c])
        b.validate()  # no raise

    def test_validate_rejects_code_with_wrong_project_id(self) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        rogue = Code.new(project_id="bbbbbbbbbbbb", name="Rogue")
        b = ProjectBundle(project=project, codes=[rogue])
        with pytest.raises(ProjectFormatError):
            b.validate()

    def test_validate_rejects_duplicate_code_ids(self) -> None:
        project = Project.new(name="X")
        c1 = Code.new(
            project_id=project.id, name="A", code_id="abcdef012345"
        )
        c2 = Code.new(
            project_id=project.id, name="B", code_id="abcdef012345"
        )
        b = ProjectBundle(project=project, codes=[c1, c2])
        with pytest.raises(ProjectFormatError):
            b.validate()


class TestLoadBundleWithCodes:
    def test_loads_persisted_codes(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        c1 = _new_code(project, name="Trying to fit in")
        c2 = _new_code(project, name="Coping silently")
        save_code(tmp_path, c1)
        save_code(tmp_path, c2)
        bundle = load_project_bundle(tmp_path, project.id)
        loaded_ids = {c.id for c in bundle.codes}
        assert loaded_ids == {c1.id, c2.id}

    def test_empty_codes_when_none_on_disk(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        bundle = load_project_bundle(tmp_path, project.id)
        assert bundle.codes == []


class TestSaveBundleWithCodes:
    def test_persists_codes(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        c1 = _new_code(project, name="One")
        c2 = _new_code(project, name="Two")
        bundle = ProjectBundle(project=project, codes=[c1, c2])
        save_project_bundle(tmp_path, bundle)
        on_disk = list_codes(tmp_path, project.id)
        assert {c.id for c in on_disk} == {c1.id, c2.id}

    def test_save_does_not_clobber_codes_not_in_bundle(
        self, tmp_path: Path
    ) -> None:
        # Mirrors the sampling-log "append-only" stance: an empty
        # bundle.codes must not erase pre-existing codes on disk.
        project = _saved_project(tmp_path)
        existing = _new_code(project, name="Existing")
        save_code(tmp_path, existing)
        bundle = ProjectBundle(project=project, codes=[])
        save_project_bundle(tmp_path, bundle)
        on_disk = list_codes(tmp_path, project.id)
        assert [c.id for c in on_disk] == [existing.id]

    def test_round_trip_through_load_then_save(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        original = _new_code(project, name="Roundtrip")
        save_code(tmp_path, original)
        bundle = load_project_bundle(tmp_path, project.id)
        assert [c.id for c in bundle.codes] == [original.id]
        # Re-save via bundle path; codes must still load.
        save_project_bundle(tmp_path, bundle)
        again = load_project_bundle(tmp_path, project.id)
        assert [c.id for c in again.codes] == [original.id]


class TestArchiveIncludesCodes:
    def test_export_archive_includes_codes_directory(
        self, tmp_path: Path
    ) -> None:
        # F3.1: codes/<id>.json should ride inside the project
        # archive automatically (it lives under the project root,
        # which the archive walks recursively).
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        c = _new_code(project, name="Archived code")
        save_code(projects_root, c)

        out = tmp_path / f"{project.id}{ARCHIVE_SUFFIX}"
        export_project_archive(projects_root, project.id, out)

        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
        assert f"{project.id}/codes/{c.id}.json" in names

    def test_import_archive_restores_codes(self, tmp_path: Path) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        c = _new_code(project, name="To be restored")
        save_code(projects_root, c)
        archive = tmp_path / f"{project.id}{ARCHIVE_SUFFIX}"
        export_project_archive(projects_root, project.id, archive)

        # Fresh root.
        target_root = tmp_path / "restored"
        target_root.mkdir()
        bundle = import_project_archive(target_root, archive)
        assert [code.id for code in bundle.codes] == [c.id]
        assert [code.name for code in bundle.codes] == ["To be restored"]


class TestProjectSettingsRoundTripThroughBundle:
    def test_save_load_preserves_settings(self, tmp_path: Path) -> None:
        # F3.1: project-level settings travel with the project
        # through the bundle save/load round trip.
        project = Project.new(
            name="Settings Project",
            settings={
                "default_coder": "Luke",
                "ai": {"enabled": False, "model": "phi-4"},
            },
        )
        bundle = ProjectBundle(project=project)
        save_project_bundle(tmp_path, bundle)
        loaded = load_project_bundle(tmp_path, project.id)
        assert loaded.project.settings == project.settings

    def test_archive_round_trip_preserves_settings(
        self, tmp_path: Path
    ) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        # Mutate settings and save the project.json again.
        project.apply_update({"settings": {"default_coder": "Sam"}})
        from scribe.projects import save_project as _save
        _save(projects_root, project)
        archive = tmp_path / f"{project.id}{ARCHIVE_SUFFIX}"
        export_project_archive(projects_root, project.id, archive)
        target = tmp_path / "restored"
        target.mkdir()
        bundle = import_project_archive(target, archive)
        assert bundle.project.settings == {"default_coder": "Sam"}


# --------------------------------------------------------------------------- #
# F3.2 — source attribute schema integration
# --------------------------------------------------------------------------- #


class TestSourceSchemaInBundle:
    def test_default_components_include_source_schema(self) -> None:
        # The manifest's default-component dict must mention the schema
        # so external readers know where to find it.
        assert COMPONENT_SOURCE_SCHEMA == "source_schema"
        assert (
            DEFAULT_COMPONENT_PATHS[COMPONENT_SOURCE_SCHEMA]
            == SCHEMA_FILENAME
        )

    def test_bundle_load_when_schema_absent(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        bundle = load_project_bundle(tmp_path, project.id)
        assert bundle.source_schema is None

    def test_bundle_load_when_schema_present(self, tmp_path: Path) -> None:
        project = _saved_project(tmp_path)
        schema = SourceAttributeSchema.new(
            project_id=project.id,
            attributes=[AttributeDefinition(key="site", label="Site")],
        )
        save_source_schema(tmp_path, schema)
        bundle = load_project_bundle(tmp_path, project.id)
        assert bundle.source_schema is not None
        assert [a.key for a in bundle.source_schema.attributes] == ["site"]

    def test_bundle_save_persists_schema(self, tmp_path: Path) -> None:
        project = Project.new(name="X")
        schema = SourceAttributeSchema.new(
            project_id=project.id,
            attributes=[AttributeDefinition(key="round", type="number")],
        )
        bundle = ProjectBundle(project=project, source_schema=schema)
        save_project_bundle(tmp_path, bundle)
        # Read back through the source-schema persistence layer too.
        loaded = load_source_schema(tmp_path, project.id)
        assert [a.key for a in loaded.attributes] == ["round"]

    def test_bundle_save_omitting_schema_does_not_delete(
        self, tmp_path: Path
    ) -> None:
        # Mirrors the codebook + sampling-log "append-only by default":
        # an empty bundle must not erase the schema on disk.
        project = _saved_project(tmp_path)
        schema = SourceAttributeSchema.new(
            project_id=project.id,
            attributes=[AttributeDefinition(key="site")],
        )
        save_source_schema(tmp_path, schema)
        bundle = ProjectBundle(project=project)  # source_schema=None
        save_project_bundle(tmp_path, bundle)
        # Schema file still there.
        assert source_schema_path(tmp_path, project.id).exists()

    def test_bundle_validate_schema_project_id_mismatch(self) -> None:
        project = Project.new(name="X", project_id="aaaaaaaaaaaa")
        rogue = SourceAttributeSchema.new(project_id="bbbbbbbbbbbb")
        bundle = ProjectBundle(project=project, source_schema=rogue)
        with pytest.raises(ProjectFormatError):
            bundle.validate()

    def test_archive_round_trip_preserves_schema(
        self, tmp_path: Path
    ) -> None:
        projects_root, project = _build_demo_project_on_disk(tmp_path)
        schema = SourceAttributeSchema.new(
            project_id=project.id,
            attributes=[
                AttributeDefinition(key="site", type="select",
                                    options=["A", "B"], required=True),
                AttributeDefinition(key="round", type="number"),
            ],
        )
        save_source_schema(projects_root, schema)
        archive = tmp_path / f"{project.id}{ARCHIVE_SUFFIX}"
        export_project_archive(projects_root, project.id, archive)
        target = tmp_path / "restored"
        target.mkdir()
        bundle = import_project_archive(target, archive)
        assert bundle.source_schema is not None
        keys = [a.key for a in bundle.source_schema.attributes]
        assert keys == ["site", "round"]
