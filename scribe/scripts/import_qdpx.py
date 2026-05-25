"""Import a QDPX archive into a Scribe project (F6.6).

Run with::

    .venv/bin/python -m scribe.scripts.import_qdpx \\
        --projects-root projects \\
        --archive study.qdpx

The script reads the archive, parses every entity REFI-QDA describes
(project metadata, codebook with hierarchy, sources, applications,
memos, coders), and persists them under
``<projects-root>/<new-project-id>/...`` using the same on-disk layout
the rest of Scribe writes — so the imported project is immediately
indistinguishable from one created in-app.

A new project id is always minted; we don't try to reuse a Scribe-
origin project's id because two installs sharing a workstation could
collide silently. Within the new project, individual entity ids are
preserved when the archive was Scribe-origin (the F6.4 export pads
ids into REFI-QDA GUIDs in a reversible way) and freshly minted when
the archive came from another QDA tool.

Imported plain-text source bodies are also persisted: each source's
text and tokenised segments land under
``<projects-root>/<project>/imported_sources/<source-id>.json`` so the
editor / coding views can later pull them up. The transcript-level
integration with the editor (rendering imported text in the player
view) is F10.3's job; F6.6 just lays the data down on disk.

The script prints a one-line summary on stderr and the new project id
on stdout, so a downstream shell script can pipe::

    pid="$(.venv/bin/python -m scribe.scripts.import_qdpx \\
              --projects-root projects --archive study.qdpx)"
    open "http://localhost:8000/projects/$pid"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import applications as _applications
from .. import code_versions as _code_versions
from .. import coders as _coders
from .. import codes as _codes
from .. import memos as _memos
from .. import projects as _projects
from .. import refi_qda_import as importer
from .. import sources as _sources


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.import_qdpx",
        description=(
            "Import a REFI-QDA / QDPX archive into a Scribe project "
            "(F6.6). Writes project + codebook + sources + applications "
            "+ memos + coders to <projects-root>/<new-project-id>/. The "
            "new project id is printed on stdout."
        ),
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=Path("projects"),
        help="Directory holding ``<project-id>/project.json`` "
             "(default: ./projects). Will be created if missing.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the .qdpx file to import.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary on stderr.",
    )
    return parser


def _persist_imported_source_text(
    projects_root: Path,
    project_id: str,
    entry: importer.ImportedSourceText,
) -> Path:
    """Write an imported source's plain-text body + tokenisation.

    Lands under ``<projects-root>/<project>/imported_sources/<sid>.json``.
    The JSON shape is:

      {
        "source_id": "...",
        "text": "...",
        "segments": [
          {"speaker": "...", "words": [{"word_id": "...", "text": "...",
                                        "start": int, "end": int}, ...]},
          ...
        ]
      }

    F10.3 will pick this up as the ground-truth transcript for an
    imported source; F6.6 just persists it so nothing is lost.
    """
    target_dir = _projects.project_dir(projects_root, project_id) / "imported_sources"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{entry.source_id}.json"
    payload = {
        "source_id": entry.source_id,
        "text": entry.text,
        "segments": [
            {
                "speaker": seg.speaker,
                "words": [
                    {"word_id": w.word_id, "text": w.text,
                     "start": w.start, "end": w.end}
                    for w in seg.words
                ],
            }
            for seg in entry.segments
        ],
    }
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(target)
    return target


def _persist_code_version(
    projects_root: Path,
    version: _code_versions.CodeVersion,
) -> Path:
    """Append a CodeVersion to its code's JSONL log.

    Mirrors what ``record_code_version`` does, but preserves the
    caller-supplied version id (the importer mints one synthetic
    version per imported code, and we want that exact id on disk so
    the imported applications' ``definition_version_id_at_apply``
    references resolve cleanly).
    """
    cvd = _code_versions.code_versions_dir(projects_root, version.project_id)
    cvd.mkdir(parents=True, exist_ok=True)
    target = _code_versions.code_versions_path(
        projects_root, version.project_id, version.code_id
    )
    line = json.dumps(version.to_dict(), ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(line)
    return target


def persist_import_result(
    projects_root: Path,
    result: importer.ImportResult,
) -> int:
    """Write every entity in ``result`` to disk under ``projects_root``.

    Returns the count of files written (project.json plus per-entity
    files). Exposed so tests don't have to redo the persistence dance
    by hand.
    """
    written = 0
    _projects.save_project(projects_root, result.project)
    written += 1

    for s in result.sources:
        _sources.save_source(projects_root, s)
        written += 1
    for c in result.codes:
        _codes.save_code(projects_root, c)
        written += 1
    for cd in result.coders:
        _coders.save_coder(projects_root, cd)
        written += 1
    for m in result.memos:
        _memos.save_memo(projects_root, m)
        written += 1
    for v in result.code_versions:
        _persist_code_version(projects_root, v)
        written += 1
    for a in result.applications:
        _applications.save_application(projects_root, a)
        written += 1
    for entry in result.source_texts.values():
        _persist_imported_source_text(
            projects_root, result.project.id, entry
        )
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.archive.is_file():
        print(f"error: archive not found: {args.archive}", file=sys.stderr)
        return 2

    try:
        result = importer.import_qdpx(args.archive)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    args.projects_root.mkdir(parents=True, exist_ok=True)
    written = persist_import_result(args.projects_root, result)

    print(result.project.id)

    if not args.quiet:
        print(
            f"Imported project {result.project.id} ({result.project.name!r}): "
            f"{len(result.sources)} source(s), "
            f"{len(result.codes)} code(s), "
            f"{len(result.coders)} coder(s), "
            f"{len(result.memos)} memo(s), "
            f"{len(result.applications)} application(s); "
            f"{written} file(s) written.",
            file=sys.stderr,
        )
        if result.warnings:
            print(
                f"warning: {len(result.warnings)} non-fatal issue(s) during import:",
                file=sys.stderr,
            )
            for w in result.warnings[:20]:
                print(f"  - {w}", file=sys.stderr)
            if len(result.warnings) > 20:
                print(
                    f"  ... and {len(result.warnings) - 20} more.",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
