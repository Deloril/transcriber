"""Export a project as a REFI-QDA / QDPX archive (F6.4).

Run with::

    .venv/bin/python -m scribe.scripts.export_qdpx \\
        --projects-root projects \\
        --project <project-id> \\
        --outputs-root outputs \\
        --out study.qdpx

The script gathers every entity that survives in REFI-QDA (project
metadata, codebook, sources, applications, memos, coders) plus each
source's transcript (looked up under ``--outputs-root`` via the
source's ``transcript_job_id``) and produces a QDPX zip a downstream
QDA tool (Atlas.ti, MAXQDA, NVivo, QDA Miner, Quirkos, Dedoose) will
import.

Sources without a discoverable transcript still appear in the
``project.qde`` (so a downstream tool sees them) but won't carry
``<PlainTextSelection>`` children — without the transcript we can't
compute REFI-QDA's char-offset anchors. Those sources' applications
are silently skipped; F4.5's orphan queue is the right place to
surface that, not this exporter.

Stdout / file:

  * ``--out PATH`` writes the archive bytes to ``PATH``.
  * Omitted: writes the archive bytes to stdout (handy for piping
    into another tool, but stdout must be a binary stream — most
    shells handle this transparently for ``foo > out.qdpx``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import applications as _applications
from .. import coders as _coders
from .. import codes as _codes
from .. import memos as _memos
from .. import projects as _projects
from .. import refi_qda_project as qdpx
from .. import sources as _sources


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.export_qdpx",
        description=(
            "Export a project as a REFI-QDA / QDPX archive (F6.4). "
            "Bundles project + codebook + sources (with plain-text "
            "transcripts) + applications + memos + coders into a "
            "single zip any major QDA tool can import."
        ),
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=Path("projects"),
        help="Directory holding ``<project-id>/project.json`` "
             "(default: ./projects).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project id (12-char hex). Must exist under --projects-root.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory holding ``<job-id>/`` transcript artefacts "
             "(default: ./outputs). Sources whose transcripts can't "
             "be found are still emitted, but without selections.",
    )
    parser.add_argument(
        "--origin",
        default=qdpx.REFI_QDA_PROJECT_ORIGIN_DEFAULT,
        help="``origin`` attribute on the QDE root (default: 'Scribe').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path (.qdpx). Omit to write the archive "
             "bytes to stdout.",
    )
    return parser


def _load_segments_for_source(
    outputs_root: Path,
    source: "_sources.Source",
) -> list[dict] | None:
    """Pull the transcript segments for a source, preferring edits.

    Looks under ``<outputs_root>/<job_id>/edited.json`` first, then
    falls back to any ``<outputs_root>/<job_id>/<stem>.json``. Returns
    ``None`` when no parseable transcript is available — the QDPX
    exporter handles that case gracefully.
    """
    if not source.transcript_job_id:
        return None
    job_dir = outputs_root / source.transcript_job_id
    if not job_dir.is_dir():
        return None

    # Prefer edited.json (the editor's authoritative version).
    edited = job_dir / "edited.json"
    candidates: list[Path] = []
    if edited.is_file():
        candidates.append(edited)
    # Fall back to any *.json sidecar produced by the engine. We sort
    # so the order is deterministic across runs and pick the first
    # one that parses + has a "segments" key.
    candidates.extend(
        sorted(p for p in job_dir.glob("*.json") if p.name != "edited.json")
    )
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return data["segments"]
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        project = _projects.load_project(args.projects_root, args.project)
    except _projects.ProjectValidationError as e:
        print(f"error: invalid project id: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(
            f"error: project {args.project!r} not found under "
            f"{args.projects_root}",
            file=sys.stderr,
        )
        return 2

    sources = _sources.list_sources(args.projects_root, args.project)
    codes = _codes.list_codes(args.projects_root, args.project)
    apps = _applications.list_applications(args.projects_root, args.project)
    memos = _memos.list_memos(args.projects_root, args.project)
    coders = _coders.list_coders(args.projects_root, args.project)

    rendered_sources: list[qdpx.RenderedSource] = []
    missing_transcripts = 0
    for s in sources:
        segs = _load_segments_for_source(args.outputs_root, s)
        if segs is None:
            missing_transcripts += 1
            continue
        rendered_sources.append(qdpx.render_source_plain_text(s.id, segs))

    archive = qdpx.to_qdpx(
        project=project,
        sources=sources,
        codes=codes,
        applications=apps,
        memos=memos,
        coders=coders,
        rendered_sources=rendered_sources,
        origin=args.origin,
    )

    if args.out is None:
        # Binary stream needed for the archive bytes.
        sys.stdout.buffer.write(archive)
        sys.stdout.flush()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_bytes(archive)
        tmp.replace(args.out)
        print(
            f"Wrote QDPX archive ({len(archive):,} bytes) → {args.out}",
            file=sys.stderr,
        )

    if missing_transcripts:
        print(
            f"warning: {missing_transcripts} source(s) had no "
            f"discoverable transcript under {args.outputs_root}; "
            "their applications were skipped.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
