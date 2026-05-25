"""Export a project's coded-segment retrieval report (F6.2).

Run with::

    .venv/bin/python -m scribe.scripts.export_retrieval_report \\
        --projects-root projects \\
        --project <project-id> \\
        --format markdown \\
        --group-by code \\
        --out segments.md

If ``--out`` is omitted the rendered report is written to stdout.

Filters (``--code``, ``--source``, ``--coder``, ``--participant``)
may be passed multiple times and are AND-combined: a row is included
only if it matches every filter set. Each filter takes a 12-char
hex id (the same shape stored on disk).

Transcript text is not loaded — the script renders rows from the
applications + entity store on disk and leaves the ``text`` column
empty. The eventual HTTP endpoint will hydrate transcript text from
``outputs/<job_id>/edited.json`` (or the fallback original
transcript) before rendering, but a CLI used in pipelines often
doesn't need text and shouldn't pay the cost of loading it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import applications as _applications
from .. import coders as _coders
from .. import codes as _codes
from .. import participants as _participants
from .. import projects as _projects
from .. import retrieval_report as rr
from .. import sources as _sources


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.export_retrieval_report",
        description=(
            "Render a project's coded-segment retrieval report in CSV "
            "/ Markdown / RTF (F6.2). Filters by code / source / coder "
            "/ participant; groups by code (default), source, "
            "participant, or none."
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
        "--format",
        default=rr.EXPORT_FORMAT_CSV,
        help=(
            "Export format: csv (default), markdown, rtf. "
            "Aliases: md → markdown; word/doc/docx → rtf."
        ),
    )
    parser.add_argument(
        "--group-by",
        default=rr.GROUP_BY_CODE,
        help=(
            "Group rows by: code (default), source, participant, none. "
            "Aliases: codes/sources/participants accepted; flat → none. "
            "Ignored for CSV (the schema is flat by contract)."
        ),
    )
    parser.add_argument(
        "--code",
        action="append",
        default=None,
        help="Code id filter (repeatable). AND-combined with other filters.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source id filter (repeatable).",
    )
    parser.add_argument(
        "--coder",
        action="append",
        default=None,
        help="Coder id filter (repeatable).",
    )
    parser.add_argument(
        "--participant",
        action="append",
        default=None,
        help="Participant id filter (repeatable). Focus-group rows "
             "match if any of their participants is in the set.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Omit to write to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        fmt = rr.normalise_format(args.format)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        group_by = rr.normalise_group_by(args.group_by)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

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

    apps = _applications.list_applications(args.projects_root, args.project)
    codes = _codes.list_codes(args.projects_root, args.project)
    sources = _sources.list_sources(args.projects_root, args.project)
    coders = _coders.list_coders(args.projects_root, args.project)
    parts = _participants.list_participants(
        args.projects_root, args.project
    )

    rows = rr.build_retrieval_rows(
        applications=apps,
        codes=codes,
        sources=sources,
        coders=coders,
        participants=parts,
    )

    if any(
        flt is not None for flt in (args.code, args.source, args.coder, args.participant)
    ):
        rows = rr.filter_rows(
            rows,
            code_ids=args.code,
            source_ids=args.source,
            coder_ids=args.coder,
            participant_ids=args.participant,
        )

    if args.out is None:
        text = rr.render_report(
            fmt, rows, project=project, group_by=group_by
        )
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0

    target = rr.write_report(
        args.out, fmt, rows, project=project, group_by=group_by
    )
    print(
        f"Wrote {len(rows)} segment(s) → {target}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
