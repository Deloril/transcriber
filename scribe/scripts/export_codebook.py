"""Export a project's codebook to CSV / Markdown / RTF (F6.1).

Run with::

    .venv/bin/python -m scribe.scripts.export_codebook \\
        --projects-root projects \\
        --project <project-id> \\
        --format csv \\
        --out codebook.csv

If ``--out`` is omitted the rendered codebook is written to stdout.
That makes the script usable in pipelines (``... --format markdown |
pandoc -o codebook.docx``) and gives a quick way to inspect the
output during development.

REFI-QDA XML is intentionally not exposed here — F6.5 owns that
button so it can grow project-archive metadata that the lighter
codebook-only surface doesn't need.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import codebook_export as ce
from .. import codes as _codes
from .. import projects as _projects


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.export_codebook",
        description=(
            "Render a project's codebook in CSV / Markdown / RTF "
            "(F6.1). Pure dispatcher over the F2.6 exporters."
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
        default=ce.EXPORT_FORMAT_CSV,
        help=(
            "Export format: csv (default), markdown, rtf. "
            "Aliases: md → markdown; word/doc/docx → rtf."
        ),
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
        fmt = ce.normalise_format(args.format)
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

    codes = _codes.list_codes(args.projects_root, args.project)

    if args.out is None:
        text = ce.render_codebook(fmt, codes, project=project)
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0

    target = ce.write_codebook(args.out, fmt, codes, project=project)
    print(f"Wrote {len(codes)} code(s) → {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
