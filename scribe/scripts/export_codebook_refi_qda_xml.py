"""Export a project's codebook as REFI-QDA Codebook XML (F6.5).

The companion to ``scribe.scripts.export_codebook`` (F6.1). F6.1
covers the CSV / Markdown / RTF formats; this script owns the
REFI-QDA Codebook 1.0 XML format kept off the F6.1 surface (so the
F6.1 endpoint's accepted format set stays stable, and the REFI-QDA
button can grow project-archive metadata that the lighter codebook-
only formats don't need).

Run with::

    .venv/bin/python -m scribe.scripts.export_codebook_refi_qda_xml \\
        --projects-root projects \\
        --project <project-id> \\
        --out codebook.refi-qda.xml

If ``--out`` is omitted the rendered XML is written to stdout. That
makes the script usable in pipelines (``... | xmllint --format -``)
and gives a quick way to inspect the output during development.

The file body is the REFI-QDA Codebook 1.0 XML produced by
:func:`scribe.codebook_export.render_refi_qda_codebook_xml`, which
includes a leading XML comment with the project's research question,
methodology, sensitising concepts, codebook stage, and timestamps.
REFI-QDA importers ignore comments, so the file remains
schema-conformant in any downstream QDA tool.
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
        prog="scribe.scripts.export_codebook_refi_qda_xml",
        description=(
            "Render a project's codebook as REFI-QDA Codebook 1.0 "
            "XML (F6.5). Pure dispatcher over "
            "scribe.codebook_export.render_refi_qda_codebook_xml."
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
        "--out",
        type=Path,
        default=None,
        help="Output file path. Omit to write to stdout.",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=ce.REFI_QDA_ORIGIN_DEFAULT,
        help=(
            "Value to put on the <CodeBook origin=...> attribute. "
            f"Defaults to {ce.REFI_QDA_ORIGIN_DEFAULT!r}; some "
            "downstream tools log the origin string for provenance."
        ),
    )
    return parser


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

    codes = _codes.list_codes(args.projects_root, args.project)

    if args.out is None:
        text = ce.render_refi_qda_codebook_xml(
            codes, project=project, origin=args.origin
        )
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0

    target = ce.write_refi_qda_codebook_xml(
        args.out, codes, project=project, origin=args.origin
    )
    print(f"Wrote {len(codes)} code(s) → {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
