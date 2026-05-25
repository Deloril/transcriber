"""Export a project's matrix views to CSV / XLSX (F6.3).

Run with::

    .venv/bin/python -m scribe.scripts.export_matrix \\
        --projects-root projects \\
        --project <project-id> \\
        --kind code-by-source \\
        --format xlsx \\
        --out frequency.xlsx

If ``--out`` is omitted the rendered matrix is written to stdout.
For XLSX (a binary format) the bytes go to ``stdout.buffer`` so
shell pipelines (``... --format xlsx > matrix.xlsx``) work.

Three matrix kinds (mirrors :data:`scribe.matrix_export.MATRIX_KINDS`):

* ``code-by-source`` (alias ``frequency``) — how often each code
  appears in each source.
* ``code-by-code`` (alias ``cooccurrence``) — symmetric co-occurrence
  count under a chosen scope (``--scope source / segment / paragraph``).
* ``code-by-attribute`` (alias ``cross-tab``) — cross-tab against a
  source attribute (F3.2) or participant demographic (F1.3).

Filenames are slugified from the project name (``my-pilot-code-by-source-matrix.csv``)
so a directory of exports stays organised when you've run several.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import applications as _applications
from .. import codes as _codes
from .. import matrix as _matrix
from .. import matrix_export as me
from .. import participants as _participants
from .. import projects as _projects
from .. import sources as _sources
from .. import speaker_map as _speaker_map


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.export_matrix",
        description=(
            "Render a project's matrix view (frequency / co-occurrence "
            "/ cross-tab) as CSV or XLSX (F6.3). Pure dispatcher over "
            "the F3.6 matrix builders + F6.3 renderers."
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
        "--kind",
        default=me.MATRIX_KIND_CODE_BY_SOURCE,
        help=(
            "Matrix kind: code-by-source (default), code-by-code, "
            "code-by-attribute. Aliases: frequency / co-occurrence / "
            "cross-tab also accepted."
        ),
    )
    parser.add_argument(
        "--format",
        default=me.EXPORT_FORMAT_CSV,
        help=(
            "Export format: csv (default), xlsx. "
            "Aliases: xls / excel / spreadsheet → xlsx."
        ),
    )
    parser.add_argument(
        "--scope",
        default="source",
        help=(
            "code-by-code only: scope for co-occurrence "
            "(source / segment / paragraph). Default: source."
        ),
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.0,
        help=(
            "code-by-code only, paragraph scope: maximum gap (in the "
            "anchor units used by the applications) between two "
            "applications still considered co-occurring. Default: 0."
        ),
    )
    parser.add_argument(
        "--attribute-key",
        default=None,
        help=(
            "code-by-attribute only: which attribute to cross-tab on. "
            "Source-attribute key for ``--attribute-kind=source`` (F3.2) "
            "or participant demographic key for "
            "``--attribute-kind=participant`` (F1.3)."
        ),
    )
    parser.add_argument(
        "--attribute-kind",
        default="source",
        choices=list(_matrix.ATTRIBUTE_KINDS),
        help=(
            "code-by-attribute only: ``source`` (default) reads from "
            "source.custom_attributes; ``participant`` resolves the "
            "speaker label to a participant and reads from "
            "participant.demographics."
        ),
    )
    parser.add_argument(
        "--no-include-missing",
        action="store_true",
        help=(
            "code-by-attribute only: drop applications whose attribute "
            "value is missing instead of bucketing them into a "
            "``(missing)`` column."
        ),
    )
    parser.add_argument(
        "--no-totals",
        action="store_true",
        help="Suppress the totals row + column in the output.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help=(
            "Use the raw row/col keys (code id, source id) in the "
            "output instead of the human-readable names."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Drop empty rows / columns (rows/cols whose total is 0) "
            "before rendering. Useful when a large codebook only has "
            "a handful of applied codes."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Omit to write to stdout.",
    )
    return parser


def _build_matrix(
    kind: str,
    args: argparse.Namespace,
) -> _matrix.Matrix:
    """Dispatch to the right F3.6 builder for ``kind``."""
    apps = _applications.list_applications(
        args.projects_root, args.project
    )
    codes = _codes.list_codes(args.projects_root, args.project)

    if kind == me.MATRIX_KIND_CODE_BY_SOURCE:
        sources = _sources.list_sources(
            args.projects_root, args.project
        )
        return _matrix.code_by_source_matrix(
            applications=apps, codes=codes, sources=sources
        )

    if kind == me.MATRIX_KIND_CODE_BY_CODE:
        return _matrix.code_by_code_matrix(
            applications=apps,
            codes=codes,
            scope=args.scope,
            max_gap=float(args.max_gap),
        )

    # code-by-attribute
    if not args.attribute_key or not args.attribute_key.strip():
        raise ValueError(
            "--attribute-key is required for kind=code-by-attribute"
        )
    if args.attribute_kind == "source":
        sources = _sources.list_sources(
            args.projects_root, args.project
        )
        return _matrix.code_by_attribute_matrix(
            applications=apps,
            codes=codes,
            attribute_key=args.attribute_key,
            attribute_kind="source",
            sources=sources,
            include_missing=not args.no_include_missing,
        )
    # participant
    parts = _participants.list_participants(
        args.projects_root, args.project
    )
    smaps_list = _speaker_map.list_speaker_maps(
        args.projects_root, args.project
    )
    smaps = {sm.source_id: sm for sm in smaps_list}
    return _matrix.code_by_attribute_matrix(
        applications=apps,
        codes=codes,
        attribute_key=args.attribute_key,
        attribute_kind="participant",
        participants=parts,
        speaker_maps=smaps,
        include_missing=not args.no_include_missing,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        fmt = me.normalise_format(args.format)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        kind = me.normalise_matrix_kind(args.kind)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        project = _projects.load_project(
            args.projects_root, args.project
        )
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

    try:
        matrix = _build_matrix(kind, args)
    except (ValueError, _matrix.MatrixError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.compact:
        matrix = matrix.compact()

    payload = me.render_matrix(
        fmt,
        matrix,
        use_titles=not args.no_titles,
        include_totals=not args.no_totals,
    )

    if args.out is None:
        # XLSX is binary; route through stdout.buffer so we don't try
        # to encode-decode the ZIP body. CSV is text but we still go
        # through the buffer to keep behaviour uniform — UTF-8-encoded
        # bytes are what the user expects on stdout regardless.
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        return 0

    target = me.write_matrix(
        args.out,
        fmt,
        matrix,
        use_titles=not args.no_titles,
        include_totals=not args.no_totals,
    )
    print(
        f"Wrote {len(matrix.rows)}×{len(matrix.cols)} matrix → "
        f"{target} (project={project.name or project.id})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
