"""Export a project's audit trail in CSV / Markdown / RTF (F9.7).

Run with::

    .venv/bin/python -m scribe.scripts.export_audit_trail \\
        --projects-root projects \\
        --project <project-id> \\
        --format markdown \\
        --out audit-trail.md

If ``--out`` is omitted the rendered audit trail is written to
stdout. That makes the script usable in pipelines (``... --format
markdown | pandoc -o audit-trail.docx``) and gives a quick way to
inspect the output during development.

Filters mirror :func:`scribe.audit_export.build_audit_trail` and AND-
combine. ``--kind`` may be supplied multiple times to restrict to a
subset of source kinds (``event`` / ``ai_invocation``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import audit_export as ae
from .. import projects as _projects


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.export_audit_trail",
        description=(
            "Render a project's audit trail in CSV / Markdown / RTF "
            "(F9.7). Composes F9.1 events + F9.6 AI invocations."
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
        default=ae.EXPORT_FORMAT_MARKDOWN,
        help=(
            "Export format: markdown (default), csv, rtf. "
            "Aliases: md → markdown; word/doc/docx → rtf."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Omit to write to stdout.",
    )

    # Filters.
    parser.add_argument(
        "--since",
        default=None,
        help="Inclusive ISO-8601 lower bound on row timestamps.",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Inclusive ISO-8601 upper bound on row timestamps.",
    )
    parser.add_argument(
        "--kind",
        action="append",
        default=None,
        help=(
            "Restrict to a source kind (event / ai_invocation). "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--actor",
        default=None,
        help="12-char hex coder id; restrict to one human's activity.",
    )
    parser.add_argument(
        "--entity-type",
        default=None,
        help=(
            "Restrict to a F9.1 entity type (project / source / code / "
            "application / memo / coder / saved_query / sampling_log / "
            "codebook / snapshot / checkpoint / other)."
        ),
    )
    parser.add_argument(
        "--action",
        default=None,
        help=(
            "Restrict to an F9.1 action (create / update / delete / "
            "rename / merge / split / retire / promote / lock / "
            "unlock / snapshot / checkpoint / import / export / other)."
        ),
    )
    parser.add_argument(
        "--feature",
        default=None,
        help=(
            "Restrict to an F9.6 AI feature (code_suggestion / "
            "new_code_suggestion / quote_similarity / transcript_review / "
            "second_coder / memo_draft / other)."
        ),
    )
    parser.add_argument(
        "--decision",
        default=None,
        help=(
            "Restrict to an F9.6 decision (pending / accepted / "
            "modified / rejected / request_only)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        fmt = ae.normalise_format(args.format)
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

    try:
        rows = ae.build_audit_trail(
            args.projects_root,
            args.project,
            since=args.since,
            until=args.until,
            kinds=args.kind,
            actor_coder_id=args.actor,
            entity_type=args.entity_type,
            action=args.action,
            feature=args.feature,
            decision=args.decision,
        )
    except _projects.ProjectValidationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.out is None:
        text = ae.render_audit_trail(fmt, rows, project=project)
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0

    target = ae.write_audit_trail(args.out, fmt, rows, project=project)
    print(f"Wrote {len(rows)} row(s) → {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
