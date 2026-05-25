"""Export an anonymised QDPX archive (F6.7).

Run with::

    .venv/bin/python -m scribe.scripts.export_anonymised_qdpx \\
        --projects-root projects \\
        --project <project-id> \\
        --outputs-root outputs \\
        --rules my-rules.json \\
        --out study-anon.qdpx

The script gathers the same entities as :mod:`scribe.scripts.export_qdpx`
plus the project's participants and per-source speaker maps, builds a
:class:`scribe.anonymise.RedactionPlan` from
``Participant.name → Participant.pseudonym`` mappings (and any custom
rules supplied via ``--rules``), runs every text-bearing entity and
every transcript through the plan, and bundles the result into a QDPX
archive with a ``Redactions/manifest.json`` listing what was replaced
and how often.

The ``--rules`` JSON file is an array of ``RedactionRule.to_dict()``
shapes, e.g.::

    [
      {"pattern": "Mercy General Hospital",
       "replacement": "[hospital]"},
      {"pattern": "\\\\b\\\\d{3}-\\\\d{4}\\\\b",
       "replacement": "[phone]",
       "regex": true}
    ]

Stdout / file:

  * ``--out PATH`` writes the redacted archive bytes to ``PATH``.
  * Omitted: writes the archive bytes to stdout (handy for piping;
    your shell must handle binary streams, ``foo > out.qdpx``).
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
from .. import participants as _participants
from .. import projects as _projects
from .. import sources as _sources
from .. import speaker_map as _speaker_map
from ..anonymise import (
    RedactionRule,
    build_anonymised_qdpx,
)
from ..refi_qda_project import REFI_QDA_PROJECT_ORIGIN_DEFAULT
from .export_qdpx import _load_segments_for_source


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Exposed so tests can introspect."""
    parser = argparse.ArgumentParser(
        prog="scribe.scripts.export_anonymised_qdpx",
        description=(
            "Export a project as an *anonymised* REFI-QDA / QDPX "
            "archive (F6.7). Applies a rule-based redaction pass to "
            "every text-bearing entity and every transcript before "
            "bundling. Participants' name → pseudonym mappings are "
            "auto-included; additional rules may be supplied via "
            "--rules. The bundle ships with a Redactions/manifest.json "
            "listing replacements + match counts (never the originals)."
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
        "--rules",
        type=Path,
        default=None,
        help="JSON file of additional redaction rules (array of "
             "RedactionRule objects). Optional.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-form note recorded in the manifest header.",
    )
    parser.add_argument(
        "--origin",
        default=REFI_QDA_PROJECT_ORIGIN_DEFAULT,
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


def load_custom_rules(rules_path: Path | None) -> list[RedactionRule]:
    """Load + parse a ``--rules`` JSON file. Empty list if path is None."""
    if rules_path is None:
        return []
    try:
        raw = json.loads(rules_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: failed to read --rules file: {exc}")
    if not isinstance(raw, list):
        raise SystemExit("error: --rules JSON must be a list of rule objects")
    out: list[RedactionRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise SystemExit(
                "error: each entry in --rules must be an object with "
                "'pattern' and 'replacement' keys"
            )
        try:
            out.append(RedactionRule.from_dict(entry))
        except ValueError as exc:
            raise SystemExit(f"error: invalid rule in --rules: {exc}")
    return out


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
    participants = _participants.list_participants(
        args.projects_root, args.project
    )

    speaker_maps: list[_speaker_map.SpeakerMap] = []
    for s in sources:
        try:
            speaker_maps.append(
                _speaker_map.load_speaker_map(
                    args.projects_root, args.project, s.id
                )
            )
        except FileNotFoundError:
            continue

    custom_rules = load_custom_rules(args.rules)

    segments_by_source_id: dict[str, list[dict]] = {}
    missing_transcripts = 0
    for s in sources:
        segs = _load_segments_for_source(args.outputs_root, s)
        if segs is None:
            missing_transcripts += 1
            continue
        segments_by_source_id[s.id] = segs

    bundle = build_anonymised_qdpx(
        project=project,
        sources=sources,
        codes=codes,
        applications=apps,
        memos=memos,
        coders=coders,
        participants=participants,
        speaker_maps=speaker_maps,
        segments_by_source_id=segments_by_source_id,
        custom_rules=custom_rules,
        note=args.note,
        origin=args.origin,
    )

    if args.out is None:
        sys.stdout.buffer.write(bundle.archive)
        sys.stdout.flush()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_bytes(bundle.archive)
        tmp.replace(args.out)
        print(
            f"Wrote anonymised QDPX archive ({len(bundle.archive):,} bytes) "
            f"→ {args.out}",
            file=sys.stderr,
        )
        print(
            f"  rules applied: {bundle.manifest['rule_count']}; "
            f"total substitutions: {bundle.manifest['total_substitutions']}",
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
