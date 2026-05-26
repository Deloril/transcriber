"""Library view (F10.1) — pure helpers that summarise persisted
transcription jobs into row-shaped dicts for the home-page library
list.

Today, every transcription job creates ``outputs/<job_id>/`` and the
only way to get back to one is the ``/edit/<job_id>`` URL the user
happens to remember. F10.1 closes that gap: ``GET /api/jobs`` (wired
in :mod:`scribe.server`) calls :func:`summarise_jobs` to return a
light-weight description of every persisted job — filename, duration,
mode, speaker count, language, created date, status — so the UI can
render a sortable, searchable list.

The module is deliberately stand-alone: no FastAPI, no engine
imports. Anything that touches the filesystem belongs in
:mod:`scribe.server`; this module just reduces a Job-shaped object
(or its on-disk dict) to the fields a row needs.

Conventions:

* Inputs are tolerant. We accept either a live :class:`Job`-shaped
  object (anything with ``to_state()``) or a plain dict in the same
  shape — that lets tests build rows without spinning the FastAPI
  app up, and lets the loader pass freshly-parsed ``job.json`` data
  straight in.
* Outputs are normalised. Every row has the same keys, even when the
  underlying job is missing fields; the UI then doesn't have to
  defensively check for ``undefined``. Length-bounded so a
  hand-edited ``job.json`` with absurdly long fields can't blow the
  page up.
"""

from __future__ import annotations

from typing import Any, Iterable


# Length cap on any free-form text field we surface. Generous, but
# bounded — matches the spirit of the project-entity validators.
_FIELD_CAP = 4000


# Values that the rest of the codebase writes for the ``status`` field.
# Repeated here so consumers can use it without importing from server.
JOB_STATUSES: tuple[str, ...] = ("queued", "running", "done", "error")


# Strings users / hand-edited config / older serialisers might write for a
# falsy boolean. Plain ``bool("false")`` returns ``True`` because the string
# is non-empty — exactly the bug we hit on a real user's library where rows
# rendered as "media discarded" even though the source media was still on
# disk. Any other truthy *string* (e.g. "true", "yes", "1") still goes
# through ``bool()`` so we don't change behaviour for those.
_FALSY_STRINGS = {"", "false", "no", "0", "off", "none", "null"}


def _to_bool(value: Any) -> bool:
    """Coerce arbitrary persisted values into a Python bool.

    Handles the ``"false"``-string footgun: ``bool("false")`` is True.
    For anything that *looks* like a stringified false (case-insensitive,
    whitespace-trimmed) we return False. Everything else falls through
    to Python's normal truthiness rules, so ``1``, ``"yes"``, ``[1]`` —
    anything genuinely truthy — still resolves True.
    """
    if isinstance(value, str):
        if value.strip().lower() in _FALSY_STRINGS:
            return False
    return bool(value)


def _job_state(job: Any) -> dict[str, Any]:
    """Coerce a Job-shaped input into a plain dict.

    Accepts either an object with a ``to_state()`` method (the live
    :class:`scribe.server.Job`) or a plain dict (the on-disk shape).
    Anything else raises :class:`TypeError` — silent type-coercion
    here would hide real bugs, since the rest of this module is only
    safe on dict-shaped data.
    """
    if hasattr(job, "to_state") and callable(job.to_state):
        d = job.to_state()
    elif isinstance(job, dict):
        d = job
    else:
        raise TypeError(
            f"summarise_job: expected Job or dict, got {type(job).__name__}"
        )
    if not isinstance(d, dict):
        raise TypeError("summarise_job: to_state() did not return a dict")
    return d


def _str_field(d: dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    if v is None:
        return ""
    s = str(v)
    return s[:_FIELD_CAP]


def _coerce_speakers(result: Any) -> list[str]:
    """Pull a clean list of speaker labels out of a ``result`` dict.

    The on-disk shape is ``result["speakers"]`` (plural) — written
    by :class:`TranscriptionResult.to_dict` or by the editor's PUT
    pass-through of the edited transcript JSON.

    The editor lets the user rename a speaker without changing its
    canonical id: e.g. SPEAKER_00 stays SPEAKER_00 in
    ``result["speakers"]`` but the user-visible label is mapped via
    ``result["speaker_names"]`` (``{"SPEAKER_00": "Luke"}``). The
    library row should show the *renamed* label so a user who
    relabels their speakers in the editor sees those names in the
    list, not the raw model-assigned ids. Anything in
    ``speaker_names`` overrides the matching id; missing entries
    fall through to the canonical id.

    Falls back to the empty list if missing or malformed; never raises.
    """
    if not isinstance(result, dict):
        return []
    raw = result.get("speakers")
    if not isinstance(raw, list):
        return []
    name_map = result.get("speaker_names")
    if not isinstance(name_map, dict):
        name_map = {}
    out: list[str] = []
    for s in raw:
        if not isinstance(s, str) or not s.strip():
            continue
        canonical = s.strip()
        # Prefer the user-renamed label when it's a non-empty string.
        renamed = name_map.get(canonical)
        if isinstance(renamed, str) and renamed.strip():
            out.append(renamed.strip()[:_FIELD_CAP])
        else:
            out.append(canonical[:_FIELD_CAP])
    return out


def _duration_from_result(result: Any) -> float | None:
    """Compute duration as ``max(segment.end)`` across all segments.

    Returns ``None`` if no segments are available — the UI then
    shows ``—`` instead of ``0:00``.
    """
    if not isinstance(result, dict):
        return None
    segs = result.get("segments")
    if not isinstance(segs, list) or not segs:
        return None
    end = 0.0
    for s in segs:
        if not isinstance(s, dict):
            continue
        v = s.get("end")
        if isinstance(v, (int, float)) and v > end:
            end = float(v)
    return end if end > 0 else None


def _resolve_language(d: dict[str, Any], result: Any) -> str:
    """Prefer the *detected* language from the result over the
    *requested* language from the form, since ``en`` plus
    auto-detect is a common combo and the detected value is what
    the user actually got."""
    if isinstance(result, dict):
        lang = result.get("language")
        if isinstance(lang, str) and lang.strip():
            return lang.strip()[:_FIELD_CAP]
    return _str_field(d, "language", "")


def summarise_job(job: Any) -> dict[str, Any]:
    """Return the row-shaped dict for a single job.

    The returned dict has *exactly* these keys:

    ``id``                — the 12-hex-char job id.
    ``input_filename``    — original uploaded filename (display only).
    ``mode``              — ``auto`` / ``multi-track`` / ``diarize``.
    ``language``          — detected (preferred) or requested language.
    ``model``             — Whisper / Parakeet model id used.
    ``status``            — ``queued`` / ``running`` / ``done`` / ``error``.
    ``progress``          — 0.0..1.0; useful for in-progress rows.
    ``message``           — last status message from the worker.
    ``created_at``        — ISO timestamp string (sort key).
    ``started_at``        — epoch seconds (or None) when worker began.
    ``finished_at``       — epoch seconds (or None) when worker stopped.
    ``audio_streams``     — number of audio tracks detected at upload.
    ``speakers``          — list of speaker labels from the result.
    ``speaker_count``     — ``len(speakers)``; convenience for sorting.
    ``duration_seconds``  — derived from segments; None if not done.
    ``has_outputs``       — True when at least one sidecar was written.
    ``media_discarded``   — True when the user reclaimed disk space by
                            dropping the source recording (F10.2). The
                            transcript is still readable; the library
                            row renders a small "media discarded" icon
                            and the editor degrades to a no-playback
                            mode.
    ``error``             — short error message; None on a healthy row.

    The shape is stable: subsequent F10.x features layer on by adding
    *new* keys, never by mutating these.
    """
    d = _job_state(job)
    result = d.get("result") or None
    speakers = _coerce_speakers(result)
    return {
        "id": _str_field(d, "id"),
        "input_filename": _str_field(d, "input_filename"),
        "mode": _str_field(d, "mode"),
        "language": _resolve_language(d, result),
        "model": _str_field(d, "model"),
        "status": _str_field(d, "status", "done") or "done",
        "progress": float(d.get("progress") or 0.0),
        "message": _str_field(d, "message"),
        "created_at": _str_field(d, "created_at"),
        "started_at": d.get("started_at"),
        "finished_at": d.get("finished_at"),
        "audio_streams": int(d.get("audio_streams") or 0),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "duration_seconds": _duration_from_result(result),
        "has_outputs": bool(d.get("output_paths")),
        "media_discarded": _to_bool(d.get("media_discarded", False)),
        "error": d.get("error"),
    }


def summarise_jobs(jobs: Iterable[Any]) -> list[dict[str, Any]]:
    """Summarise a sequence of jobs, sorted by ``created_at`` desc
    (newest first). Jobs missing a ``created_at`` sink to the
    bottom; ties on the timestamp break by ``id`` descending so the
    output is deterministic.
    """
    rows = [summarise_job(j) for j in jobs]
    # Two groups: rows with a created_at and rows without. Within
    # the timestamped group we want newest first; the bare group
    # sorts by id descending so the output is stable across runs.
    has_created = [r for r in rows if r.get("created_at")]
    no_created = [r for r in rows if not r.get("created_at")]
    has_created.sort(
        key=lambda r: (r.get("created_at") or "", r.get("id") or ""),
        reverse=True,
    )
    no_created.sort(key=lambda r: r.get("id") or "", reverse=True)
    return has_created + no_created


def matches_query(row: dict[str, Any], query: str) -> bool:
    """Case-insensitive substring match across filename + speakers +
    mode + status + language + model. Empty query matches
    everything.

    Used as the server-side filter for ``GET /api/jobs?q=...``; the
    client mirrors the same logic via :func:`searchLibraryRows` in
    ``helpers.mjs`` so live filtering works without a round-trip.
    """
    q = (query or "").strip().lower()
    if not q:
        return True
    parts: list[str] = [
        str(row.get("input_filename") or ""),
        str(row.get("status") or ""),
        str(row.get("mode") or ""),
        str(row.get("language") or ""),
        str(row.get("model") or ""),
        " ".join(row.get("speakers") or []),
    ]
    haystack = " ".join(parts).lower()
    return q in haystack


def filter_rows(rows: Iterable[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Keep rows that match ``query``; preserves input order so the
    caller controls sort."""
    return [r for r in rows if matches_query(r, query)]
