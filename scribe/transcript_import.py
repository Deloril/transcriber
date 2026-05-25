"""Transcript import (F10.3) — pure parsers for already-finished
transcripts in the four shapes researchers actually have on disk:

1. **Plain text + speaker labels** — what we export as ``.txt``::

       [00:01] LUKE: Hello there.

       [00:03] GUEST: Hi back.

2. **SRT subtitle files** — numbered cues with ``HH:MM:SS,mmm``
   timestamps. Speaker prefixes (``LUKE: ...``) are honoured when
   present; otherwise the cue uses the default speaker.
3. **WebVTT subtitle files** — same idea as SRT but with ``.``
   instead of ``,`` for the millisecond separator and an optional
   ``<v Speaker>`` voice tag we recognise as the speaker label.
4. **Scribe JSON** — the same shape :class:`TranscriptionResult.to_dict`
   produces. Word-level timestamps come through verbatim if present.

The parsers are deliberately framework-free. They take a string
(file content) and return a normalised dict in the same shape as
``TranscriptionResult.to_dict()``::

    {
        "language": "en",
        "mode": "diarize",            # imported transcripts are
        "speakers": ["LUKE", "GUEST"], # treated as a single track
        "segments": [
            {
                "text": "Hello there.",
                "start": 1.0,
                "end": 3.0,
                "speaker": "LUKE",
                "words": [
                    {"text": "Hello", "start": 1.0, "end": 2.0,
                     "speaker": "LUKE", "score": null},
                    {"text": "there.", "start": 2.0, "end": 3.0,
                     "speaker": "LUKE", "score": null},
                ],
            },
            ...
        ],
    }

Word-level timestamps are *synthesised* when the source format
doesn't carry them (TXT, SRT, VTT) by spreading the segment's
duration evenly across its tokens — the same fallback the editor
uses after a manual edit (`spreadTokensAcrossSpan` in
``helpers.mjs``). Scribe JSON files keep their existing word
timings if they have any.

The server thinly wraps these parsers behind ``POST /api/import``;
nothing in this module touches the filesystem or the ``Job``
runtime.
"""

from __future__ import annotations

import json
import re
from typing import Any


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #

#: Recognised import formats.
KNOWN_FORMATS: tuple[str, ...] = ("scribe-json", "srt", "vtt", "txt")


def _strip_bom(text: str) -> str:
    """Drop a leading UTF-8 BOM if present.

    Subtitle files and JSON exports from Windows tools sometimes
    carry one; downstream parsers don't expect it.
    """
    if text.startswith("﻿"):
        return text[1:]
    return text


def sniff_format(filename: str | None, content: str) -> str:
    """Return the best guess at the import format for ``content``.

    Strategy: extension first (it's authoritative when present),
    content-shape sniff as a fallback. ``txt`` is the catch-all
    when nothing else matches — the plain-text parser is permissive
    enough to handle "no timestamps, no speaker labels" gracefully.
    """
    head = _strip_bom(content).lstrip()

    # Extension wins when it's a known one.
    if filename:
        lower = filename.lower()
        if lower.endswith(".srt"):
            return "srt"
        if lower.endswith(".vtt"):
            return "vtt"
        if lower.endswith(".json"):
            return "scribe-json"
        if lower.endswith(".txt"):
            # Still need to confirm — a Scribe JSON saved as ``.txt``
            # by a confused user should still parse correctly.
            if head.startswith("{"):
                return "scribe-json"
            if head.upper().startswith("WEBVTT"):
                return "vtt"
            return "txt"

    # Content sniff for the no-extension case.
    if head.startswith("{"):
        return "scribe-json"
    if head.upper().startswith("WEBVTT"):
        return "vtt"
    # SRT files start with a numeric cue index on its own line; we
    # confirm that by looking for the timestamp arrow on the next.
    first_lines = head.splitlines()[:4]
    if (
        len(first_lines) >= 2
        and first_lines[0].strip().isdigit()
        and "-->" in first_lines[1]
    ):
        return "srt"
    return "txt"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

#: Cap the number of segments any single import can produce.  A
#: maliciously huge VTT could otherwise consume unbounded memory at
#: parse time. 100 000 cues at ~5 s each is ≈ 138 hours of audio,
#: well past anything a researcher would import.
MAX_SEGMENTS = 100_000

#: Cap any single segment's text length so the resulting page is
#: navigable. The longest line in a normal interview transcript is
#: a few hundred characters; 50 000 is paranoia, not a real limit.
MAX_TEXT_LEN = 50_000

# Speaker prefix at the start of a line: ``LUKE:``, ``Guest 2:``,
# ``Speaker A:``. We accept letters, digits, spaces, hyphens and a
# couple of other low-risk separators; the colon is required so we
# don't mistake the first capitalised word of the sentence for a
# speaker label.  Bounded to 64 chars on the speaker side.
_SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z][\w \-.'_/&]{0,63}):\s+(.*)$")

# Optional ``[HH:MM:SS]`` or ``[MM:SS]`` clock prefix in plain-text
# transcripts. We tolerate either bracket style; the format we
# *write* is square-bracketed so that's the canonical one.
_TIME_PREFIX_RE = re.compile(
    r"^\s*[\[(]"
    r"(?:(\d{1,2}):)?"           # optional H
    r"(\d{1,2}):(\d{2})"          # M:SS
    r"(?:[.,](\d{1,3}))?"        # optional .mmm
    r"[\])]\s*"
)


def _parse_clock_prefix(line: str) -> tuple[float | None, str]:
    """If ``line`` begins with ``[hh:mm:ss]``-style timestamp, peel
    it off and return ``(seconds, remainder)``. Otherwise return
    ``(None, line)``.
    """
    m = _TIME_PREFIX_RE.match(line)
    if not m:
        return None, line
    h = int(m.group(1) or 0)
    mn = int(m.group(2))
    s = int(m.group(3))
    ms_raw = m.group(4) or "0"
    # Pad/truncate so ".5" parses as 500 ms, ".50" as 500 ms.
    ms = int((ms_raw + "000")[:3])
    secs = h * 3600 + mn * 60 + s + ms / 1000.0
    return secs, line[m.end():]


def _parse_speaker_prefix(line: str) -> tuple[str | None, str]:
    """If ``line`` begins with ``SPEAKER: ...`` peel it off."""
    m = _SPEAKER_PREFIX_RE.match(line)
    if not m:
        return None, line
    return m.group(1).strip(), m.group(2)


def _parse_clock_str(s: str) -> float:
    """Parse ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm`` (or shorter).

    Used by both the SRT and VTT cue parsers; raising ValueError
    lets the dispatcher surface a clean 400.
    """
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 2:
        h = 0
        m, rest = parts
    elif len(parts) == 3:
        h, m, rest = parts
    else:
        raise ValueError(f"Invalid timestamp: {s!r}")
    if "." in rest:
        sec, ms = rest.split(".", 1)
    else:
        sec, ms = rest, "0"
    ms = (ms + "000")[:3]
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def _tokenise(text: str) -> list[str]:
    """Split a segment's text into space-separated tokens.

    Mirrors ``spreadTokensAcrossSpan`` (helpers.mjs): the function
    that takes ``"Hello world"`` and stretches it across the
    segment's [start, end) interval. Empty tokens are dropped.
    """
    return [t for t in text.strip().split() if t]


def _spread_words(
    text: str, start: float, end: float, speaker: str
) -> list[dict[str, Any]]:
    """Synthesise word-level timestamps by spreading the tokens
    evenly across the [start, end) interval.

    Mirrors the JS helper of the same shape (intentionally —
    transcript edits in the editor synthesise word timings the same
    way, so an imported transcript and a hand-edited one feel
    identical to downstream code).
    """
    tokens = _tokenise(text)
    if not tokens:
        return []
    span = max(0.05, end - start)
    per = span / len(tokens)
    return [
        {
            "text": tok,
            "start": round(start + i * per, 6),
            "end": round(start + (i + 1) * per, 6),
            "speaker": speaker,
            "score": None,
        }
        for i, tok in enumerate(tokens)
    ]


def _segment(
    text: str, start: float, end: float, speaker: str
) -> dict[str, Any]:
    text = (text or "").strip()
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]
    if end <= start:
        # Subtitle authors sometimes write zero- or negative-length
        # cues. We bump the end by a tiny ε so word spreading has a
        # non-zero span to work with; the editor's resync helper
        # does the same.
        end = start + 0.05
    return {
        "text": text,
        "start": round(float(start), 6),
        "end": round(float(end), 6),
        "speaker": speaker or "Speaker 1",
        "words": _spread_words(text, start, end, speaker or "Speaker 1"),
    }


def _collect_speakers(segments: list[dict[str, Any]]) -> list[str]:
    """Return distinct speaker labels, in first-occurrence order."""
    seen: list[str] = []
    for s in segments:
        sp = s.get("speaker")
        if isinstance(sp, str) and sp and sp not in seen:
            seen.append(sp)
    return seen


def _finalise(
    segments: list[dict[str, Any]],
    *,
    language: str = "en",
    mode: str = "diarize",
    speakers: list[str] | None = None,
) -> dict[str, Any]:
    """Wrap a list of segments in the standard result envelope."""
    if len(segments) > MAX_SEGMENTS:
        raise ValueError(
            f"Imported transcript has {len(segments)} segments; "
            f"max is {MAX_SEGMENTS}."
        )
    sps = speakers if speakers is not None else _collect_speakers(segments)
    return {
        "language": language,
        "mode": mode,
        "speakers": sps,
        "segments": segments,
    }


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #


def parse_txt(content: str) -> dict[str, Any]:
    """Parse plain-text transcripts of the shape Scribe writes::

        [00:01] LUKE: Hello there.

        [00:03] GUEST: Hi back.

    Tolerant: missing timestamps, missing speaker labels, blank
    lines between or inside paragraphs are all handled. When no
    timestamps are present at all, segments are placed back-to-back
    starting from 0.0 with a 4-second guess per segment so the
    editor has *something* to anchor to; the user can fix the
    timing afterwards by re-syncing against an audio file or just
    trusting the order.

    Each non-empty paragraph (separated by a blank line) becomes a
    single segment. Within a paragraph we honour optional ``[time]``
    and ``SPEAKER:`` prefixes.
    """
    text = _strip_bom(content).replace("\r\n", "\n").replace("\r", "\n")
    # Split paragraphs on one-or-more blank lines.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    segments: list[dict[str, Any]] = []
    last_speaker: str | None = None
    cursor = 0.0
    DEFAULT_SEG_DUR = 4.0

    for para in paragraphs:
        # Collapse newlines inside a paragraph; the writer doesn't
        # produce them but pasted-in text might.
        first_line, *rest = para.split("\n", 1)
        body = first_line + (("\n" + rest[0]) if rest else "")

        secs, remainder = _parse_clock_prefix(body.strip())
        speaker, remainder = _parse_speaker_prefix(remainder)
        # Re-split on newlines now that we've peeled the prefixes
        # off.  Multi-line paragraphs (rare) get joined with a
        # single space.
        body_text = " ".join(t.strip() for t in remainder.splitlines() if t.strip())
        if not body_text:
            continue

        if secs is None:
            secs = cursor
        speaker = speaker or last_speaker or "Speaker 1"
        last_speaker = speaker

        # Tentative end: bump for the next segment in the same
        # paragraph chain. Will get fixed up below once we know the
        # next start.
        seg = _segment(body_text, secs, secs + DEFAULT_SEG_DUR, speaker)
        segments.append(seg)
        cursor = secs + DEFAULT_SEG_DUR

    # Stitch end-of-segment to the next start so playback is
    # contiguous when timestamps are present.
    for i in range(len(segments) - 1):
        nxt_start = segments[i + 1]["start"]
        if nxt_start > segments[i]["start"]:
            segments[i] = _segment(
                segments[i]["text"],
                segments[i]["start"],
                nxt_start,
                segments[i]["speaker"],
            )

    return _finalise(segments)


# --------------------------------------------------------------------------- #
# SRT
# --------------------------------------------------------------------------- #


_SRT_TS_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_srt(content: str) -> dict[str, Any]:
    """Parse SRT subtitle files into the standard envelope.

    Speaker labels at the start of a cue (``LUKE: hello``) are
    pulled out and dropped from the visible text. Voice tags from
    SRT-extended dialects (``<v Speaker>``) are also honoured for
    leniency.
    """
    text = _strip_bom(content).replace("\r\n", "\n").replace("\r", "\n")
    # Cues are separated by blank lines.
    blocks = re.split(r"\n\s*\n", text)

    segments: list[dict[str, Any]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        # Drop the optional cue index on line 0.
        if lines and lines[0].strip().isdigit() and len(lines) >= 2:
            lines = lines[1:]
        if not lines:
            continue
        m = _SRT_TS_RE.search(lines[0])
        if not m:
            continue
        start = _parse_clock_str(m.group(1))
        end = _parse_clock_str(m.group(2))
        body_lines = lines[1:]
        body = "\n".join(body_lines).strip()
        body = _strip_voice_tag_speaker(body)
        speaker, body = _split_speaker_inline(body)
        # Collapse internal newlines — SRT line breaks are display
        # hints, not structural.
        body = " ".join(t.strip() for t in body.splitlines() if t.strip())
        if not body:
            continue
        segments.append(_segment(body, start, end, speaker or "Speaker 1"))

    return _finalise(segments)


# --------------------------------------------------------------------------- #
# WebVTT
# --------------------------------------------------------------------------- #


_VTT_TS_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}\.\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{1,3})"
)
_VOICE_TAG_RE = re.compile(r"<v\s+([^>]+?)>(.*?)(?=<v\s|$)", re.DOTALL)
_VOICE_TAG_OPEN_RE = re.compile(r"<v\s+([^>]+?)>")


def _strip_voice_tag_speaker(body: str) -> str:
    """Replace ``<v Name>text</v>`` (or unclosed ``<v Name>text``)
    with ``Name: text`` so the speaker prefix path picks it up.
    Multiple voice tags within the same cue concatenate."""
    if "<v " not in body:
        return body
    out_parts: list[str] = []
    matches = list(_VOICE_TAG_OPEN_RE.finditer(body))
    if not matches:
        return body
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        seg_start = m.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[seg_start:seg_end]
        # Drop trailing </v> if present.
        chunk = re.sub(r"</v>", "", chunk).strip()
        if chunk:
            out_parts.append(f"{speaker}: {chunk}")
    return "\n".join(out_parts) if out_parts else body


def _split_speaker_inline(body: str) -> tuple[str | None, str]:
    """Like :func:`_parse_speaker_prefix` but works on multi-line
    cue bodies — we only check the first line."""
    if not body:
        return None, body
    first, *rest = body.split("\n", 1)
    speaker, remainder = _parse_speaker_prefix(first)
    if speaker is None:
        return None, body
    if rest:
        return speaker, remainder + "\n" + rest[0]
    return speaker, remainder


def parse_vtt(content: str) -> dict[str, Any]:
    """Parse WebVTT subtitle files."""
    text = _strip_bom(content).replace("\r\n", "\n").replace("\r", "\n")
    if not text.lstrip().upper().startswith("WEBVTT"):
        raise ValueError("Not a WebVTT file (missing WEBVTT signature).")
    blocks = re.split(r"\n\s*\n", text)
    # First block is the WEBVTT header (and any STYLE/REGION blocks
    # we don't care about); skip those.
    segments: list[dict[str, Any]] = []
    for block in blocks:
        block = block.strip()
        if not block or block.upper().startswith("WEBVTT"):
            continue
        if block.startswith(("NOTE", "STYLE", "REGION")):
            continue
        lines = block.splitlines()
        # Optional cue identifier on line 0.
        if lines and "-->" not in lines[0]:
            lines = lines[1:]
        if not lines:
            continue
        m = _VTT_TS_RE.search(lines[0])
        if not m:
            continue
        start = _parse_clock_str(m.group(1))
        end = _parse_clock_str(m.group(2))
        body = "\n".join(lines[1:]).strip()
        body = _strip_voice_tag_speaker(body)
        speaker, body = _split_speaker_inline(body)
        body = " ".join(t.strip() for t in body.splitlines() if t.strip())
        if not body:
            continue
        segments.append(_segment(body, start, end, speaker or "Speaker 1"))
    return _finalise(segments)


# --------------------------------------------------------------------------- #
# Scribe JSON
# --------------------------------------------------------------------------- #


def parse_scribe_json(content: str) -> dict[str, Any]:
    """Parse a Scribe-shaped JSON export back into the standard
    envelope. Word-level timestamps survive verbatim if present;
    otherwise they're synthesised the same way as for TXT/SRT/VTT.
    """
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg} at line {e.lineno}")
    if not isinstance(raw, dict):
        raise ValueError("Scribe JSON must be an object at the top level.")
    raw_segs = raw.get("segments")
    if not isinstance(raw_segs, list):
        raise ValueError("Scribe JSON missing segments array.")
    if len(raw_segs) > MAX_SEGMENTS:
        raise ValueError(
            f"Scribe JSON has {len(raw_segs)} segments; max is {MAX_SEGMENTS}."
        )

    language = raw.get("language") if isinstance(raw.get("language"), str) else "en"
    mode_raw = raw.get("mode")
    mode = mode_raw if mode_raw in ("multi-track", "diarize") else "diarize"

    segments: list[dict[str, Any]] = []
    for s in raw_segs:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(s.get("start") or 0.0)
            end = float(s.get("end") or start)
        except (TypeError, ValueError):
            continue
        speaker = str(s.get("speaker") or "Speaker 1") or "Speaker 1"
        words_raw = s.get("words")
        words: list[dict[str, Any]] = []
        if isinstance(words_raw, list) and words_raw:
            for w in words_raw:
                if not isinstance(w, dict):
                    continue
                wtxt = str(w.get("text") or "").strip()
                if not wtxt:
                    continue
                try:
                    ws = float(w.get("start") or start)
                    we = float(w.get("end") or ws)
                except (TypeError, ValueError):
                    continue
                wsp = str(w.get("speaker") or speaker) or speaker
                score = w.get("score")
                if not isinstance(score, (int, float)):
                    score = None
                words.append({
                    "text": wtxt[:MAX_TEXT_LEN],
                    "start": round(float(ws), 6),
                    "end": round(float(we), 6),
                    "speaker": wsp,
                    "score": score,
                })
        if words:
            seg = {
                "text": text[:MAX_TEXT_LEN],
                "start": round(float(start), 6),
                "end": round(float(end if end > start else start + 0.05), 6),
                "speaker": speaker,
                "words": words,
            }
        else:
            seg = _segment(text, start, end, speaker)
        segments.append(seg)

    speakers_raw = raw.get("speakers")
    speakers: list[str] | None = None
    if isinstance(speakers_raw, list):
        speakers = [str(x) for x in speakers_raw if isinstance(x, str) and x.strip()]
        if not speakers:
            speakers = None
    return _finalise(segments, language=language, mode=mode, speakers=speakers)


# --------------------------------------------------------------------------- #
# Top-level dispatch
# --------------------------------------------------------------------------- #


def parse_transcript(
    filename: str | None, content: str, *, fmt: str | None = None
) -> dict[str, Any]:
    """Dispatch ``content`` to the right parser.

    ``fmt`` overrides format detection when supplied; otherwise we
    sniff from the filename + content. Returns the standard
    envelope; raises :class:`ValueError` on an unrecoverable parse
    error so the caller can surface a clean 400.
    """
    if fmt is None:
        fmt = sniff_format(filename, content)
    if fmt not in KNOWN_FORMATS:
        raise ValueError(f"Unknown transcript format: {fmt!r}")
    if fmt == "scribe-json":
        out = parse_scribe_json(content)
    elif fmt == "srt":
        out = parse_srt(content)
    elif fmt == "vtt":
        out = parse_vtt(content)
    else:
        out = parse_txt(content)
    if not out.get("segments"):
        raise ValueError(
            "No transcript segments could be parsed from this file."
        )
    return out
