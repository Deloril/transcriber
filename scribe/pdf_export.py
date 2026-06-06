"""PDF export of a coded transcript — Word-style margin annotations.

Renders one source's transcript with each coded span highlighted in
its code's colour, plus a margin annotation that shows the code name,
its definition (one line), and the coder. Mirrors the on-screen
coding view's colour palette so a printed PDF reads as the same
document the researcher annotated.

The HTML→PDF route uses ``weasyprint``; the import is lazy because
weasyprint pulls in cairo/pango which (a) aren't installed on every
dev box, and (b) live behind system packages we don't want to demand
just to import this module's pure helpers.

Public surface:

* :func:`render_html` — pure helper that builds the HTML string.
  Takes plain dicts so unit tests don't need to spin up the full
  scribe.* dataclass stack. Tests target this.
* :func:`render_pdf_bytes` — wraps ``render_html`` + weasyprint.
  Imports weasyprint lazily so a missing system dep degrades to a
  clean :class:`PdfExportError` instead of an import-time crash.

Word-id format: ``s<seg_idx>w<word_idx>`` (matches the rest of the
codebase). Applications carry ``anchor_start_word_id`` and
``anchor_end_word_id`` in this shape; ``_parse_word_id`` cracks them
into ``(seg, word)`` integers for fast comparison against word
positions in segments.
"""

from __future__ import annotations

import html as html_mod
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


class PdfExportError(RuntimeError):
    """Raised when PDF rendering can't run.

    The two main causes are (a) weasyprint isn't installed (system
    deps missing on Linux, ``brew install pango cairo`` not done on
    macOS), and (b) the input data is malformed (we don't expect
    this in production but surfaces clearly during dev).
    """


_WORD_ID_RE = re.compile(r"^s(\d+)w(\d+)$")


def _parse_word_id(word_id: str | None) -> tuple[int, int] | None:
    """``s3w12`` → ``(3, 12)``. Returns ``None`` for unparseable inputs.

    We accept ``None`` and odd shapes silently because applications
    on legacy data sometimes carry empty anchors; the caller treats
    ``None`` as "this application has no usable anchor" and skips it.
    """
    if not word_id:
        return None
    m = _WORD_ID_RE.match(word_id)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #


def code_colour(code: Mapping[str, Any]) -> dict[str, str]:
    """Mirror the coding view's ``codeColours()`` JS in Python.

    A code with a user-set ``colour`` (hex like ``#7aa7ff``) wins;
    otherwise the FNV-1a hash → HSL hue roller (skipping the
    orange/red 10–25 band so highlights don't read as errors).

    Returns a dict ``{"bg": ..., "border": ..., "swatch": ...}``
    where every value is a CSS colour string. Tuned for printed
    output: lower-alpha backgrounds, full-saturation borders.
    """
    user = (code.get("colour") or "").strip() if isinstance(code, Mapping) else ""
    if user.startswith("#") and len(user) in (4, 7):
        rgb = _hex_to_rgb(user)
        if rgb is not None:
            r, g, b = rgb
            return {
                "bg": f"rgba({r}, {g}, {b}, 0.18)",
                "border": user,
                "swatch": user,
            }
    key = (code.get("id") or code.get("name") or "") if isinstance(code, Mapping) else ""
    h = 2_166_136_261
    for ch in str(key).encode("utf-8"):
        h ^= ch
        h = (h * 16_777_619) & 0xFFFFFFFF
    hue = (h % 335 + 25) % 360
    # On a white page we want stronger separation than the dark-theme
    # palette uses — bump saturation, drop lightness slightly.
    return {
        "bg": f"hsla({hue}, 70%, 78%, 0.55)",
        "border": f"hsl({hue}, 70%, 50%)",
        "swatch": f"hsl({hue}, 70%, 50%)",
    }


def _hex_to_rgb(s: str) -> tuple[int, int, int] | None:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Run-slicing — split a segment into ``coded`` and ``plain`` runs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Run:
    """One contiguous slice of a segment's words.

    ``application_ids`` is the set of applications that overlap this
    slice. Empty → plain prose; non-empty → highlighted with each
    code's colour stacked.
    """

    seg_idx: int
    start_word_idx: int
    end_word_idx: int  # inclusive
    text: str
    application_ids: tuple[str, ...]


def _segment_word_count(seg: Mapping[str, Any]) -> int:
    """Number of word-indices the segment exposes via word-ids.

    Real Scribe segments carry a ``words`` array; legacy segments
    sometimes only have ``text``. For the second shape we synthesise
    one virtual word that spans the whole segment so word-id-based
    application anchors still resolve.
    """
    words = seg.get("words") or []
    if isinstance(words, list) and words:
        return len(words)
    return 1


def _segment_word_text(seg: Mapping[str, Any], word_idx: int) -> str:
    words = seg.get("words") or []
    if isinstance(words, list) and 0 <= word_idx < len(words):
        w = words[word_idx]
        if isinstance(w, Mapping):
            return str(w.get("text") or w.get("word") or "")
    # Fallback for the synthesised single-word case.
    return str(seg.get("text") or "")


def split_segment_into_runs(
    seg: Mapping[str, Any],
    seg_idx: int,
    applications: Sequence[Mapping[str, Any]],
) -> list[_Run]:
    """Walk the segment's words, grouping consecutive words by their
    *application stack* (the set of applications that overlap each
    word).

    Returns a list of :class:`_Run` covering every word in the
    segment, with no gaps. Plain runs have ``application_ids == ()``;
    coded runs carry one or more ids in stable, sorted order.

    Two design notes worth pinning:

    * **Application bounds.** ``anchor_start_word_id`` and
      ``anchor_end_word_id`` are inclusive. A single-word
      application has equal start + end.
    * **Cross-segment applications.** An application can start in
      segment 4 and end in segment 6. For seg_idx in [4..6] we
      compute the per-segment word range it covers and treat that
      as the local span. Applications that don't touch this
      segment at all are filtered out before this function runs.
    """
    n = _segment_word_count(seg)
    if n <= 0:
        return []
    # Per-word stack of application ids.
    per_word: list[list[str]] = [[] for _ in range(n)]
    for app in applications:
        app_id = str(app.get("id") or "")
        if not app_id:
            continue
        start = _parse_word_id(app.get("anchor_start_word_id"))
        end = _parse_word_id(app.get("anchor_end_word_id"))
        if start is None or end is None:
            continue
        s_seg, s_word = start
        e_seg, e_word = end
        if s_seg > e_seg or (s_seg == e_seg and s_word > e_word):
            # Backwards anchor — engine bug or hand edit; skip.
            continue
        # Compute the [lo, hi] inclusive word range in *this* segment.
        if seg_idx < s_seg or seg_idx > e_seg:
            continue
        lo = s_word if seg_idx == s_seg else 0
        hi = e_word if seg_idx == e_seg else n - 1
        lo = max(0, lo)
        hi = min(n - 1, hi)
        for i in range(lo, hi + 1):
            per_word[i].append(app_id)

    # Group consecutive words by identical sorted-tuple of ids.
    runs: list[_Run] = []
    cur_key: tuple[str, ...] | None = None
    cur_lo = 0
    for i in range(n):
        key = tuple(sorted(per_word[i]))
        if cur_key is None:
            cur_key = key
            cur_lo = i
            continue
        if key != cur_key:
            runs.append(_make_run(seg, seg_idx, cur_lo, i - 1, cur_key))
            cur_key = key
            cur_lo = i
    if cur_key is not None:
        runs.append(_make_run(seg, seg_idx, cur_lo, n - 1, cur_key))
    return runs


def _make_run(
    seg: Mapping[str, Any], seg_idx: int,
    lo: int, hi: int, key: tuple[str, ...],
) -> _Run:
    text = " ".join(
        _segment_word_text(seg, i).strip() for i in range(lo, hi + 1)
    ).strip()
    return _Run(
        seg_idx=seg_idx,
        start_word_idx=lo,
        end_word_idx=hi,
        text=text,
        application_ids=key,
    )


# --------------------------------------------------------------------------- #
# Annotation packaging
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _AnnotationGroup:
    """One margin annotation tied to a segment.

    ``codes`` is the ordered list of code mappings (with colour) that
    apply to the run. ``coder_label`` is whichever coder the
    application records — Word-style margin annotations are scoped
    by user.
    """

    seg_idx: int
    run_idx: int   # which run within the segment
    codes: tuple[Mapping[str, Any], ...]
    coder_label: str
    quote: str


def _format_time(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "0:00"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #


_BASE_CSS = """
@page {
    size: A4;
    margin: 18mm 18mm 22mm 18mm;
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: "Helvetica Neue", "Helvetica", Arial, sans-serif;
        font-size: 9pt;
        color: #888;
    }
    @bottom-left {
        content: string(footer-title);
        font-family: "Helvetica Neue", "Helvetica", Arial, sans-serif;
        font-size: 9pt;
        color: #888;
    }
}
body {
    font-family: "Helvetica Neue", "Helvetica", Arial, sans-serif;
    color: #1d1d1d;
    font-size: 10.5pt;
    line-height: 1.5;
    margin: 0;
}
h1 {
    font-size: 19pt;
    font-weight: 600;
    margin: 0 0 4pt 0;
    string-set: footer-title content();
}
.subtitle { color: #555; font-size: 10pt; margin: 0 0 4pt 0; }
.metaline { color: #555; font-size: 9pt; margin: 0 0 16pt 0; }
.legend {
    border-top: 1px solid #ccc;
    border-bottom: 1px solid #ccc;
    padding: 8pt 0;
    margin: 0 0 18pt 0;
    page-break-inside: avoid;
}
.legend h2 {
    font-size: 11pt; font-weight: 600;
    margin: 0 0 6pt 0; color: #333;
}
.legend ul { list-style: none; margin: 0; padding: 0; }
.legend li {
    display: inline-block;
    font-size: 9pt;
    padding: 2pt 8pt 2pt 4pt;
    margin: 0 8pt 4pt 0;
    border-left: 3pt solid currentColor;
}
.legend .swatch {
    display: inline-block;
    width: 9pt; height: 9pt;
    margin-right: 5pt;
    border-radius: 2pt;
    vertical-align: middle;
}
/* Two-column layout: transcript on the left, margin annotations on
   the right. The right column starts with a vertical rule so it
   reads as a margin column rather than a body column. */
.body-grid {
    display: grid;
    grid-template-columns: 1fr 220pt;
    gap: 14pt;
    align-items: start;
}
.transcript .seg {
    display: grid;
    grid-template-columns: 60pt 1fr 18pt;
    gap: 6pt;
    margin: 0 0 8pt 0;
    page-break-inside: auto;
}
.transcript .seg .meta {
    color: #555;
    font-size: 8.5pt;
    padding-top: 1pt;
}
.transcript .seg .meta .speaker {
    color: #1d1d1d; font-weight: 600;
    display: block;
}
.transcript .seg .meta .ts {
    font-variant-numeric: tabular-nums;
    color: #777;
}
.transcript .seg .text { font-size: 10.5pt; }
.transcript .seg .lineno {
    font-size: 8pt;
    color: #aaa;
    text-align: right;
    font-variant-numeric: tabular-nums;
    padding-top: 1pt;
}
/* Coded inline runs — borrow the on-screen highlight shape: a
   subtle background + a stronger left border so the eye picks
   the start of the span. */
.coded {
    padding: 0 1pt;
    border-radius: 2pt;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}
.coded.callout {
    border-left-width: 2.5pt;
    border-left-style: solid;
    padding-left: 3pt;
}
/* Margin column. Each annotation pins to the segment that owns it
   via vertical position; weasyprint's grid layout keeps them
   roughly aligned even if the text column wraps. */
.margin .ann {
    border-left: 2.5pt solid currentColor;
    padding: 3pt 4pt 4pt 6pt;
    margin: 0 0 8pt 0;
    font-size: 8.5pt;
    color: #1d1d1d;
    background: #fafafa;
    page-break-inside: avoid;
}
.margin .ann .codes {
    font-weight: 600;
    margin-bottom: 1pt;
}
.margin .ann .codes .swatch {
    display: inline-block;
    width: 7pt; height: 7pt;
    margin-right: 3pt;
    border-radius: 2pt;
    vertical-align: middle;
}
.margin .ann .definition {
    color: #555;
    font-size: 8pt;
    margin: 1pt 0 2pt 0;
    font-style: italic;
}
.margin .ann .quote {
    color: #777;
    font-size: 8pt;
    border-top: 1px dotted #ccc;
    padding-top: 2pt;
    margin-top: 2pt;
}
.margin .ann .coder {
    color: #888; font-size: 7.5pt;
    margin-top: 1pt;
}
"""


def _esc(s: Any) -> str:
    return html_mod.escape(str(s if s is not None else ""), quote=True)


def render_html(
    *,
    project: Mapping[str, Any],
    source: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    applications: Sequence[Mapping[str, Any]],
    codes: Sequence[Mapping[str, Any]],
    coders: Sequence[Mapping[str, Any]] = (),
    speaker_names: Mapping[str, str] | None = None,
    now: str | None = None,
) -> str:
    """Build the full HTML document.

    The structure is one ``<section>`` per segment, with a parallel
    ``<aside>`` carrying its margin annotations. Inputs are plain
    dicts so tests don't have to drag in the dataclass machinery.
    """
    code_by_id: dict[str, Mapping[str, Any]] = {
        str(c.get("id") or ""): c for c in codes if isinstance(c, Mapping)
    }
    coder_by_id: dict[str, str] = {}
    for c in coders or ():
        if not isinstance(c, Mapping):
            continue
        cid = str(c.get("id") or "")
        if not cid:
            continue
        coder_by_id[cid] = str(c.get("name") or c.get("id") or "")

    # Pre-bucket applications by segment so the per-segment loop is
    # O(seg) instead of O(seg * apps).
    apps_by_seg: dict[int, list[Mapping[str, Any]]] = {}
    for app in applications:
        if not isinstance(app, Mapping):
            continue
        start = _parse_word_id(app.get("anchor_start_word_id"))
        end = _parse_word_id(app.get("anchor_end_word_id"))
        if start is None or end is None:
            continue
        s_seg, e_seg = start[0], end[0]
        for sidx in range(min(s_seg, e_seg), max(s_seg, e_seg) + 1):
            apps_by_seg.setdefault(sidx, []).append(app)

    speaker_names = speaker_names or {}

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<style>" + _BASE_CSS + "</style></head><body>")

    # Header.
    project_name = _esc(project.get("name") or "Project")
    source_name = _esc(source.get("name") or source.get("id") or "Transcript")
    parts.append(f"<h1>{source_name}</h1>")
    research_question = (project.get("research_question") or "").strip()
    if research_question:
        parts.append(
            f"<p class='subtitle'>{_esc(project_name)} — {_esc(research_question)}</p>"
        )
    else:
        parts.append(f"<p class='subtitle'>{_esc(project_name)}</p>")
    stamp = now or time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    parts.append(
        f"<p class='metaline'>Exported {_esc(stamp)} · "
        f"{len(segments)} segments · {len(applications)} coded spans · "
        f"{len(codes)} codes in codebook</p>"
    )

    # Legend — codes that actually appear in this source.
    used_code_ids = sorted({
        str(app.get("code_id") or "")
        for app in applications if isinstance(app, Mapping)
    })
    used_codes = [code_by_id[cid] for cid in used_code_ids if cid in code_by_id]
    if used_codes:
        parts.append("<div class='legend'><h2>Codes used in this transcript</h2><ul>")
        for c in used_codes:
            colours = code_colour(c)
            parts.append(
                f"<li style='color:{colours['border']}'>"
                f"<span class='swatch' style='background:{colours['swatch']}'></span>"
                f"<span style='color:#1d1d1d'>{_esc(c.get('name'))}</span>"
                f"</li>"
            )
        parts.append("</ul></div>")

    parts.append("<div class='body-grid'>")
    parts.append("<div class='transcript'>")

    annotations_html: list[str] = ["<div class='margin'>"]

    for seg_idx, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            continue
        sp = str(seg.get("speaker") or "")
        sp_label = speaker_names.get(sp, sp) or "—"
        # Some legacy transcripts carry non-numeric ``start`` values
        # (or none at all). Coerce defensively — a malformed segment
        # shouldn't kill the whole export.
        try:
            start_secs = float(seg.get("start") or 0.0)
        except (TypeError, ValueError):
            start_secs = 0.0
        ts = _format_time(start_secs)
        seg_apps = apps_by_seg.get(seg_idx, ())
        runs = split_segment_into_runs(seg, seg_idx, seg_apps)

        text_html: list[str] = []
        for run in runs:
            run_text = _esc(run.text)
            if not run.application_ids:
                text_html.append(run_text)
                continue
            # Multiple overlapping codes → use the first code's colour
            # for the highlight + show every overlapping code in the
            # margin annotation. Most transcripts have at most one
            # code per span, but the data model allows stacks so we
            # don't crash on them.
            first_code = code_by_id.get(_app_code_id(seg_apps, run.application_ids[0])) or {}
            colours = code_colour(first_code)
            text_html.append(
                f"<span class='coded callout' "
                f"style='background:{colours['bg']};border-color:{colours['border']}'>"
                f"{run_text}</span>"
            )

        parts.append(
            "<section class='seg'>"
            "<div class='meta'>"
            f"<span class='speaker'>{_esc(sp_label)}</span>"
            f"<span class='ts'>{_esc(ts)}</span>"
            "</div>"
            f"<div class='text'>{' '.join(text_html)}</div>"
            f"<div class='lineno'>{seg_idx + 1}</div>"
            "</section>"
        )

        # One annotation per coded run. Walk the runs in render order
        # so the margin annotations cluster near their text.
        for run_idx, run in enumerate(runs):
            if not run.application_ids:
                continue
            run_codes: list[Mapping[str, Any]] = []
            run_coder_labels: list[str] = []
            for app_id in run.application_ids:
                app = _find_app(seg_apps, app_id)
                if app is None:
                    continue
                code = code_by_id.get(str(app.get("code_id") or ""))
                if code is None:
                    continue
                run_codes.append(code)
                coder_id = str(app.get("coder_id") or "")
                run_coder_labels.append(coder_by_id.get(coder_id, coder_id))
            if not run_codes:
                continue
            # Margin annotation HTML.
            colours = code_colour(run_codes[0])
            code_names = "".join(
                f"<span><span class='swatch' "
                f"style='background:{code_colour(c)['swatch']}'></span>"
                f"{_esc(c.get('name'))}</span> "
                for c in run_codes
            )
            definition = (run_codes[0].get("definition") or "").strip()
            coder_str = ", ".join(c for c in run_coder_labels if c) or "(unknown coder)"
            annotations_html.append(
                f"<div class='ann' style='color:{colours['border']}'>"
                f"<div class='codes'>{code_names}</div>"
                + (f"<div class='definition'>{_esc(definition)}</div>" if definition else "")
                + f"<div class='quote'>“{_esc(run.text)}”</div>"
                f"<div class='coder'>by {_esc(coder_str)}</div>"
                "</div>"
            )

    parts.append("</div>")  # transcript
    annotations_html.append("</div>")
    parts.extend(annotations_html)
    parts.append("</div>")  # body-grid
    parts.append("</body></html>")
    return "".join(parts)


def _find_app(apps: Iterable[Mapping[str, Any]], app_id: str) -> Mapping[str, Any] | None:
    for a in apps:
        if str(a.get("id") or "") == app_id:
            return a
    return None


def _app_code_id(apps: Iterable[Mapping[str, Any]], app_id: str) -> str:
    a = _find_app(apps, app_id)
    return str(a.get("code_id") or "") if a else ""


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def render_pdf_bytes(html: str) -> bytes:
    """Convert HTML to PDF via weasyprint.

    Imported lazily so the module's pure helpers (and their tests)
    don't require the system Cairo/Pango libraries. Surfaces a
    clear :class:`PdfExportError` if the import fails.
    """
    try:
        # weasyprint also depends on libpango / libcairo at runtime;
        # the import succeeds when the wheels are present but the
        # write may still fail later if those system libs are
        # missing. We catch both at the boundary.
        from weasyprint import HTML  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise PdfExportError(
            "weasyprint isn't available. Install it with the wheel + "
            "system dependencies (Linux: apt-get install "
            "libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libffi-dev; "
            f"macOS: brew install pango cairo). Underlying: {e}"
        ) from e
    try:
        return HTML(string=html).write_pdf()
    except Exception as e:  # noqa: BLE001
        raise PdfExportError(
            f"weasyprint failed to render PDF: {e}"
        ) from e


def render_pdf_for_source(
    *,
    project: Mapping[str, Any],
    source: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    applications: Sequence[Mapping[str, Any]],
    codes: Sequence[Mapping[str, Any]],
    coders: Sequence[Mapping[str, Any]] = (),
    speaker_names: Mapping[str, str] | None = None,
    now: str | None = None,
) -> bytes:
    """End-to-end: build HTML + render to PDF bytes."""
    html = render_html(
        project=project,
        source=source,
        segments=segments,
        applications=applications,
        codes=codes,
        coders=coders,
        speaker_names=speaker_names,
        now=now,
    )
    return render_pdf_bytes(html)
