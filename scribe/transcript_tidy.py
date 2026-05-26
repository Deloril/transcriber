"""Transcript tidy-up — group consecutive same-speaker segments,
hand the run to a local LLM for grammar / capitalisation / paragraph
breaks, then realign the proposed words back onto the original
audio timestamps so word-level playback highlighting still works.

The module is **pure**: no FastAPI, no engine, no filesystem. It
operates on Job-result-shaped dicts (matching what
``/api/job/<id>/transcript`` returns and what ``editor.html``
already renders). The HTTP wrapper in :mod:`scribe.server` calls into
these helpers; the editor JS calls the wrapper.

Why "runs"?
-----------
Speech-to-text models emit segments at acoustic boundaries
(silences, breath pauses, sentence-final intonation). When one
person talks for thirty seconds, the model returns six or seven
fragmentary segments — fine for follow-along playback, awful to
read on a page. The grammar bot operates one *run* at a time: a
maximal sequence of consecutive segments that share a speaker.
Inside a run, the model can rewrite freely; segment boundaries
across speakers are sacrosanct (they encode "who's talking").

Realignment
-----------
The model only rewrites words; it does *not* know about
timestamps. We use ``difflib.SequenceMatcher`` on the
lowercased-stripped tokens to find which proposed words match
the original. ``equal`` blocks keep their original
``start``/``end`` exactly. The remaining proposed words land in
gaps between matched anchors and get **linearly interpolated**
into that gap's wall-clock duration. The result is a word list
whose timestamps are monotone and span the same total wall-clock
range as the input — so the playback cursor still tracks the
audio, even after a heavy rewrite.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# Lower bound on a "run" — the editor will not offer to tidy a single
# segment with no neighbours, since the LLM has nothing to merge.
MIN_RUN_SEGMENTS = 2

# Caps on what we'll accept as input. The transcripts we work with
# are seconds- to minutes-long monologues; an entire 90-minute interview
# would be one giant run if every segment shared a speaker, which is
# both unrealistic and would tank an LLM's context window. We cap at
# ~6000 words / ~30 minutes per run for sanity.
MAX_RUN_WORDS = 6000
MAX_RUN_DURATION_S = 30 * 60.0


# Punctuation we strip when normalising for the diff.
_NORMALISE_PUNCT_RE = re.compile(r"[^\w']+", flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Run:
    """One maximal sequence of consecutive same-speaker segments.

    ``segment_indices`` lets the apply step splice the proposed
    paragraphs back in the right place; we store indices rather than
    cloning segments so callers can decide whether to keep the
    originals on reject.
    """

    speaker: str
    segment_indices: tuple[int, ...]
    start: float
    end: float
    text: str
    word_count: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "segment_indices": list(self.segment_indices),
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "word_count": self.word_count,
            "duration_s": self.duration_s,
        }


@dataclass
class TidiedSegment:
    """One segment of the proposed (tidied) output. Mirrors the
    on-disk segment shape used by the editor + ``put_transcript``.
    """

    speaker: str
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": list(self.words),
        }


# --------------------------------------------------------------------------- #
# Run grouping
# --------------------------------------------------------------------------- #


def group_runs(segments: Sequence[dict[str, Any]]) -> list[Run]:
    """Return the maximal consecutive same-speaker runs in ``segments``.

    Singleton runs (``len < MIN_RUN_SEGMENTS``) and runs that exceed
    :data:`MAX_RUN_WORDS` or :data:`MAX_RUN_DURATION_S` are dropped —
    they aren't usable inputs to the tidy flow. The remaining runs are
    returned in the order they appear in the source.
    """
    runs: list[Run] = []
    if not segments:
        return runs

    cur_speaker: str | None = None
    cur_indices: list[int] = []

    def _flush() -> None:
        if len(cur_indices) < MIN_RUN_SEGMENTS:
            return
        run = _build_run(segments, tuple(cur_indices))
        if run.word_count > MAX_RUN_WORDS:
            return
        if run.duration_s > MAX_RUN_DURATION_S:
            return
        runs.append(run)

    for i, seg in enumerate(segments):
        sp = (seg.get("speaker") or "").strip() or "SPEAKER_??"
        if cur_speaker is None or sp != cur_speaker:
            _flush()
            cur_speaker = sp
            cur_indices = [i]
        else:
            cur_indices.append(i)
    _flush()
    return runs


def _build_run(segments: Sequence[dict[str, Any]], indices: tuple[int, ...]) -> Run:
    speaker = (segments[indices[0]].get("speaker") or "").strip() or "SPEAKER_??"
    parts: list[str] = []
    word_count = 0
    starts: list[float] = []
    ends: list[float] = []
    for i in indices:
        seg = segments[i]
        text = (seg.get("text") or "").strip()
        if text:
            parts.append(text)
        words = seg.get("words") or []
        word_count += sum(1 for w in words if str(w.get("text", "")).strip())
        if isinstance(seg.get("start"), (int, float)):
            starts.append(float(seg["start"]))
        if isinstance(seg.get("end"), (int, float)):
            ends.append(float(seg["end"]))
    return Run(
        speaker=speaker,
        segment_indices=indices,
        start=min(starts) if starts else 0.0,
        end=max(ends) if ends else 0.0,
        text=" ".join(parts),
        word_count=word_count,
    )


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #


TIDY_SYSTEM_PROMPT = (
    "You are a careful editor preparing a verbatim research interview "
    "transcript for readability. The speech may be ungrammatical, "
    "interrupted, or repetitive. Your task:\n\n"
    "  1. Fix obvious grammar, punctuation, and capitalisation.\n"
    "  2. Break the speech into natural paragraphs at topic shifts; "
    "separate paragraphs with a single blank line.\n"
    "  3. PRESERVE meaning verbatim. Do not invent content, do not "
    "translate, do not summarise, do not add facts.\n"
    "  4. You may drop disfluencies (\"um\", \"uh\", repeated false "
    "starts) where doing so doesn't change meaning.\n"
    "  5. Output ONLY the cleaned text — no preamble, no commentary, "
    "no markdown formatting, no list bullets.\n"
)


def build_tidy_prompt(run_text: str) -> str:
    """Compose the prompt body the LLM sees.

    Kept as a separate function so tests can pin the exact text that
    goes to the model and so future tweaks (different prompts for
    different languages, etc.) have one place to live.
    """
    return f"{TIDY_SYSTEM_PROMPT}\n---\n{run_text.strip()}\n---\n"


def parse_tidied_paragraphs(response: str) -> list[str]:
    """Split the model's response into paragraph strings.

    Strips leading/trailing whitespace, collapses internal blank-line
    runs to a single blank, drops empty paragraphs. If the model
    ignored the "no markdown" rule and returned bullets, we strip a
    leading ``- `` / ``* `` so the words still tokenise cleanly.
    """
    body = (response or "").strip()
    if not body:
        return []
    # Some models like to wrap output in a fenced block; strip it if
    # present (``` ... ``` or ```text\n ... \n```).
    body = re.sub(r"^```[a-zA-Z]*\n", "", body)
    body = re.sub(r"\n```$", "", body)
    paragraphs: list[str] = []
    for p in re.split(r"\n\s*\n", body):
        line = p.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        # Collapse internal newlines to single spaces — paragraph
        # breaks come from the blank-line split, not soft wraps.
        line = re.sub(r"\s*\n\s*", " ", line).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


# --------------------------------------------------------------------------- #
# Tokenisation + diff-based realignment
# --------------------------------------------------------------------------- #


_TOKEN_RE = re.compile(r"\S+", flags=re.UNICODE)


def _normalise_token(token: str) -> str:
    """Lowercase + strip punctuation for the diff comparison."""
    return _NORMALISE_PUNCT_RE.sub("", token.lower())


def _flatten_old_words(run: Run, segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate every word across the run's segments into one
    flat list, in transcript order. Skips empty / whitespace-only
    word records.
    """
    out: list[dict[str, Any]] = []
    for i in run.segment_indices:
        for w in segments[i].get("words") or []:
            text = str(w.get("text", "")).strip()
            if not text:
                continue
            out.append({
                "text": text,
                "start": float(w.get("start", run.start)) if isinstance(w.get("start"), (int, float)) else None,
                "end": float(w.get("end", run.end)) if isinstance(w.get("end"), (int, float)) else None,
                "speaker": w.get("speaker") or run.speaker,
                "score": w.get("score"),
            })
    return out


def _tokenise_paragraphs(paragraphs: Sequence[str]) -> list[dict[str, Any]]:
    """One flat token list across all paragraphs, each carrying its
    paragraph index so we can split back at the end."""
    out: list[dict[str, Any]] = []
    for p_idx, p in enumerate(paragraphs):
        for m in _TOKEN_RE.finditer(p):
            out.append({
                "text": m.group(0),
                "paragraph_idx": p_idx,
                "start": None,
                "end": None,
            })
    return out


def realign_words(
    old_words: Sequence[dict[str, Any]],
    paragraphs: Sequence[str],
    *,
    fallback_start: float,
    fallback_end: float,
) -> list[list[dict[str, Any]]]:
    """Map the proposed paragraphs' tokens back onto the original word
    timestamps.

    Returns a list-of-lists: one inner list per paragraph, each
    inner list a sequence of word records ``{text, start, end,
    speaker?, score?}``.

    The realignment uses :class:`difflib.SequenceMatcher` on
    case- and punctuation-stripped tokens. ``equal`` blocks keep the
    original timestamps verbatim. Other blocks are anchored on either
    side by the most recent ``equal`` end and the next ``equal``
    start; the new tokens fall in that interval at evenly-spaced
    durations. With no equal anchors at all, every token is spread
    evenly across ``[fallback_start, fallback_end]`` — which is the
    run's own ``[start, end]`` from the source segments.
    """
    new_tokens = _tokenise_paragraphs(paragraphs)

    # Trivial cases: no work to do.
    if not new_tokens:
        return []
    if not old_words:
        # No timestamps at all — distribute evenly across the run.
        return _split_by_paragraph(
            _spread_tokens(new_tokens, fallback_start, fallback_end),
            len(paragraphs),
        )

    # 1) Match equal tokens via SequenceMatcher.
    old_norm = [_normalise_token(w["text"]) for w in old_words]
    new_norm = [_normalise_token(t["text"]) for t in new_tokens]
    matcher = difflib.SequenceMatcher(a=old_norm, b=new_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            old = old_words[i1 + k]
            tok = new_tokens[j1 + k]
            tok["start"] = old.get("start")
            tok["end"] = old.get("end")
            tok["speaker"] = old.get("speaker")
            if old.get("score") is not None:
                tok["score"] = old["score"]

    # 2) Fill remaining tokens by interpolation between anchors.
    _interpolate_unaligned(new_tokens, fallback_start, fallback_end)

    return _split_by_paragraph(new_tokens, len(paragraphs))


def _spread_tokens(
    tokens: list[dict[str, Any]], lo: float, hi: float,
) -> list[dict[str, Any]]:
    """Distribute every token evenly across [lo, hi]. Used when the
    diff has no equal-block anchors at all."""
    n = len(tokens)
    if n <= 0 or hi <= lo:
        for tok in tokens:
            tok["start"] = lo
            tok["end"] = lo
        return tokens
    span = (hi - lo) / n
    for i, tok in enumerate(tokens):
        tok["start"] = lo + i * span
        tok["end"] = lo + (i + 1) * span
    return tokens


def _interpolate_unaligned(
    tokens: list[dict[str, Any]],
    fallback_start: float,
    fallback_end: float,
) -> None:
    """For each maximal run of tokens with ``start is None``, fill
    them by spreading evenly between the surrounding anchors.

    Mutates ``tokens`` in place.
    """
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i]["start"] is not None:
            i += 1
            continue
        # Find the run [i, j) of consecutive None-starts.
        j = i
        while j < n and tokens[j]["start"] is None:
            j += 1
        # Anchor on the previous aligned token's end (or fallback).
        if i > 0 and tokens[i - 1].get("end") is not None:
            anchor_lo = float(tokens[i - 1]["end"])
        else:
            anchor_lo = fallback_start
        # Anchor on the next aligned token's start (or fallback).
        if j < n and tokens[j].get("start") is not None:
            anchor_hi = float(tokens[j]["start"])
        else:
            anchor_hi = fallback_end
        if anchor_hi < anchor_lo:
            # Pathological — shouldn't happen on monotonic input. Pin
            # everything to the lower anchor so we don't go backwards.
            anchor_hi = anchor_lo
        gap_n = j - i
        # Even split across [anchor_lo, anchor_hi]. Each token gets a
        # window of size (anchor_hi - anchor_lo) / gap_n.
        if anchor_hi == anchor_lo:
            for k in range(gap_n):
                tokens[i + k]["start"] = anchor_lo
                tokens[i + k]["end"] = anchor_lo
        else:
            step = (anchor_hi - anchor_lo) / gap_n
            for k in range(gap_n):
                tokens[i + k]["start"] = anchor_lo + k * step
                tokens[i + k]["end"] = anchor_lo + (k + 1) * step
        i = j


def _split_by_paragraph(
    tokens: list[dict[str, Any]], n_paragraphs: int,
) -> list[list[dict[str, Any]]]:
    """Group the flat token list back into per-paragraph sublists.

    Strips the internal ``paragraph_idx`` field from each record on
    the way out so the output matches the on-disk word shape
    ``{text, start, end, speaker?, score?}`` that
    ``put_transcript`` expects.
    """
    paragraphs: list[list[dict[str, Any]]] = [[] for _ in range(n_paragraphs)]
    for tok in tokens:
        idx = int(tok.get("paragraph_idx", 0))
        if idx < 0 or idx >= n_paragraphs:
            continue
        out_tok = {k: v for k, v in tok.items() if k != "paragraph_idx"}
        paragraphs[idx].append(out_tok)
    return paragraphs


# --------------------------------------------------------------------------- #
# Assembly: paragraphs + realigned words → tidied segments
# --------------------------------------------------------------------------- #


def assemble_tidied_segments(
    *,
    paragraphs: Sequence[str],
    paragraph_words: Sequence[Sequence[dict[str, Any]]],
    speaker: str,
    fallback_start: float,
    fallback_end: float,
) -> list[TidiedSegment]:
    """Build one :class:`TidiedSegment` per paragraph, each with its
    own start / end / words.

    Empty paragraphs (no words after tokenisation) are dropped. If a
    paragraph somehow has zero words after realignment it falls back
    to the run's own start/end so the editor doesn't render NaNs.
    """
    segs: list[TidiedSegment] = []
    if len(paragraphs) != len(paragraph_words):
        raise ValueError(
            "paragraphs and paragraph_words must have the same length; "
            f"got {len(paragraphs)} vs {len(paragraph_words)}"
        )
    for text, words in zip(paragraphs, paragraph_words):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        if not words:
            segs.append(TidiedSegment(
                speaker=speaker,
                start=fallback_start,
                end=fallback_end,
                text=cleaned,
                words=[],
            ))
            continue
        starts = [w["start"] for w in words if isinstance(w.get("start"), (int, float))]
        ends = [w["end"] for w in words if isinstance(w.get("end"), (int, float))]
        seg = TidiedSegment(
            speaker=speaker,
            start=min(starts) if starts else fallback_start,
            end=max(ends) if ends else fallback_end,
            text=cleaned,
            words=[
                {
                    "text": w["text"],
                    "start": float(w.get("start") or 0.0),
                    "end": float(w.get("end") or 0.0),
                    "speaker": w.get("speaker") or speaker,
                    **({"score": float(w["score"])} if w.get("score") is not None else {}),
                }
                for w in words
            ],
        )
        segs.append(seg)
    return segs


# --------------------------------------------------------------------------- #
# Splice — given an apply payload, produce the edited transcript
# --------------------------------------------------------------------------- #


def splice_run(
    transcript: dict[str, Any],
    *,
    segment_indices: Sequence[int],
    new_segments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return a copy of ``transcript`` with the segments at
    ``segment_indices`` replaced by ``new_segments``.

    ``segment_indices`` must be a contiguous block (as
    :func:`group_runs` always returns). The function rebuilds the
    full segments list rather than mutating in place so callers can
    keep the original around for "Reject" / undo.
    """
    if not isinstance(transcript, dict):
        raise TypeError("transcript must be a dict")
    segs = list(transcript.get("segments") or [])
    if not segment_indices:
        raise ValueError("segment_indices is empty")
    indices = sorted(set(int(i) for i in segment_indices))
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("segment_indices must be a contiguous block")
    if indices[0] < 0 or indices[-1] >= len(segs):
        raise ValueError(
            f"segment_indices out of range: {indices} (transcript has {len(segs)} segs)"
        )
    out: dict[str, Any] = {**transcript}
    out["segments"] = (
        segs[: indices[0]] + list(new_segments) + segs[indices[-1] + 1 :]
    )
    return out
