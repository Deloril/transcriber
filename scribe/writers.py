"""Write a TranscriptionResult to JSON, plain text, SRT, and VTT."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import TranscriptionResult


def _fmt_time(seconds: float, *, sep: str = ",") -> str:
    if seconds < 0:
        seconds = 0
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    whole = int(s)
    ms = int(round((s - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    return f"{int(h):02d}:{int(m):02d}:{whole:02d}{sep}{ms:03d}"


def _fmt_clock(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def write_json(result: TranscriptionResult, out_path: Path) -> None:
    out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def write_txt(result: TranscriptionResult, out_path: Path) -> None:
    """Group consecutive same-speaker segments into paragraphs."""
    lines: list[str] = []
    current_speaker: str | None = None
    current_chunks: list[str] = []
    block_start: float = 0.0

    def flush() -> None:
        if current_speaker is None or not current_chunks:
            return
        text = " ".join(c.strip() for c in current_chunks if c.strip())
        if not text:
            return
        lines.append(f"[{_fmt_clock(block_start)}] {current_speaker}: {text}")

    for seg in result.segments:
        if seg.speaker != current_speaker:
            flush()
            current_speaker = seg.speaker
            current_chunks = []
            block_start = seg.start
        current_chunks.append(seg.text)
    flush()

    out_path.write_text("\n\n".join(lines) + ("\n" if lines else ""))


def write_srt(result: TranscriptionResult, out_path: Path) -> None:
    parts: list[str] = []
    for i, seg in enumerate(result.segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        parts.append(
            f"{i}\n"
            f"{_fmt_time(seg.start)} --> {_fmt_time(seg.end)}\n"
            f"{seg.speaker}: {text}\n"
        )
    out_path.write_text("\n".join(parts))


def write_vtt(result: TranscriptionResult, out_path: Path) -> None:
    parts: list[str] = ["WEBVTT", ""]
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        parts.append(
            f"{_fmt_time(seg.start, sep='.')} --> {_fmt_time(seg.end, sep='.')}\n"
            f"<v {seg.speaker}>{text}\n"
        )
    out_path.write_text("\n".join(parts))


def write_all(result: TranscriptionResult, base_path: Path) -> dict[str, Path]:
    """Write json/txt/srt/vtt next to base_path. Returns the written paths."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": base_path.with_suffix(".json"),
        "txt": base_path.with_suffix(".txt"),
        "srt": base_path.with_suffix(".srt"),
        "vtt": base_path.with_suffix(".vtt"),
    }
    write_json(result, paths["json"])
    write_txt(result, paths["txt"])
    write_srt(result, paths["srt"])
    write_vtt(result, paths["vtt"])
    return paths
