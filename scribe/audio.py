"""Audio inspection and extraction via ffmpeg/ffprobe."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"{tool} not found on PATH. Install with: brew install ffmpeg")
    return path


@dataclass
class AudioStream:
    index: int
    channels: int
    title: str | None
    language: str | None
    codec: str


def probe_audio_streams(input_path: Path) -> list[AudioStream]:
    """Return all audio streams in the input, in declaration order."""
    ffprobe = _require("ffprobe")
    out = subprocess.check_output(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            str(input_path),
        ]
    )
    data = json.loads(out)
    streams: list[AudioStream] = []
    for s in data.get("streams", []):
        tags = s.get("tags") or {}
        streams.append(
            AudioStream(
                index=int(s["index"]),
                channels=int(s.get("channels") or 1),
                title=tags.get("title"),
                language=tags.get("language"),
                codec=s.get("codec_name", "unknown"),
            )
        )
    return streams


def extract_track_to_wav(
    input_path: Path,
    out_path: Path,
    stream_index: int | None = None,
    sample_rate: int = 16000,
) -> Path:
    """
    Extract a single audio track to mono 16-bit PCM WAV at `sample_rate`,
    preserving the stream's absolute start time relative to t=0 of the
    container.

    Critical for multi-track / multi-stream recordings: each stream in
    a container can carry its own ``start_time`` (OBS, field
    recorders, and any container that pads with silence-before-speech
    do this). Without ``-copyts -start_at_zero`` the stream rebases
    its internal time so the first audio sample lands at 0:00 in the
    output WAV — Whisper then transcribes timestamps that are
    *relative to the stream's first sample*, not absolute against the
    recording. The result: each speaker appears to start talking at
    "00:00" and the transcript ends up out of order when sorted by
    start.

    With these flags, the output WAV has leading silence padded to
    match the original timeline so downstream timestamps are
    comparable across tracks.

    `stream_index` is the absolute ffprobe stream index (e.g. 1 for
    the second stream overall). ``None`` picks the default audio
    stream.
    """
    ffmpeg = _require("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        # ``-copyts`` keeps the original container timestamps on the
        # output instead of rebasing to zero. ``-start_at_zero``
        # then pads the leading-silence gap so the output WAV's
        # samples line up with the recording's t=0. Together they
        # mean a stream that starts speaking at 30s in the original
        # produces a WAV that has 30s of silence before any speech,
        # so downstream timestamps are absolute.
        "-copyts",
        "-start_at_zero",
        "-i", str(input_path),
    ]
    if stream_index is not None:
        cmd += ["-map", f"0:{stream_index}"]
    else:
        cmd += ["-map", "0:a:0"]
    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _has_video(input_path: Path) -> bool:
    """Best-effort: True if the file has at least one video stream."""
    ffprobe = _require("ffprobe")
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v", "error",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v",
                str(input_path),
            ]
        )
    except subprocess.CalledProcessError:
        return False
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        return False
    return bool(data.get("streams"))


def build_playback_mix(
    input_path: Path,
    out_path: Path,
    *,
    selected_stream_indices: list[int] | None = None,
) -> Path:
    """Produce a single-audio-track playback file for the editor's ``<video>``.

    Browsers' built-in media element only plays the *default* audio
    stream of a multi-track file. For OBS-style recordings where
    each speaker lives on their own track, that means the user only
    hears track 0 — silently missing every other speaker.

    This helper merges the chosen audio tracks (default: all of them)
    into one stereo track via ffmpeg's ``amix`` filter and re-muxes
    (or re-encodes for audio-only inputs) so the resulting file has
    exactly one audio stream that contains all the selected
    speakers. The function is destination-aware — if the source has
    video, we keep the video track via ``-c:v copy`` and just
    re-encode audio to AAC; if it doesn't, we write an MP3.

    ``selected_stream_indices`` is the set of absolute ffprobe stream
    indices to include in the mix. ``None`` mixes every audio stream
    in the file. Indices not present in the file raise
    :class:`ValueError`.

    Idempotent: if ``out_path`` already exists and is newer than
    ``input_path``, we return it without re-running ffmpeg. Callers
    can force a rebuild by deleting the cache file first.
    """
    streams = probe_audio_streams(input_path)
    if not streams:
        raise ValueError(f"No audio streams in {input_path}")

    if selected_stream_indices is None:
        chosen = streams
    else:
        valid = {s.index for s in streams}
        bad = [i for i in selected_stream_indices if i not in valid]
        if bad:
            raise ValueError(
                f"selected_stream_indices contains entries not in "
                f"{input_path}: {bad} (valid: {sorted(valid)})"
            )
        chosen = [s for s in streams if s.index in selected_stream_indices]
        if not chosen:
            raise ValueError(
                "selected_stream_indices is empty after filtering"
            )

    # Cache hit: if the destination already exists and is newer than
    # the input, reuse it. Cheap stat call avoids the 30+ second mix
    # on every page load.
    if out_path.is_file():
        try:
            if out_path.stat().st_mtime >= input_path.stat().st_mtime:
                return out_path
        except OSError:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _require("ffmpeg")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(input_path)]

    # ``amix`` filter combines N audio streams into one. We feed each
    # selected stream as a labelled input by absolute ffprobe index,
    # then mix to stereo. ``normalize=0`` keeps each track at its
    # original loudness instead of dividing by the input count
    # (otherwise a quiet speaker on a 4-track recording becomes
    # inaudible after the divide-by-4 default).
    inputs = "".join(f"[0:{s.index}]" for s in chosen)
    if len(chosen) == 1:
        # Trivial case: one track, just re-mux that stream as the
        # only audio. amix with one input would still work but
        # we'd burn a re-encode for nothing.
        filter_args = []
        map_args = ["-map", "0:v?", "-map", f"0:{chosen[0].index}"]
    else:
        filter_complex = (
            f"{inputs}amix=inputs={len(chosen)}:duration=longest:"
            "normalize=0[mixed]"
        )
        filter_args = ["-filter_complex", filter_complex]
        map_args = ["-map", "0:v?", "-map", "[mixed]"]

    cmd.extend(filter_args)
    cmd.extend(map_args)

    if _has_video(input_path):
        # Keep video as-is; encode the mixed audio to AAC inside the
        # original container (mp4/mkv both accept AAC).
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    else:
        # Audio-only path: write an MP3 regardless of source codec
        # so the editor's <audio> always finds a playable stream.
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]

    cmd.append(str(out_path))
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to build playback mix for {input_path}: "
            f"{stderr or proc.returncode}"
        )
    return out_path


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _parse_fraction(v: Any) -> float | None:
    """ffprobe returns frame rates as 'num/den'."""
    if not v or v == "0/0":
        return None
    if isinstance(v, str) and "/" in v:
        try:
            num, den = v.split("/", 1)
            d = float(den)
            return float(num) / d if d else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return _safe_float(v)


def probe_media_info(input_path: Path) -> dict[str, Any]:
    """
    Single ffprobe call returning duration, size, format, plus per-stream
    summaries for video and audio. Used for the upload "stats" panel.
    """
    ffprobe = _require("ffprobe")
    out = subprocess.check_output(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(input_path),
        ]
    )
    data = json.loads(out)
    fmt = data.get("format") or {}
    streams = data.get("streams") or []

    duration = _safe_float(fmt.get("duration"))
    bit_rate = _safe_int(fmt.get("bit_rate"))
    size = _safe_int(fmt.get("size")) or input_path.stat().st_size

    videos: list[dict[str, Any]] = []
    audios: list[dict[str, Any]] = []
    for s in streams:
        tags = s.get("tags") or {}
        if s.get("codec_type") == "video":
            videos.append({
                "index": _safe_int(s.get("index")),
                "codec": s.get("codec_name"),
                "width": _safe_int(s.get("width")),
                "height": _safe_int(s.get("height")),
                "fps": _parse_fraction(s.get("avg_frame_rate") or s.get("r_frame_rate")),
                "pix_fmt": s.get("pix_fmt"),
                "bit_rate": _safe_int(s.get("bit_rate")),
                "duration": _safe_float(s.get("duration")),
            })
        elif s.get("codec_type") == "audio":
            audios.append({
                "index": _safe_int(s.get("index")),
                "codec": s.get("codec_name"),
                "channels": _safe_int(s.get("channels")),
                "channel_layout": s.get("channel_layout"),
                "sample_rate": _safe_int(s.get("sample_rate")),
                "bit_rate": _safe_int(s.get("bit_rate")),
                "duration": _safe_float(s.get("duration")),
                "title": tags.get("title"),
                "language": tags.get("language"),
            })

    return {
        "filename": input_path.name,
        "size_bytes": size,
        "duration_seconds": duration,
        "format_name": fmt.get("format_long_name") or fmt.get("format_name"),
        "bit_rate": bit_rate,
        "video": videos,
        "audio": audios,
    }


def compute_waveform(input_path: Path, bins: int = 1000, sample_rate: int = 8000) -> list[float]:
    """
    Decode the input's audio to mono PCM at `sample_rate`, downsample into
    `bins` peak-amplitude buckets in [0.0, 1.0]. Cheap (a few hundred ms
    even for hour-long files) and never holds the full file in memory at once.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    ffmpeg = _require("ffmpeg")
    proc = subprocess.Popen(
        [
            ffmpeg, "-loglevel", "error",
            "-i", str(input_path),
            "-vn",
            "-map", "0:a:0",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None

    # We don't know the total sample count up front; collect peaks per
    # block and merge into the final bin layout once we know the total.
    blocks: list[tuple[int, list[int]]] = []  # (sample count in block, peaks per micro-bin)
    micro_bins_per_read = 64
    chunk_samples = 1 << 16  # 64Ki samples = 128 KiB
    total_samples = 0
    while True:
        raw = proc.stdout.read(chunk_samples * 2)
        if not raw:
            break
        n = len(raw) // 2
        if n == 0:
            break
        samples = struct.unpack(f"<{n}h", raw[: n * 2])
        # downsample this chunk into micro_bins_per_read peaks
        if n >= micro_bins_per_read:
            step = n / micro_bins_per_read
            peaks = []
            for i in range(micro_bins_per_read):
                lo = int(i * step)
                hi = int((i + 1) * step) if i < micro_bins_per_read - 1 else n
                if lo >= hi:
                    peaks.append(0)
                    continue
                peaks.append(max(abs(s) for s in samples[lo:hi]))
        else:
            peaks = [max(abs(s) for s in samples)] * micro_bins_per_read
        blocks.append((n, peaks))
        total_samples += n

    proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    if not blocks or total_samples == 0:
        return [0.0] * bins

    # Merge all micro-bins into the requested bin count by sample-weighted
    # peak picking.
    micro_total = sum(len(b[1]) for b in blocks)
    flat: list[int] = []
    for _, peaks in blocks:
        flat.extend(peaks)

    out: list[float] = []
    step = micro_total / bins
    for i in range(bins):
        lo = int(i * step)
        hi = int((i + 1) * step) if i < bins - 1 else micro_total
        if lo >= hi:
            out.append(0.0)
            continue
        peak = max(flat[lo:hi]) if hi > lo else 0
        out.append(peak / 32768.0)
    return out
