"""Audio inspection and extraction via ffmpeg/ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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
    Extract a single audio track to mono 16-bit PCM WAV at `sample_rate`.

    `stream_index` is the absolute stream index from ffprobe (e.g. 1 for the
    second stream overall). If None, ffmpeg picks the default audio stream.
    """
    ffmpeg = _require("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
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
