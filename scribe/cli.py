"""Scribe CLI entry — `python -m scribe.cli <input>`."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from .engine import transcribe
from .writers import write_all


def _print_progress(msg: str, frac: float) -> None:
    pct = max(0.0, min(1.0, frac)) * 100
    sys.stderr.write(f"\r[{pct:5.1f}%] {msg:<60}")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="scribe",
        description="Offline interview transcription with speaker identification.",
    )
    parser.add_argument("input", type=Path, help="Audio or video file")
    parser.add_argument(
        "--mode",
        choices=("auto", "multi-track", "diarize"),
        default="auto",
        help="auto picks multi-track if ≥2 audio streams detected",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help="Comma-separated speaker names for multi-track mode, in track order. "
             "Example: --speakers Luke,Guest",
    )
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="(diarize mode) exact number of speakers")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output base path (no extension). Defaults to input path.")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep extracted WAV files after transcription")

    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    speakers = None
    if args.speakers:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]

    out_base = args.out or args.input.with_suffix("")

    tmp = Path(tempfile.mkdtemp(prefix="scribe-"))
    try:
        result = transcribe(
            args.input,
            work_dir=tmp,
            mode=args.mode,
            speaker_labels=speakers,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            model_name=args.model,
            language=args.language,
            batch_size=args.batch_size,
            hf_token=os.environ.get("HF_TOKEN"),
            progress=_print_progress,
        )
        paths = write_all(result, out_base)
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"Temp dir kept: {tmp}", file=sys.stderr)

    print()
    print(f"Mode:     {result.mode}")
    print(f"Language: {result.language}")
    print(f"Speakers: {', '.join(result.speaker_labels) or '(none detected)'}")
    print("Outputs:")
    for kind, p in paths.items():
        print(f"  {kind:4s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
