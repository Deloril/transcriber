"""
Smoke test for the AMD/ROCm path.

Run with:
    .venv/bin/python -m scribe.scripts.check_rocm

Generates a 5-second silent WAV, loads CT2 (small Whisper variant) on the
detected GPU backend, runs alignment, and reports timing. Useful for triaging
"my AMD setup is broken" support questions because it exercises every layer
of the stack quickly.

Will work on CUDA / ROCm / CPU; on MPS the CT2 path falls back to CPU as
designed. Doesn't load the diarization model — that requires an HF token,
and we want this script to run for any user.
"""

from __future__ import annotations

import sys
import tempfile
import time
import wave
from pathlib import Path


def _make_silent_wav(path: Path, seconds: float = 5.0, sr: int = 16000) -> None:
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def main() -> int:
    from scribe.engine import (
        _whisper_device_and_compute,
        _torch_device,
        gpu_backend,
    )

    print(f"Backend: {gpu_backend()}")
    w_dev, w_compute = _whisper_device_and_compute()
    print(f"Whisper device: {w_dev} compute={w_compute}")
    print(f"Torch device: {_torch_device()}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "silent.wav"
        _make_silent_wav(wav)

        print(">> Loading whisperx + tiny model …")
        t0 = time.time()
        try:
            import whisperx
            asr = whisperx.load_model(
                "tiny",
                device=w_dev,
                compute_type=w_compute,
                language="en",
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAILED to load model: {type(e).__name__}: {e}")
            return 2
        load_secs = time.time() - t0
        print(f"   model loaded in {load_secs:.1f}s")

        print(">> Transcribing 5 s of silence …")
        t0 = time.time()
        try:
            audio = whisperx.load_audio(str(wav))
            asr.transcribe(audio, batch_size=1)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED to transcribe: {type(e).__name__}: {e}")
            return 3
        tx_secs = time.time() - t0
        print(f"   transcribe done in {tx_secs:.1f}s")

        print(">> Loading wav2vec2 alignment model …")
        t0 = time.time()
        try:
            am, meta = whisperx.load_align_model(language_code="en", device=_torch_device())
        except Exception as e:  # noqa: BLE001
            print(f"FAILED to load alignment model: {type(e).__name__}: {e}")
            return 4
        align_secs = time.time() - t0
        print(f"   align model loaded in {align_secs:.1f}s")

    print()
    print("All layers reached without crashing. Backend looks healthy.")
    print("Note: this script does not exercise pyannote diarization, which")
    print("requires an HF_TOKEN. Try a real transcription via ./run.sh next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
