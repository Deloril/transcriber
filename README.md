# Scribe

Offline interview transcription with speaker identification. Runs 100% locally on **Apple Silicon (macOS)** or **Linux (NVIDIA CUDA or CPU)**. No data leaves your machine.

Built to fix the long-monologue truncation problem in Scriberr / vanilla Whisper by using VAD-based chunking and forced alignment.

## What it does

- Accepts video or audio files (anything ffmpeg can read)
- Transcribes with `whisper-large-v3` via WhisperX
- Word-level timestamps via wav2vec2 forced alignment
- **Multi-track mode**: if your recording has separate audio tracks per speaker, each track is transcribed independently and merged on the timeline — perfect speaker labels
- **Single-track mode**: pyannote-based AI diarization
- Outputs: JSON, plain text, SRT, VTT
- Local web UI with live progress
- All models cached locally after first download; no internet needed for subsequent runs

## Why this is more accurate than Scriberr for long monologues

Whisper has a 30-second context window. Most tools (including Scriberr) feed it fixed-length chunks; if a monologue spans a chunk boundary, the model can drop or hallucinate text. WhisperX uses pyannote VAD to find actual speech segments and chunks **on silences only**, so monologues stay intact end-to-end. Combined with forced alignment, you also get accurate per-word timestamps.

## Requirements

- One of:
  - **macOS on Apple Silicon** (M1/M2/M3/M4) — uses MPS for alignment/diarization, CPU+int8 for Whisper
  - **Linux + NVIDIA GPU** (8 GB+ VRAM ideal; 6 GB works with int8 quantization)
  - **Linux CPU-only** — works fine, just slow (5–10× realtime for large-v3)
- Python 3.10 or 3.11 (3.12+ has dependency issues with some pinned versions)
- `ffmpeg`
- ~10 GB free disk for model weights (one-time download)
- A free Hugging Face account + access token (only needed for diarization in single-track mode; not needed for multi-track)

## Install

### macOS (Apple Silicon)

```bash
brew install ffmpeg python@3.11   # if you don't have them
cd scribe
./setup.sh
```

### Linux (NVIDIA GPU)

```bash
sudo apt install ffmpeg python3.11 python3.11-venv     # Debian/Ubuntu
# or: sudo dnf install ffmpeg python3.11                  Fedora/RHEL

# Make sure the NVIDIA driver and CUDA runtime are installed:
nvidia-smi                                              # should print your GPU

cd scribe
./setup.sh
```

By default `setup.sh` lets pip pick the right PyTorch wheel from PyPI, which on Linux x86_64 is the CUDA build. If you need a specific CUDA version or a CPU-only build, set `SCRIBE_TORCH` before running setup:

```bash
SCRIBE_TORCH=cu124 ./setup.sh    # PyTorch built against CUDA 12.4
SCRIBE_TORCH=cu121 ./setup.sh    # PyTorch built against CUDA 12.1
SCRIBE_TORCH=cpu   ./setup.sh    # CPU-only Linux
```

### Linux (CPU-only)

```bash
sudo apt install ffmpeg python3.11 python3.11-venv
cd scribe
SCRIBE_TORCH=cpu ./setup.sh
```

The installer creates a venv, installs PyTorch + WhisperX + FastAPI, then prints the device config it will use. You can rerun that check any time:

```bash
source .venv/bin/activate
python -m scribe.devices
```

You should see something like:

```
Selected backends:
  Whisper (CTranslate2):  device=cuda  compute=float16        # or cpu/int8 on Mac/CPU
  Alignment (torch):      device=cuda                         # or mps
  Diarization (pyannote): device=cuda                         # or cpu
```

### Hugging Face token (only for single-track diarization)

If you'll record everything multi-track (recommended below), skip this.

Otherwise:
1. Create a free account at https://huggingface.co
2. Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept the license at https://huggingface.co/pyannote/segmentation-3.0
4. Create a token at https://huggingface.co/settings/tokens (read scope is enough)
5. Save it: `echo "HF_TOKEN=hf_xxx" > .env`

## Run

```bash
./run.sh
```

Open http://localhost:8765 in your browser.

Drag a video/audio file onto the page. Pick a mode. Click Transcribe.

The first run downloads ~3 GB of model weights. After that, it's fully offline.

## CLI

```bash
source .venv/bin/activate
python -m scribe.cli path/to/recording.mp4 --mode multi-track --speakers Luke,Guest
python -m scribe.cli path/to/recording.mp4 --mode diarize --num-speakers 2
```

Outputs land next to the input file as `<name>.json`, `<name>.srt`, `<name>.txt`.

---

## OBS setup for perfect speaker identification

This is the highest-leverage change you can make. With per-speaker audio tracks, AI diarization is unnecessary — each track is one speaker, and the result is essentially perfect.

The OBS UI is the same on macOS and Linux. The only platform-specific part is **how you capture system audio** (the other end of the Zoom call), which is covered in two short subsections below.

### One-time OBS configuration

1. **Sources panel** — make sure you have at least two audio sources:
   - Your mic (e.g. "Mic/Aux")
   - Desktop audio (captures the other end of Zoom/Meet/etc.) — e.g. "Desktop Audio"

   If you only see one, add the missing one: **`+`** under Sources → **Audio Input Capture** (for mic) or **Audio Output Capture** (for system audio). System audio capture is the part that varies by OS — see below.

   **macOS:** OBS cannot capture system audio directly. Install a virtual loopback driver:
   - [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) — free, recommended. After install, create a **Multi-Output Device** in `Audio MIDI Setup.app` that includes both your speakers/headphones AND BlackHole 2ch, set that Multi-Output as your system output, then add an **Audio Input Capture** in OBS pointed at "BlackHole 2ch".
   - Or [Loopback](https://rogueamoeba.com/loopback/) (paid, easier).

   **Linux (PipeWire — modern Ubuntu / Fedora / Arch):** OBS can capture per-app audio natively. Add an **Audio Output Capture (PipeWire)** source in OBS — it will let you pick a specific application (Zoom, Firefox, etc.) or "Default Output Device". No extra drivers needed. Confirm you're on PipeWire with `pactl info | grep "Server Name"` (should say `PulseAudio (on PipeWire …)`).

   **Linux (PulseAudio — older distros):** add a regular **Audio Output Capture** source; you'll capture whatever sink your speakers are using. To capture *only* the Zoom call rather than every system sound, use `pavucontrol` to route Zoom's output to a dedicated null sink and capture that sink. Easier path: upgrade to PipeWire if your distro supports it.

2. **Route each source to its own track**:
   - In the Audio Mixer panel, click the **⚙ gear** next to your mic → **Advanced Audio Properties**
   - Under "Tracks", uncheck everything except **Track 1** for the mic
   - For Desktop Audio, uncheck everything except **Track 2**
   - (If you have other audio sources you don't want recorded, make sure none of them are on Track 1 or 2)

3. **Tell OBS to record both tracks**:
   - **Settings → Output**
   - Set **Output Mode** to **Advanced**
   - Go to the **Recording** tab
   - **Recording Format**: `mkv` (safest — MP4 corrupts if OBS crashes; you can remux to MP4 later) or `hybrid_mp4`
   - **Audio Track**: tick **1** AND **2** (and any others you've configured)
   - **Audio Encoder**: keep AAC or whatever default
   - **Audio Bitrate** (per track): 192 kbps is plenty for speech

4. **Set sample rate to 48 kHz**:
   - **Settings → Audio → Sample Rate: 48 kHz** (Whisper internally resamples to 16 kHz, but starting from a clean 48 kHz signal is better than upsampled 44.1)

5. **Save the profile** so you don't have to redo this: **Profile → Rename / Duplicate**.

### Per-recording checklist (30 seconds)

- Check Audio Mixer: mic and Desktop both showing levels
- Hit Record
- Talk briefly, confirm both meters move
- After the interview, the resulting `.mkv` will have **two audio streams** — Scribe detects this automatically and uses Multi-track Mode

### Verify a recording has multiple tracks

```bash
ffprobe -v error -show_entries stream=index,codec_type,channels:stream_tags=title -of default recording.mkv
```

You should see two `audio` streams. If you only see one, OBS isn't routing correctly — recheck step 2.

### Mic technique tips that improve transcription accuracy

- Use a directional mic (cardioid) close to your mouth (~6 inches). Lavaliers are fine. Built-in laptop mics are the single biggest accuracy killer.
- Record in a quiet, soft-furnished room. Hard walls = reverb = errors.
- For the remote person on Zoom, ask them to use a headset; their audio is what hits Track 2 directly via Zoom.
- Avoid noise suppression filters in OBS or Zoom during recording — they distort speech in ways Whisper handles worse than the original noise.

---

## Editor & playback review

After a transcription finishes, click **Open in editor →** on the result card (or visit `/edit/<job_id>`). The editor lets you review and correct the transcription with the recording in sync:

- **Word-level highlighting** — every word lights up as the recording plays. The current segment is also marked. Auto-scroll keeps the active word centred (toggle "Follow" off to scroll freely).
- **Click-to-jump** — click any word or segment timestamp to seek playback there.
- **Inline editing** — every segment is contenteditable. Edits resync timestamps proportionally across the segment span on blur. Plain `Enter` closes editing without a line break; `Shift+Enter` splits at the cursor.
- **Segment menu (⋮)** — split, merge with previous/next, insert after, reassign speaker, add annotation, delete.
- **Speakers bar** — click any chip to rename a speaker (rename propagates to all segments). Add new speakers with `+ Add`. Reassign a segment's speaker by clicking its label, or with `Ctrl+1..9` while focused.
- **Annotations** — insert observational notes like `[laughs]` or `[unintelligible]` from the segment menu. They render in italic and export as bracketed text.
- **Search** — `Ctrl/⌘+F`. Enter / Shift+Enter for next / previous.
- **Undo/redo** — `Ctrl/⌘+Z` and `Shift+Ctrl/⌘+Z`. ~80 levels.
- **Autosave** — edits are debounced and PUT'd to the server every 1.5s; TXT/SRT/VTT/JSON exports are regenerated on every save. There's also `Ctrl/⌘+S`.
- **Export** — Export menu offers all four formats. They reflect the latest edits.
- **Persistence** — jobs survive a server restart. Every job's state is saved under `outputs/<job_id>/job.json`; edits live alongside in `edited.json`.

Keyboard summary (also visible in-app via `?`):

| Key | Action |
|-----|--------|
| `Space` | Play / pause |
| `Ctrl+←` / `Ctrl+→` | Skip 5 seconds |
| `Esc` | Pause + blur |
| `Shift+Enter` | Split segment at cursor |
| `Ctrl+1..9` | Set speaker N for focused segment |
| `Ctrl/⌘+F` | Search |
| `Ctrl/⌘+Z` / `Shift+Ctrl/⌘+Z` | Undo / redo |
| `Ctrl/⌘+S` | Save now |

## Output formats

After processing, you get several files next to the original recording:

- `<name>.json` — full structured output: word-level timestamps, speaker per word, confidence scores
- `<name>.txt` — plain readable transcript: `[00:01:23] LUKE: Hello there.`
- `<name>.srt` — subtitles
- `<name>.vtt` — WebVTT subtitles

## Troubleshooting

**Slow on first run.** First time WhisperX runs it downloads ~3 GB of weights. Subsequent runs are offline.

**"MPS backend out of memory." (macOS)** Lower batch size: `--batch-size 4` or `--batch-size 2`.

**"CUDA out of memory." (Linux)** Lower batch size: `--batch-size 4` or `--batch-size 2`. If still tight, force int8: `SCRIBE_COMPUTE_TYPE=int8_float16 ./run.sh`.

**`python -m scribe.devices` says CUDA is not available but I have an NVIDIA GPU.** Your PyTorch was likely installed as the CPU-only build. Wipe the venv and rerun setup with the CUDA index: `rm -rf .venv && SCRIBE_TORCH=cu124 ./setup.sh` (use `cu121` for CUDA 12.1). Then `nvidia-smi` to confirm the driver is loaded.

**Diarization "permission denied" / 401.** Make sure you accepted the licenses on both pyannote model pages and your `HF_TOKEN` is set in `.env`.

**Diarization is slow on Apple Silicon.** Pyannote on MPS sometimes falls back to CPU silently. The default uses CPU for diarization on Apple Silicon (it's actually faster than the partial-MPS path). Force MPS if you want: `SCRIBE_DIARIZE_DEVICE=mps ./run.sh`.

**Audio file has weird format.** Run `ffmpeg -i input.xyz -c copy out.mkv` first to remux.

**Two-speaker interview but diarization says 3.** Use `--num-speakers 2`. Or switch to multi-track mode.

## Architecture

```
input.mp4 (or .mkv with multiple audio tracks)
        │
        ▼
   ffmpeg ── extract each audio track to 16 kHz mono WAV
        │
        ▼
   Multi-track mode:           Single-track mode:
   for each track:              ┌─ pyannote VAD → speech segments
     WhisperX large-v3          ├─ pyannote diarization → speaker turns
     (VAD-chunked,              └─ WhisperX large-v3 (VAD-chunked)
      forced-aligned)              + alignment (wav2vec2)
        │                             + speaker assignment per word
        ▼                             │
   merge all tracks ◄──────────────────┘
   on the timeline
        │
        ▼
   write JSON / TXT / SRT / VTT
```
