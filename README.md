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

### Linux (AMD GPU / ROCm)

The full Whisper + alignment + diarization pipeline runs on AMD GPUs via the
PyTorch ROCm 6.3 build and CTranslate2 v4.7.0+'s ROCm wheel. **Linux only**;
RDNA 3 / RDNA 4 (RX 7000, RX 9000) are first-class; RDNA 2 (RX 6000) works
with auto-applied env-var workarounds; Parakeet (NeMo) is NVIDIA-only and
will be hidden in the UI.

```bash
sudo apt install ffmpeg python3.11 python3.11-venv unzip curl
cd scribe
./setup.sh              # one-time, installs the CUDA/CPU stack first
./setup.sh --rocm       # swap in ROCm PyTorch + CTranslate2 ROCm wheel
```

Verify it took:

```bash
.venv/bin/python -m scribe.devices    # should show "GPU backend: rocm"
.venv/bin/python -m scribe.scripts.check_rocm   # smoke-tests the full layer
```

Caveats:
- AMD officially supports Ubuntu 22.04/24.04 and RHEL 9.7/10.1 for consumer
  Radeons. Other distros (Fedora, Arch, Debian) work in practice but aren't in
  AMD's matrix.
- RDNA 2 cards may also need `export HSA_OVERRIDE_GFX_VERSION=10.3.0` if they
  hit allocator faults; Scribe sets `CT2_CUDA_ALLOCATOR=cub_caching` for you.
- See `docs/research/amd-rocm-research.md` for the full hardware/distro
  picture and known upstream bugs.

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

## Choosing a model

The model dropdown groups two engine families.

### Whisper (multilingual, default)

The path this project was built around. Best general-purpose accuracy, multilingual.

| Option | When to use |
|---|---|
| `large-v3` | Best raw accuracy. Default. |
| `large-v3-turbo` | Same accuracy as large-v3, **~8× faster**. Free upgrade for English-heavy or single-language work; recommended on most hardware. |
| `large-v2` | Older. Use only if you have specific accuracy regressions on v3. |
| `medium.en` | Faster, English-only, slightly lower accuracy. |
| `distil-large-v3` | A distilled student of large-v3. ~6× faster, ~2% WER worse on English. EN only. |

### NVIDIA Parakeet (English-only, GPU)

Optional engine path. For English-only recordings on a CUDA GPU, Parakeet TDT 0.6B is materially faster than Whisper and competitive on accuracy (top of the Hugging Face Open ASR leaderboard at the time of writing).

| Option | Notes |
|---|---|
| `parakeet-tdt-0.6b-v2` | Stable, ~30× faster than Whisper large-v3 on RTX cards. **English only.** |
| `parakeet-tdt-0.6b-v3` | Newer, marginal improvements. |

**Install** (one-time, separate from main install):

```bash
source .venv/bin/activate
pip install -r requirements-parakeet.txt    # ~4 GB of NVIDIA NeMo
```

The first run will download Parakeet weights (~2.4 GB) from Hugging Face. After that it's offline like everything else.

**Pipeline.** Parakeet replaces only the *transcription* step. Voice-activity chunking (pyannote VAD), word-level forced alignment (wav2vec2), and speaker diarization (pyannote) are all unchanged — so word highlighting in the editor and multi-track / diarize modes work identically. The long-monologue fix carries over.

**Trade-offs:**

- English only. Pick a Whisper model for any other language.
- Requires CUDA. CPU-only Parakeet is technically possible but extremely slow, not worth it.
- Slightly more VRAM than int8-quantized Whisper. The 6 GB RTX 1000 used for testing this comfortably runs Parakeet TDT 0.6B alongside the alignment + diarization models loaded sequentially.
- NeMo carries large transitive dependencies (PyTorch Lightning's full ecosystem, Numba, Hydra, etc.), which is why the install is opt-in.

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

## Advanced settings & profiles

The upload card has an **Advanced settings** panel (collapsed by default). Every knob has an inline tooltip; here's a deeper guide.

| Setting | Default | What it does |
|---|---|---|
| **Beam size** | 5 | Beam-search width. 1 disables beam search (greedy decode, fastest). Higher = slower but more accurate. Diminishing returns past 8. |
| **Best of** | 5 | Sampling candidates considered when temperature > 0. Used by Whisper's fallback path when a chunk fails the hallucination guards. |
| **Temperature** | 0.0 | 0 = deterministic. Whisper auto-falls-back to higher temperatures internally if a chunk trips the guards; setting it > 0 just nudges the starting point. |
| **No-speech threshold** | 0.45 | If the model's no-speech probability exceeds this, the chunk is dropped. Raise toward 0.6 if you're getting hallucinated text in silent stretches; lower if real speech is being skipped. |
| **Compression ratio cap** | 2.4 | gzip ratio of decoded text. Above this, the chunk is treated as broken. Lower (e.g. 2.0) is stricter — useful if you see the same phrase repeated in a loop. |
| **Condition on previous text** | off | When on, prior text feeds into the next chunk's context. Off by default — it's the main source of repeat-loop hallucinations on long speech. Turn on only if you see context-loss across boundaries. |
| **Chunk size (s)** | 30 | Hard cap on chunk length. Whisper's training window is 30s; below that has no upside. Lowering (e.g. 20) can help with very fast speakers. |
| **VAD onset** | 0.500 | Speech-start detection threshold. Higher = stricter (skips quieter starts). |
| **VAD offset** | 0.363 | Speech-end detection threshold. Higher = stricter. |
| **Initial prompt** | empty | Free text prepended to the model's context. **The single highest-leverage knob for domain accuracy.** Drop in speaker names, technical terms, product names — Whisper will bias toward them. |
| **Hotwords** | empty | Comma- or space-separated bias words. A lighter version of `initial_prompt` for proper-noun nudges. |
| **Batch size** | 8 | How many chunks faster-whisper processes in parallel. Lower if you hit OOM on a small GPU; raise to 16/32 on big GPUs. |

### Tips for accuracy on tough recordings

- **Domain jargon (legal/medical/technical):** put the most-misheard terms in `Initial prompt`, e.g. *"This interview discusses 21 CFR Part 11, GxP, validation lifecycles, and OQ/PQ documentation."*
- **Whisper repeating the same phrase in a loop:** lower `Compression ratio cap` to 2.0–2.2 and ensure `Condition on previous text` is OFF.
- **Whisper fabricating sentences in long silences:** raise `No-speech threshold` to 0.55–0.65.
- **Ultra-fast speaker getting truncated:** lower `Chunk size` to 20 and beam size 8.

### Profiles

Above the advanced fields is a profile dropdown. Save the current settings as a named profile (stored on disk in `profiles.json` next to the project), then apply it on future recordings — useful for "Customer interview", "Legal deposition", "Internal meeting" presets.

- **Apply** loads the selected profile's settings into the form (and into your browser's persisted state).
- **Save as profile…** prompts for a name and writes the current form state to disk.
- **Delete** removes the profile from disk.

The form also remembers your last-used values per browser (localStorage) — profiles are the cross-browser layer on top of that.

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

## Testing

Scribe has two test suites — a fast Python unit suite and a fast JS unit suite. A pre-commit hook runs both before allowing a commit.

### Python (pytest)

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest                       # fast suite (no models, no GPU)
.venv/bin/pytest -m slow               # real-model integration tests
.venv/bin/pytest -m gpu                # tests requiring CUDA
.venv/bin/pytest --cov=scribe          # with coverage
```

The default invocation deselects `slow` and `gpu` markers so the suite stays under a few seconds. Real-model tests live behind `-m slow` and load whisperx/pyannote/NeMo, so you'll only run them locally during integration work.

### JavaScript (Vitest + jsdom)

```bash
npm install        # one-time
npm test           # run once
npm run test:watch # watch mode
npm run test:coverage
```

JS tests live in `tests/js/**/*.test.mjs` and exercise the pure helpers in `scribe/static/js/helpers.mjs` (formatters, ETA math, word-highlight search, etc).

### Pre-commit hook

After cloning, install the pre-commit hook so tests run before every commit:

```bash
./scripts/install-hooks.sh
```

It runs the fast pytest suite + Vitest. If either fails, the commit is blocked. Bypass with `git commit --no-verify` if you must, but the suite is fast enough (~5s for both) that there's rarely a reason to.

## Troubleshooting

**Slow on first run.** First time WhisperX runs it downloads ~3 GB of weights. Subsequent runs are offline.

**"MPS backend out of memory." (macOS)** Lower batch size: `--batch-size 4` or `--batch-size 2`.

**"CUDA out of memory." (Linux)** Lower batch size: `--batch-size 4` or `--batch-size 2`. If still tight, force int8: `SCRIBE_COMPUTE_TYPE=int8_float16 ./run.sh`.

**`python -m scribe.devices` says CUDA is not available but I have an NVIDIA GPU.** Your PyTorch was likely installed as the CPU-only build. Wipe the venv and rerun setup with the CUDA index: `rm -rf .venv && SCRIBE_TORCH=cu124 ./setup.sh` (use `cu121` for CUDA 12.1). Then `nvidia-smi` to confirm the driver is loaded.

**`Could not load library libcudnn_ops_infer.so.8` (Linux + CUDA).** ctranslate2 4.4.x links against cuDNN 8, but modern PyTorch wheels ship cuDNN 9. requirements.txt pins ctranslate2 ≥ 4.6 which uses cuDNN 9 — if you have a stale install, run `./setup.sh --realign`.

**`TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'`.** huggingface_hub 1.x dropped the `use_auth_token` kwarg, but pyannote-audio 3.4 still passes it. requirements.txt pins `huggingface_hub<1.0` and `transformers<5` together to keep both happy. If pip resolved different versions (typically after installing NeMo or another package that upgrades transformers), run `./setup.sh --realign` to force-reinstall the pinned versions.

### Realigning a drifted venv

```bash
./setup.sh --realign
```

This force-reinstalls the three packages that drift in practice (`huggingface_hub`, `transformers`, `ctranslate2`) at their pinned versions, then runs `pip install -r requirements.txt` to bring everything else in line, then prints the device + version report so you can confirm the fix took. Use this any time `python -m scribe.devices` shows a package outside `requirements.txt`'s constraints.

**`UnpicklingError: Weights only load failed ... Unsupported global: omegaconf.listconfig.ListConfig`.** PyTorch 2.6 flipped the default of `torch.load` to `weights_only=True`, and pyannote / whisperx checkpoints contain config containers that the strict loader rejects. Scribe handles this automatically: it allowlists the known-safe globals via `torch.serialization.add_safe_globals` and falls back to the legacy load path for any remaining cases (the model files come from HuggingFace via verified hashes). To enforce strict mode anyway and surface the failure, set `SCRIBE_STRICT_TORCH_LOAD=1` before launching the server.

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
