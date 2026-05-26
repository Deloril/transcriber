# Scribe — Claude Code context

This file is for future Claude Code sessions arriving at this repo
cold. It's the same kind of "if I had to pick this up tomorrow"
brief I'd write for myself. Read `PLANNING.md` and `README.md`
alongside it; they cover *what we're building* and *how a user runs
it* respectively. This file covers *how the codebase is organised
and how I've worked on it*.

## What this project is

A local-first transcription + qualitative coding tool. Replaces
Scriberr (which fails on long monologues). Scope has grown well
beyond simple transcription:

1. **Transcription** — Whisper / WhisperX / Parakeet, with VAD
   chunking, alignment, diarization. Mac (MPS), NVIDIA (CUDA),
   AMD (ROCm), CPU fallback. Multi-track or auto-diarize modes.
2. **Editor** — word-level transcript editor with media playback
   (synchronised word highlighting), speaker rename, find/replace,
   inline split / merge-up / annotate buttons. Recently added:
   ✨ Tidy speech with AI (grammar bot — see "Recent additions").
3. **Academic coding** — full constructivist-grounded-theory
   tooling: projects → sources → participants, codes with
   versioned definitions, applications with anchor offsets,
   memos, multi-coder ICR, REFI-QDA export.
4. **Local AI** — Ollama-backed code suggestions, new-code
   proposals, second-coder pass, memo drafts. F8.10 gates AI
   until enough hand coding has happened.

The user is the sole researcher building this for their own PhD
work. They write to me as a senior collaborator — terse responses,
no excess preamble.

## How the code is laid out

```
scribe/
  server.py                  # FastAPI app — ~4080 lines, the seam
  engine.py                  # transcription pipeline (Whisper/Parakeet)
  parakeet.py
  audio.py
  devices.py                 # CUDA/ROCm/MPS/CPU detection
  model_tiers.py             # tier picker (F8.11)
  model_recommendations.py   # default model picks (F8.12)

  # Project model (F1)
  projects.py
  sources.py
  participants.py
  participant_sources.py

  # Codebook (F2)
  codes.py                   # Code entity
  code_versions.py           # immutable version history (F2.2/F9.2)
  code_lifecycle.py          # merge/split/rename/retire (F2.3)
  codebook_lock.py           # F2.4 locked-codebook stage
  codebook_lock_audit.py
  codebook_snapshots.py      # F9.3
  codebook_export.py
  coders.py
  applications.py            # Code-applied-to-span; carries AIProvenance
  application_spans.py
  application_reanchor.py    # keep anchors valid through edits
  application_gutter.py      # editor sidebar
  application_playback.py
  application_provenance_display.py
  definition_at_apply.py     # F9.2 round-trip helper

  # Memos (F5)
  memos.py
  memo_canvas.py
  memo_context.py
  memo_drafts.py             # AI memo drafting (F8.8)
  memo_export.py
  memo_promote.py            # promote a memo to a code

  # AI (F8)
  ai_backend.py              # ModelBackend ABC, Ollama provider
  ai_provenance.py           # canonical AI_FEATURES + AI_DECISIONS
  ai_invocation_log.py       # F9.6
  ai_gate.py                 # F8.10
  ai_second_coder.py         # F8.7
  global_ai_backend.py       # NEW: project-less BackendConfig
                             # for the editor's grammar bot
  embedding_index.py
  code_suggestions.py        # F8.3 — suggest existing code for span
  new_code_suggestions.py    # F8.4 — propose new codes
  transcript_tidy.py         # NEW: grammar bot's pure logic

  # Reliability + reproducibility (F9)
  event_log.py               # F9.1 append-only
  audit_export.py            # F9.7
  project_checkpoints.py     # F9.4
  refi_qda_project.py        # REFI-QDA XML export

  # Other
  library.py                 # F10.1 home-page job summaries
  transcript_import.py       # F10.3 import existing transcripts
  anonymise.py
  speaker_map.py
  matrix.py / matrix_export.py
  icr.py                     # Cohen's kappa, Krippendorff's alpha

  templates/
    editor.html              # ~2200 lines
    source_coding.html       # ~970 lines (the coding view)
    codebook_editor.html
    project_subpage.html     # generic wireframe shell
    home.html / library.html
    _shell.html              # base template
  static/js/
    helpers.mjs              # pure JS helpers (find/replace, formatters)

tests/
  conftest.py
  fixtures/
  js/                        # Vitest tests
  test_*.py                  # 5235 passing pytest tests
```

## Working agreements I've internalised

- **Pre-commit hook runs the fast pytest suite + vitest.** Both
  must pass. Don't bypass with `--no-verify` — investigate
  failures.
- **Pure modules first.** Most features are: pure Python module
  with deep unit tests, then a thin FastAPI wrapper in
  `server.py`, then template + JS. The pure module is the
  contract; wrappers + UI follow it.
- **Tests next to the module.** `scribe/foo.py` →
  `tests/test_foo.py`. Endpoints in `server.py` go in
  `tests/test_server_*.py` (one file per feature surface — e.g.
  `test_server_ai_suggestions.py`, `test_server_tidy.py`).
- **Use `_*_override` module-level globals for backend injection
  in tests.** Patterns:
  - `_ai_backend_transport_override` — swaps the urllib transport.
  - `_ai_suggest_backend_override` — `(BackendConfig, FakeBackend)`
    tuple for the AI suggestion endpoints.
  - `_tidy_backend_override` — same shape for the grammar bot.
  - `_model_tiers_snapshot_override` — for tier picker tests.
  Reset every override in the test fixture.
- **Storage paths via module-globals**: `UPLOAD_DIR`,
  `OUTPUT_DIR`, `ROOT`, `PROJECTS_DIR`. Tests `monkeypatch.setattr`
  them onto a tmp dir.
- **Keep changes scoped.** A bug fix isn't a refactor. The user
  has explicitly asked for tight, focused changes.
- **No emojis in code unless the user asks.** They have asked,
  so the editor + coding view do use them (✨, ✂, ⬆, ＋, ⋮, 🗂,
  🗑, 📼). Stick to that vocabulary.

## Patterns to follow when adding a feature

1. **Author the pure module + tests.** Run them in isolation.
2. **Add the FastAPI route** in `server.py`. Convention: keep
   route handlers thin; they extract from `request.json()`,
   call into the pure module, return `JSONResponse`.
3. **Add `tests/test_server_<feature>.py`** with a `TestClient`
   fixture that monkeypatches storage globals + relevant
   overrides.
4. **Wire into the template** (editor / coding view / home).
   Keep new HTML next to the most-related existing block.
5. **Add a Vitest test** in `tests/js/` if the new JS is non-
   trivial — pure helpers go in `helpers.mjs` for testability.
6. **Update `PLANNING.md`** if you've completed an enumerated
   feature ID. Don't add new IDs without checking — the user
   curates the backlog.

## Recent significant additions (in commit order, newest first)

- **`00d8c4d` ✨ Tidy speech with AI** — grammar bot in the editor.
  - `scribe/global_ai_backend.py` — project-less BackendConfig at
    `~/.scribe/ai_backend.json` (atomic write, defensive parsing).
    The editor isn't bound to a project so it needs its own
    config. To configure: drop a JSON file with at least
    `default_model` set; same shape as project AI backend
    settings. Honours `SCRIBE_HOME` env var for tests.
  - `scribe/transcript_tidy.py` — pure module:
    - `group_runs(segments)` — maximal consecutive same-speaker
      blocks, ≥2 segments, capped at 6000 words / 30 min.
    - `realign_words(old_words, paragraphs, ...)` — uses
      `difflib.SequenceMatcher` on case- and punctuation-stripped
      tokens. `equal` blocks keep timestamps verbatim; new tokens
      interpolate into surrounding gaps. **This is what keeps
      playback word highlighting working after a rewrite.**
    - `assemble_tidied_segments(...)` — one segment per paragraph.
    - `splice_run(transcript, segment_indices, new_segments)` —
      replaces a contiguous run; no mutation.
  - `server.py` endpoints (search for "Transcript tidy-up"):
    - `GET /api/job/<id>/tidy/runs` — list candidate runs.
    - `POST /api/job/<id>/tidy/preview` — call LLM, return
      proposed paragraphs + realigned segments. Does not persist.
    - `POST /api/job/<id>/tidy/apply` — splice + write via the
      same path as `PUT /transcript`; sidecars (.txt/.srt/.vtt)
      regenerate.
  - `templates/editor.html` — `✨ Tidy` button in the topbar.
    Modal is 960px wide, side-by-side raw vs editable proposed
    text; Suggest / Skip / Accept / Close. Walks runs one at a
    time; refreshes the candidate list after each apply (indices
    shift).
  - **F8.10 gate is intentionally skipped** for this feature —
    that gate is about hand-coding before AI coding, irrelevant
    to transcript cleanup.

- **`c033c9d` F8.3 / F8.4 AI code suggestions** — wired into the
  coding view's apply-popover. Endpoints under
  `/api/projects/<pid>/ai/suggestions(/<sid>/{accept,reject})`.
  Acceptance creates an Application with `AIProvenance` (F8.9);
  rejection keeps the audit row (F9.6). UI: ✨ row in the
  popover, inline panel of ranked candidates with confidence %.
  Uses `record_decision()` from `code_suggestions.py` to seal
  the audit chain.

- **`14f4504` Editor: speaker rename → library + inline reassign
  + merge-up button**:
  - `library._coerce_speakers()` reads `result["speaker_names"]`
    so renamed speakers show up in library rows. Tests for the
    full override matrix in `test_library.py`.
  - Inline `speaker-popover` replaces the modal speaker-reassign
    flow — fewer clicks, anchors to the clicked label.
  - ⬆ merge-up button in `seg-actions` next to ✂ split.

- **`06c3626` Coding view: create-code-on-the-fly + per-code
  colours**:
  - Apply-popover: typing a non-existing code name shows a
    "+ Create code: foo" row. On click, POSTs `/codes`, then
    applies.
  - `codeColours(code)` in `source_coding.html` — FNV-1a hash
    on `code.id` → HSL hue. Skips the orange/red 10–25 band so
    highlights don't read as errors. Returns `{bg, border,
    swatch}`.

- **`20a67d2` Make the academic coding flow actually work
  end-to-end** — replaced wireframes with real working pages for
  project create, projects list, codebook editor, source picker,
  project home, and source coding view. Before this commit the
  loop reported "76/76 done" but most UI was placeholder-only;
  the user said "WTF" and chose to make the project flow real
  rather than continue feature-checking against commit subjects.

- **G1–G6 (commits `0ffc716`–`0ed2a8a`)** — AMD ROCm support.
  4-state device labels (cuda/rocm/mps/cpu), CT2 ROCm wheel
  installer, pyannote LSTM dropout patch for MIOpen, RDNA 2
  workarounds (`CT2_CUDA_ALLOCATOR=cub_caching`,
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`), Parakeet hidden on ROCm,
  smoke test + benchmark. CT2 takes `device="cuda"` on ROCm
  via the HIP shim — engine wrappers translate `rocm` → CT2
  `"cuda"` while keeping the user-facing label honest.

## Errors I've already debugged (don't re-debug)

- **PyTorch `weights_only=True` UnpicklingError on pyannote
  load.** Fix: `_register_safe_globals()` allowlists
  omegaconf/numpy/pyannote types AND a force-override of
  `torch.load` and `torch.serialization.load` to
  `weights_only=False` regardless of caller. Mac required
  `--realign` because pyannote calls `torch.load(weights_only=
  True)` explicitly.
- **`hf_hub_download() got unexpected kwarg 'use_auth_token'`.**
  Fix: pin `huggingface_hub<1.0` in `requirements.txt` +
  defensive shim in `engine.py` translating
  `use_auth_token=` → `token=`. Required `pip install -U
  "huggingface_hub<1.0" "transformers<5"`.
- **`libcudnn_ops_infer.so.8` not found.** ctranslate2 4.4 needs
  cuDNN 8; torch 2.6+ ships cuDNN 9. Fix: pin
  `ctranslate2>=4.6,<4.8`. `setup.sh --realign` exists for this.
- **False "media discarded" indicator** in library rows. Two
  fixes: `library._to_bool()` handles string `"false"` (which
  `bool()` would treat as truthy), AND filesystem-truth
  reconciliation only flags when the upload dir is actually
  missing.
- **Replace-all bug `"ababab" → "Xabab"`.** Right-to-left
  replacement loop was applying overlapping matches incorrectly.
  Fix: separate single-word matches (apply via single regex pass
  per word) from multi-word matches (splice array). See
  `helpers.mjs::replaceInSegmentWords`.

## Test conventions

- Run fast suite: `.venv/bin/python -m pytest tests/
  --ignore=tests/test_engine_pipeline_slow.py
  -m "not slow and not gpu" -q`
- Run a single file: `.venv/bin/python -m pytest tests/test_X.py -x`
- Vitest: invoked by the pre-commit hook; manually
  `npx vitest run` from repo root.
- Slow tests (`@pytest.mark.slow`) hit real models. GPU tests
  (`@pytest.mark.gpu`) need a GPU. Don't run these by default.
- Tests target ~5–15s for the fast suite. Anything that hits a
  network or a real Ollama daemon belongs behind an override.

## Things that are NOT done (per `PLANNING.md`)

These features have IDs in `PLANNING.md` but no implementation
yet, or only partial implementation. Don't claim them as done
without checking the current code:

- **F8.5** Find similar quotes — embedding index exists, UI does
  not.
- **F8.6** Whole-transcript AI review pass as a background job —
  not wired.
- **F8.7** AI second-coder pass — pure module exists
  (`ai_second_coder.py`); UI plumbing not done.
- **F8.11** Model-tier picker UI with download manager — partial.
- **F9.4** Project checkpoints — module exists, UI not wired.
- **F9.7** Audit export — module exists, no download endpoint.
- **F9.8** Time-travel view — not started.

## Things that ARE done that you might mistake for not-done

- **F1.x project entity** — fully implemented; project create
  form works end-to-end.
- **F2.x codebook** — versioning, lifecycle ops, locked-codebook
  stage, ICR all exist as pure modules with tests.
- **F8.3 + F8.4** — endpoints + UI exist (commit `c033c9d`).
- **F10.1 library** — works at `/library` and on the home page.
- **F10.2 discard media** — works including the small "📼
  discarded" icon in library rows.
- **F10.3 import transcript** — endpoint exists at `/api/import`.
- **F11.1 inline buttons** (✂ ⬆ ＋ ⋮) — all wired in
  `editor.html` `renderSegment()`.
- **F11.2 find/replace** — works in the editor; helpers in
  `helpers.mjs`.

## Loop scripts

- `scripts/feature-loop.sh` — overnight `claude -p` loop driver.
  No budget cap; max 200 iterations. Driven by `PLANNING.md`.
- `scripts/feature-implementer-prompt.md` — prompt the loop uses.
- `.loop-pid` — lockfile for the running loop. Safe to delete
  if `kill` confirms the PID is gone.
- The loop's "feature done" heuristic was once
  `git log | grep ^F\d+\.\d+`. That heuristic counted commit
  *subjects* as completion and falsely reported "76/76 done"
  while UI was wireframes. **Trust working features, not commit
  counts.**

## Memory / state

The user's `~/.claude` memory tracks: their role (researcher,
writing for their own PhD), terse-response preference, the
"actually work end-to-end" demand, and the inability to run CU
on Linux (so `claude -p` shell loop is the unattended-runs
strategy). Persistent collaboration preferences live there;
project-specific architectural facts live in this file.

## Quick orientation questions to ask the user if unclear

- "Is this a feature ID from PLANNING.md or new scope?"
- "Should the AI use the project-scoped backend or the global
  one?" (transcript editor → global; everything else → project.)
- "Does this need an audit-trail row?" (F9.6 says rejected AI
  suggestions are evidence too — most AI features answer yes.)
- "Should it pass through the F8.10 gate?" (yes for code work,
  no for transcript cleanup.)
