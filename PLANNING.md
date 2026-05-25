# Scribe — Planning

This document captures the next-phase feature backlog. It's a working planning doc, not a contract. Update as decisions land.

## Status

- ✅ Local transcription pipeline (Whisper + Parakeet, VAD chunking, alignment, diarization)
- ✅ Multi-track + diarize modes
- ✅ Web UI: upload, advanced settings, profiles, recording stats + waveform
- ✅ Transcript editor with playback, word highlighting, inline edit, search, export
- ✅ Persistence across server restarts; per-job error logs
- 🚧 **Test infrastructure** (in progress — pytest + Vitest set up; per-module test files still being written)
- ⬜ **AMD / ROCm GPU support** (this doc)
- ⬜ **Transcription management** (this doc)
- ⬜ **Editor UX polish** (this doc)
- ⬜ **Academic coding engine** (this doc)

---

# Feature: AMD / ROCm GPU support

Until CTranslate2 v4.7.0 (Feb 2026), AMD users had no path to GPU-accelerated Whisper at all. That changed: with the right wheels, **the full Scribe pipeline can run on AMD GPUs on Linux today**. This section captures the plan; deep research is in `docs/research/amd-rocm-research.md`.

## Tiering

The honest story:

| Tier | Hardware | Strategy | Status |
|------|----------|----------|--------|
| **Tier 1** | Linux + RDNA 3 / RDNA 4 (RX 7000, RX 9000) | CTranslate2 ROCm wheel + PyTorch ROCm 6.3 + LSTM dropout patch for pyannote. Full pipeline GPU-accelerated. | Primary path |
| **Tier 2** | Linux + RDNA 2 (RX 6000) | Same stack + `CT2_CUDA_ALLOCATOR=cub_caching` + `HSA_OVERRIDE_GFX_VERSION=10.3.0` env vars. AMD-officially-unsupported but works. | Best-effort |
| **Tier 3** | Windows AMD, RDNA 1, dropped cards | whisper.cpp Vulkan backend. Loses CT2 int8 quants and our existing faster-whisper integration; uses GGUF Whisper weights. | Future, not blocking |
| **Out of scope** | Parakeet on AMD | NeMo is NVIDIA-only; no community fork. Hide Parakeet in the UI when AMD is the active backend. | Won't fix |

## Backlog

### Foundation: detection + device routing

- **G1.1** Detect ROCm at runtime via `torch.version.hip` (PyTorch's ROCm wheels alias `torch.cuda.*` to HIP, so `is_available()` is True on both — `torch.version.hip` is the discriminator).
- **G1.2** Extend `_torch_device()` / `_diarization_device()` / `_whisper_device_and_compute()` to return a 4-state result: `cuda` / `rocm` / `mps` / `cpu`. CTranslate2 still takes `device="cuda"` on a ROCm wheel (HIP shim), so the engine wrappers translate `rocm` → CT2 `"cuda"` while keeping the user-facing label honest.
- **G1.3** `scribe.devices` reports ROCm details (HIP version, gfx target, OS) so support tickets show the right context.
- **G1.4** UI shows the active backend (cuda / rocm / mps / cpu) on the Recording details card.

### CTranslate2 ROCm install

- **G2.1** Installer subcommand or `setup.sh --rocm` that fetches the right ZIP from the CT2 GitHub release (currently v4.7.2), unzips, pip-installs the wheel. Pin a known-good version; surface it in `scribe.devices` so we can spot drift.
- **G2.2** Mirror the wheel on our own host as a fallback if GitHub Releases is unavailable. (Optional, future.)
- **G2.3** Document Linux distro support: Ubuntu 22/24 first-class, RHEL 9/10 supported, Fedora/Arch/Debian best-effort. Windows is Tier 3 only.

### pyannote LSTM dropout patch

- **G3.1** When `torch.version.hip` is set, patch the segmentation model's LSTM dropout to 0.0 after `Pipeline.from_pretrained(...).to(device)`. Inference behaviour is unchanged (dropout is a no-op outside training); avoids the open MIOpen header bug ([pyannote-audio#1995](https://github.com/pyannote/pyannote-audio/issues/1995)). Hide behind the same engine-load shim.

### RDNA 2 workarounds

- **G4.1** Auto-detect gfx103x via `torch.cuda.get_device_name()` and set `CT2_CUDA_ALLOCATOR=cub_caching` in the worker process before loading CT2.
- **G4.2** Document `HSA_OVERRIDE_GFX_VERSION=10.3.0` for users hitting allocator faults on RX 6600/6700.

### Engine routing

- **G5.1** When the active backend is `rocm`, hide the Parakeet model options in the UI dropdown (NeMo doesn't run on ROCm). Show a tooltip explaining why.
- **G5.2** Prefer `int8_float16` on AMD cards with <8 GB (matches our existing CUDA tiering). RDNA 3 24 GB cards (7900 XTX) get `float16`.

### Validation

- **G6.1** Smoke test script (`scribe/scripts/check_rocm.py`) that loads a tiny Whisper model, runs alignment, runs diarization, reports timing. Run on user's machine after install.
- **G6.2** In-house benchmark vs CUDA on a representative file before publishing performance claims.

### Known issues to monitor (live upstream bugs)

- [CT2 #2021](https://github.com/OpenNMT/CTranslate2/issues/2021) RX 9070 XT (gfx1201) crashes on Fedora 43 + ROCm 7.2. Open. Workaround: use Ubuntu 24.04 or wait for upstream fix.
- [CT2 #2038](https://github.com/OpenNMT/CTranslate2/issues/2038) Windows HIP allocator deadlock on `del model`. Confirms Windows is Tier 3 only.
- [pyannote-audio #1995](https://github.com/pyannote/pyannote-audio/issues/1995) LSTM dropout MIOpen header missing on ROCm ≥ 6.1.1. Our G3.1 patch covers this.

## Phasing

- **Phase A** — Detection + device routing + Parakeet hiding + scribe.devices reporting (tasks G1.1–G1.4, G5.1–G5.2). No install changes; just makes the existing code paths AMD-aware. Ships a dev-time win even before the wheel is installed.
- **Phase B** — `setup.sh --rocm` installer that fetches the CT2 ROCm wheel (G2.1, G2.3). LSTM dropout patch (G3.1). RDNA 2 workarounds (G4.1, G4.2). This is the milestone that says "AMD GPU support shipped."
- **Phase C** — Smoke-test script + in-house benchmark (G6.1, G6.2). Mirror the wheel (G2.2). Optional whisper.cpp Vulkan backend for Tier 3 if there's demand.

---

# Feature: Transcription management

Today, every transcription job creates `outputs/<job_id>/` and `uploads/<job_id>/` and the only way to get back to one is the `/edit/<job_id>` URL the user happens to remember. These three features close that gap so the home screen becomes a real library, source media can be reclaimed without losing transcripts, and externally-produced transcripts can join the set.

## Backlog

- **F10.1** Library view: a list of all completed (and in-progress) transcriptions on the home page, or one click away under `/library`. Shows filename, duration, mode (multi-track / diarize), speaker count, language, created date, status, and per-row actions (open editor, download, delete). Sortable, with a search box that matches filenames and detected speakers. Backed by the persisted `outputs/<id>/job.json` files we already write — no schema changes for the basic view. Surfaces interrupted-by-restart jobs (status=`error`) so the user sees them and can clean them up.

- **F10.2** Delete the source media while keeping the transcript. Today, source files (`uploads/<id>/<file>`) are needed for the editor's media playback (`/api/job/<id>/media`). After F10.2:
  - A **"Discard source media"** action on each library row and on the editor page.
  - Confirms once with a clear "playback will no longer work for this transcription" warning.
  - Removes `uploads/<id>/` and rewrites `job.json` with a `media_discarded: true` flag.
  - The editor degrades gracefully when the flag is set: hides the player, shows a notice, keeps everything else (transcript text, word-level edit, search, export).
  - The library view shows a small icon when source media has been discarded.
  - Outputs (`outputs/<id>/*.json/.txt/.srt/.vtt/edited.json/waveform_*.json`) are untouched.
  - Disk-space rationale: video sources can be hundreds of MB; once the transcript is final and reviewed, users want to reclaim that space without losing their edited transcript or coding work.

- **F10.3** Import an existing transcript. Three import paths researchers actually have:
  1. **Plain text + speaker labels** (`[00:01] LUKE: Hello.` style — what we export as `.txt`).
  2. **SRT / VTT subtitle files**.
  3. **Scribe JSON** (the same shape we produce in `outputs/<id>/<name>.json`, including word-level timestamps if present).
  - Drag-and-drop on the home page (alongside the existing audio/video drop), with a content-type sniff to decide which parser to use.
  - Optionally upload a *companion media file* alongside the transcript so playback works. If no media is provided, the resulting job lands with `media_discarded: true` from the start (F10.2 already gave us that flag).
  - Treat the import as a finished job: skip the engine entirely, populate `result` directly from the parsed transcript, write the standard sidecars (`.txt/.srt/.vtt/.json`), assign a normal job id, set `status=done`, jump straight to the editor on success.
  - Word-level timestamps when missing (TXT, SRT, VTT) are synthesised proportionally across each segment span using the same `spreadTokensAcrossSpan` helper the editor already uses for resync after edits.

- **F10.4** Fix the library-row action overflow. On a 14" MacBook (and any window narrower than ~1180 px), the third action button — **Delete** — gets clipped off the right edge because `td.actions` has `white-space: nowrap` and no width budgeting. The Delete button is functionally invisible to the user. Fix:
  - **Primary**: shrink the action labels and let the column wrap on narrow viewports. `td.actions` should keep buttons readable but allow `white-space: normal` below ~1100 px, or condense labels to icons + tooltip (▶ Open · 🗑 Discard media · ✕ Delete) to keep them on one line.
  - **Stretch**: collapse secondary actions into a per-row `⋮` dropdown when the table doesn't fit. Open stays as a primary button; Discard media + Delete move into the dropdown. Mirrors the editor's F11.1 inline-button + dropdown pattern.
  - **Test**: a pytest assertion that the rendered library row does not depend on horizontal overflow to display every action — i.e. the `<td class="actions">` doesn't have `white-space: nowrap` once viewport-aware CSS lands. Alternatively a Vitest+jsdom test for the action layout if we extract the row into a helper.
  - Files: `scribe/templates/library.html` only. No backend change.

## Phasing

- **Phase A — Library + delete source** (F10.1, F10.2). Needs only persistence + UI work; no parsing.
- **Phase B — Import** (F10.3). Builds on F10.2's `media_discarded` flag for the no-media path.
- **Phase C — Library polish** (F10.4). Pure CSS / layout fix for the action-button cutoff on 14" screens.

---

# Feature: Editor UX polish

The transcript editor today hides every segment-level action behind the `⋮` dropdown. Two of those actions — **split at cursor** and **add annotation** — are used so frequently during review that hunting for them in a menu is friction. Promote them to inline buttons; leave the rest in the dropdown.

## Backlog

- **F11.1** Per-line inline action buttons for the two high-frequency segment ops:
  - **Split at cursor** (✂) — splits the segment at the current caret position. Same behaviour as `Shift+Enter` and the `⋮ → Split at cursor` menu item; this just gives the action visible affordance.
  - **Add annotation** (＋) — opens the annotation modal with the current segment pre-selected, same as `⋮ → Add annotation`.
  - Visible on hover (and when the segment is focused), keyboard-accessible, paired with the existing `⋮` button. The dropdown keeps merge/insert/reassign/delete to avoid cluttering the row with 7 buttons.
  - No new keyboard shortcuts; the existing `Shift+Enter` (split) and `?` shortcut help still apply.
  - Implemented in `scribe/templates/editor.html` — adjusts the existing `.seg-menu-btn` rendering to add two siblings, reuses `splitAtCursor()` and `openNoteFor()` directly. Vitest only if extracting helpers; otherwise just a small edit.

---

# Feature: Academic coding engine

A Scribe project after a transcription is "done" should also be a **research project** that can be coded using the constructivist grounded theory (Charmaz) workflow. Multiple transcripts share a single codebook. Codebooks evolve across coding stages (initial → focused → axial → theoretical). Locally-running AI models can suggest codes, similar quotes, or memo drafts — *never auto-apply*. Every coded segment carries an audit trail sufficient for a methodologically transparent thesis or paper.

## Core principles

These shape every feature decision below.

1. **Local-only AI.** No cloud calls, ever. Models run via Ollama / llama.cpp / transformers under the user's control. Same constraint as the rest of Scribe.
2. **AI suggests, the human applies.** Every code application has a human author. AI suggestions are logged whether accepted or rejected, but never become applications without an explicit accept event.
3. **Word-level anchoring.** Code applications anchor to Scribe's existing word IDs from the ASR pipeline, not raw character offsets. Transcript edits don't break the audit trail.
4. **REFI-QDA / QDPX is the lingua franca.** Export from day one; import as a later milestone. No file-format lock-in.
5. **Audit trail from day one.** Append-only event log; code definition versioning; AI invocation log; codebook snapshots. The data model has to support this from the start — bolting it on later is painful.
6. **Charmaz-aligned defaults but methodologically pluralistic.** Gerund-form code-name suggestions, axial coding optional not required, reflexive memos as a first-class memo type. But don't lock users into Charmaz: Strauss/Corbin and Glaser users should be able to use the same tool.

## Glossary

- **Project** — a research corpus: multiple transcripts, one shared codebook, project-wide metadata.
- **Source** — a single transcript (or later: field notes, document, image).
- **Code** — a label with a definition, criteria, exemplars, version history.
- **Application** — one specific instance of a code attached to a span of text in a source. (Other tools call this a "coded segment" or "quotation.")
- **Memo** — free-form analytic note; can be code-attached, source-attached, project-attached, or free-floating.
- **Codebook stage** — initial / focused / axial / theoretical / locked.

---

## Backlog

Feature IDs match the deep-research report (`docs/research/coding-engine-research.md`, to be checked in alongside).

### Foundation: data model + project shell

- **F1.1** Project entity (name, research question, methodology, sensitising concepts, created/modified, current codebook stage).
- **F1.2** Source entity (transcript ID linkage, type, language, recording date, custom attributes).
- **F1.3** Participant entity (one participant ↔ many sources; demographic columns user-defined).
- **F1.4** Sampling log on the project (which sources added when, why — theoretical-sampling justification).
- **F1.5** Project file format on disk: JSON manifests + the existing `outputs/<job>/` artefacts. Designed to round-trip through REFI-QDA.

### Codebook (§2)

- **F2.1** Code entity with full field set: UUID, name, definition, inclusion criteria, exclusion criteria, exemplars, parent code, related codes (typed), theoretical memo, stage, colour, status, provenance.
- **F2.2** Code revision history. Every definition edit creates a new immutable version; applications record version-at-apply.
- **F2.3** Lifecycle ops: merge, split, rename, retire, promote/demote in hierarchy. All preserve back-pointers and don't destroy history.
- **F2.4** Locked-codebook stage marker. Toggle prevents new codes / edits but allows applications. Unlock requires a methodological memo with a reason.
- **F2.5** Multi-coder mode. Per-coder application; ICR computation (Cohen's kappa first, Krippendorff's alpha later); reconciliation UI.
- **F2.6** Codebook export: CSV, structured Markdown, RTF/Word (with definitions and exemplars), REFI-QDA Codebook XML.

### Multi-transcript projects (§3)

- **F3.1** Project shell holds {sources, participants, codebook, memos, settings}.
- **F3.2** Source attribute schema (user-defined columns per source).
- **F3.3** Participant ↔ source mapping; one participant can have multiple sources.
- **F3.4** Speaker awareness in queries (separate interviewer from interviewee; focus group support).
- **F3.5** Query builder: code filter + source filter + participant attribute filter + speaker filter + boolean code combinator + proximity (within span / paragraph / source).
- **F3.6** Matrix views: code × source (frequency); code × code (co-occurrence); code × attribute (cross-tab).
- **F3.7** Saved queries (named, re-runnable).

### Highlighting and selection (§4)

- **F4.1** Application = (code_id, source_id, anchor_start_word_id, anchor_end_word_id, optional sub-word char offsets, coder_id, created_at, optional confidence/provenance, definition_version_id_at_apply).
- **F4.2** Multiple non-contiguous applications per (code, source).
- **F4.3** Unlimited overlapping codes on a span; gutter/margin renderer.
- **F4.4** Snap-to-word / sentence / paragraph selection helpers.
- **F4.5** Re-anchoring strategy on transcript edit; "orphaned application" review queue when anchors are deleted.
- **F4.6** One-click playback from any coded segment (reuse the editor's word→time map).

### Memos (§5)

- **F5.1** Memo entity with type (code / theoretical / methodological / reflexive / quote / source / project), rich text body, multi-target links.
- **F5.2** Right-click memo creation from any context with link pre-populated.
- **F5.3** Memo-sorting canvas (drag/group cards, link memo→memo, link memo→category).
- **F5.4** "Export all memos" filtered by type / linked-to.
- **F5.5** Promote a memo into a code definition (one click).

### Export (§6)

- **F6.1** Codebook export (CSV / Markdown / Word).
- **F6.2** Coded-segment retrieval report (per code, filterable, grouped by source / participant).
- **F6.3** Frequency and co-occurrence matrix exports (CSV / XLSX).
- **F6.4** REFI-QDA / QDPX project export (priority — this is the "no lock-in" feature).
- **F6.5** REFI-QDA Codebook XML export.
- **F6.6** REFI-QDA / QDPX project import (later milestone).
- **F6.7** Anonymised export — regenerate transcripts with a redaction pass before bundling.

### Local AI (§8)

- **F8.1** Pluggable model backend abstraction (Ollama HTTP API first; llama.cpp / transformers later). User selects model from a list.
- **F8.2** Embedding index of every coded segment + every uncoded paragraph. Built on import; refreshed on edit.
- **F8.3** "Suggest codes from existing codebook" action on a highlighted span. Ranked list combining embedding similarity to existing exemplars + LLM-driven analysis. Explicit accept / modify / reject buttons.
- **F8.4** "Suggest a *new* code" action — separate command, requires explicit invocation. The phrasing in the UI nudges toward gerund-form Charmaz-style names.
- **F8.5** "Find similar quotes" action on any quote (semantic search on the embedding index).
- **F8.6** Whole-transcript AI review pass as a background job. Produces a list of suggestions for review; never auto-applies.
- **F8.7** AI second-coder pass on a locked codebook. Diffs against human coding; ICR view.
- **F8.8** Memo-draft action on a code (LLM seeds with the code's exemplars; researcher rewrites).
- **F8.9** AI provenance fields on every application + every event.
- **F8.10** First-N-transcripts AI-off mode. Configurable threshold (default: AI suggestions disabled until codebook has ≥ 8 codes AND ≥ 2 transcripts hand-coded). Rationale: protects the inductive opening of grounded theory.
- **F8.11** Model-tier picker with hardware autodetection. Tiers: small (3B / laptop / 8GB GPU or CPU), mid (8–14B / 16GB GPU), large (32–70B / 24GB GPU). Includes a download manager.
- **F8.12** Model recommendations baked in: laptop default Llama 3.2 3B or Phi-3.5 3.8B; mid-tier Phi-4 14B or Mistral Nemo 12B; large-tier Qwen 2.5 32B or Llama 3.3 70B. Embedding default `bge-m3` (multilingual) or `nomic-embed-text-v1.5`.

### Trust & reproducibility (§9)

- **F9.1** Append-only event log. Every operation becomes an event with timestamp, actor, and full payload diff. Never deletable.
- **F9.2** Code definition versioning. Each edit creates a new version; applications record version-at-apply. Reports can show "this code's definition at the time this application was made."
- **F9.3** Named codebook snapshots ("Initial coding done 2026-04-12"). Reports can be regenerated against any snapshot.
- **F9.4** Project checkpoints (full project state save; git-like).
- **F9.5** Locked-codebook mode with reason-to-unlock memo.
- **F9.6** AI invocation log including *rejected* suggestions (rejected suggestions are evidence too).
- **F9.7** Audit trail export (chronological Markdown / Word; filterable).
- **F9.8** "Time-travel" view — display the project read-only as it was on date Y.
- **F9.9** Per-application provenance display on hover.

---

## Phasing

Suggested order for implementation. Each phase is a shipping milestone; nothing in a later phase should be required for earlier phases to be useful.

### Phase A — Foundation (no AI)

The minimum viable academic coding tool. Researchers can start using it immediately for line-by-line coding without any AI involvement, which is also the methodologically right starting point.

- F1.1, F1.2, F1.3, F1.5 — project shell + sources + persistence
- F2.1, F2.2, F2.3 — codebook with revision history and lifecycle ops
- F4.1, F4.2, F4.3, F4.6 — applications with overlap, gutter renderer, audio playback
- F5.1, F5.2, F5.4 — memos
- F6.1, F6.2 — codebook + retrieval report export (CSV / Markdown / Word)
- F9.1, F9.2 — event log + code definition versioning

### Phase B — Cross-corpus analysis

- F2.4, F2.6 — locked stages, codebook XML export
- F3.4, F3.5, F3.6, F3.7 — speaker awareness, query builder, matrix views, saved queries
- F5.3, F5.5 — memo sorting canvas, promote-memo-to-definition
- F6.3, F6.4, F6.5 — matrix exports, REFI-QDA project + codebook export
- F9.3, F9.6, F9.7 — codebook snapshots, AI invocation log (still no AI yet, but log is in place), audit export

### Phase C — Local AI as suggester

- F8.1 — Ollama backend abstraction
- F8.2 — embedding index
- F8.5 — find similar quotes (the safest AI feature; no category judgement)
- F8.10 — AI-off-by-default gating
- F8.11, F8.12 — model picker + recommendations

### Phase D — AI deeper integration

- F8.3, F8.4 — code suggestion (existing codebook + new-code modes)
- F8.6 — whole-transcript review pass
- F8.7 — AI second-coder / ICR pass
- F8.8 — memo drafts

### Phase E — Reliability + collaboration

- F2.5 — multi-coder mode + ICR statistics
- F4.4, F4.5 — selection helpers, orphan re-anchoring
- F6.6, F6.7 — REFI-QDA import, anonymised export
- F9.4, F9.5, F9.8, F9.9 — checkpoints, locked-with-reason, time-travel view, provenance display

---

## Open questions

- Project file format: a single `.scribe` archive (zip) or an `outputs/<project_id>/` directory tree like jobs use today? Tree is easier to inspect, archive is easier to share.
- Many-projects-per-Scribe-install or one? Probably many; affects the URL routing scheme (`/projects/<id>/...`).
- Should transcripts be coded *only* after the editor pass, or live during editing? Researchers typically want to clean before coding.
- Memo Markdown vs rich text. Markdown is simpler and exports cleanly; rich text is what NVivo users expect.
- AI: Ollama-only or also support llama.cpp embedded? Ollama is simpler to ship; embedded is one fewer dependency on the user.
