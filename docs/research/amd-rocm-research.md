# AMD GPU Support for Scribe — Research Report

Source material for the AMD/ROCm planning section of `PLANNING.md`. Written 2026-05-25.

## Bottom line up front

The story is **much better than 12 months ago**. As of February 2026, **CTranslate2 has merged official ROCm support** (PR [#1989](https://github.com/OpenNMT/CTranslate2/pull/1989), released in [v4.7.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.0) on 2026-02-03), and AMD's RDNA 3 / RDNA 4 cards are first-class on PyTorch + ROCm 6.3. RDNA 2 works with one allocator workaround. **Scribe can ship "real" AMD GPU support today** for the full Whisper + alignment + diarization pipeline — but only on Linux, with caveats per architecture, and Parakeet stays NVIDIA-only.

---

## A. CTranslate2 + ROCm

**Status: official, shipping, but young.**

- **PR [#1989](https://github.com/OpenNMT/CTranslate2/pull/1989)** "Introduce AMD GPU support with ROCm HIP" was merged by jordimas on **2026-02-02** by author `sssshhhhhh`, building on the long-running community fork [arlo-phoenix/CTranslate2-rocm](https://github.com/arlo-phoenix/CTranslate2-rocm). It closed the 3-year-old feature request [#1072](https://github.com/OpenNMT/CTranslate2/issues/1072).
- Released as part of **[v4.7.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.0)** (2026-02-03), updated to ROCm 7.2.1 in **[v4.7.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.2)** (2026-05-19).
- **Wheels are published as ZIP archives** on the releases page, *not* on PyPI:
  - `rocm-python-wheels-Linux.zip` (~284 MB)
  - `rocm-python-wheels-Windows.zip` (~137 MB)
- The Linux `.so` ships with **gfx803 → gfx1201 baked in** per nabe2030's report on PR #1989 (covers GCN3 through RDNA 4, including gfx1150/gfx1151).

**Known live bugs (open as of May 2026):**

| Issue | Hardware | Symptom |
|---|---|---|
| [#2021](https://github.com/OpenNMT/CTranslate2/issues/2021) | RX 9070 XT (gfx1201), Fedora 43, ROCm 7.2.0 | faster-whisper crashes with "Memory access fault by GPU node-1" in fp16 and int8. Open, no triage. |
| [#2038](https://github.com/OpenNMT/CTranslate2/issues/2038) | gfx1100 on Windows ROCm 7.2.1 | `del model` deadlocks in HIP allocator free path. |
| [#2016](https://github.com/OpenNMT/CTranslate2/issues/2016) | Windows | Wheels built against ROCm 7.2 but the public Windows HIP SDK only ships 7.1.1. |

**Known confirmed-working configs:**

- [#2011](https://github.com/OpenNMT/CTranslate2/issues/2011) — RDNA 1 (gfx1010) on Windows ROCm.
- [#2012](https://github.com/OpenNMT/CTranslate2/issues/2012) — RX 6600 / gfx1032 (RDNA 2) **with workaround**: must export `CT2_CUDA_ALLOCATOR=cub_caching` (default `MallocAsync` crashes on RDNA 2 with "illegal memory access"). Plus `HSA_OVERRIDE_GFX_VERSION=10.3.0` to map gfx1032→gfx1030. Reported 5.5× realtime on Whisper Large-V3 / int8.

So: **CTranslate2 ROCm is production-usable on RDNA 3 and RDNA 4 (gfx1100/1101/1200), shaky on RDNA 4 gfx1201, working on RDNA 2 with one env var, and unproven on Windows.**

---

## B. PyTorch + ROCm on consumer Radeon

**Stable wheels:** PyTorch 2.7.0 ships **ROCm 6.3** wheels via `--index-url https://download.pytorch.org/whl/rocm6.3` ([pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)). **Linux only.** No Windows or macOS ROCm wheels exist on the official index.

**AMD's own [System Requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)** for ROCm 7.2.3 lists, for consumer Radeon:

| Architecture | Cards | Status |
|---|---|---|
| **RDNA 4** (gfx1200/1201) | RX 9060/9060 XT, RX 9070/9070 XT/GRE | Officially supported |
| **RDNA 3** (gfx1100/1101) | RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT/7700 | Officially supported |
| **RDNA 2** (gfx1030/1031/1032) | **No RX 6000 consumer card listed** — only Radeon PRO W6800 (workstation) | Officially unsupported, community-enabled |
| **Vega 20** (gfx906) | Radeon VII | Officially **unsupported** |

OS support is narrowed to **Ubuntu 24.04.4, Ubuntu 22.04.5, RHEL 10.1, RHEL 9.7**. Other distros work in practice via the upstream packages and `HSA_OVERRIDE_GFX_VERSION`, but they are not in the support matrix.

**Windows ROCm + PyTorch:** Not available in the official PyTorch index. AMD has an experimental Windows HIP SDK (capped at 7.1.1) but PyTorch wheels for Windows ROCm don't exist as of May 2026. **Treat Windows AMD as "Vulkan-via-whisper.cpp only" for Scribe.**

---

## C. pyannote.audio on ROCm

**Works in principle, has one live blocker on ROCm ≥ 6.1.1.**

[pyannote-audio#1995](https://github.com/pyannote/pyannote-audio/issues/1995) (opened 2026-03-20, **open, untriaged**):

> *"MIOpen(HIP): fatal error: 'hiprand/hiprand_xorwow.h' file not found … RuntimeError: miopenStatusUnknownError"*

PyanNet's segmentation model uses `nn.LSTM(dropout=0.5, num_layers=4)`. ROCm 6.1.1+ removed a header MIOpen needs to compile the dropout kernel. **Workaround (safe, used in the reporter's `amd-gpu-patch` fork):** after `Pipeline.from_pretrained(...).to(...)`, walk into the segmentation model and force LSTM `dropout = 0.0`. Inference behaviour is unchanged because dropout is a no-op outside training.

**Scribe action:** Ship that monkey-patch behind an `if torch.version.hip` guard.

---

## D. wav2vec2 alignment on ROCm

**Should work; no specific blockers found, but unverified.**

- `torchaudio` ROCm support is not documented on the official install page (CPU and CUDA only) but in practice torchaudio is installed alongside the ROCm PyTorch wheel and inherits the HIP-aliased `cuda` namespace.
- Hugging Face `transformers` Wav2Vec2 uses standard `.to(device)` / `.cuda()` — no HIP-specific code needed.
- No known showstopper on the wav2vec2 path on ROCm. Treat as "expected to work, must validate on real hardware."

---

## E. NeMo / Parakeet on ROCm

**Don't bother. NeMo is NVIDIA-only.**

- NVIDIA/NeMo issues for ROCm/AMD/HIP turn up nothing substantive — only incidental matches. No official policy, no community fork shipping.
- NeMo depends on NVIDIA-specific stacks (CUDA Graphs, TensorRT-LLM, FP8 via Transformer Engine).
- **Recommendation for Scribe:** when the user picks AMD GPU, **hide Parakeet** (or fall back to CPU-only Parakeet via ONNX if you want to invest, but that's a separate project).

---

## F. Alternative Whisper backends on AMD

| Backend | AMD support | Maturity | Notes |
|---|---|---|---|
| **CTranslate2 + ROCm** | Official since v4.7.0 (Feb 2026) | Young but functional | First choice on supported Linux Radeons. Same model files, same int8/fp16 quants, same `faster-whisper` Python API. |
| **whisper.cpp + Vulkan** | Yes, cross-vendor | Mature | Works on AMD/Intel/NVIDIA *and* Windows. Build flag `-DGGML_VULKAN=1`. **Best fallback for Windows AMD and unsupported RDNA 2 cards.** |
| **whisper.cpp + HIP/ROCm** | Yes, dedicated backend | Mature | Build flag `-DGGML_HIP=1 -DAMDGPU_TARGETS="gfx1100"`. Faster than Vulkan on ROCm-supported cards. |
| **transformers Whisper on ROCm** | Yes (PyTorch) | Slow | Works but significantly slower than CT2. Useful only as a debug fallback. |
| **AMD-maintained Whisper fork** | Doesn't exist | — | Searched; nothing canonical. |

**Accuracy:** No clean head-to-head WER benchmark of whisper.cpp vs faster-whisper at matching model sizes from an authoritative source. Anecdotally they are within rounding for non-quantised builds (both consume the same OpenAI weights). Quantised whisper.cpp models (`q5_0`, `q8_0`) have shown small WER regressions in community benchmarks but remain usable.

---

## G. Runtime detection

Canonical idiom — `torch.version.hip` is the discriminator:

```python
import torch

def detect_gpu_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    # PyTorch ROCm aliases torch.cuda.* to HIP, so is_available() is True on both.
    # torch.version.hip is None on CUDA builds, a string on ROCm builds.
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"
```

Notes:
- Don't rely on `torch.version.cuda` alone — ROCm builds also populate it (with a CUDA-compat string).
- For CTranslate2, the device flag is still `"cuda"` on a ROCm wheel (because CT2 uses HIP's CUDA-runtime API shim). So `Whisper(model_path, device="cuda", compute_type="float16")` is the correct call on ROCm.
- `rocm-smi` exists but shelling out is brittle; prefer the `torch.version.hip` check.

---

## H. Install path

**Linux, supported distro, RDNA 3/4:**

```bash
# 1. Install ROCm 6.3 system packages per AMD docs (Ubuntu 22.04 / 24.04, RHEL 9/10).
# 2. PyTorch + torchaudio:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
# 3. CTranslate2 ROCm wheel (NOT on PyPI):
curl -LO https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.2/rocm-python-wheels-Linux.zip
unzip rocm-python-wheels-Linux.zip
pip install ctranslate2-4.7.2-*-linux_x86_64.whl
# 4. faster-whisper, pyannote, etc. as today.
pip install faster-whisper pyannote.audio
```

**ROCm version pin:** Pin to **ROCm 6.3** (PyTorch's stable target) in Scribe's installer documentation. CT2 v4.7.2 was built against ROCm 7.2.1, but practice on PR #1989 shows the runtime-side HIP shim is forward/backward compatible enough for 6.3 to load it.

**Distro reality:**
- **Ubuntu 22.04/24.04** — first-class.
- **Fedora 41/42/43** — works in practice, not officially supported. Need `HSA_OVERRIDE_GFX_VERSION` for some cards.
- **Arch** — `extra/rocm-hip-sdk` is well maintained, works.
- **RHEL 9.7 / 10.1** — supported per AMD.
- Anything else — document as "best-effort."

**Windows:** Not viable for the PyTorch path. **whisper.cpp Vulkan** is the answer for Windows AMD users.

---

## I. What can actually ship vs vapourware

| Claim | Reality |
|---|---|
| "CTranslate2 supports ROCm" | **TRUE.** Official since 2026-02. |
| "AMD acceleration for Whisper, alignment, diarization" | **TRUE on Linux RDNA 3/4**; needs LSTM-dropout patch for pyannote. |
| "AMD acceleration for Parakeet" | **FALSE.** NeMo is NVIDIA-only. |
| "Works on Windows AMD" | **FALSE for the PyTorch path.** True only via whisper.cpp Vulkan. |
| "Works on RX 6000 series" | **TRUE with workarounds** (`CT2_CUDA_ALLOCATOR=cub_caching`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`). Officially unsupported by AMD. |

So "complete AMD support" honestly means:

- **Tier 1 (full pipeline acceleration):** Linux + RDNA 3/4 + ROCm 6.3 PyTorch + CT2 v4.7.2 ROCm wheel. Whisper, alignment, diarization all GPU-accelerated. Parakeet hidden.
- **Tier 2 (works with workarounds):** Linux + RDNA 2. Same stack + two env vars. Document loudly.
- **Tier 3 (Whisper only, cross-platform):** whisper.cpp Vulkan path. Loses ctranslate2's int8 quants and our existing faster-whisper integration. But ships on Windows and on cards ROCm has dropped.
- **Out of scope:** Parakeet on AMD, macOS Intel + AMD eGPU.

---

## J. Recommended path for Scribe

**1. GPU-accelerating the post-Whisper pipeline alone is not worth shipping.** Whisper inference dominates wall time (60–80% on long files). A "Whisper on CPU + alignment/diarization on GPU" config delivers maybe 20–30% improvement and dramatically complicates device selection.

**2. Primary AMD path: CTranslate2 ROCm wheel + PyTorch ROCm + LSTM dropout patch.** This gets us full-pipeline GPU acceleration with the same `device="cuda"` code path and the same model artefacts.

**3. Secondary AMD path: whisper.cpp Vulkan, behind a feature flag.** This is the answer for Windows AMD, RDNA 1, and any card AMD has dropped. Costs a separate Whisper backend abstraction. Defer.

**4. Should we wait for CTranslate2 ROCm to mature?** **No.** Three months in, with confirmed working users, clear bug reports, and an active maintainer (jordimas). Open issues are specific and tractable.

**5. Risk register:**

- ROCm 6.3 → 7.x transition. PyTorch's stable wheels lag AMD's releases.
- Issue #2021 (RX 9070 XT crash on Fedora) unresolved.
- CT2 ROCm wheels are ZIP-only on GitHub. Mirror later if needed.
- Issue #1995 (pyannote LSTM dropout) — watch upstream.

---

## Key references (chronological)

- 2023-02-09 — [CT2 #1072](https://github.com/OpenNMT/CTranslate2/issues/1072) feature request opened.
- 2024-02-04 — [CT2 #1615](https://github.com/OpenNMT/CTranslate2/issues/1615) DirectML alternative request (still open).
- 2026-02-02 — [CT2 PR #1989](https://github.com/OpenNMT/CTranslate2/pull/1989) ROCm HIP merged.
- 2026-02-03 — [CT2 v4.7.0](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.0) released with ROCm wheels.
- 2026-02-08 — [CT2 #2012](https://github.com/OpenNMT/CTranslate2/issues/2012) RDNA 2 allocator workaround documented.
- 2026-02-28 — [CT2 #2021](https://github.com/OpenNMT/CTranslate2/issues/2021) RX 9070 XT crash, open.
- 2026-03-20 — [pyannote-audio #1995](https://github.com/pyannote/pyannote-audio/issues/1995) LSTM dropout MIOpen failure on ROCm 6.1.1+.
- 2026-05-08 — [CT2 #2038](https://github.com/OpenNMT/CTranslate2/issues/2038) Windows HIP allocator deadlock.
- 2026-05-19 — [CT2 v4.7.2](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.2) ROCm 7.2.1.

---

**Things to verify before shipping:**

- Concrete Vulkan-vs-HIP-vs-CUDA whisper.cpp benchmark numbers — must benchmark in-house.
- Whether the v4.7.2 Linux ROCm wheel actually loads cleanly on PyTorch's ROCm 6.3 build vs requiring system ROCm 7.2.x.
- WER parity between CT2 large-v3 fp16 and whisper.cpp large-v3 q5_0/q8_0.
