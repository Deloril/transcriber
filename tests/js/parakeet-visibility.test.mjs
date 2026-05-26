// Tests for the Parakeet-on-AMD visibility helpers (G5.1).
//
// NVIDIA NeMo (the runtime that loads Parakeet) is CUDA-only — there's
// no AMD/ROCm support and no community fork. The upload page hides the
// Parakeet optgroup entirely when the active backend is ROCm, and
// surfaces a tooltip naming the backend when the user has Parakeet
// pre-selected from a saved profile. These helpers are the pure
// decision logic behind that surface; the renderer in index.html is a
// thin wrapper around them.

import { describe, it, expect } from "vitest";
import {
  shouldHideParakeetOptgroup,
  parakeetModelHint,
} from "../../scribe/static/js/helpers.mjs";

describe("shouldHideParakeetOptgroup", () => {
  it("hides Parakeet on AMD ROCm — NeMo doesn't run there", () => {
    expect(shouldHideParakeetOptgroup("rocm")).toBe(true);
  });

  it("keeps Parakeet visible on CUDA (the supported path)", () => {
    expect(shouldHideParakeetOptgroup("cuda")).toBe(false);
  });

  it("keeps Parakeet visible on CPU (slow but technically possible)", () => {
    expect(shouldHideParakeetOptgroup("cpu")).toBe(false);
  });

  it("keeps Parakeet visible on Apple Silicon (MPS)", () => {
    // The runtime hint will warn the user; the optgroup itself is
    // not removed, matching the existing behaviour that's been live
    // for MPS users.
    expect(shouldHideParakeetOptgroup("mps")).toBe(false);
  });

  it("is case- and whitespace-tolerant", () => {
    expect(shouldHideParakeetOptgroup("ROCm")).toBe(true);
    expect(shouldHideParakeetOptgroup(" ROCM ")).toBe(true);
  });

  it("treats unknown / null / empty as 'don't hide'", () => {
    expect(shouldHideParakeetOptgroup(null)).toBe(false);
    expect(shouldHideParakeetOptgroup(undefined)).toBe(false);
    expect(shouldHideParakeetOptgroup("")).toBe(false);
    expect(shouldHideParakeetOptgroup("xpu")).toBe(false);
  });
});

describe("parakeetModelHint", () => {
  it("returns kind=none for non-Parakeet models", () => {
    const r = parakeetModelHint({
      model: "large-v3",
      backend: "cuda",
      parakeet: { available: true, installed: true, blocked_by_backend: false },
    });
    expect(r).toEqual({ kind: "none", tone: null, html: null });
  });

  it("returns kind=none when no model is selected", () => {
    expect(parakeetModelHint({}).kind).toBe("none");
    expect(parakeetModelHint({ model: null }).kind).toBe("none");
    expect(parakeetModelHint({ model: "" }).kind).toBe("none");
  });

  it("returns kind=blocked with the backend label on ROCm (G5.1 surface)", () => {
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v2",
      backend: "rocm",
      parakeet: {
        available: false,
        installed: true,
        blocked_by_backend: true,
      },
    });
    expect(r.kind).toBe("blocked");
    expect(r.tone).toBe("warn");
    expect(r.html).toContain("Parakeet (NVIDIA NeMo) doesn't run");
    expect(r.html).toContain("<strong>ROCm</strong>");
    expect(r.html).toContain("Pick a Whisper model");
  });

  it("returns kind=blocked with the MPS label on Apple Silicon", () => {
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v3",
      backend: "mps",
      parakeet: {
        available: false,
        installed: true,
        blocked_by_backend: true,
      },
    });
    expect(r.kind).toBe("blocked");
    expect(r.html).toContain("<strong>MPS</strong>");
  });

  it("blocked > missing: blocked path wins when both flags are set", () => {
    // If NeMo isn't installed *and* the backend is unsupported, the
    // backend reason is the more useful one to show — installing
    // won't help.
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v2",
      backend: "rocm",
      parakeet: {
        available: false,
        installed: false,
        blocked_by_backend: true,
      },
    });
    expect(r.kind).toBe("blocked");
  });

  it("returns kind=missing when NeMo isn't installed on a supported backend", () => {
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v2",
      backend: "cuda",
      parakeet: {
        available: false,
        installed: false,
        blocked_by_backend: false,
      },
    });
    expect(r.kind).toBe("missing");
    expect(r.tone).toBe("warn");
    expect(r.html).toContain("NVIDIA NeMo isn't installed");
    expect(r.html).toContain("requirements-parakeet.txt");
  });

  it("returns kind=info when Parakeet is selected and ready to go", () => {
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v2",
      backend: "cuda",
      parakeet: {
        available: true,
        installed: true,
        blocked_by_backend: false,
      },
    });
    expect(r.kind).toBe("info");
    expect(r.tone).toBe("muted");
    expect(r.html).toContain("English only");
    expect(r.html).toContain("CUDA GPU recommended");
  });

  it("HTML-escapes the backend label so a poisoned value can't inject markup", () => {
    // Defence in depth: formatBackendLabel falls back to "CPU" on
    // unknown inputs, so this can't actually happen in practice —
    // but the helper still escapes its output, and the test pins
    // that contract.
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v2",
      backend: "<script>alert(1)</script>",
      parakeet: { blocked_by_backend: true },
    });
    expect(r.html).not.toContain("<script>");
  });

  it("treats a missing parakeet payload as 'not installed'", () => {
    // The capabilities call can fail; the upload page caches a null
    // payload and the hint should still degrade gracefully when the
    // user has Parakeet selected.
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-0.6b-v2",
      backend: "cuda",
      parakeet: null,
    });
    expect(r.kind).toBe("missing");
  });

  it("matches Parakeet by substring (handles future model name variants)", () => {
    const r = parakeetModelHint({
      model: "nvidia/parakeet-tdt-1.1b-v4",
      backend: "cuda",
      parakeet: { available: true, installed: true, blocked_by_backend: false },
    });
    expect(r.kind).toBe("info");
  });

  it("is case-insensitive on the model id", () => {
    const r = parakeetModelHint({
      model: "NVIDIA/Parakeet-TDT-0.6b-v2",
      backend: "cuda",
      parakeet: { available: true, installed: true, blocked_by_backend: false },
    });
    expect(r.kind).toBe("info");
  });
});
