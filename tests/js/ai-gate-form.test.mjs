// Tests for the F8.10 AI-gate UI helpers (project_ai.html).
//
// Pure functions only — they format the progress string and build the
// PUT /api/projects/<pid>/ai/gate payload. The page-side wiring
// (loadGate / saveGate / submitGateForm) is exercised by the pytest
// integration tests in tests/test_server_project_ai.py.

import { describe, it, expect } from "vitest";
import {
  formatGateProgress,
  gateFormPayload,
  gateForceOnPayload,
} from "../../scribe/static/js/helpers.mjs";


describe("formatGateProgress", () => {
  it("formats a closed-gate status with progress + override + reason", () => {
    const out = formatGateProgress({
      allowed: false,
      reason: "insufficient_both",
      message: "fresh project",
      code_count: 3,
      min_codes: 8,
      hand_coded_source_count: 0,
      min_hand_coded_sources: 2,
      override: "auto",
      enabled: true,
    });
    expect(out).toBe(
      "codes 3/8 · hand-coded 0/2 · override = auto · reason = insufficient_both",
    );
  });

  it("flags a disabled-policy state explicitly", () => {
    const out = formatGateProgress({
      allowed: true,
      reason: "disabled",
      code_count: 0, min_codes: 8,
      hand_coded_source_count: 0, min_hand_coded_sources: 2,
      override: "auto", enabled: false,
    });
    expect(out).toContain("policy disabled");
    expect(out).toContain("reason = disabled");
  });

  it("returns the empty string for null / non-object input", () => {
    expect(formatGateProgress(null)).toBe("");
    expect(formatGateProgress(undefined)).toBe("");
    expect(formatGateProgress(42)).toBe("");
    expect(formatGateProgress("hello")).toBe("");
  });

  it("renders sensible zeros when counts are missing", () => {
    const out = formatGateProgress({});
    expect(out).toBe("codes 0/0 · hand-coded 0/0");
  });

  it("preserves reason='threshold_met' in the open-state line", () => {
    const out = formatGateProgress({
      allowed: true,
      reason: "threshold_met",
      code_count: 9, min_codes: 8,
      hand_coded_source_count: 3, min_hand_coded_sources: 2,
      override: "auto", enabled: true,
    });
    expect(out).toBe(
      "codes 9/8 · hand-coded 3/2 · override = auto · reason = threshold_met",
    );
  });
});


// Build a duck-typed "form" with the same shape gateFormPayload reads.
function fakeForm(values) {
  const els = {};
  for (const [name, def] of Object.entries(values)) {
    if (def === null) continue;
    if (def.kind === "check") {
      els[name] = { checked: def.checked, value: def.checked ? "on" : "" };
    } else {
      els[name] = { value: String(def.value) };
    }
  }
  return {
    elements: {
      namedItem: (name) => els[name] || null,
    },
  };
}


describe("gateFormPayload", () => {
  it("builds the canonical PUT body from a populated form", () => {
    const form = fakeForm({
      min_codes: { value: 12 },
      min_hand_coded_sources: { value: 4 },
      override: { value: "force_on" },
      enabled: { kind: "check", checked: true },
    });
    expect(gateFormPayload(form)).toEqual({
      min_codes: 12,
      min_hand_coded_sources: 4,
      override: "force_on",
      enabled: true,
    });
  });

  it("falls back to spec defaults (8 / 2 / auto / true) on missing fields", () => {
    expect(gateFormPayload({ elements: { namedItem: () => null } })).toEqual({
      min_codes: 8,
      min_hand_coded_sources: 2,
      override: "auto",
      enabled: true,
    });
  });

  it("coerces NaN inputs to defaults", () => {
    const form = fakeForm({
      min_codes: { value: "not a number" },
      min_hand_coded_sources: { value: "" },
      override: { value: "auto" },
      enabled: { kind: "check", checked: false },
    });
    expect(gateFormPayload(form)).toEqual({
      min_codes: 8,
      min_hand_coded_sources: 2,
      override: "auto",
      enabled: false,
    });
  });

  it("returns defaults on null / no form at all", () => {
    expect(gateFormPayload(null)).toEqual({
      min_codes: 8,
      min_hand_coded_sources: 2,
      override: "auto",
      enabled: true,
    });
    expect(gateFormPayload(undefined)).toEqual({
      min_codes: 8,
      min_hand_coded_sources: 2,
      override: "auto",
      enabled: true,
    });
  });
});


describe("gateForceOnPayload", () => {
  it("flips override to force_on while preserving thresholds", () => {
    expect(gateForceOnPayload({
      min_codes: 5,
      min_hand_coded_sources: 1,
      override: "auto",
      enabled: true,
    })).toEqual({
      min_codes: 5,
      min_hand_coded_sources: 1,
      override: "force_on",
      enabled: true,
    });
  });

  it("falls back to spec defaults when the cfg is missing", () => {
    expect(gateForceOnPayload(null)).toEqual({
      min_codes: 8,
      min_hand_coded_sources: 2,
      override: "force_on",
      enabled: true,
    });
  });

  it("preserves enabled=false when the policy was disabled", () => {
    expect(gateForceOnPayload({
      min_codes: 8, min_hand_coded_sources: 2,
      override: "auto", enabled: false,
    })).toEqual({
      min_codes: 8, min_hand_coded_sources: 2,
      override: "force_on", enabled: false,
    });
  });

  it("treats unset enabled as the safe default (true)", () => {
    expect(gateForceOnPayload({
      min_codes: 8, min_hand_coded_sources: 2, override: "auto",
    }).enabled).toBe(true);
  });
});
