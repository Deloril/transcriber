// Tests for the F8.13 inline AI-gate block helpers (source_coding.html).
//
// The helpers are pure: extractGateStatus() pulls the AIGateStatus out
// of a 412 response body (handling FastAPI's outer {detail: …}
// envelope), and renderInlineGateBlockHtml() builds the HTML block
// that replaces the old plain-text "AI gate not satisfied" status.
//
// Page-side wiring (the click handler that PUTs /ai/gate and retries
// the original action) is exercised by the pytest integration tests in
// tests/test_server_ai_gate_inline.py.

import { describe, it, expect } from "vitest";
import {
  extractGateStatus,
  renderInlineGateBlockHtml,
} from "../../scribe/static/js/helpers.mjs";


function _gateClosed() {
  // Canonical AIGateStatus.to_dict() shape for a fresh project.
  // Mirrors scribe.ai_gate.evaluate_project_ai_gate's payload.
  return {
    allowed: false,
    reason: "insufficient_both",
    message: "AI suggestions disabled until you have ≥ 8 codes and ≥ 2 hand-coded transcripts.",
    code_count: 0,
    hand_coded_source_count: 0,
    min_codes: 8,
    min_hand_coded_sources: 2,
    override: "auto",
    enabled: true,
    feature: "code_suggestion",
    feature_exempt: false,
  };
}


describe("extractGateStatus", () => {
  it("unwraps FastAPI's {detail: {detail, gate}} envelope", () => {
    const status = _gateClosed();
    const out = extractGateStatus({
      detail: { detail: "AI gate not satisfied", gate: status },
    });
    expect(out).not.toBeNull();
    expect(out.gate).toEqual(status);
    expect(out.message).toBe(status.message);
  });

  it("accepts an already-unwrapped {detail, gate} body", () => {
    const status = _gateClosed();
    const out = extractGateStatus({
      detail: "AI gate not satisfied", gate: status,
    });
    expect(out).not.toBeNull();
    expect(out.gate).toBe(status);
  });

  it("falls back to inner.detail when gate.message is empty", () => {
    const out = extractGateStatus({
      detail: {
        detail: "AI suggestions disabled.",
        gate: { ..._gateClosed(), message: "" },
      },
    });
    expect(out).not.toBeNull();
    expect(out.message).toBe("AI suggestions disabled.");
  });

  it("falls back to a generic message when both are empty", () => {
    const out = extractGateStatus({
      detail: {
        detail: "",
        gate: { ..._gateClosed(), message: "" },
      },
    });
    expect(out).not.toBeNull();
    expect(out.message).toBe("AI gate not satisfied");
  });

  it("returns null when the body has no gate payload", () => {
    expect(extractGateStatus(null)).toBeNull();
    expect(extractGateStatus(undefined)).toBeNull();
    expect(extractGateStatus({})).toBeNull();
    expect(extractGateStatus({ detail: "Some unrelated 400 error" })).toBeNull();
    expect(extractGateStatus({ detail: { detail: "no gate" } })).toBeNull();
  });

  it("returns null when gate is not an object", () => {
    expect(extractGateStatus({ detail: { gate: "blocked" } })).toBeNull();
    expect(extractGateStatus({ detail: { gate: 42 } })).toBeNull();
  });
});


describe("renderInlineGateBlockHtml", () => {
  it("returns the empty string when status is missing", () => {
    expect(renderInlineGateBlockHtml(null, {})).toBe("");
    expect(renderInlineGateBlockHtml(undefined)).toBe("");
    expect(renderInlineGateBlockHtml("blocked")).toBe("");
  });

  it("renders the canonical block with progress + force-on + settings link", () => {
    const html = renderInlineGateBlockHtml(_gateClosed(), { projectId: "proj-abc" });
    // Wrapper carries the test id and the reason marker so a future
    // browser script can scope-pick the block.
    expect(html).toContain('data-test-id="ai-gate-inline-block"');
    expect(html).toContain('data-gate-reason="insufficient_both"');
    // Message survives unmodified in escaped form.
    expect(html).toContain("AI suggestions disabled until you have ≥ 8 codes");
    // Progress is the formatGateProgress() output, escaped.
    expect(html).toContain("codes 0/8");
    expect(html).toContain("hand-coded 0/2");
    expect(html).toContain("override = auto");
    expect(html).toContain("reason = insufficient_both");
    // Force-on action button + settings link.
    expect(html).toContain('data-test-id="ai-gate-inline-force-on"');
    expect(html).toContain('data-gate-action="force-on"');
    expect(html).toContain('href="/projects/proj-abc/ai"');
  });

  it("URL-encodes the project id in the settings link", () => {
    const html = renderInlineGateBlockHtml(_gateClosed(), { projectId: "p&id" });
    expect(html).toContain('href="/projects/p%26id/ai"');
  });

  it("omits the settings link when no projectId is given", () => {
    const html = renderInlineGateBlockHtml(_gateClosed(), {});
    expect(html).not.toContain('data-test-id="ai-gate-inline-settings-link"');
    // Force-on button is still present (project-id-independent).
    expect(html).toContain('data-test-id="ai-gate-inline-force-on"');
  });

  it("omits the force-on button when allowForceOn is explicitly false", () => {
    const html = renderInlineGateBlockHtml(_gateClosed(), {
      projectId: "p", allowForceOn: false,
    });
    expect(html).not.toContain('data-test-id="ai-gate-inline-force-on"');
    // Settings link is still present so the user has a way to act.
    expect(html).toContain('data-test-id="ai-gate-inline-settings-link"');
  });

  it("escapes HTML in the gate message and reason", () => {
    const html = renderInlineGateBlockHtml({
      ..._gateClosed(),
      message: "Bad <script>alert(1)</script>",
      reason: "weird&stuff",
    }, { projectId: "p" });
    expect(html).not.toContain("<script>alert(1)</script>");
    expect(html).toContain("&lt;script&gt;");
    // Reason is rendered into a data-attribute; the ampersand is
    // escaped.
    expect(html).toContain('data-gate-reason="weird&amp;stuff"');
  });

  it("renders a force_off-state status with no force-on button by default", () => {
    // force_off = the policy is permanently closed for this project.
    // The user can still flip override→force_on to bypass; helper keeps
    // the button visible (the button writes force_on regardless of the
    // current override) so users can recover.
    const html = renderInlineGateBlockHtml({
      ..._gateClosed(),
      reason: "force_off",
      override: "force_off",
      message: "AI is force-disabled for this project.",
    }, { projectId: "p" });
    expect(html).toContain('data-gate-reason="force_off"');
    expect(html).toContain("override = force_off");
    expect(html).toContain('data-test-id="ai-gate-inline-force-on"');
  });

  it("renders even when message is missing (defaults to canonical text)", () => {
    const html = renderInlineGateBlockHtml({
      ..._gateClosed(),
      message: "",
    }, { projectId: "p" });
    expect(html).toContain("AI gate not satisfied");
  });
});
