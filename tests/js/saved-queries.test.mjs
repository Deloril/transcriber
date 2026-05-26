// Vitest for the F3.7 saved-queries helpers in
// scribe/static/js/helpers.mjs.
//
// `buildSavedQueryPayload` is the JS-side translation between the
// queries.html form (free-form name + free-form description + the
// already-built F3.5 query payload) and the JSON shape that
// scribe.server.create_saved_query_endpoint accepts. Pulling it into
// helpers.mjs (rather than the template) so the round-trip is
// covered by the same vitest run as the matrix / query helpers.
//
// `formatSavedQueryRunSummary` formats a SavedQuery's run-tracking
// metadata into a one-liner for the panel ("never run" / "last run
// 5 min ago · 3 ×"). nowFn is injectable so we can pin Date.now()
// without timing flake.

import { describe, expect, it } from "vitest";

import {
  buildSavedQueryPayload,
  formatSavedQueryRunSummary,
} from "../../scribe/static/js/helpers.mjs";


// 12-hex fixture ids that match scribe.projects.PROJECT_ID_RE etc.
const PID = "0123456789ab";
const C1  = "111111111111";


function _innerQuery(extra = {}) {
  return {
    project_id: PID,
    name: "Power quotes",
    ...extra,
  };
}


describe("buildSavedQueryPayload", () => {
  it("requires a query object", () => {
    expect(() => buildSavedQueryPayload({ name: "X" }))
      .toThrow(/query is required/);
  });

  it("rejects a non-object query", () => {
    expect(() => buildSavedQueryPayload({ query: "string", name: "X" }))
      .toThrow(/query is required/);
  });

  it("requires a non-blank name", () => {
    expect(() => buildSavedQueryPayload({ query: _innerQuery() }))
      .toThrow(/name is required/);
    expect(() => buildSavedQueryPayload({
      query: _innerQuery(), name: "   ",
    })).toThrow(/name is required/);
  });

  it("emits the minimal shape with just query + name", () => {
    const out = buildSavedQueryPayload({
      query: _innerQuery(),
      name: "Quotes about power",
    });
    expect(out).toEqual({
      query: _innerQuery(),
      name: "Quotes about power",
    });
  });

  it("trims the top-level name", () => {
    const out = buildSavedQueryPayload({
      query: _innerQuery(),
      name: "  Quotes about power  ",
    });
    expect(out.name).toBe("Quotes about power");
  });

  it("emits description only when non-empty", () => {
    const out = buildSavedQueryPayload({
      query: _innerQuery(),
      name: "Q",
      description: "",
    });
    expect(out.description).toBeUndefined();
    const out2 = buildSavedQueryPayload({
      query: _innerQuery(),
      name: "Q",
      description: "Initial coding pass",
    });
    expect(out2.description).toBe("Initial coding pass");
  });

  it("passes the inner query through verbatim", () => {
    const inner = _innerQuery({
      codes: { expr: { op: "code", code_id: C1 } },
    });
    const out = buildSavedQueryPayload({ query: inner, name: "C1" });
    expect(out.query).toBe(inner);  // same reference: route stamps pid
    expect(out.query.codes.expr.code_id).toBe(C1);
  });
});


describe("formatSavedQueryRunSummary", () => {
  it("returns '' for null / non-object", () => {
    expect(formatSavedQueryRunSummary(null)).toBe("");
    expect(formatSavedQueryRunSummary("X")).toBe("");
    expect(formatSavedQueryRunSummary(undefined)).toBe("");
  });

  it("returns 'never run' when run_count = 0 and no last_run_at", () => {
    expect(formatSavedQueryRunSummary({})).toBe("never run");
    expect(formatSavedQueryRunSummary({ run_count: 0 })).toBe("never run");
  });

  it("returns the count when run_count > 0 but no last_run_at", () => {
    expect(formatSavedQueryRunSummary({ run_count: 1, last_run_at: "" }))
      .toBe("1 run");
    expect(formatSavedQueryRunSummary({ run_count: 5, last_run_at: "" }))
      .toBe("5 runs");
  });

  it("formats 'just now' when last_run_at is < 60 s ago", () => {
    // Pin Date.now() to 30 seconds after the run.
    const last = "2026-05-27T00:00:00Z";
    const nowMs = Date.parse(last) + 30 * 1000;
    const out = formatSavedQueryRunSummary(
      { run_count: 1, last_run_at: last },
      { nowFn: () => nowMs },
    );
    expect(out).toMatch(/just now/);
    expect(out).toMatch(/1 ×$/);
  });

  it("formats 'min ago' between 1 min and 1 h", () => {
    const last = "2026-05-27T00:00:00Z";
    const nowMs = Date.parse(last) + 5 * 60 * 1000;  // 5 minutes
    const out = formatSavedQueryRunSummary(
      { run_count: 3, last_run_at: last },
      { nowFn: () => nowMs },
    );
    expect(out).toMatch(/5 min ago/);
    expect(out).toMatch(/3 ×$/);
  });

  it("formats 'h ago' between 1 h and 1 d", () => {
    const last = "2026-05-27T00:00:00Z";
    const nowMs = Date.parse(last) + 3 * 3600 * 1000;
    const out = formatSavedQueryRunSummary(
      { run_count: 7, last_run_at: last },
      { nowFn: () => nowMs },
    );
    expect(out).toMatch(/3 h ago/);
    expect(out).toMatch(/7 ×$/);
  });

  it("formats 'd ago' beyond 1 day", () => {
    const last = "2026-05-25T00:00:00Z";
    const nowMs = Date.parse(last) + 2 * 86400 * 1000;
    const out = formatSavedQueryRunSummary(
      { run_count: 12, last_run_at: last },
      { nowFn: () => nowMs },
    );
    expect(out).toMatch(/2 d ago/);
    expect(out).toMatch(/12 ×$/);
  });

  it("falls back to the raw ISO string for unparseable dates", () => {
    const out = formatSavedQueryRunSummary(
      { run_count: 1, last_run_at: "not-a-date" },
    );
    expect(out).toMatch(/not-a-date/);
    expect(out).toMatch(/1 ×$/);
  });
});
