// Vitest for the F3.5 query-builder helpers in
// scribe/static/js/helpers.mjs. Mirrors the pure-Python contract in
// scribe/query.py (Query / CodeExpr / SourceFilter / SpeakerFilter
// / ProximityFilter) so the JS-side translation can't drift.

import { describe, expect, it } from "vitest";

import {
  buildQueryPayload,
  groupApplicationsBySource,
} from "../../scribe/static/js/helpers.mjs";


// 12-hex fixture ids that match the validators in scribe.projects /
// scribe.codes / scribe.sources.
const PID = "0123456789ab";
const C1  = "111111111111";
const C2  = "222222222222";
const C3  = "333333333333";
const S1  = "aaaaaaaaaaaa";
const S2  = "bbbbbbbbbbbb";


describe("buildQueryPayload", () => {
  it("requires a project id", () => {
    expect(() => buildQueryPayload({})).toThrow(/projectId/);
  });

  it("returns the minimal payload for an empty form", () => {
    const out = buildQueryPayload({ projectId: PID });
    expect(out).toEqual({ project_id: PID });
  });

  it("emits a single-code leaf CodeExpr", () => {
    const out = buildQueryPayload({ projectId: PID, codeIds: [C1] });
    expect(out.codes).toEqual({ expr: { op: "code", code_id: C1 } });
  });

  it("emits an OR combinator for multiple codes", () => {
    const out = buildQueryPayload({ projectId: PID, codeIds: [C1, C2, C3] });
    expect(out.codes.expr.op).toBe("or");
    expect(out.codes.expr.children).toHaveLength(3);
    expect(out.codes.expr.children[0]).toEqual({ op: "code", code_id: C1 });
    expect(out.codes.expr.children[2]).toEqual({ op: "code", code_id: C3 });
  });

  it("does not emit a codes filter when zero codes are selected", () => {
    const out = buildQueryPayload({ projectId: PID, codeIds: [] });
    expect(out.codes).toBeUndefined();
  });

  it("emits a SourceFilter when source_ids is non-empty", () => {
    const out = buildQueryPayload({ projectId: PID, sourceIds: [S1, S2] });
    expect(out.sources).toEqual({ source_ids: [S1, S2] });
  });

  it("does not emit sources when none picked", () => {
    const out = buildQueryPayload({ projectId: PID });
    expect(out.sources).toBeUndefined();
  });

  it("emits a SpeakerFilter when role is set", () => {
    const out = buildQueryPayload({ projectId: PID, speakerRole: "interviewee" });
    expect(out.speakers).toEqual({
      roles: ["interviewee"],
      labels: [],
      participant_ids: [],
    });
  });

  it("does not emit a SpeakerFilter when role + labels + pids all empty", () => {
    const out = buildQueryPayload({ projectId: PID, speakerRole: "" });
    expect(out.speakers).toBeUndefined();
  });

  it("merges role with explicit labels list", () => {
    const out = buildQueryPayload({
      projectId: PID,
      speakerRole: "interviewer",
      speakerLabels: ["LUKE"],
    });
    expect(out.speakers).toEqual({
      roles: ["interviewer"],
      labels: ["LUKE"],
      participant_ids: [],
    });
  });

  it("emits a ProximityFilter when scope + required ids set", () => {
    const out = buildQueryPayload({
      projectId: PID,
      proximity: {
        scope: "source",
        requiredCodeIds: [C1, C2],
        maxGap: 0,
      },
    });
    expect(out.proximity).toEqual({
      scope: "source",
      required_code_ids: [C1, C2],
      max_gap: 0,
    });
  });

  it("omits proximity when required ids is empty", () => {
    const out = buildQueryPayload({
      projectId: PID,
      proximity: { scope: "source", requiredCodeIds: [], maxGap: 0 },
    });
    expect(out.proximity).toBeUndefined();
  });

  it("composes every dimension simultaneously", () => {
    const out = buildQueryPayload({
      projectId: PID,
      codeIds: [C1, C2],
      sourceIds: [S1],
      speakerRole: "interviewee",
      proximity: {
        scope: "paragraph",
        requiredCodeIds: [C1, C2],
        maxGap: 25.5,
      },
    });
    expect(out.project_id).toBe(PID);
    expect(out.codes.expr.op).toBe("or");
    expect(out.sources.source_ids).toEqual([S1]);
    expect(out.speakers.roles).toEqual(["interviewee"]);
    expect(out.proximity.scope).toBe("paragraph");
    expect(out.proximity.max_gap).toBe(25.5);
  });

  it("returns a fresh array on every call (no aliasing)", () => {
    const codes = [C1];
    const out = buildQueryPayload({ projectId: PID, codeIds: codes });
    codes.push(C2);
    // The previously-produced payload's leaf must still reflect a
    // single-id form despite the caller mutating the input array.
    expect(out.codes.expr).toEqual({ op: "code", code_id: C1 });
  });
});


describe("groupApplicationsBySource", () => {
  it("returns an empty Map for an empty array", () => {
    const m = groupApplicationsBySource([]);
    expect(m.size).toBe(0);
  });

  it("preserves first-seen source order", () => {
    const apps = [
      { id: "a", source_id: S2 },
      { id: "b", source_id: S1 },
      { id: "c", source_id: S2 },
    ];
    const m = groupApplicationsBySource(apps);
    expect(Array.from(m.keys())).toEqual([S2, S1]);
    expect(m.get(S2).map(a => a.id)).toEqual(["a", "c"]);
    expect(m.get(S1).map(a => a.id)).toEqual(["b"]);
  });

  it("tolerates non-array input", () => {
    expect(groupApplicationsBySource(null).size).toBe(0);
    expect(groupApplicationsBySource(undefined).size).toBe(0);
  });

  it("skips falsy / non-object entries", () => {
    const m = groupApplicationsBySource([null, undefined, { source_id: S1 }]);
    expect(m.size).toBe(1);
    expect(m.get(S1)).toHaveLength(1);
  });
});
