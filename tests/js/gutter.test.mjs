// Tests for the JS gutter / margin layout helpers (F4.3).
//
// Mirrors the Python tests in tests/test_application_gutter.py:
// the algorithm and outputs MUST agree with
// scribe.application_gutter.assign_lanes for the renderer's lane
// numbering to be consistent across server↔client.

import { describe, it, expect } from "vitest";
import {
  assignLanes,
  assignLanesPerSource,
  sortApplicationsByAnchor,
  parseWordId,
} from "../../scribe/static/js/helpers.mjs";

// --------------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------------- //

function hexId(n) {
  return n.toString(16).padStart(12, "0");
}

const SRC1 = "1".repeat(12);
const SRC2 = "2".repeat(12);

function app({
  id,
  sourceId = SRC1,
  start,
  end,
  startOffset = null,
  endOffset = null,
} = {}) {
  return {
    id,
    sourceId,
    anchorStartWordId: start,
    anchorEndWordId: end,
    startCharOffset: startOffset,
    endCharOffset: endOffset,
  };
}

// --------------------------------------------------------------------------- //
// parseWordId
// --------------------------------------------------------------------------- //

describe("parseWordId", () => {
  it("parses canonical ids", () => {
    expect(parseWordId("s0w0")).toEqual([0, 0]);
    expect(parseWordId("s12w345")).toEqual([12, 345]);
  });
  it("returns null for malformed inputs", () => {
    expect(parseWordId("")).toBeNull();
    expect(parseWordId("S0W0")).toBeNull();
    expect(parseWordId("s0")).toBeNull();
    expect(parseWordId(null)).toBeNull();
    expect(parseWordId(undefined)).toBeNull();
    expect(parseWordId(42)).toBeNull();
  });
});

// --------------------------------------------------------------------------- //
// sortApplicationsByAnchor
// --------------------------------------------------------------------------- //

describe("sortApplicationsByAnchor", () => {
  it("sorts in document order", () => {
    const a = app({ id: hexId(3), start: "s0w10", end: "s0w15" });
    const b = app({ id: hexId(2), start: "s0w0", end: "s0w5" });
    const c = app({ id: hexId(1), start: "s0w3", end: "s0w8" });
    const out = sortApplicationsByAnchor([a, b, c]);
    expect(out.map((x) => x.id)).toEqual([b.id, c.id, a.id]);
  });

  it("breaks ties by application id", () => {
    const a = app({ id: hexId(0xFFFF), start: "s0w0", end: "s0w5" });
    const b = app({ id: hexId(0x0001), start: "s0w0", end: "s0w5" });
    const out = sortApplicationsByAnchor([a, b]);
    expect(out.map((x) => x.id)).toEqual([b.id, a.id]);
  });

  it("does not mutate input", () => {
    const list = [
      app({ id: hexId(2), start: "s0w5", end: "s0w9" }),
      app({ id: hexId(1), start: "s0w0", end: "s0w3" }),
    ];
    const before = list.map((x) => x.id);
    sortApplicationsByAnchor(list);
    expect(list.map((x) => x.id)).toEqual(before);
  });
});

// --------------------------------------------------------------------------- //
// assignLanes — empty / trivial
// --------------------------------------------------------------------------- //

describe("assignLanes — empty / trivial", () => {
  it("handles empty input", () => {
    expect(assignLanes([])).toEqual({
      sourceId: "",
      placements: [],
      laneCount: 0,
      maxStackDepth: 0,
    });
  });

  it("places a solo application in lane 0", () => {
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w5" });
    const layout = assignLanes([a]);
    expect(layout.sourceId).toBe(SRC1);
    expect(layout.laneCount).toBe(1);
    expect(layout.maxStackDepth).toBe(0);
    expect(layout.placements).toEqual([
      { applicationId: a.id, lane: 0, stackDepth: 0 },
    ]);
  });

  it("shares one lane for disjoint applications", () => {
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w3" });
    const b = app({ id: hexId(2), start: "s0w5", end: "s0w9" });
    const layout = assignLanes([a, b]);
    expect(layout.laneCount).toBe(1);
    expect(layout.placements.every((p) => p.lane === 0)).toBe(true);
    expect(layout.maxStackDepth).toBe(0);
  });
});

// --------------------------------------------------------------------------- //
// assignLanes — overlap behaviour
// --------------------------------------------------------------------------- //

describe("assignLanes — overlap", () => {
  it("uses two lanes for an overlapping pair", () => {
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w5" });
    const b = app({ id: hexId(2), start: "s0w3", end: "s0w8" });
    const layout = assignLanes([a, b]);
    expect(layout.laneCount).toBe(2);
    expect(layout.placements[0].lane).toBe(0);
    expect(layout.placements[1].lane).toBe(1);
    expect(layout.placements.every((p) => p.stackDepth === 1)).toBe(true);
  });

  it("opens a separate lane for each of six co-located codes", () => {
    // Six codes on the same 30-word utterance — the design's
    // motivating example.
    const apps = [];
    for (let i = 1; i <= 6; i++) {
      apps.push(app({ id: hexId(i), start: "s0w0", end: "s0w29" }));
    }
    const layout = assignLanes(apps);
    expect(layout.laneCount).toBe(6);
    const lanes = new Set(layout.placements.map((p) => p.lane));
    expect(lanes.size).toBe(6);
    expect(layout.placements.every((p) => p.stackDepth === 5)).toBe(true);
    expect(layout.maxStackDepth).toBe(5);
  });

  it("re-uses lane 0 once the previous occupant ends", () => {
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w3" });
    const b = app({ id: hexId(2), start: "s0w2", end: "s0w7" });
    const c = app({ id: hexId(3), start: "s0w10", end: "s0w15" });
    const layout = assignLanes([a, b, c]);
    expect(layout.laneCount).toBe(2);
    expect(layout.placements[0].lane).toBe(0);
    expect(layout.placements[1].lane).toBe(1);
    expect(layout.placements[2].lane).toBe(0);
  });

  it("touching at a single point can share a lane", () => {
    // F4.2: touching at a point is NOT overlap.
    const a = app({
      id: hexId(1),
      start: "s0w5",
      end: "s0w5",
      startOffset: 0,
      endOffset: 3,
    });
    const b = app({
      id: hexId(2),
      start: "s0w5",
      end: "s0w5",
      startOffset: 3,
      endOffset: 8,
    });
    const layout = assignLanes([a, b]);
    expect(layout.laneCount).toBe(1);
    expect(layout.placements.every((p) => p.stackDepth === 0)).toBe(true);
  });

  it("computes asymmetric stack depths in a chain", () => {
    // A↔B overlap; B↔C overlap; A and C don't touch directly.
    // Depth: A=1, B=2, C=1.
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w3" });
    const b = app({ id: hexId(2), start: "s0w2", end: "s0w7" });
    const c = app({ id: hexId(3), start: "s0w6", end: "s0w10" });
    const layout = assignLanes([a, b, c]);
    const depthById = Object.fromEntries(
      layout.placements.map((p) => [p.applicationId, p.stackDepth])
    );
    expect(depthById[a.id]).toBe(1);
    expect(depthById[b.id]).toBe(2);
    expect(depthById[c.id]).toBe(1);
    expect(layout.maxStackDepth).toBe(2);
  });
});

// --------------------------------------------------------------------------- //
// assignLanes — sub-word offsets
// --------------------------------------------------------------------------- //

describe("assignLanes — sub-word offsets", () => {
  it("subword overlap on a single word forces two lanes", () => {
    const a = app({
      id: hexId(1),
      start: "s0w5",
      end: "s0w5",
      startOffset: 0,
      endOffset: 4,
    });
    const b = app({
      id: hexId(2),
      start: "s0w5",
      end: "s0w5",
      startOffset: 2,
      endOffset: 8,
    });
    expect(assignLanes([a, b]).laneCount).toBe(2);
  });

  it("subword disjoint on a single word shares a lane", () => {
    const a = app({
      id: hexId(1),
      start: "s0w5",
      end: "s0w5",
      startOffset: 0,
      endOffset: 3,
    });
    const b = app({
      id: hexId(2),
      start: "s0w5",
      end: "s0w5",
      startOffset: 4,
      endOffset: 7,
    });
    expect(assignLanes([a, b]).laneCount).toBe(1);
  });
});

// --------------------------------------------------------------------------- //
// assignLanes — determinism / ordering
// --------------------------------------------------------------------------- //

describe("assignLanes — determinism", () => {
  it("input order does not affect the layout", () => {
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w5" });
    const b = app({ id: hexId(2), start: "s0w3", end: "s0w8" });
    const c = app({ id: hexId(3), start: "s0w10", end: "s0w15" });
    const forwards = assignLanes([a, b, c]);
    const reversed = assignLanes([c, b, a]);
    expect(forwards).toEqual(reversed);
  });

  it("placements are in document order", () => {
    const c = app({ id: hexId(3), start: "s0w10", end: "s0w15" });
    const a = app({ id: hexId(1), start: "s0w0", end: "s0w5" });
    const b = app({ id: hexId(2), start: "s0w3", end: "s0w8" });
    const layout = assignLanes([c, a, b]);
    expect(layout.placements.map((p) => p.applicationId)).toEqual([
      a.id,
      b.id,
      c.id,
    ]);
  });
});

// --------------------------------------------------------------------------- //
// Cross-source guard
// --------------------------------------------------------------------------- //

describe("assignLanes — cross-source guard", () => {
  it("throws on mixed sources", () => {
    const a = app({ id: hexId(1), sourceId: SRC1 });
    const b = app({ id: hexId(2), sourceId: SRC2 });
    expect(() => assignLanes([a, b])).toThrow(/single-source/);
  });
});

// --------------------------------------------------------------------------- //
// assignLanesPerSource
// --------------------------------------------------------------------------- //

describe("assignLanesPerSource", () => {
  it("returns an empty object on no input", () => {
    expect(assignLanesPerSource([])).toEqual({});
    expect(assignLanesPerSource(undefined)).toEqual({});
  });

  it("buckets applications by sourceId", () => {
    const a = app({ id: hexId(1), sourceId: SRC1 });
    const b = app({ id: hexId(2), sourceId: SRC2 });
    const out = assignLanesPerSource([a, b]);
    expect(Object.keys(out).sort()).toEqual([SRC1, SRC2].sort());
    expect(out[SRC1].laneCount).toBe(1);
    expect(out[SRC2].laneCount).toBe(1);
  });

  it("numbers lanes independently per source", () => {
    const a1 = app({
      id: hexId(1),
      sourceId: SRC1,
      start: "s0w0",
      end: "s0w3",
    });
    const a2 = app({
      id: hexId(2),
      sourceId: SRC1,
      start: "s0w1",
      end: "s0w5",
    });
    const b1 = app({
      id: hexId(3),
      sourceId: SRC2,
      start: "s0w0",
      end: "s0w3",
    });
    const out = assignLanesPerSource([a1, a2, b1]);
    expect(out[SRC1].laneCount).toBe(2);
    expect(out[SRC2].laneCount).toBe(1);
  });
});

// --------------------------------------------------------------------------- //
// End-to-end realistic case (mirror of Python)
// --------------------------------------------------------------------------- //

describe("assignLanes — realistic mixed gutter", () => {
  it("matches the Python end-to-end fixture", () => {
    const apps = [
      app({ id: hexId(1), start: "s0w0", end: "s2w10" }), // long topic
      app({ id: hexId(2), start: "s0w2", end: "s0w9" }),  // first emotion
      app({ id: hexId(3), start: "s0w4", end: "s0w6" }),  // in-vivo
      app({ id: hexId(4), start: "s1w2", end: "s1w7" }),  // second emotion
      app({ id: hexId(5), start: "s1w3", end: "s1w5" }),  // reflexive
      app({ id: hexId(6), start: "s2w5", end: "s2w9" }),  // disjoint
    ];
    const layout = assignLanes(apps);
    expect(layout.laneCount).toBe(3);
    const laneById = Object.fromEntries(
      layout.placements.map((p) => [p.applicationId, p.lane])
    );
    expect(laneById[apps[0].id]).toBe(0);
    expect(laneById[apps[1].id]).toBe(1);
    expect(laneById[apps[2].id]).toBe(2);
    expect(laneById[apps[3].id]).toBe(1);
    expect(laneById[apps[4].id]).toBe(2);
    expect(laneById[apps[5].id]).toBe(1);
    // Long topic overlaps all 5 others.
    const depthById = Object.fromEntries(
      layout.placements.map((p) => [p.applicationId, p.stackDepth])
    );
    expect(depthById[apps[0].id]).toBe(5);
    expect(layout.maxStackDepth).toBe(5);
  });
});
