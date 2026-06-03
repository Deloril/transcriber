import {describe, it, expect} from "vitest";
import {reorderSegments} from "../../scribe/static/js/helpers.mjs";


describe("reorderSegments", () => {
  // Build a tiny segment list keyed by text so the assertion error
  // messages stay readable when an index calculation is off by one.
  const _segs = (...labels) => labels.map(t => ({text: t}));

  it("moves a row up (before earlier neighbour)", () => {
    const out = reorderSegments(_segs("A", "B", "C", "D"), 2, 0, true);
    expect(out.map(s => s.text)).toEqual(["C", "A", "B", "D"]);
  });

  it("moves a row up (after earlier neighbour)", () => {
    const out = reorderSegments(_segs("A", "B", "C", "D"), 2, 0, false);
    expect(out.map(s => s.text)).toEqual(["A", "C", "B", "D"]);
  });

  it("moves a row down (before later neighbour)", () => {
    const out = reorderSegments(_segs("A", "B", "C", "D"), 0, 3, true);
    expect(out.map(s => s.text)).toEqual(["B", "C", "A", "D"]);
  });

  it("moves a row down (after later neighbour)", () => {
    const out = reorderSegments(_segs("A", "B", "C", "D"), 0, 3, false);
    expect(out.map(s => s.text)).toEqual(["B", "C", "D", "A"]);
  });

  it("dropping onto self is a no-op", () => {
    const segs = _segs("A", "B", "C");
    const out = reorderSegments(segs, 1, 1, true);
    expect(out.map(s => s.text)).toEqual(["A", "B", "C"]);
  });

  it("returns a new array — input is not mutated", () => {
    const segs = _segs("A", "B", "C");
    reorderSegments(segs, 2, 0, true);
    expect(segs.map(s => s.text)).toEqual(["A", "B", "C"]);
  });

  it("rejects non-integer indices", () => {
    expect(() => reorderSegments(_segs("A", "B"), 0.5, 1, true)).toThrow();
    expect(() => reorderSegments(_segs("A", "B"), 0, "1", true)).toThrow();
  });

  it("rejects out-of-range indices", () => {
    expect(() => reorderSegments(_segs("A", "B"), 5, 0, true)).toThrow();
    expect(() => reorderSegments(_segs("A", "B"), 0, -1, true)).toThrow();
  });

  it("rejects non-array segments", () => {
    expect(() => reorderSegments(null, 0, 1, true)).toThrow();
    expect(() => reorderSegments({segments: []}, 0, 1, true)).toThrow();
  });

  it("preserves segment objects by reference", () => {
    const segs = _segs("A", "B");
    const out = reorderSegments(segs, 0, 1, false);
    // Same object instances — the editor relies on this so any
    // unrelated state attached to the segment (e.g. word arrays,
    // temporary highlighting) survives a reorder.
    expect(out[0]).toBe(segs[1]);
    expect(out[1]).toBe(segs[0]);
  });
});
