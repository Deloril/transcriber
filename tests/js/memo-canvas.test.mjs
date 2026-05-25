// Tests for the F5.3 memo-sorting canvas helpers in helpers.mjs.
//
// Mirrors tests/test_memo_canvas.py for the pure-helper subset
// (clamp / snap / hit-test) plus the wire-payload builders. Constants
// here MUST match the Python module so a drag computed in the editor
// produces the same canvas state regardless of who runs the math.

import { describe, it, expect } from "vitest";
import {
  CANVAS_MAX_COORD,
  CANVAS_MAX_LABEL_LEN,
  buildAddCategoryPayload,
  buildAssignCardPayload,
  buildMoveCardPayload,
  clampToBounds,
  hitTestCard,
  snapToGrid,
} from "../../scribe/static/js/helpers.mjs";

const MEMO_A = "a".repeat(12);
const MEMO_B = "b".repeat(12);

// --------------------------------------------------------------------------- //
// clampToBounds
// --------------------------------------------------------------------------- //

describe("clampToBounds", () => {
  it("passes through values inside bounds", () => {
    expect(
      clampToBounds(10, 20, { minX: 0, minY: 0, maxX: 100, maxY: 100 }),
    ).toEqual([10, 20]);
  });

  it("clamps to lower bound", () => {
    expect(
      clampToBounds(-5, -5, { minX: 0, minY: 0, maxX: 100, maxY: 100 }),
    ).toEqual([0, 0]);
  });

  it("clamps to upper bound", () => {
    expect(
      clampToBounds(500, 500, { minX: 0, minY: 0, maxX: 100, maxY: 100 }),
    ).toEqual([100, 100]);
  });

  it("default bounds use ±CANVAS_MAX_COORD", () => {
    expect(clampToBounds(123456, -789)).toEqual([123456, -789]);
  });

  it("rejects NaN", () => {
    expect(() => clampToBounds(NaN, 0)).toThrow();
  });

  it("rejects Infinity", () => {
    expect(() => clampToBounds(0, Infinity)).toThrow();
  });

  it("rejects bool", () => {
    expect(() => clampToBounds(true, 0)).toThrow();
  });

  it("rejects inverted bounds", () => {
    expect(() => clampToBounds(0, 0, { minX: 10, maxX: 0 })).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// snapToGrid
// --------------------------------------------------------------------------- //

describe("snapToGrid", () => {
  it("default grid is round-to-int", () => {
    expect(snapToGrid(3.4, 7.6)).toEqual([3, 8]);
  });

  it("grid of 16", () => {
    expect(snapToGrid(22, 25, { grid: 16 })).toEqual([16, 32]);
  });

  it("rejects zero grid", () => {
    expect(() => snapToGrid(1, 1, { grid: 0 })).toThrow();
  });

  it("rejects negative grid", () => {
    expect(() => snapToGrid(1, 1, { grid: -4 })).toThrow();
  });

  it("rejects NaN coord", () => {
    expect(() => snapToGrid(NaN, 0)).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// hitTestCard
// --------------------------------------------------------------------------- //

describe("hitTestCard", () => {
  it("empty cards returns null", () => {
    expect(hitTestCard([], 0, 0)).toBeNull();
  });

  it("inside box returns memo_id", () => {
    const cards = [{ memo_id: MEMO_A, x: 100, y: 100 }];
    expect(
      hitTestCard(cards, 110, 95, { halfWidth: 80, halfHeight: 50 }),
    ).toBe(MEMO_A);
  });

  it("outside box returns null", () => {
    const cards = [{ memo_id: MEMO_A, x: 100, y: 100 }];
    expect(
      hitTestCard(cards, 300, 300, { halfWidth: 80, halfHeight: 50 }),
    ).toBeNull();
  });

  it("topmost (later in list) wins", () => {
    const cards = [
      { memo_id: MEMO_A, x: 100, y: 100 },
      { memo_id: MEMO_B, x: 110, y: 110 },
    ];
    expect(
      hitTestCard(cards, 105, 105, { halfWidth: 80, halfHeight: 50 }),
    ).toBe(MEMO_B);
  });

  it("accepts camelCase memoId", () => {
    const cards = [{ memoId: MEMO_A, x: 0, y: 0 }];
    expect(hitTestCard(cards, 0, 0)).toBe(MEMO_A);
  });

  it("zero half-extent throws", () => {
    expect(() =>
      hitTestCard([{ memo_id: MEMO_A, x: 0, y: 0 }], 0, 0, { halfWidth: 0 }),
    ).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// Payload builders
// --------------------------------------------------------------------------- //

describe("buildMoveCardPayload", () => {
  it("emits x/y", () => {
    expect(buildMoveCardPayload({ x: 10, y: 20 })).toEqual({ x: 10, y: 20 });
  });

  it("rejects NaN", () => {
    expect(() => buildMoveCardPayload({ x: NaN, y: 0 })).toThrow();
  });

  it("rejects out-of-range", () => {
    expect(() =>
      buildMoveCardPayload({ x: CANVAS_MAX_COORD * 2, y: 0 }),
    ).toThrow();
  });
});

describe("buildAddCategoryPayload", () => {
  it("emits label only by default", () => {
    expect(buildAddCategoryPayload({ label: "Care" })).toEqual({
      label: "Care",
      x: 0,
      y: 0,
    });
  });

  it("trims label", () => {
    expect(buildAddCategoryPayload({ label: "   Care   " })).toEqual({
      label: "Care",
      x: 0,
      y: 0,
    });
  });

  it("emits color when given", () => {
    expect(
      buildAddCategoryPayload({ label: "Care", color: "#FF8800" }),
    ).toEqual({
      label: "Care",
      x: 0,
      y: 0,
      color: "#ff8800",
    });
  });

  it("rejects bad color", () => {
    expect(() =>
      buildAddCategoryPayload({ label: "Care", color: "red" }),
    ).toThrow();
  });

  it("rejects empty label", () => {
    expect(() => buildAddCategoryPayload({ label: "   " })).toThrow();
  });

  it("rejects too-long label", () => {
    expect(() =>
      buildAddCategoryPayload({ label: "x".repeat(CANVAS_MAX_LABEL_LEN + 1) }),
    ).toThrow();
  });

  it("emits x/y when given", () => {
    expect(buildAddCategoryPayload({ label: "Care", x: 5, y: 6 })).toEqual({
      label: "Care",
      x: 5,
      y: 6,
    });
  });
});

describe("buildAssignCardPayload", () => {
  it("returns empty object — the ids live on the path", () => {
    expect(buildAssignCardPayload()).toEqual({});
  });
});
