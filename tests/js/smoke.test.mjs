// Smoke test: prove the helpers module loads and a basic function works.
// Real coverage lives in the per-feature test files (#58–#61).
import { describe, it, expect } from "vitest";
import { fmtElapsed } from "../../scribe/static/js/helpers.mjs";

describe("helpers smoke", () => {
  it("fmtElapsed handles seconds", () => {
    expect(fmtElapsed(5)).toBe("5s");
  });
});
