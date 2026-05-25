// Tests for ETA + elapsed math.
//
// The bug we fixed in commit f84ad6d: ETA was computed as
//   remaining = elapsed × (1/progress − 1)
// which mechanically grew between progress updates, causing the counter
// to climb up and then snap down each time a new SSE event arrived.
// The fix holds ETA as an absolute predicted finish time so the
// countdown is monotonic between updates. These tests pin that.

import { describe, it, expect } from "vitest";
import {
  fmtElapsed,
  predictFinishTime,
} from "../../scribe/static/js/helpers.mjs";

describe("fmtElapsed", () => {
  it.each([
    [0, "0s"],
    [1, "1s"],
    [59, "59s"],
    [60, "1m 00s"],
    [125, "2m 05s"],
    [3599, "59m 59s"],
    [3600, "1h 00m 00s"],
    [3661, "1h 01m 01s"],
  ])("formats %d seconds as %s", (input, expected) => {
    expect(fmtElapsed(input)).toBe(expected);
  });

  it("clamps negative values to zero", () => {
    expect(fmtElapsed(-100)).toBe("0s");
  });

  it("rounds to nearest second", () => {
    expect(fmtElapsed(1.4)).toBe("1s");
    expect(fmtElapsed(1.5)).toBe("2s");
    expect(fmtElapsed(59.9)).toBe("1m 00s");
  });
});

describe("predictFinishTime", () => {
  it("returns null before any progress update", () => {
    expect(predictFinishTime({
      startedAt: 100, lastProgress: 0, lastProgressTime: null,
    })).toBeNull();
  });

  it("returns null when progress is below threshold", () => {
    // Below 5% the projection is meaningless — refuse to estimate.
    expect(predictFinishTime({
      startedAt: 100, lastProgress: 0.04, lastProgressTime: 110,
    })).toBeNull();
  });

  it("returns null at exactly the threshold (>0.05 required)", () => {
    expect(predictFinishTime({
      startedAt: 100, lastProgress: 0.05, lastProgressTime: 110,
    })).toBeNull();
  });

  it("projects from a single progress reading", () => {
    // 10s elapsed at 25% complete → ETA total 40s → finish at startedAt+40
    const finish = predictFinishTime({
      startedAt: 100, lastProgress: 0.25, lastProgressTime: 110,
    });
    expect(finish).toBe(140);
  });

  it("projects from later progress reading", () => {
    // 30s in at 50% complete → 60s total
    const finish = predictFinishTime({
      startedAt: 1000, lastProgress: 0.5, lastProgressTime: 1030,
    });
    expect(finish).toBe(1060);
  });

  it("returns null when lastProgressTime is before startedAt (clock skew)", () => {
    expect(predictFinishTime({
      startedAt: 1000, lastProgress: 0.2, lastProgressTime: 500,
    })).toBeNull();
  });

  it("returns null when lastProgressTime equals startedAt (zero elapsed)", () => {
    expect(predictFinishTime({
      startedAt: 1000, lastProgress: 0.5, lastProgressTime: 1000,
    })).toBeNull();
  });
});

describe("ETA monotonic countdown — regression test for the climbing bug", () => {
  // Simulate the buggy formula vs the fixed one to make sure the new code
  // genuinely produces a monotonically-decreasing remaining time between
  // progress updates.

  function remainingAt(state, now) {
    const finish = predictFinishTime(state);
    if (finish == null) return null;
    return Math.max(0, finish - now);
  }

  it("counts down between updates, doesn't climb", () => {
    // Progress update at T=10, lastProgress=0.25 → predicted finish at T=40.
    const state = { startedAt: 0, lastProgress: 0.25, lastProgressTime: 10 };
    const samples = [10, 12, 14, 16, 18, 20, 22, 24];
    const remainings = samples.map(t => remainingAt(state, t));
    // Expect 30, 28, 26, 24, 22, 20, 18, 16 — strictly decreasing.
    for (let i = 1; i < remainings.length; i++) {
      expect(remainings[i]).toBeLessThan(remainings[i - 1]);
    }
    expect(remainings[0]).toBe(30);
    expect(remainings[remainings.length - 1]).toBe(16);
  });

  it("revises on new progress reading", () => {
    // Initial reading at T=10, p=0.25 → predicted finish T=40.
    const initial = { startedAt: 0, lastProgress: 0.25, lastProgressTime: 10 };
    expect(remainingAt(initial, 20)).toBe(20);

    // Better reading at T=20, p=0.6 → predicted finish T=33.33.
    // (Job is going faster than we initially estimated.)
    const updated = { startedAt: 0, lastProgress: 0.6, lastProgressTime: 20 };
    expect(remainingAt(updated, 20)).toBeCloseTo(13.33, 1);
  });
});
