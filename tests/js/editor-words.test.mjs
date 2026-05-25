// Tests for editor word-highlighting helpers.

import { describe, it, expect } from "vitest";
import {
  findActiveWord,
  spreadTokensAcrossSpan,
} from "../../scribe/static/js/helpers.mjs";

describe("findActiveWord", () => {
  const words = [
    { start: 0.0, end: 0.5 },   // 0
    { start: 0.6, end: 1.0 },   // 1
    { start: 1.5, end: 2.0 },   // 2 (gap before)
    { start: 3.0, end: 3.4 },   // 3 (big gap before)
    { start: 3.4, end: 3.8 },   // 4 (no gap)
  ];

  describe("monotonic playback", () => {
    it("finds the active word from full search", () => {
      expect(findActiveWord(words, 0.25, -1)).toBe(0);
      expect(findActiveWord(words, 0.7, -1)).toBe(1);
      expect(findActiveWord(words, 1.7, -1)).toBe(2);
    });

    it("uses lastActive shortcut when within span", () => {
      // Already on word 1 → return same index.
      expect(findActiveWord(words, 0.7, 1)).toBe(1);
    });

    it("advances when playback crosses to next word", () => {
      // From word 1, time advances into word 2.
      expect(findActiveWord(words, 1.7, 1)).toBe(2);
    });

    it("returns -1 when in a gap (regression for the climbing-highlight bug)", () => {
      // Between word 1 (ends 1.0) and word 2 (starts 1.5) — must clear.
      expect(findActiveWord(words, 1.25, 1)).toBe(-1);
      // Same via binary-search path (lastActive=-1).
      expect(findActiveWord(words, 1.25, -1)).toBe(-1);
    });

    it("stays on word during the small post-end grace window", () => {
      // GAP=0.05 — within that window we keep the highlight.
      expect(findActiveWord(words, 0.52, 0)).toBe(0);
      // Beyond the window, in a true gap, clear.
      expect(findActiveWord(words, 0.58, 0)).toBe(-1);
    });

    it("handles the no-gap case at word boundaries", () => {
      // Word 4 starts exactly where word 3 ends — no gap.
      expect(findActiveWord(words, 3.4, 3)).toBe(3);  // still in word 3's grace
      expect(findActiveWord(words, 3.5, 3)).toBe(4);
    });
  });

  describe("seek (non-monotonic)", () => {
    it("binary-searches when lastActive is invalid", () => {
      // Seek straight to word 4 from -1.
      expect(findActiveWord(words, 3.5, -1)).toBe(4);
    });

    it("returns -1 when seeking before any word", () => {
      // Time before word 0 starts.
      expect(findActiveWord(words, -0.5, -1)).toBe(-1);
    });

    it("returns -1 when seeking after the last word ends", () => {
      // Far beyond word 4's end.
      expect(findActiveWord(words, 100.0, -1)).toBe(-1);
    });

    it("handles empty word list", () => {
      expect(findActiveWord([], 5.0, -1)).toBe(-1);
      expect(findActiveWord([], 5.0, 0)).toBe(-1);
    });
  });
});

describe("spreadTokensAcrossSpan", () => {
  it("returns empty list for no tokens", () => {
    expect(spreadTokensAcrossSpan([], 0, 1, "A")).toEqual([]);
  });

  it("spreads tokens evenly across the span", () => {
    const out = spreadTokensAcrossSpan(["a", "b", "c", "d"], 0, 4, "A");
    expect(out).toHaveLength(4);
    expect(out[0]).toEqual({ text: "a", start: 0, end: 1, speaker: "A", score: null });
    expect(out[1]).toEqual({ text: "b", start: 1, end: 2, speaker: "A", score: null });
    expect(out[3]).toEqual({ text: "d", start: 3, end: 4, speaker: "A", score: null });
  });

  it("uses minimum 0.05s span for very short ranges", () => {
    // start === end: would otherwise be div-by-zero.
    const out = spreadTokensAcrossSpan(["a", "b"], 1.0, 1.0, "S");
    expect(out).toHaveLength(2);
    // Each token gets 0.025s out of the 0.05 minimum.
    expect(out[0].start).toBe(1.0);
    expect(out[0].end).toBeCloseTo(1.025);
    expect(out[1].start).toBeCloseTo(1.025);
    expect(out[1].end).toBeCloseTo(1.05);
  });

  it("preserves the speaker label", () => {
    const out = spreadTokensAcrossSpan(["x"], 0, 1, "GUEST");
    expect(out[0].speaker).toBe("GUEST");
  });

  it("clears confidence score on resynthesised words", () => {
    const out = spreadTokensAcrossSpan(["x", "y"], 0, 1, "A");
    expect(out[0].score).toBeNull();
    expect(out[1].score).toBeNull();
  });

  it("single token covers the full span", () => {
    const out = spreadTokensAcrossSpan(["only"], 5.0, 8.0, "Z");
    expect(out).toEqual([
      { text: "only", start: 5.0, end: 8.0, speaker: "Z", score: null },
    ]);
  });
});
