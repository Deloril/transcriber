// Tests for the search-highlighter token-range mapper.

import { describe, it, expect } from "vitest";
import { rangeForMatch } from "../../scribe/static/js/helpers.mjs";

describe("rangeForMatch", () => {
  it("returns empty list when needle isn't present", () => {
    expect(rangeForMatch(["the", "quick", "fox"], "zebra")).toEqual([]);
  });

  it("returns single-word match", () => {
    expect(rangeForMatch(["the", "quick", "fox"], "quick")).toEqual([
      { firstWord: 1, lastWord: 1 },
    ]);
  });

  it("returns multi-word match spanning two words", () => {
    expect(rangeForMatch(["the", "quick", "brown", "fox"], "quick brown")).toEqual([
      { firstWord: 1, lastWord: 2 },
    ]);
  });

  it("returns multi-word match spanning three words", () => {
    expect(rangeForMatch(["the", "quick", "brown", "fox"], "the quick brown")).toEqual([
      { firstWord: 0, lastWord: 2 },
    ]);
  });

  it("is case-insensitive", () => {
    expect(rangeForMatch(["The", "Quick", "Fox"], "quick")).toEqual([
      { firstWord: 1, lastWord: 1 },
    ]);
    expect(rangeForMatch(["the", "quick"], "THE QUICK")).toEqual([
      { firstWord: 0, lastWord: 1 },
    ]);
  });

  it("finds multiple non-overlapping matches", () => {
    const out = rangeForMatch(
      ["yes", "no", "yes", "maybe", "yes"], "yes",
    );
    expect(out).toEqual([
      { firstWord: 0, lastWord: 0 },
      { firstWord: 2, lastWord: 2 },
      { firstWord: 4, lastWord: 4 },
    ]);
  });

  it("finds substring matches inside a single word", () => {
    // "fox" appears inside "foxes" — the match should still light up that word.
    expect(rangeForMatch(["the", "foxes"], "fox")).toEqual([
      { firstWord: 1, lastWord: 1 },
    ]);
  });

  it("handles word at the start", () => {
    expect(rangeForMatch(["alpha", "beta", "gamma"], "alpha")).toEqual([
      { firstWord: 0, lastWord: 0 },
    ]);
  });

  it("handles word at the end", () => {
    expect(rangeForMatch(["alpha", "beta", "gamma"], "gamma")).toEqual([
      { firstWord: 2, lastWord: 2 },
    ]);
  });

  it("returns empty for empty token list", () => {
    expect(rangeForMatch([], "anything")).toEqual([]);
  });

  it("matches across the joining-space boundary", () => {
    // Joined string is "alpha beta gamma" — "ha be" spans the gap.
    const out = rangeForMatch(["alpha", "beta", "gamma"], "ha be");
    expect(out).toEqual([{ firstWord: 0, lastWord: 1 }]);
  });
});
