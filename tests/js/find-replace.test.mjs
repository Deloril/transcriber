import { describe, it, expect } from "vitest";
import {
  replaceInSegmentWords,
  rebuildSegmentText,
} from "../../scribe/static/js/helpers.mjs";

const W = (text, start, end) => ({
  text,
  start,
  end,
  speaker: "S",
  score: 0.95,
});

describe("replaceInSegmentWords — single-word matches", () => {
  it("replaces a substring inside a single word, keeping timestamps", () => {
    const words = [W("Hello", 0, 0.5), W("world", 0.6, 1.0)];
    const { words: out, replacements } = replaceInSegmentWords(words, "Hell", "Salut");
    expect(replacements).toBe(1);
    expect(out).toHaveLength(2);
    expect(out[0].text).toBe("Saluto");
    // Timestamps preserved.
    expect(out[0].start).toBe(0);
    expect(out[0].end).toBe(0.5);
    // Speaker and score preserved.
    expect(out[0].speaker).toBe("S");
    expect(out[0].score).toBe(0.95);
  });

  it("matches case-insensitively but writes the replacement verbatim", () => {
    const words = [W("HELLO", 0, 1)];
    const { words: out } = replaceInSegmentWords(words, "hello", "world");
    expect(out[0].text).toBe("world");
  });

  it("handles multiple matches in the same word", () => {
    const words = [W("ababab", 0, 1)];
    const { words: out, replacements } = replaceInSegmentWords(words, "ab", "X");
    expect(replacements).toBe(3);
    expect(out[0].text).toBe("XXX");
  });

  it("drops a word entirely when the replacement empties it", () => {
    const words = [W("um", 0, 0.3), W("hello", 0.4, 1)];
    const { words: out } = replaceInSegmentWords(words, "um", "");
    expect(out).toHaveLength(1);
    expect(out[0].text).toBe("hello");
  });

  it("returns input unchanged on empty needle", () => {
    const words = [W("hi", 0, 1)];
    const { words: out, replacements } = replaceInSegmentWords(words, "", "X");
    expect(replacements).toBe(0);
    expect(out).toHaveLength(1);
    expect(out[0].text).toBe("hi");
  });

  it("returns input unchanged when needle is not found", () => {
    const words = [W("hi", 0, 1)];
    const { words: out, replacements } = replaceInSegmentWords(words, "world", "X");
    expect(replacements).toBe(0);
    expect(out[0].text).toBe("hi");
  });
});

describe("replaceInSegmentWords — multi-word matches", () => {
  it("collapses a 2-word match into one word spanning both timestamps", () => {
    const words = [
      W("the", 0, 0.2),
      W("quick", 0.3, 0.6),
      W("brown", 0.7, 1.0),
      W("fox", 1.1, 1.4),
    ];
    const { words: out, replacements } = replaceInSegmentWords(
      words, "quick brown", "agile",
    );
    expect(replacements).toBe(1);
    expect(out).toHaveLength(3);
    expect(out[0].text).toBe("the");
    expect(out[1].text).toBe("agile");
    // Spans both original words' timestamps.
    expect(out[1].start).toBe(0.3);
    expect(out[1].end).toBe(1.0);
    // Confidence cleared because it's synthetic.
    expect(out[1].score).toBeNull();
    expect(out[2].text).toBe("fox");
  });

  it("preserves prefix / suffix when the match doesn't cover the full first/last word", () => {
    const words = [W("foo", 0, 0.5), W("bar", 0.6, 1.0)];
    // "oo ba" spans the suffix of "foo" + the joining space + the prefix of "bar".
    const { words: out, replacements } = replaceInSegmentWords(words, "oo ba", "X");
    expect(replacements).toBe(1);
    expect(out).toHaveLength(1);
    expect(out[0].text).toBe("fXr");
    // Timestamp range still spans both source words.
    expect(out[0].start).toBe(0);
    expect(out[0].end).toBe(1.0);
  });

  it("drops the run entirely when the replacement empties it", () => {
    const words = [W("um", 0, 0.2), W("uh", 0.3, 0.5), W("hello", 0.6, 1.0)];
    const { words: out } = replaceInSegmentWords(words, "um uh", "");
    expect(out).toHaveLength(1);
    expect(out[0].text).toBe("hello");
  });

  it("handles overlapping potential matches by advancing past each hit", () => {
    // "abab" replacing "ab" → 2 hits, not infinite.
    const words = [W("abab", 0, 1)];
    const { words: out, replacements } = replaceInSegmentWords(words, "ab", "X");
    expect(replacements).toBe(2);
    expect(out[0].text).toBe("XX");
  });
});

describe("rebuildSegmentText", () => {
  it("joins word texts with single spaces", () => {
    const words = [W("hello", 0, 1), W("world", 1, 2)];
    expect(rebuildSegmentText(words)).toBe("hello world");
  });
  it("collapses inner whitespace from multi-token replacements", () => {
    // After a multi-word replace the synthetic word may legitimately
    // contain spaces (the user typed a phrase). rebuildSegmentText
    // collapses runs of whitespace so seg.text stays readable.
    const words = [W("hi  there", 0, 1)];
    expect(rebuildSegmentText(words)).toBe("hi there");
  });
  it("returns empty string on empty/null input", () => {
    expect(rebuildSegmentText([])).toBe("");
    expect(rebuildSegmentText(null)).toBe("");
  });
});
