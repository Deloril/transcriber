// Tests for the JS snap helpers (F4.4).
//
// Mirrors the Python tests in tests/test_selection_snap.py — the
// snap functions in scribe/static/js/helpers.mjs MUST agree with
// scribe.selection_snap on every shared input.

import { describe, it, expect } from "vitest";
import {
  isSentenceFinal,
  paragraphRanges,
  sentenceRangesInSegment,
  snapToParagraph,
  snapToSentence,
  snapToWord,
} from "../../scribe/static/js/helpers.mjs";

// --------------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------------- //

function seg(words, speaker = null) {
  return { speaker, words: words.map((t) => ({ text: t })) };
}

function sel(startWordId, endWordId, startCharOffset = null, endCharOffset = null) {
  return { startWordId, endWordId, startCharOffset, endCharOffset };
}

// --------------------------------------------------------------------------- //
// isSentenceFinal
// --------------------------------------------------------------------------- //

describe("isSentenceFinal", () => {
  it("recognises the three terminators", () => {
    expect(isSentenceFinal("hello.")).toBe(true);
    expect(isSentenceFinal("really?")).toBe(true);
    expect(isSentenceFinal("stop!")).toBe(true);
  });
  it("strips closing quotes / brackets", () => {
    expect(isSentenceFinal('hello."')).toBe(true);
    expect(isSentenceFinal("(stop!)")).toBe(true);
    expect(isSentenceFinal("done.]")).toBe(true);
    expect(isSentenceFinal("quiet?'")).toBe(true);
  });
  it("treats ellipsis as final", () => {
    expect(isSentenceFinal("trailing...")).toBe(true);
  });
  it("rejects non-final words", () => {
    expect(isSentenceFinal("hello")).toBe(false);
    expect(isSentenceFinal("comma,")).toBe(false);
    expect(isSentenceFinal("dash-")).toBe(false);
    expect(isSentenceFinal("colon:")).toBe(false);
  });
  it("rejects empty / whitespace / pure punctuation", () => {
    expect(isSentenceFinal("")).toBe(false);
    expect(isSentenceFinal("   ")).toBe(false);
    expect(isSentenceFinal('"')).toBe(false);
    expect(isSentenceFinal(")")).toBe(false);
  });
  it("returns false for non-strings", () => {
    expect(isSentenceFinal(null)).toBe(false);
    expect(isSentenceFinal(undefined)).toBe(false);
    expect(isSentenceFinal(42)).toBe(false);
  });
});

// --------------------------------------------------------------------------- //
// sentenceRangesInSegment
// --------------------------------------------------------------------------- //

describe("sentenceRangesInSegment", () => {
  it("single sentence", () => {
    expect(sentenceRangesInSegment([{ text: "Hello" }, { text: "world." }]))
      .toEqual([[0, 1]]);
  });
  it("multiple sentences", () => {
    const words = [
      { text: "Hello." },
      { text: "How" },
      { text: "are" },
      { text: "you?" },
      { text: "Goodbye!" },
    ];
    expect(sentenceRangesInSegment(words)).toEqual([[0, 0], [1, 3], [4, 4]]);
  });
  it("trailing words without terminator form one final sentence", () => {
    const words = [{ text: "Hello." }, { text: "And" }, { text: "then" }];
    expect(sentenceRangesInSegment(words)).toEqual([[0, 0], [1, 2]]);
  });
  it("no terminators at all", () => {
    const words = [{ text: "no" }, { text: "punctuation" }, { text: "here" }];
    expect(sentenceRangesInSegment(words)).toEqual([[0, 2]]);
  });
  it("empty input", () => {
    expect(sentenceRangesInSegment([])).toEqual([]);
  });
  it("partition invariant", () => {
    const words = [
      { text: "a." }, { text: "b" }, { text: "c?" }, { text: "d" },
      { text: "e" }, { text: "f!" },
    ];
    const ranges = sentenceRangesInSegment(words);
    const flat = ranges.flatMap(([a, b]) => {
      const out = [];
      for (let i = a; i <= b; i++) out.push(i);
      return out;
    });
    expect(flat).toEqual([0, 1, 2, 3, 4, 5]);
  });
});

// --------------------------------------------------------------------------- //
// paragraphRanges
// --------------------------------------------------------------------------- //

describe("paragraphRanges", () => {
  it("single speaker run", () => {
    expect(paragraphRanges([
      seg(["a"], "S0"), seg(["b"], "S0"), seg(["c"], "S0"),
    ])).toEqual([[0, 2]]);
  });
  it("speaker change breaks paragraph", () => {
    expect(paragraphRanges([
      seg(["a"], "S0"),
      seg(["b"], "S1"),
      seg(["c"], "S1"),
      seg(["d"], "S0"),
    ])).toEqual([[0, 0], [1, 2], [3, 3]]);
  });
  it("missing speaker is its own paragraph", () => {
    expect(paragraphRanges([
      seg(["a"], "S0"),
      seg(["b"], null),
      seg(["c"], null),
      seg(["d"], "S0"),
    ])).toEqual([[0, 0], [1, 1], [2, 2], [3, 3]]);
  });
  it("empty input", () => {
    expect(paragraphRanges([])).toEqual([]);
  });
  it("single segment", () => {
    expect(paragraphRanges([seg(["a"], "S0")])).toEqual([[0, 0]]);
  });
});

// --------------------------------------------------------------------------- //
// snapToWord
// --------------------------------------------------------------------------- //

describe("snapToWord", () => {
  it("drops offsets", () => {
    const out = snapToWord(sel("s0w0", "s0w5", 2, 4));
    expect(out.startWordId).toBe("s0w0");
    expect(out.endWordId).toBe("s0w5");
    expect(out.startCharOffset).toBeNull();
    expect(out.endCharOffset).toBeNull();
  });
  it("idempotent + returns input when already snapped", () => {
    const s = sel("s0w0", "s0w5");
    const once = snapToWord(s);
    const twice = snapToWord(once);
    expect(once).toEqual(twice);
    expect(once).toBe(s);
  });
  it("validates word ids", () => {
    expect(() => snapToWord(sel("invalid", "s0w0"))).toThrow();
    expect(() => snapToWord(sel("s0w0", "S0W0"))).toThrow();
  });
  it("rejects non-objects", () => {
    expect(() => snapToWord(null)).toThrow();
    expect(() => snapToWord("nope")).toThrow();
  });
  it("partial offsets are dropped", () => {
    const out = snapToWord(sel("s0w0", "s0w5", 3, null));
    expect(out.startCharOffset).toBeNull();
    expect(out.endCharOffset).toBeNull();
  });
});

// --------------------------------------------------------------------------- //
// snapToSentence
// --------------------------------------------------------------------------- //

function transcript() {
  return [
    seg(["Hello.", "How", "are", "you?", "Goodbye!"], "S0"),
    seg(["I", "think", "so."], "S0"),
    seg(["And", "then", "we", "left", "and", "went", "home"], "S0"),
  ];
}

describe("snapToSentence", () => {
  it("extends to sentence within segment", () => {
    const out = snapToSentence(sel("s0w2", "s0w2"), transcript());
    expect(out.startWordId).toBe("s0w1");
    expect(out.endWordId).toBe("s0w3");
  });
  it("endpoint already at boundary stays put", () => {
    const out = snapToSentence(sel("s0w1", "s0w3"), transcript());
    expect(out.startWordId).toBe("s0w1");
    expect(out.endWordId).toBe("s0w3");
  });
  it("drops offsets", () => {
    const out = snapToSentence(sel("s0w2", "s0w2", 1, 2), transcript());
    expect(out.startCharOffset).toBeNull();
    expect(out.endCharOffset).toBeNull();
  });
  it("cross-segment", () => {
    const out = snapToSentence(sel("s0w2", "s1w1"), transcript());
    expect(out.startWordId).toBe("s0w1");
    expect(out.endWordId).toBe("s1w2");
  });
  it("unterminated segment", () => {
    const out = snapToSentence(sel("s2w3", "s2w3"), transcript());
    expect(out.startWordId).toBe("s2w0");
    expect(out.endWordId).toBe("s2w6");
  });
  it("idempotent", () => {
    const once = snapToSentence(sel("s0w2", "s0w4"), transcript());
    const twice = snapToSentence(once, transcript());
    expect(once).toEqual(twice);
  });
  it("first sentence starts at word 0", () => {
    const out = snapToSentence(sel("s0w0", "s0w0"), transcript());
    expect(out.startWordId).toBe("s0w0");
    expect(out.endWordId).toBe("s0w0");
  });
  it("rejects out-of-range word", () => {
    expect(() => snapToSentence(sel("s0w99", "s0w99"), transcript())).toThrow();
  });
  it("rejects out-of-range segment", () => {
    expect(() => snapToSentence(sel("s9w0", "s9w0"), transcript())).toThrow();
  });
  it("rejects start after end", () => {
    expect(() => snapToSentence(sel("s0w3", "s0w1"), transcript())).toThrow();
  });
  it("rejects bad segments", () => {
    expect(() => snapToSentence(sel("s0w0", "s0w0"), "not array")).toThrow();
  });
  it("rejects non-selection", () => {
    expect(() => snapToSentence(null, transcript())).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// snapToParagraph
// --------------------------------------------------------------------------- //

function paraTranscript() {
  return [
    seg(["Hello.", "How", "are"], "S0"),
    seg(["you?", "Yes."], "S0"),
    seg(["I'm", "fine."], "S1"),
    seg(["Are", "you", "sure?"], "S1"),
    seg(["OK", "good."], "S0"),
  ];
}

describe("snapToParagraph", () => {
  it("extends to paragraph within speaker run", () => {
    const out = snapToParagraph(sel("s0w1", "s0w2"), paraTranscript());
    expect(out.startWordId).toBe("s0w0");
    expect(out.endWordId).toBe("s1w1");
  });
  it("single-segment paragraph", () => {
    const out = snapToParagraph(sel("s4w0", "s4w0"), paraTranscript());
    expect(out.startWordId).toBe("s4w0");
    expect(out.endWordId).toBe("s4w1");
  });
  it("cross-paragraph", () => {
    const out = snapToParagraph(sel("s1w0", "s2w0"), paraTranscript());
    expect(out.startWordId).toBe("s0w0");
    expect(out.endWordId).toBe("s3w2");
  });
  it("drops offsets", () => {
    const out = snapToParagraph(sel("s0w1", "s0w2", 2, 3), paraTranscript());
    expect(out.startCharOffset).toBeNull();
    expect(out.endCharOffset).toBeNull();
  });
  it("missing speaker is its own paragraph", () => {
    const segs = [
      seg(["a", "b"], "S0"),
      seg(["c", "d"], null),
      seg(["e", "f"], "S0"),
    ];
    const out = snapToParagraph(sel("s1w0", "s1w0"), segs);
    expect(out.startWordId).toBe("s1w0");
    expect(out.endWordId).toBe("s1w1");
  });
  it("idempotent", () => {
    const once = snapToParagraph(sel("s0w1", "s0w2"), paraTranscript());
    const twice = snapToParagraph(once, paraTranscript());
    expect(once).toEqual(twice);
  });
  it("rejects out-of-range word", () => {
    expect(() => snapToParagraph(sel("s0w99", "s0w99"), paraTranscript())).toThrow();
  });
  it("rejects start after end", () => {
    expect(() => snapToParagraph(sel("s2w0", "s0w0"), paraTranscript())).toThrow();
  });
  it("rejects non-selection", () => {
    expect(() => snapToParagraph(null, paraTranscript())).toThrow();
  });
  it("rejects bad segments", () => {
    expect(() => snapToParagraph(sel("s0w0", "s0w0"), "not array")).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// Snap composition (parallel of TestSnapComposition in Python)
// --------------------------------------------------------------------------- //

describe("snap composition", () => {
  it("widens monotonically: word ⊆ sentence ⊆ paragraph", () => {
    const segs = [
      seg(["Hello.", "How", "are", "you?", "Goodbye!"], "S0"),
      seg(["I'm", "fine."], "S1"),
    ];
    const s = sel("s0w2", "s0w2", 1, 2);
    const w = snapToWord(s);
    const sent = snapToSentence(s, segs);
    const para = snapToParagraph(s, segs);
    expect([w.startWordId, w.endWordId]).toEqual(["s0w2", "s0w2"]);
    expect([sent.startWordId, sent.endWordId]).toEqual(["s0w1", "s0w3"]);
    expect([para.startWordId, para.endWordId]).toEqual(["s0w0", "s0w4"]);
  });
});
