// Tests for the JS playback helpers (F4.6).
//
// Mirrors the Python tests in tests/test_application_playback.py —
// the playback functions in scribe/static/js/helpers.mjs MUST agree
// with scribe.application_playback on every shared input.

import { describe, it, expect } from "vitest";
import {
  buildWordTimeMap,
  playbackRangeForApplication,
  playbackRangesForApplications,
} from "../../scribe/static/js/helpers.mjs";

// --------------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------------- //

const HEX_PROJECT = "0".repeat(12);
const HEX_CODE = "a".repeat(12);
const HEX_SOURCE_1 = "1".repeat(12);
const HEX_SOURCE_2 = "2".repeat(12);
const HEX_CODER = "d".repeat(12);
const HEX_VERSION = "e".repeat(12);

function hexId(seed) {
  return seed.toString(16).padStart(12, "0");
}

function word(text, start, end, speaker = "S0") {
  return { text, start, end, speaker };
}

function seg(words, start = null, end = null, speaker = "S0") {
  if (start == null && words.length && words[0].start != null) start = words[0].start;
  if (end == null && words.length && words[words.length - 1].end != null) {
    end = words[words.length - 1].end;
  }
  return { speaker, start, end, words };
}

function app({
  id = hexId(1),
  sourceId = HEX_SOURCE_1,
  startId = "s0w0",
  endId = "s0w2",
  startCharOffset = null,
  endCharOffset = null,
} = {}) {
  return {
    id,
    projectId: HEX_PROJECT,
    codeId: HEX_CODE,
    sourceId,
    coderId: HEX_CODER,
    anchorStartWordId: startId,
    anchorEndWordId: endId,
    definitionVersionIdAtApply: HEX_VERSION,
    startCharOffset,
    endCharOffset,
  };
}

function basicTranscript() {
  return [
    seg(
      [
        word("Hello,", 0.0, 0.5),
        word("world.", 0.5, 1.0),
        word("How", 1.2, 1.4),
        word("are", 1.4, 1.6),
        word("you?", 1.6, 2.0),
      ],
      null, null, "A",
    ),
    seg(
      [
        word("I'm", 3.0, 3.2),
        word("fine,", 3.2, 3.5),
        word("thanks.", 3.5, 4.0),
      ],
      null, null, "B",
    ),
  ];
}

// --------------------------------------------------------------------------- //
// buildWordTimeMap
// --------------------------------------------------------------------------- //

describe("buildWordTimeMap", () => {
  it("flattens a simple transcript", () => {
    const m = buildWordTimeMap(basicTranscript());
    expect(m["s0w0"]).toEqual({ start: 0.0, end: 0.5, text: "Hello," });
    expect(m["s0w4"]).toEqual({ start: 1.6, end: 2.0, text: "you?" });
    expect(m["s1w0"]).toEqual({ start: 3.0, end: 3.2, text: "I'm" });
    expect(m["s1w2"]).toEqual({ start: 3.5, end: 4.0, text: "thanks." });
  });

  it("preserves document order", () => {
    const keys = Object.keys(buildWordTimeMap(basicTranscript()));
    expect(keys).toEqual([
      "s0w0", "s0w1", "s0w2", "s0w3", "s0w4",
      "s1w0", "s1w1", "s1w2",
    ]);
  });

  it("returns empty map for empty segments", () => {
    expect(buildWordTimeMap([])).toEqual({});
  });

  it("skips words missing timing", () => {
    const segs = [
      seg([
        word("first", 0.0, 0.5),
        word("untimed", null, null),
        word("third", 1.0, 1.2),
      ]),
    ];
    const m = buildWordTimeMap(segs);
    expect("s0w0" in m).toBe(true);
    expect("s0w1" in m).toBe(false);
    expect("s0w2" in m).toBe(true);
  });

  it("skips words with partial timing", () => {
    const segs = [
      seg([
        word("a", 0.0, null),
        word("b", null, 1.0),
        word("c", 1.0, 2.0),
      ]),
    ];
    expect(Object.keys(buildWordTimeMap(segs))).toEqual(["s0w2"]);
  });

  it("skips reversed timing", () => {
    const segs = [seg([word("backwards", 1.0, 0.5)])];
    expect(buildWordTimeMap(segs)).toEqual({});
  });

  it("skips NaN / Infinity", () => {
    const segs = [
      seg(
        [
          word("a", NaN, 0.5),
          word("b", 0.0, Infinity),
          word("c", 0.0, 0.5),
        ],
        0.0, 0.5,
      ),
    ];
    expect(Object.keys(buildWordTimeMap(segs))).toEqual(["s0w2"]);
  });

  it("skips bool timing", () => {
    const segs = [
      { speaker: "A", start: 0.0, end: 1.0, words: [{ text: "x", start: true, end: 0.5 }] },
    ];
    expect(buildWordTimeMap(segs)).toEqual({});
  });

  it("skips non-object words", () => {
    const segs = [
      { speaker: "A", start: 0.0, end: 1.0, words: ["raw-string", word("ok", 0.0, 0.5)] },
    ];
    const m = buildWordTimeMap(segs);
    expect("s0w0" in m).toBe(false);
    expect("s0w1" in m).toBe(true);
  });

  it("skips segments without a words array", () => {
    const segs = [
      { speaker: "A", start: 0.0, end: 1.0 },
      seg([word("ok", 1.0, 2.0)]),
    ];
    expect(Object.keys(buildWordTimeMap(segs))).toEqual(["s1w0"]);
  });

  it("throws on non-array segments", () => {
    expect(() => buildWordTimeMap(null)).toThrow();
    expect(() => buildWordTimeMap("not a list")).toThrow();
  });

  it("coerces non-string text to empty string", () => {
    const segs = [seg([{ text: null, start: 0.0, end: 0.5 }])];
    const m = buildWordTimeMap(segs);
    expect(m["s0w0"].text).toBe("");
  });
});

// --------------------------------------------------------------------------- //
// playbackRangeForApplication — whole-word
// --------------------------------------------------------------------------- //

describe("playbackRangeForApplication — whole-word", () => {
  it("single-word anchor", () => {
    const segs = basicTranscript();
    const a = app({ startId: "s0w0", endId: "s0w0" });
    const r = playbackRangeForApplication(a, segs);
    expect(r).not.toBeNull();
    expect(r.applicationId).toBe(a.id);
    expect(r.sourceId).toBe(HEX_SOURCE_1);
    expect(r.start).toBeCloseTo(0.0, 9);
    expect(r.end).toBeCloseTo(0.5, 9);
  });

  it("multi-word within segment", () => {
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w2" }),
      basicTranscript(),
    );
    expect(r.start).toBeCloseTo(0.0, 9);
    expect(r.end).toBeCloseTo(1.4, 9);
  });

  it("crosses segments", () => {
    const r = playbackRangeForApplication(
      app({ startId: "s0w3", endId: "s1w1" }),
      basicTranscript(),
    );
    expect(r.start).toBeCloseTo(1.4, 9);
    expect(r.end).toBeCloseTo(3.5, 9);
  });

  it("returns null when no timing anywhere", () => {
    const segs = [
      {
        speaker: null,
        words: [word("a", null, null), word("b", null, null)],
      },
    ];
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w1" }),
      segs,
    );
    expect(r).toBeNull();
  });

  it("falls back to segment start when start word untimed", () => {
    const segs = [
      seg(
        [
          word("a", null, null),
          word("b", 0.5, 1.0),
          word("c", 1.0, 1.5),
        ],
        0.1, 1.5,
      ),
    ];
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w2" }),
      segs,
    );
    expect(r.start).toBeCloseTo(0.1, 9);
    expect(r.end).toBeCloseTo(1.5, 9);
  });

  it("falls back to segment end when last word untimed", () => {
    const segs = [
      seg(
        [
          word("a", 0.0, 0.5),
          word("b", null, null),
        ],
        0.0, 2.0,
      ),
    ];
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w1" }),
      segs,
    );
    expect(r.start).toBeCloseTo(0.0, 9);
    expect(r.end).toBeCloseTo(2.0, 9);
  });

  it("clamps end >= start when transcript is weird", () => {
    const segs = [
      seg(
        [
          word("a", 5.0, 6.0),
          word("b", 1.0, 2.0),
        ],
        1.0, 6.0,
      ),
    ];
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w1" }),
      segs,
    );
    expect(r.start).toBeCloseTo(5.0, 9);
    expect(r.end).toBeGreaterThanOrEqual(r.start);
  });
});

// --------------------------------------------------------------------------- //
// playbackRangeForApplication — sub-word
// --------------------------------------------------------------------------- //

describe("playbackRangeForApplication — sub-word", () => {
  it("interpolates start_char_offset proportionally", () => {
    // "Hello," 6 chars, [0.0, 0.5], offset 3 → 0.25.
    const r = playbackRangeForApplication(
      app({
        startId: "s0w0",
        endId: "s0w0",
        startCharOffset: 3,
        endCharOffset: 6,
      }),
      basicTranscript(),
    );
    expect(r.start).toBeCloseTo(0.25, 9);
    expect(r.end).toBeCloseTo(0.5, 9);
  });

  it("interpolates end_char_offset proportionally", () => {
    // "world." 6 chars, [0.5, 1.0], end_offset 3 → 0.75.
    const r = playbackRangeForApplication(
      app({
        startId: "s0w1",
        endId: "s0w1",
        startCharOffset: 0,
        endCharOffset: 3,
      }),
      basicTranscript(),
    );
    expect(r.start).toBeCloseTo(0.5, 9);
    expect(r.end).toBeCloseTo(0.75, 9);
  });

  it("clamps offset to word bounds", () => {
    const r = playbackRangeForApplication(
      app({
        startId: "s0w0",
        endId: "s0w0",
        startCharOffset: 0,
        endCharOffset: 200,
      }),
      basicTranscript(),
    );
    expect(r.start).toBeCloseTo(0.0, 9);
    expect(r.end).toBeCloseTo(0.5, 9); // clamped
  });

  it("zero-length text falls back to word bounds", () => {
    const segs = [seg([{ text: "", start: 1.0, end: 2.0 }])];
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w0" }),
      segs,
    );
    expect(r.start).toBeCloseTo(1.0, 9);
    expect(r.end).toBeCloseTo(2.0, 9);
  });
});

// --------------------------------------------------------------------------- //
// playbackRangeForApplication — validation
// --------------------------------------------------------------------------- //

describe("playbackRangeForApplication — validation", () => {
  it("throws when start anchor segment is out of range", () => {
    expect(() =>
      playbackRangeForApplication(
        app({ startId: "s5w0", endId: "s5w1" }),
        basicTranscript(),
      ),
    ).toThrow();
  });

  it("throws when end anchor segment is out of range", () => {
    expect(() =>
      playbackRangeForApplication(
        app({ startId: "s0w0", endId: "s9w0" }),
        basicTranscript(),
      ),
    ).toThrow();
  });

  it("throws when anchor word ids are malformed", () => {
    expect(() =>
      playbackRangeForApplication(
        { ...app(), anchorStartWordId: "garbage" },
        basicTranscript(),
      ),
    ).toThrow();
  });

  it("uses caller-supplied wordTimeMap", () => {
    const wmap = {
      "s0w0": { start: 10.0, end: 11.0, text: "Hello," },
      "s0w2": { start: 12.0, end: 13.0, text: "How" },
    };
    const r = playbackRangeForApplication(
      app({ startId: "s0w0", endId: "s0w2" }),
      basicTranscript(),
      wmap,
    );
    expect(r.start).toBeCloseTo(10.0, 9);
    expect(r.end).toBeCloseTo(13.0, 9);
  });

  it("throws on non-array segments", () => {
    expect(() =>
      playbackRangeForApplication(app(), null),
    ).toThrow();
  });

  it("throws on non-object app", () => {
    expect(() =>
      playbackRangeForApplication(null, basicTranscript()),
    ).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// playbackRangesForApplications
// --------------------------------------------------------------------------- //

describe("playbackRangesForApplications", () => {
  it("buckets per source", () => {
    const segs1 = basicTranscript();
    const segs2 = [
      seg([word("alpha", 0.0, 0.4), word("beta", 0.4, 0.8)], null, null, "X"),
    ];
    const a1 = app({
      id: hexId(1),
      sourceId: HEX_SOURCE_1,
      startId: "s0w0",
      endId: "s0w0",
    });
    const a2 = app({
      id: hexId(2),
      sourceId: HEX_SOURCE_2,
      startId: "s0w0",
      endId: "s0w1",
    });
    const out = playbackRangesForApplications(
      [a1, a2],
      { [HEX_SOURCE_1]: segs1, [HEX_SOURCE_2]: segs2 },
    );
    expect(Object.keys(out).sort()).toEqual([a1.id, a2.id].sort());
    expect(out[a1.id].sourceId).toBe(HEX_SOURCE_1);
    expect(out[a1.id].end).toBeCloseTo(0.5, 9);
    expect(out[a2.id].sourceId).toBe(HEX_SOURCE_2);
    expect(out[a2.id].end).toBeCloseTo(0.8, 9);
  });

  it("skips applications with unknown source", () => {
    const out = playbackRangesForApplications(
      [app({ sourceId: HEX_SOURCE_2 })],
      { [HEX_SOURCE_1]: basicTranscript() },
    );
    expect(out).toEqual({});
  });

  it("skips applications returning null", () => {
    const segs = [{ speaker: null, words: [word("a", null, null)] }];
    const out = playbackRangesForApplications(
      [app({ startId: "s0w0", endId: "s0w0" })],
      { [HEX_SOURCE_1]: segs },
    );
    expect(out).toEqual({});
  });

  it("handles empty input", () => {
    expect(playbackRangesForApplications([], {})).toEqual({});
  });

  it("computes ranges for multiple apps on the same source", () => {
    const segs = basicTranscript();
    const a1 = app({ id: hexId(1), startId: "s0w0", endId: "s0w0" });
    const a2 = app({ id: hexId(2), startId: "s0w2", endId: "s0w4" });
    const out = playbackRangesForApplications(
      [a1, a2],
      { [HEX_SOURCE_1]: segs },
    );
    expect(out[a1.id].start).toBeCloseTo(0.0, 9);
    expect(out[a2.id].start).toBeCloseTo(1.2, 9);
  });
});
