// Tests for library-view (F10.1) helpers in helpers.mjs.
//
// These run server-side filtering / sorting / formatting logic in the
// browser without round-tripping the API; the tests here cover the
// same edge cases the Python summariser does, plus the formatting
// helpers that the table renderer uses.

import { describe, it, expect } from "vitest";
import {
  matchesLibraryQuery,
  searchLibraryRows,
  compareLibraryRows,
  formatLibraryDate,
  formatLibrarySpeakers,
} from "../../scribe/static/js/helpers.mjs";

const ROW = (overrides = {}) => ({
  id: "abc123def456",
  input_filename: "Interview.wav",
  mode: "diarize",
  language: "en",
  model: "large-v3",
  status: "done",
  speakers: ["Luke", "Maria"],
  speaker_count: 2,
  duration_seconds: 60,
  created_at: "2026-05-25T10:00:00Z",
  has_outputs: true,
  ...overrides,
});

describe("matchesLibraryQuery", () => {
  it("matches everything for empty query", () => {
    expect(matchesLibraryQuery(ROW(), "")).toBe(true);
    expect(matchesLibraryQuery(ROW(), "   ")).toBe(true);
    expect(matchesLibraryQuery(ROW(), null)).toBe(true);
    expect(matchesLibraryQuery(ROW(), undefined)).toBe(true);
  });

  it("matches the filename", () => {
    expect(matchesLibraryQuery(ROW(), "interview")).toBe(true);
    expect(matchesLibraryQuery(ROW(), "INTERVIEW")).toBe(true);
    expect(matchesLibraryQuery(ROW(), "view")).toBe(true);
  });

  it("matches a speaker name", () => {
    expect(matchesLibraryQuery(ROW(), "luke")).toBe(true);
    expect(matchesLibraryQuery(ROW(), "Maria")).toBe(true);
  });

  it("matches the mode and status", () => {
    expect(matchesLibraryQuery(ROW(), "diarize")).toBe(true);
    expect(matchesLibraryQuery(ROW(), "done")).toBe(true);
    expect(matchesLibraryQuery(ROW({ status: "error" }), "error")).toBe(true);
  });

  it("matches the language and model", () => {
    expect(matchesLibraryQuery(ROW(), "en")).toBe(true);
    expect(matchesLibraryQuery(ROW(), "large-v3")).toBe(true);
  });

  it("returns false on no match", () => {
    expect(matchesLibraryQuery(ROW(), "zzz-nope")).toBe(false);
    expect(matchesLibraryQuery(ROW(), "queued")).toBe(false);
  });

  it("handles missing fields without throwing", () => {
    const sparse = { id: "abc", input_filename: null };
    expect(() => matchesLibraryQuery(sparse, "anything")).not.toThrow();
    expect(matchesLibraryQuery(sparse, "abc")).toBe(false);  // id isn't in haystack
  });

  it("handles non-array speakers without throwing", () => {
    const sparse = { id: "abc", input_filename: "x.wav", speakers: undefined };
    expect(matchesLibraryQuery(sparse, "x.wav")).toBe(true);
  });
});

describe("searchLibraryRows", () => {
  it("preserves input order when filtering", () => {
    const rows = [
      ROW({ id: "aaaaaaaaaaaa", input_filename: "zzz.wav" }),
      ROW({ id: "bbbbbbbbbbbb", input_filename: "aaa-zzz.wav" }),
    ];
    const out = searchLibraryRows(rows, "zzz");
    expect(out.map(r => r.id)).toEqual(["aaaaaaaaaaaa", "bbbbbbbbbbbb"]);
  });

  it("returns all rows for empty query", () => {
    const rows = [ROW({ id: "a" }), ROW({ id: "b" })];
    expect(searchLibraryRows(rows, "")).toEqual(rows);
  });

  it("returns [] for null/undefined input", () => {
    expect(searchLibraryRows(null, "x")).toEqual([]);
    expect(searchLibraryRows(undefined, "x")).toEqual([]);
  });
});

describe("compareLibraryRows", () => {
  it("sorts strings case-insensitively", () => {
    const a = ROW({ id: "1", input_filename: "Beta.wav" });
    const b = ROW({ id: "2", input_filename: "alpha.wav" });
    const asc = [a, b].slice().sort((x, y) => compareLibraryRows(x, y, "input_filename", "asc"));
    expect(asc.map(r => r.input_filename)).toEqual(["alpha.wav", "Beta.wav"]);
    const desc = [a, b].slice().sort((x, y) => compareLibraryRows(x, y, "input_filename", "desc"));
    expect(desc.map(r => r.input_filename)).toEqual(["Beta.wav", "alpha.wav"]);
  });

  it("sorts numbers numerically", () => {
    // String comparison would put "10" before "2"; numeric comparison
    // must do the right thing.
    const rows = [
      ROW({ id: "1", duration_seconds: 2 }),
      ROW({ id: "2", duration_seconds: 10 }),
      ROW({ id: "3", duration_seconds: 60 }),
    ];
    const out = rows.slice().sort((a, b) => compareLibraryRows(a, b, "duration_seconds", "asc"));
    expect(out.map(r => r.duration_seconds)).toEqual([2, 10, 60]);
  });

  it("nulls and missing always sink to the bottom regardless of direction", () => {
    const rows = [
      ROW({ id: "1", duration_seconds: null }),
      ROW({ id: "2", duration_seconds: 10 }),
      ROW({ id: "3", duration_seconds: 5 }),
    ];
    const asc = rows.slice().sort((a, b) => compareLibraryRows(a, b, "duration_seconds", "asc"));
    expect(asc.map(r => r.id)).toEqual(["3", "2", "1"]);
    const desc = rows.slice().sort((a, b) => compareLibraryRows(a, b, "duration_seconds", "desc"));
    expect(desc.map(r => r.id)).toEqual(["2", "3", "1"]);
  });

  it("breaks ties on id deterministically", () => {
    // Two rows with identical sort key — output order must match
    // the id ordering, not the input order.
    const a = ROW({ id: "bbb", duration_seconds: 10 });
    const b = ROW({ id: "aaa", duration_seconds: 10 });
    const out = [a, b].slice().sort((x, y) => compareLibraryRows(x, y, "duration_seconds", "asc"));
    expect(out.map(r => r.id)).toEqual(["aaa", "bbb"]);
    // Same on desc — tie-break is *always* ascending id, which is
    // the property the UI relies on for stable sorting.
    const out2 = [a, b].slice().sort((x, y) => compareLibraryRows(x, y, "duration_seconds", "desc"));
    expect(out2.map(r => r.id)).toEqual(["aaa", "bbb"]);
  });

  it("sorts ISO dates lexicographically (== chronologically)", () => {
    const rows = [
      ROW({ id: "1", created_at: "2026-05-25T10:00:00Z" }),
      ROW({ id: "2", created_at: "2025-12-01T00:00:00Z" }),
      ROW({ id: "3", created_at: "2026-01-15T00:00:00Z" }),
    ];
    const desc = rows.slice().sort((a, b) => compareLibraryRows(a, b, "created_at", "desc"));
    expect(desc.map(r => r.id)).toEqual(["1", "3", "2"]);
  });

  it("handles unknown sort key by tie-breaking on id", () => {
    const rows = [ROW({ id: "bbb" }), ROW({ id: "aaa" })];
    const out = rows.slice().sort((a, b) => compareLibraryRows(a, b, "garbage", "asc"));
    expect(out.map(r => r.id)).toEqual(["aaa", "bbb"]);
  });

  it("treats empty strings as missing", () => {
    const rows = [
      ROW({ id: "1", language: "" }),
      ROW({ id: "2", language: "en" }),
    ];
    const asc = rows.slice().sort((a, b) => compareLibraryRows(a, b, "language", "asc"));
    expect(asc.map(r => r.id)).toEqual(["2", "1"]);
  });
});

describe("formatLibraryDate", () => {
  it("returns empty string for blank input", () => {
    expect(formatLibraryDate("")).toBe("");
    expect(formatLibraryDate(null)).toBe("");
    expect(formatLibraryDate(undefined)).toBe("");
  });

  it("returns the raw input on parse failure", () => {
    expect(formatLibraryDate("not-a-date")).toBe("not-a-date");
  });

  it("formats a valid ISO timestamp into YYYY-MM-DD HH:MM", () => {
    // Use a timestamp at midnight UTC and assert the *shape* — exact
    // hour depends on the test runner's timezone, which we can't pin
    // without faking the Date object.
    const out = formatLibraryDate("2026-05-25T10:00:00Z");
    expect(out).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
  });
});

describe("formatLibrarySpeakers", () => {
  it("returns empty string for empty / non-array input", () => {
    expect(formatLibrarySpeakers([])).toBe("");
    expect(formatLibrarySpeakers(null)).toBe("");
    expect(formatLibrarySpeakers(undefined)).toBe("");
  });

  it("joins two or fewer speakers with comma", () => {
    expect(formatLibrarySpeakers(["Luke"])).toBe("Luke");
    expect(formatLibrarySpeakers(["Luke", "Maria"])).toBe("Luke, Maria");
  });

  it("collapses three or more into +N more suffix", () => {
    expect(formatLibrarySpeakers(["A", "B", "C"])).toBe("A, B +1 more");
    expect(formatLibrarySpeakers(["A", "B", "C", "D", "E"])).toBe("A, B +3 more");
  });
});

// F10.2 — the row carries a ``media_discarded`` flag that the page
// uses to render a small icon and hide the per-row "Discard media"
// action. The flag must NOT leak into the search haystack (it isn't
// a user-meaningful keyword) and must NOT affect the sort comparator
// (sorting by media_discarded is meaningless and there's no header
// for it).
describe("library media_discarded passthrough (F10.2)", () => {
  it("media_discarded does not affect the search filter", () => {
    const rows = [
      ROW({ id: "aaaaaaaaaaaa", media_discarded: true }),
      ROW({ id: "bbbbbbbbbbbb", media_discarded: false }),
    ];
    // Searching for the literal "discarded" matches neither row, even
    // though one carries the flag — the flag isn't part of the
    // searchable text.
    expect(searchLibraryRows(rows, "discarded").length).toBe(0);
    // Empty query still returns both rows regardless of flag.
    expect(searchLibraryRows(rows, "").length).toBe(2);
  });

  it("matchesLibraryQuery ignores the media_discarded flag entirely", () => {
    const a = ROW({ media_discarded: true });
    const b = ROW({ media_discarded: false });
    // Both rows have the same searchable text, so any non-empty query
    // either matches both or neither.
    expect(matchesLibraryQuery(a, "interview")).toBe(true);
    expect(matchesLibraryQuery(b, "interview")).toBe(true);
    expect(matchesLibraryQuery(a, "true")).toBe(false);
    expect(matchesLibraryQuery(b, "false")).toBe(false);
  });

  it("compareLibraryRows is stable when media_discarded is the only difference", () => {
    // Two rows that are otherwise identical — the comparator's
    // tie-breaker on id keeps them in id-ascending order regardless
    // of which one had its media discarded.
    const a = ROW({ id: "aaaaaaaaaaaa", media_discarded: true });
    const b = ROW({ id: "bbbbbbbbbbbb", media_discarded: false });
    const sortedA = [a, b].slice().sort((x, y) =>
      compareLibraryRows(x, y, "input_filename", "asc"),
    );
    const sortedB = [b, a].slice().sort((x, y) =>
      compareLibraryRows(x, y, "input_filename", "asc"),
    );
    expect(sortedA.map(r => r.id)).toEqual(["aaaaaaaaaaaa", "bbbbbbbbbbbb"]);
    expect(sortedB.map(r => r.id)).toEqual(["aaaaaaaaaaaa", "bbbbbbbbbbbb"]);
  });
});
