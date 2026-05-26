// Vitest for the F6.7 anonymised-rules text parser in
// scribe/static/js/helpers.mjs.
//
// The project-settings page lets a researcher supply optional custom
// redaction rules alongside the participants' name → pseudonym pairs.
// `parseAnonymisedRulesText` is the JS-side translation between the
// textarea (one rule per line, ``pattern => replacement`` with an
// optional ``re:`` prefix for regex) and the JSON shape the
// /api/projects/<pid>/qdpx/anonymised endpoint accepts:
//
//   [{ pattern, replacement, regex? }]
//
// Pulling it into helpers.mjs so the parser is exercised by Vitest
// rather than only by the manual click. The same logic lives inline
// in the template (since the script tag isn't ``type="module"``); this
// suite is the canonical reference both implementations track.

import { describe, expect, it } from "vitest";

import {
  parseAnonymisedRulesText,
} from "../../scribe/static/js/helpers.mjs";


describe("parseAnonymisedRulesText", () => {
  it("returns empty rules + null error for empty input", () => {
    const out = parseAnonymisedRulesText("");
    expect(out).toEqual({ rules: [], error: null });
  });

  it("returns empty rules + null error for whitespace-only input", () => {
    const out = parseAnonymisedRulesText("   \n  \r\n   ");
    expect(out).toEqual({ rules: [], error: null });
  });

  it("treats null and undefined as empty input", () => {
    expect(parseAnonymisedRulesText(null)).toEqual({ rules: [], error: null });
    expect(parseAnonymisedRulesText(undefined)).toEqual({ rules: [], error: null });
  });

  it("rejects non-string input with a helpful error", () => {
    const out = parseAnonymisedRulesText(123);
    expect(out.rules).toEqual([]);
    expect(out.error).toMatch(/string/);
  });

  it("parses a single literal rule", () => {
    const out = parseAnonymisedRulesText("Mercy General => [hospital]");
    expect(out.error).toBeNull();
    expect(out.rules).toEqual([
      { pattern: "Mercy General", replacement: "[hospital]" },
    ]);
  });

  it("parses multiple literal rules across lines", () => {
    const out = parseAnonymisedRulesText(
      "Jane Doe => P01\nJohn Smith => P02"
    );
    expect(out.error).toBeNull();
    expect(out.rules).toEqual([
      { pattern: "Jane Doe", replacement: "P01" },
      { pattern: "John Smith", replacement: "P02" },
    ]);
  });

  it("recognises the ``re:`` prefix as a regex flag", () => {
    const out = parseAnonymisedRulesText("re:\\d{3}-\\d{4} => [phone]");
    expect(out.error).toBeNull();
    expect(out.rules).toEqual([
      { pattern: "\\d{3}-\\d{4}", replacement: "[phone]", regex: true },
    ]);
  });

  it("does not set regex flag for non-prefixed lines", () => {
    const out = parseAnonymisedRulesText("Mercy General => [hospital]");
    expect(out.rules[0].regex).toBeUndefined();
  });

  it("skips empty lines between rules without erroring", () => {
    const out = parseAnonymisedRulesText(
      "Jane Doe => P01\n\n\nJohn Smith => P02\n\n"
    );
    expect(out.error).toBeNull();
    expect(out.rules).toHaveLength(2);
  });

  it("handles \\r\\n line endings", () => {
    const out = parseAnonymisedRulesText("a => x\r\nb => y\r\n");
    expect(out.error).toBeNull();
    expect(out.rules).toEqual([
      { pattern: "a", replacement: "x" },
      { pattern: "b", replacement: "y" },
    ]);
  });

  it("returns a parse error with line number for missing =>", () => {
    const out = parseAnonymisedRulesText("Jane Doe => P01\nbroken-line");
    expect(out.rules).toEqual([]);
    expect(out.error).toContain("line 2");
    expect(out.error).toMatch(/=>/);
  });

  it("returns a parse error with line number for empty pattern", () => {
    const out = parseAnonymisedRulesText(" => only-replacement");
    expect(out.rules).toEqual([]);
    expect(out.error).toContain("line 1");
    expect(out.error).toMatch(/empty pattern/);
  });

  it("returns a parse error with line number for empty regex pattern", () => {
    const out = parseAnonymisedRulesText("re: => x");
    expect(out.rules).toEqual([]);
    expect(out.error).toContain("line 1");
    expect(out.error).toMatch(/regex/);
  });

  it("trims whitespace around pattern and replacement", () => {
    const out = parseAnonymisedRulesText("   Jane Doe   =>   P01   ");
    expect(out.error).toBeNull();
    expect(out.rules).toEqual([
      { pattern: "Jane Doe", replacement: "P01" },
    ]);
  });

  it("allows replacement to be empty (full deletion)", () => {
    const out = parseAnonymisedRulesText("Jane Doe =>");
    expect(out.error).toBeNull();
    expect(out.rules).toEqual([
      { pattern: "Jane Doe", replacement: "" },
    ]);
  });

  it("allows the literal => to appear in the replacement value", () => {
    const out = parseAnonymisedRulesText("a => x => y");
    expect(out.error).toBeNull();
    // Splits on the first =>, so x => y is the replacement.
    expect(out.rules).toEqual([
      { pattern: "a", replacement: "x => y" },
    ]);
  });
});
