// Tests for the F5.5 promote-memo-to-code helper in helpers.mjs.
//
// Mirrors tests/test_memo_promote.py (Python). Validation rules must
// agree across both sides — a payload built by buildPromoteMemoPayload
// must be acceptable to scribe/memo_promote.py without translation.

import { describe, it, expect } from "vitest";
import {
  PROMOTE_DEFAULTS,
  buildPromoteMemoPayload,
} from "../../scribe/static/js/helpers.mjs";

const CODE_ID = "a".repeat(12);
const PARENT_ID = "b".repeat(12);
const CODER_ID = "c".repeat(12);

// --------------------------------------------------------------------------- //
// Defaults / no-arg shape
// --------------------------------------------------------------------------- //

describe("buildPromoteMemoPayload — defaults", () => {
  it("returns an empty object when called with no arguments", () => {
    expect(buildPromoteMemoPayload()).toEqual({});
  });

  it("returns an empty object when called with empty options", () => {
    expect(buildPromoteMemoPayload({})).toEqual({});
  });

  it("does not invent fields the caller didn't supply", () => {
    // Server applies its own defaults; payload should be sparse so a
    // caller can opt into individual fields without inheriting our
    // assumptions.
    const out = buildPromoteMemoPayload({});
    for (const k of [
      "name",
      "definition",
      "stage",
      "status",
      "record_back_link",
      "back_link_role",
    ]) {
      expect(Object.prototype.hasOwnProperty.call(out, k)).toBe(false);
    }
  });

  it("PROMOTE_DEFAULTS expose the conventional defaults", () => {
    expect(PROMOTE_DEFAULTS.stage).toBe("initial");
    expect(PROMOTE_DEFAULTS.status).toBe("active");
    expect(PROMOTE_DEFAULTS.recordBackLink).toBe(true);
    expect(PROMOTE_DEFAULTS.backLinkRole).toBe("promoted_to");
  });
});

// --------------------------------------------------------------------------- //
// Field forwarding (snake_case wire shape)
// --------------------------------------------------------------------------- //

describe("buildPromoteMemoPayload — field forwarding", () => {
  it("forwards name as-is", () => {
    expect(buildPromoteMemoPayload({ name: "Pacing" })).toEqual({
      name: "Pacing",
    });
  });

  it("forwards definition as-is", () => {
    expect(buildPromoteMemoPayload({ definition: "A long body." })).toEqual({
      definition: "A long body.",
    });
  });

  it("translates camelCase -> snake_case for criteria fields", () => {
    expect(
      buildPromoteMemoPayload({
        inclusionCriteria: "include this",
        exclusionCriteria: "but not that",
      }),
    ).toEqual({
      inclusion_criteria: "include this",
      exclusion_criteria: "but not that",
    });
  });

  it("forwards exemplars as a list of strings", () => {
    expect(
      buildPromoteMemoPayload({ exemplars: ["a", "b", 3] }),
    ).toEqual({ exemplars: ["a", "b", "3"] });
  });

  it("forwards parent_code_id (validating shape)", () => {
    expect(
      buildPromoteMemoPayload({ parentCodeId: PARENT_ID }),
    ).toEqual({ parent_code_id: PARENT_ID });
  });

  it("forwards related_codes (translating camelCase keys)", () => {
    expect(
      buildPromoteMemoPayload({
        relatedCodes: [
          { codeId: CODE_ID, relationType: "associated" },
          { code_id: PARENT_ID, relation_type: "broader" },
        ],
      }),
    ).toEqual({
      related_codes: [
        { code_id: CODE_ID, relation_type: "associated" },
        { code_id: PARENT_ID, relation_type: "broader" },
      ],
    });
  });

  it("forwards stage and status when in vocabulary", () => {
    expect(
      buildPromoteMemoPayload({ stage: "focused", status: "draft" }),
    ).toEqual({ stage: "focused", status: "draft" });
  });

  it("forwards colour when valid", () => {
    expect(buildPromoteMemoPayload({ colour: "#abc" }).colour).toBe("#abc");
    expect(buildPromoteMemoPayload({ colour: "#0a0a0a" }).colour).toBe(
      "#0a0a0a",
    );
  });

  it("accepts an empty colour as a noop reset", () => {
    // Same convention as Code.colour — empty string clears the colour.
    expect(buildPromoteMemoPayload({ colour: "" })).toEqual({ colour: "" });
  });

  it("forwards extra_provenance when keys are non-reserved", () => {
    expect(
      buildPromoteMemoPayload({
        extraProvenance: { promoted_by: CODER_ID },
      }),
    ).toEqual({ extra_provenance: { promoted_by: CODER_ID } });
  });

  it("forwards codeId / changeNote / recordBackLink / backLinkRole", () => {
    const out = buildPromoteMemoPayload({
      codeId: CODE_ID,
      changeNote: "Promoted by Luke",
      recordBackLink: false,
      backLinkRole: "became",
    });
    expect(out).toEqual({
      code_id: CODE_ID,
      change_note: "Promoted by Luke",
      record_back_link: false,
      back_link_role: "became",
    });
  });

  it("preserves recordBackLink=true on the wire", () => {
    expect(
      buildPromoteMemoPayload({ recordBackLink: true }).record_back_link,
    ).toBe(true);
  });

  it("forwards theoreticalMemo override", () => {
    expect(
      buildPromoteMemoPayload({ theoreticalMemo: "custom" }),
    ).toEqual({ theoretical_memo: "custom" });
  });
});

// --------------------------------------------------------------------------- //
// Validation
// --------------------------------------------------------------------------- //

describe("buildPromoteMemoPayload — validation", () => {
  it("rejects a non-string name", () => {
    expect(() => buildPromoteMemoPayload({ name: 42 })).toThrow();
  });

  it("rejects a non-string definition", () => {
    expect(() => buildPromoteMemoPayload({ definition: 42 })).toThrow();
  });

  it("rejects an exemplars value that isn't an array", () => {
    expect(() =>
      buildPromoteMemoPayload({ exemplars: "not-a-list" }),
    ).toThrow();
  });

  it("rejects an unknown stage", () => {
    expect(() => buildPromoteMemoPayload({ stage: "garbage" })).toThrow();
  });

  it("rejects an unknown status", () => {
    expect(() => buildPromoteMemoPayload({ status: "garbage" })).toThrow();
  });

  it("rejects a malformed colour", () => {
    expect(() => buildPromoteMemoPayload({ colour: "red" })).toThrow();
    expect(() => buildPromoteMemoPayload({ colour: "#abcd" })).toThrow();
  });

  it("rejects a malformed parent_code_id", () => {
    expect(() =>
      buildPromoteMemoPayload({ parentCodeId: "not-hex" }),
    ).toThrow();
  });

  it("rejects related_codes entries missing code_id", () => {
    expect(() =>
      buildPromoteMemoPayload({
        relatedCodes: [{ relation_type: "associated" }],
      }),
    ).toThrow();
  });

  it("rejects related_codes entries missing relation_type", () => {
    expect(() =>
      buildPromoteMemoPayload({ relatedCodes: [{ code_id: CODE_ID }] }),
    ).toThrow();
  });

  it("rejects related_codes entries with malformed code_id", () => {
    expect(() =>
      buildPromoteMemoPayload({
        relatedCodes: [{ codeId: "nope", relationType: "associated" }],
      }),
    ).toThrow();
  });

  it("rejects extra_provenance with a reserved key 'source'", () => {
    expect(() =>
      buildPromoteMemoPayload({ extraProvenance: { source: "human" } }),
    ).toThrow();
  });

  it("rejects extra_provenance with a reserved key 'memo_id'", () => {
    expect(() =>
      buildPromoteMemoPayload({ extraProvenance: { memo_id: "x" } }),
    ).toThrow();
  });

  it("rejects extra_provenance that isn't an object", () => {
    expect(() =>
      buildPromoteMemoPayload({ extraProvenance: ["not", "obj"] }),
    ).toThrow();
  });

  it("rejects malformed code_id", () => {
    expect(() =>
      buildPromoteMemoPayload({ codeId: "not-hex" }),
    ).toThrow();
  });

  it("rejects malformed back_link_role (e.g. starts with digit)", () => {
    // Memo.LINK_ROLE_RE requires a letter start.
    expect(() =>
      buildPromoteMemoPayload({ backLinkRole: "1bad" }),
    ).toThrow();
  });

  it("treats empty back_link_role as 'do not set'", () => {
    expect(buildPromoteMemoPayload({ backLinkRole: "" })).toEqual({});
    expect(buildPromoteMemoPayload({ backLinkRole: "   " })).toEqual({});
  });
});
