// Tests for the F5.2 right-click memo helpers in helpers.mjs.
//
// Mirrors tests/test_memo_context.py — defaults must agree across
// Python and JS or the right-click flow's "default type" preselection
// would differ depending on whether the editor or the server built
// the payload.

import { describe, it, expect } from "vitest";
import {
  DEFAULT_MEMO_TYPE_BY_TARGET,
  MEMO_LINK_TARGET_TYPES,
  MEMO_TYPES,
  buildMemoContextPayload,
  buildMemoDraftPayload,
  defaultMemoTypeForTarget,
} from "../../scribe/static/js/helpers.mjs";

const PROJECT_ID = "0".repeat(12);
const CODE_ID = "a".repeat(12);
const SOURCE_ID = "b".repeat(12);
const APP_ID = "d".repeat(12);
const CODER_ID = "c".repeat(12);

// --------------------------------------------------------------------------- //
// defaultMemoTypeForTarget
// --------------------------------------------------------------------------- //

describe("defaultMemoTypeForTarget", () => {
  it("maps code → code", () => {
    expect(defaultMemoTypeForTarget("code")).toBe("code");
  });

  it("maps application → quote", () => {
    expect(defaultMemoTypeForTarget("application")).toBe("quote");
  });

  it("maps source → source", () => {
    expect(defaultMemoTypeForTarget("source")).toBe("source");
  });

  it("maps project → project", () => {
    expect(defaultMemoTypeForTarget("project")).toBe("project");
  });

  it("maps coder → methodological", () => {
    expect(defaultMemoTypeForTarget("coder")).toBe("methodological");
  });

  it("maps memo → theoretical", () => {
    expect(defaultMemoTypeForTarget("memo")).toBe("theoretical");
  });

  it("maps participant → free", () => {
    expect(defaultMemoTypeForTarget("participant")).toBe("free");
  });

  it("falls back to free for unknown targets", () => {
    expect(defaultMemoTypeForTarget("not-a-target")).toBe("free");
    expect(defaultMemoTypeForTarget("")).toBe("free");
  });

  it("throws for non-string target_type", () => {
    expect(() => defaultMemoTypeForTarget(42)).toThrow();
    expect(() => defaultMemoTypeForTarget(null)).toThrow();
  });

  it("every target type has a default", () => {
    for (const t of MEMO_LINK_TARGET_TYPES) {
      expect(DEFAULT_MEMO_TYPE_BY_TARGET[t]).toBeDefined();
      expect(MEMO_TYPES).toContain(DEFAULT_MEMO_TYPE_BY_TARGET[t]);
    }
  });
});

// --------------------------------------------------------------------------- //
// buildMemoDraftPayload
// --------------------------------------------------------------------------- //

describe("buildMemoDraftPayload", () => {
  it("returns a JSON-shaped payload with primary link first", () => {
    const p = buildMemoDraftPayload({
      targetType: "code",
      targetId: CODE_ID,
    });
    expect(p.type).toBe("code");
    expect(p.links).toHaveLength(1);
    expect(p.links[0]).toEqual({ target_type: "code", target_id: CODE_ID });
  });

  it("carries role onto the primary link when supplied", () => {
    const p = buildMemoDraftPayload({
      targetType: "application",
      targetId: APP_ID,
      role: "exemplifies",
    });
    expect(p.links[0]).toEqual({
      target_type: "application",
      target_id: APP_ID,
      role: "exemplifies",
    });
    expect(p.type).toBe("quote");
  });

  it("explicit type overrides the default", () => {
    const p = buildMemoDraftPayload({
      targetType: "code",
      targetId: CODE_ID,
      type: "theoretical",
    });
    expect(p.type).toBe("theoretical");
  });

  it("rejects unknown target_type", () => {
    expect(() =>
      buildMemoDraftPayload({ targetType: "planet", targetId: CODE_ID }),
    ).toThrow();
  });

  it("rejects bad target_id shape", () => {
    expect(() =>
      buildMemoDraftPayload({ targetType: "code", targetId: "nope" }),
    ).toThrow();
  });

  it("rejects unknown memo type", () => {
    expect(() =>
      buildMemoDraftPayload({
        targetType: "code",
        targetId: CODE_ID,
        type: "wrong",
      }),
    ).toThrow();
  });

  it("rejects bad role characters", () => {
    expect(() =>
      buildMemoDraftPayload({
        targetType: "code",
        targetId: CODE_ID,
        role: "!!nope",
      }),
    ).toThrow();
  });

  it("appends extra_links after the primary", () => {
    const p = buildMemoDraftPayload({
      targetType: "application",
      targetId: APP_ID,
      extraLinks: [
        { targetType: "code", targetId: CODE_ID, role: "applies" },
      ],
    });
    expect(p.links).toHaveLength(2);
    expect(p.links[0].target_type).toBe("application");
    expect(p.links[1]).toEqual({
      target_type: "code",
      target_id: CODE_ID,
      role: "applies",
    });
  });

  it("dedupes extra_links matching the primary triple", () => {
    const p = buildMemoDraftPayload({
      targetType: "code",
      targetId: CODE_ID,
      role: "exemplifies",
      extraLinks: [
        { targetType: "code", targetId: CODE_ID, role: "exemplifies" },
      ],
    });
    expect(p.links).toHaveLength(1);
  });

  it("keeps same target with different role as a distinct link", () => {
    const p = buildMemoDraftPayload({
      targetType: "code",
      targetId: CODE_ID,
      role: "exemplifies",
      extraLinks: [
        { targetType: "code", targetId: CODE_ID, role: "contradicts" },
      ],
    });
    expect(p.links).toHaveLength(2);
  });

  it("accepts snake_case in extraLinks too (server round-trip shape)", () => {
    const p = buildMemoDraftPayload({
      targetType: "application",
      targetId: APP_ID,
      extraLinks: [
        { target_type: "code", target_id: CODE_ID },
      ],
    });
    expect(p.links).toHaveLength(2);
    expect(p.links[1].target_type).toBe("code");
  });

  it("composer fields round-trip onto the payload", () => {
    const p = buildMemoDraftPayload({
      targetType: "application",
      targetId: APP_ID,
      title: "Note",
      body: "P3 hesitates.",
      bodyFormat: "markdown",
      authorCoderId: CODER_ID,
      tags: ["hesitation", "P3"],
      provenance: { source: "human" },
    });
    expect(p.title).toBe("Note");
    expect(p.body).toContain("P3");
    expect(p.body_format).toBe("markdown");
    expect(p.author_coder_id).toBe(CODER_ID);
    expect(p.tags).toEqual(["hesitation", "P3"]);
    expect(p.provenance).toEqual({ source: "human" });
  });

  it("omits empty-but-optional fields", () => {
    const p = buildMemoDraftPayload({
      targetType: "code",
      targetId: CODE_ID,
    });
    expect(p.author_coder_id).toBeUndefined();
    expect(p.tags).toBeUndefined();
    expect(p.provenance).toBeUndefined();
  });

  it("rejects malformed extra_links entries", () => {
    expect(() =>
      buildMemoDraftPayload({
        targetType: "code",
        targetId: CODE_ID,
        extraLinks: ["not-an-object"],
      }),
    ).toThrow();
    expect(() =>
      buildMemoDraftPayload({
        targetType: "code",
        targetId: CODE_ID,
        extraLinks: [{ target_type: "code" }], // missing id
      }),
    ).toThrow();
  });
});

// --------------------------------------------------------------------------- //
// buildMemoContextPayload — server-routed composer body
// --------------------------------------------------------------------------- //

describe("buildMemoContextPayload", () => {
  it("nests target info under the context key", () => {
    const p = buildMemoContextPayload({
      targetType: "code",
      targetId: CODE_ID,
    });
    expect(p.context).toEqual({
      target_type: "code",
      target_id: CODE_ID,
    });
    // No links field at top level — the server will derive them.
    expect(p.links).toBeUndefined();
    // No type override → server applies the default.
    expect(p.type).toBeUndefined();
  });

  it("includes role on the context when supplied", () => {
    const p = buildMemoContextPayload({
      targetType: "application",
      targetId: APP_ID,
      role: "exemplifies",
    });
    expect(p.context.role).toBe("exemplifies");
  });

  it("explicit type override appears at top level", () => {
    const p = buildMemoContextPayload({
      targetType: "code",
      targetId: CODE_ID,
      type: "theoretical",
    });
    expect(p.type).toBe("theoretical");
  });

  it("forwards extra_links and composer fields", () => {
    const p = buildMemoContextPayload({
      targetType: "application",
      targetId: APP_ID,
      title: "T",
      body: "B",
      tags: ["t1"],
      provenance: { source: "human" },
      extraLinks: [{ targetType: "source", targetId: SOURCE_ID }],
    });
    expect(p.title).toBe("T");
    expect(p.body).toBe("B");
    expect(p.tags).toEqual(["t1"]);
    expect(p.provenance).toEqual({ source: "human" });
    expect(p.extra_links).toEqual([
      { target_type: "source", target_id: SOURCE_ID },
    ]);
  });

  it("rejects bad target_type", () => {
    expect(() =>
      buildMemoContextPayload({ targetType: "planet", targetId: CODE_ID }),
    ).toThrow();
  });

  it("rejects bad target_id", () => {
    expect(() =>
      buildMemoContextPayload({ targetType: "code", targetId: "nope" }),
    ).toThrow();
  });

  it("rejects bad role characters", () => {
    expect(() =>
      buildMemoContextPayload({
        targetType: "code",
        targetId: CODE_ID,
        role: "!!bad",
      }),
    ).toThrow();
  });

  it("rejects unknown memo type", () => {
    expect(() =>
      buildMemoContextPayload({
        targetType: "code",
        targetId: CODE_ID,
        type: "wrong",
      }),
    ).toThrow();
  });
});
