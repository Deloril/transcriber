// Vitest for the F3.6 matrix-builder helpers in
// scribe/static/js/helpers.mjs:
//
//   * buildMatrixPayload — turns the form selections in queries.html
//     into the JSON body of POST /api/projects/<pid>/matrices/run.
//     Mirrors the contract the FastAPI route enforces in
//     scribe/server.py (run_project_matrix_endpoint).
//
//   * matrixToTable — projects a Matrix.to_dict() payload (server-
//     side response shape) into a 2-D array suitable for rendering
//     as an HTML table. Pure logic so the row/column shape is
//     testable without a DOM.

import { describe, expect, it } from "vitest";

import {
  buildMatrixPayload,
  matrixToTable,
} from "../../scribe/static/js/helpers.mjs";


// 12-hex fixture ids matching the scribe.codes / scribe.sources
// validators.
const PID = "0123456789ab";
const C1  = "111111111111";
const S1  = "aaaaaaaaaaaa";


describe("buildMatrixPayload", () => {
  it("requires a kind", () => {
    expect(() => buildMatrixPayload({})).toThrow(/kind/);
  });

  it("rejects an unknown kind", () => {
    expect(() => buildMatrixPayload({ kind: "bogus" })).toThrow(/unknown kind/);
  });

  it("returns the minimal payload for code-by-source", () => {
    const out = buildMatrixPayload({ kind: "code-by-source" });
    expect(out).toEqual({ kind: "code-by-source" });
  });

  it("emits scope when kind is code-by-code", () => {
    const out = buildMatrixPayload({ kind: "code-by-code", scope: "segment" });
    expect(out).toEqual({ kind: "code-by-code", scope: "segment" });
  });

  it("emits max_gap only for paragraph scope with a positive value", () => {
    const out = buildMatrixPayload({
      kind: "code-by-code", scope: "paragraph", maxGap: 2,
    });
    expect(out.scope).toBe("paragraph");
    expect(out.max_gap).toBe(2);
  });

  it("does NOT emit max_gap for non-paragraph scope", () => {
    const out = buildMatrixPayload({
      kind: "code-by-code", scope: "source", maxGap: 99,
    });
    expect(out.max_gap).toBeUndefined();
  });

  it("does NOT emit max_gap when value is zero", () => {
    const out = buildMatrixPayload({
      kind: "code-by-code", scope: "paragraph", maxGap: 0,
    });
    expect(out.max_gap).toBeUndefined();
  });

  it("ignores kind-specific fields for kinds that don't use them", () => {
    const out = buildMatrixPayload({
      kind: "code-by-source",
      scope: "segment",
      maxGap: 99,
      attributeKey: "x",
    });
    // Only kind makes it through.
    expect(out).toEqual({ kind: "code-by-source" });
  });

  it("requires attributeKey for code-by-attribute", () => {
    expect(() =>
      buildMatrixPayload({ kind: "code-by-attribute" }),
    ).toThrow(/attributeKey/);
    expect(() =>
      buildMatrixPayload({ kind: "code-by-attribute", attributeKey: "  " }),
    ).toThrow(/attributeKey/);
  });

  it("emits attribute_key + attribute_kind + include_missing for cross-tab", () => {
    const out = buildMatrixPayload({
      kind: "code-by-attribute",
      attributeKey: "setting",
      attributeKind: "source",
      includeMissing: true,
    });
    expect(out).toEqual({
      kind: "code-by-attribute",
      attribute_key: "setting",
      attribute_kind: "source",
      include_missing: true,
    });
  });

  it("trims attribute_key", () => {
    const out = buildMatrixPayload({
      kind: "code-by-attribute",
      attributeKey: "  setting  ",
    });
    expect(out.attribute_key).toBe("setting");
  });

  it("emits compact=false only when explicit", () => {
    const default_ = buildMatrixPayload({ kind: "code-by-source" });
    expect(default_.compact).toBeUndefined();
    const off = buildMatrixPayload({ kind: "code-by-source", compact: false });
    expect(off.compact).toBe(false);
  });

  it("emits the optional query payload verbatim", () => {
    const q = {
      project_id: PID,
      codes: { expr: { op: "code", code_id: C1 } },
      sources: { source_ids: [S1] },
    };
    const out = buildMatrixPayload({ kind: "code-by-source", query: q });
    expect(out.query).toEqual(q);
  });

  it("does not accept truthy non-object values for query", () => {
    const out = buildMatrixPayload({ kind: "code-by-source", query: "nope" });
    expect(out.query).toBeUndefined();
  });
});


describe("matrixToTable", () => {
  function fakeMatrix() {
    // 2 rows × 2 cols; sparse cells.
    return {
      title: "Code × Source",
      row_label: "Code",
      col_label: "Source",
      rows: ["c1", "c2"],
      cols: ["s1", "s2"],
      row_titles: { c1: "Code one", c2: "Code two" },
      col_titles: { s1: "Source A", s2: "Source B" },
      cells: [
        ["c1", "s1", 3],
        ["c2", "s2", 1],
      ],
    };
  }

  it("returns a header row with row_label + col titles", () => {
    const t = matrixToTable(fakeMatrix());
    expect(t.header).toEqual(["Code", "Source A", "Source B"]);
  });

  it("uses row_titles for the leftmost cell and 0-fills missing cells", () => {
    const t = matrixToTable(fakeMatrix());
    expect(t.body).toEqual([
      ["Code one", 3, 0],
      ["Code two", 0, 1],
    ]);
  });

  it("computes row, column, and grand totals", () => {
    const t = matrixToTable(fakeMatrix());
    expect(t.rowTotals).toEqual([3, 1]);
    expect(t.colTotals).toEqual([3, 1]);
    expect(t.grandTotal).toBe(4);
  });

  it("falls back to row/col keys when titles are missing", () => {
    const m = fakeMatrix();
    delete m.row_titles;
    delete m.col_titles;
    const t = matrixToTable(m);
    expect(t.header).toEqual(["Code", "s1", "s2"]);
    expect(t.body[0][0]).toBe("c1");
    expect(t.body[1][0]).toBe("c2");
  });

  it("produces an empty shape for null / undefined input", () => {
    expect(matrixToTable(null)).toEqual({
      header: [], body: [], rowTotals: [], colTotals: [], grandTotal: 0,
    });
    expect(matrixToTable(undefined)).toEqual({
      header: [], body: [], rowTotals: [], colTotals: [], grandTotal: 0,
    });
  });

  it("ignores malformed cell triples", () => {
    const m = fakeMatrix();
    m.cells = [
      ["c1", "s1", 7],
      "not-a-triple",
      ["c2"],            // wrong length
      ["c1", "s2", 4, 5], // wrong length
    ];
    const t = matrixToTable(m);
    expect(t.body[0]).toEqual(["Code one", 7, 0]);
    expect(t.body[1]).toEqual(["Code two", 0, 0]);
  });
});
