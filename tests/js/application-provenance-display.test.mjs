// Tests for the JS provenance-display helpers (F9.9).
//
// Mirrors the Python tests in tests/test_application_provenance_display.py —
// the field set, vocabularies, and rendering MUST agree with
// scribe/application_provenance_display.py on every shared input.

import { describe, it, expect } from "vitest";
import {
  AI_DECISION_LABELS,
  AI_FEATURE_LABELS,
  DEFAULT_PROVENANCE_SOURCE_LABEL,
  PROVENANCE_SOURCE_LABELS,
  buildProvenanceDisplay,
  formatProvenanceHtml,
  formatProvenanceText,
  provenanceSummaryLabel,
} from "../../scribe/static/js/helpers.mjs";

// --------------------------------------------------------------------------- //
// Fixtures
// --------------------------------------------------------------------------- //

const PROJECT_ID = "0".repeat(12);
const CODE_ID = "a".repeat(12);
const SOURCE_ID = "1".repeat(12);
const CODER_ID = "d".repeat(12);
const VERSION_ID = "e".repeat(12);
const APP_ID = "f".repeat(12);

function makeApp(overrides = {}) {
  return {
    id: APP_ID,
    project_id: PROJECT_ID,
    code_id: CODE_ID,
    source_id: SOURCE_ID,
    coder_id: CODER_ID,
    anchor_start_word_id: "s0w0",
    anchor_end_word_id: "s0w12",
    definition_version_id_at_apply: VERSION_ID,
    start_char_offset: null,
    end_char_offset: null,
    confidence: null,
    provenance: {},
    ai_provenance: null,
    note: "",
    created_at: "2026-04-15T10:00:00Z",
    modified_at: "2026-04-15T10:00:00Z",
    ...overrides,
  };
}

function makeCode(overrides = {}) {
  return {
    id: CODE_ID,
    project_id: PROJECT_ID,
    name: "Negotiating identity",
    definition: "Initial def",
    inclusion_criteria: "",
    exclusion_criteria: "",
    exemplars: [],
    related_codes: [],
    theoretical_memo: "",
    stage: "initial",
    colour: "#aabbcc",
    status: "active",
    provenance: {},
    ...overrides,
  };
}

function makeVersion(code, overrides = {}) {
  return {
    id: VERSION_ID,
    code_id: code.id,
    project_id: PROJECT_ID,
    version: 1,
    created_at: "2026-04-15T10:00:00Z",
    snapshot: { ...code },
    change_note: "",
    ...overrides,
  };
}

function makeCoder(overrides = {}) {
  return {
    id: CODER_ID,
    project_id: PROJECT_ID,
    name: "Alex",
    role: "researcher",
    email: "",
    colour: "",
    status: "active",
    notes: "",
    ...overrides,
  };
}

// --------------------------------------------------------------------------- //
// buildProvenanceDisplay — happy paths
// --------------------------------------------------------------------------- //

describe("buildProvenanceDisplay", () => {
  it("returns safe placeholders with no relations", () => {
    const d = buildProvenanceDisplay(makeApp());
    expect(d.applicationId).toBe(APP_ID);
    expect(d.codeName).toBe("(unknown)");
    expect(d.codeMissing).toBe(true);
    expect(d.coderName).toBe("(unknown)");
    expect(d.snapshotMissing).toBe(true);
    expect(d.versionNumberAtApply).toBe("");
    expect(d.provenanceSource).toBe("");
    expect(d.provenanceSourceLabel).toBe(DEFAULT_PROVENANCE_SOURCE_LABEL);
    expect(d.aiPresent).toBe(false);
    expect(d.driftedFields).toEqual([]);
    expect(d.definitionDrifted).toBe(false);
  });

  it("hydrates with code + version + coder", () => {
    const code = makeCode();
    const version = makeVersion(code, { version: 3 });
    const coder = makeCoder();
    const d = buildProvenanceDisplay(makeApp(), {
      code,
      codeVersion: version,
      coder,
    });
    expect(d.codeName).toBe("Negotiating identity");
    expect(d.codeColour).toBe("#aabbcc");
    expect(d.coderName).toBe("Alex");
    expect(d.coderRole).toBe("researcher");
    expect(d.versionNumberAtApply).toBe("v3");
    expect(d.snapshotMissing).toBe(false);
    expect(d.nameAtApply).toBe("Negotiating identity");
    expect(d.codeMissing).toBe(false);
    expect(d.definitionDrifted).toBe(false);
    expect(d.driftedFields).toEqual([]);
  });

  it("collapses single-word anchor", () => {
    const d = buildProvenanceDisplay(
      makeApp({ anchor_start_word_id: "s0w4", anchor_end_word_id: "s0w4" }),
    );
    expect(d.anchorLabel).toBe("s0w4");
  });

  it("renders range when sub-word offsets are set on a single word", () => {
    const d = buildProvenanceDisplay(
      makeApp({
        anchor_start_word_id: "s0w4",
        anchor_end_word_id: "s0w4",
        start_char_offset: 0,
        end_char_offset: 3,
      }),
    );
    expect(d.anchorLabel).toBe("s0w4–s0w4");
  });

  it("formats multi-word anchor with en-dash", () => {
    const d = buildProvenanceDisplay(
      makeApp({ anchor_start_word_id: "s0w0", anchor_end_word_id: "s1w7" }),
    );
    expect(d.anchorLabel).toBe("s0w0–s1w7");
  });
});

// --------------------------------------------------------------------------- //
// Provenance source vocabulary
// --------------------------------------------------------------------------- //

describe("provenance source labels", () => {
  for (const [src, expected] of Object.entries(PROVENANCE_SOURCE_LABELS)) {
    it(`${src} → ${expected}`, () => {
      const d = buildProvenanceDisplay(
        makeApp({ provenance: { source: src } }),
      );
      expect(d.provenanceSource).toBe(src);
      expect(d.provenanceSourceLabel).toBe(expected);
    });
  }

  it("defaults to Human-coded when source missing", () => {
    const d = buildProvenanceDisplay(makeApp());
    expect(d.provenanceSource).toBe("");
    expect(d.provenanceSourceLabel).toBe("Human-coded");
  });

  it("defaults to Human-coded when source unknown", () => {
    const d = buildProvenanceDisplay(
      makeApp({ provenance: { source: "weird" } }),
    );
    expect(d.provenanceSource).toBe("");
    expect(d.provenanceSourceLabel).toBe("Human-coded");
  });

  it("filters reserved keys from extras", () => {
    const d = buildProvenanceDisplay(
      makeApp({
        provenance: {
          source: "ai_accepted",
          model_id: "llama3.2:3b",
          embedding_model: "bge-m3",
          suggestion_id: "1".repeat(12),
          accepted_at: "2026-04-15T10:00:00Z",
          feature: "code_suggestion",
          backend: "ollama",
          custom: "kept",
        },
      }),
    );
    expect(d.extraProvenance).toEqual(["custom: kept"]);
  });

  it("sorts extra keys alphabetically", () => {
    const d = buildProvenanceDisplay(
      makeApp({
        provenance: {
          source: "imported",
          import_run: "run-2026-04-12",
          import_format: "qdpx",
        },
      }),
    );
    expect(d.extraProvenance).toEqual([
      "import_format: qdpx",
      "import_run: run-2026-04-12",
    ]);
  });
});

// --------------------------------------------------------------------------- //
// Drift detection
// --------------------------------------------------------------------------- //

describe("drift detection", () => {
  it("no drift when current matches snapshot", () => {
    const code = makeCode();
    const version = makeVersion(code);
    const d = buildProvenanceDisplay(makeApp(), {
      code,
      codeVersion: version,
    });
    expect(d.definitionDrifted).toBe(false);
    expect(d.driftedFields).toEqual([]);
  });

  it("drift detected when definition changed", () => {
    const original = makeCode({ definition: "A" });
    const version = makeVersion(original);
    const current = makeCode({ definition: "B" });
    const d = buildProvenanceDisplay(makeApp(), {
      code: current,
      codeVersion: version,
    });
    expect(d.definitionDrifted).toBe(true);
    expect(d.driftedFields).toContain("definition");
  });

  it("drift skipped without current code", () => {
    const code = makeCode();
    const version = makeVersion(code);
    const d = buildProvenanceDisplay(makeApp(), { codeVersion: version });
    expect(d.codeMissing).toBe(true);
    expect(d.definitionDrifted).toBe(false);
    expect(d.driftedFields).toEqual([]);
  });

  it("drift skipped without version snapshot", () => {
    const code = makeCode();
    const d = buildProvenanceDisplay(makeApp(), { code });
    expect(d.snapshotMissing).toBe(true);
    expect(d.definitionDrifted).toBe(false);
  });

  it("compares exemplar arrays element-wise", () => {
    const original = makeCode({ exemplars: ["a", "b"] });
    const version = makeVersion(original);
    const current = makeCode({ exemplars: ["a", "b", "c"] });
    const d = buildProvenanceDisplay(makeApp(), {
      code: current,
      codeVersion: version,
    });
    expect(d.driftedFields).toContain("exemplars");
  });
});

// --------------------------------------------------------------------------- //
// AI provenance
// --------------------------------------------------------------------------- //

describe("AI provenance", () => {
  const aip = {
    feature: "code_suggestion",
    backend: "ollama",
    generation_model: "llama3.2:3b",
    embedding_model: "bge-m3",
    suggestion_id: "1".repeat(12),
    decision: "accepted",
    decided_by_coder_id: CODER_ID,
    decided_at: "2026-04-15T10:00:00Z",
    confidence: 0.82,
    prompt_hash: "cafebabe1234",
    notes: "Span tightened by reviewer.",
  };

  it("populates ai section when present", () => {
    const d = buildProvenanceDisplay(makeApp({ ai_provenance: aip }));
    expect(d.aiPresent).toBe(true);
    expect(d.aiFeature).toBe("code_suggestion");
    expect(d.aiFeatureLabel).toBe(AI_FEATURE_LABELS.code_suggestion);
    expect(d.aiBackend).toBe("ollama");
    expect(d.aiGenerationModel).toBe("llama3.2:3b");
    expect(d.aiEmbeddingModel).toBe("bge-m3");
    expect(d.aiSuggestionId).toBe("1".repeat(12));
    expect(d.aiDecision).toBe("accepted");
    expect(d.aiDecisionLabel).toBe(AI_DECISION_LABELS.accepted);
    expect(d.aiDecidedByCoderId).toBe(CODER_ID);
    expect(d.aiDecidedByCoderName).toBe("(unknown)");
    expect(d.aiDecidedAt).toBe("2026-04-15T10:00:00Z");
    expect(d.aiConfidence).toBe("0.82");
    expect(d.aiPromptHash).toBe("cafebabe1234");
    expect(d.aiNotes).toBe("Span tightened by reviewer.");
  });

  it("hydrates decided-by coder name when supplied", () => {
    const d = buildProvenanceDisplay(makeApp({ ai_provenance: aip }), {
      decidedByCoder: makeCoder({ name: "Sam" }),
    });
    expect(d.aiDecidedByCoderName).toBe("Sam");
  });

  it("blank decided-by name when no coder id recorded", () => {
    const pending = { feature: "code_suggestion", decision: "pending" };
    const d = buildProvenanceDisplay(makeApp({ ai_provenance: pending }));
    expect(d.aiPresent).toBe(true);
    expect(d.aiDecidedByCoderId).toBe("");
    expect(d.aiDecidedByCoderName).toBe("");
  });

  it("formats AI confidence to 2 decimal places", () => {
    for (const [v, expected] of [
      [0.0, "0.00"],
      [1.0, "1.00"],
      [0.823, "0.82"],
      [0.8265, "0.83"],
      [null, ""],
    ]) {
      const d = buildProvenanceDisplay(
        makeApp({
          ai_provenance: { feature: "code_suggestion", confidence: v },
        }),
      );
      expect(d.aiConfidence).toBe(expected);
    }
  });
});

// --------------------------------------------------------------------------- //
// Confidence formatting (top-level)
// --------------------------------------------------------------------------- //

describe("application confidence formatting", () => {
  for (const [v, expected] of [
    [null, ""],
    [0.0, "0.00"],
    [1.0, "1.00"],
    [0.823, "0.82"],
    [0.8265, "0.83"],
  ]) {
    it(`${v} → "${expected}"`, () => {
      const d = buildProvenanceDisplay(makeApp({ confidence: v }));
      expect(d.confidence).toBe(expected);
    });
  }
});

// --------------------------------------------------------------------------- //
// provenanceSummaryLabel
// --------------------------------------------------------------------------- //

describe("provenanceSummaryLabel", () => {
  it("includes coder name + provenance label + date prefix", () => {
    const d = buildProvenanceDisplay(makeApp(), { coder: makeCoder() });
    const s = provenanceSummaryLabel(d);
    expect(s).toContain("Alex");
    expect(s).toContain("Human-coded");
    expect(s).toContain("2026-04-15");
  });

  it("skips unknown coder", () => {
    const d = buildProvenanceDisplay(makeApp());
    const s = provenanceSummaryLabel(d);
    expect(s).not.toContain("(unknown)");
    expect(s).toContain("Human-coded");
  });

  it("omits date when createdAt is blank", () => {
    const d = buildProvenanceDisplay(makeApp({ created_at: "" }), {
      coder: makeCoder(),
    });
    const s = provenanceSummaryLabel(d);
    expect(s).toBe("Alex · Human-coded");
  });
});

// --------------------------------------------------------------------------- //
// formatProvenanceText
// --------------------------------------------------------------------------- //

describe("formatProvenanceText", () => {
  it("renders title row + meta + anchor + coder", () => {
    const d = buildProvenanceDisplay(makeApp());
    const text = formatProvenanceText(d);
    const first = text.split("\n")[0];
    expect(first.startsWith("(unknown) (")).toBe(true);
    expect(text).toContain("Human-coded");
    expect(text).toContain(`anchor ${d.anchorLabel}`);
  });

  it("includes drift hint", () => {
    const original = makeCode({ definition: "A" });
    const version = makeVersion(original);
    const current = makeCode({ definition: "B" });
    const d = buildProvenanceDisplay(makeApp(), {
      code: current,
      codeVersion: version,
    });
    const text = formatProvenanceText(d);
    expect(text).toContain("Definition has changed since apply");
    expect(text).toContain("definition");
  });

  it("includes snapshot-missing warning", () => {
    const d = buildProvenanceDisplay(makeApp());
    const text = formatProvenanceText(d);
    expect(text).toContain("Definition snapshot at apply not found.");
  });

  it("includes AI section", () => {
    const aip = {
      feature: "code_suggestion",
      backend: "ollama",
      generation_model: "llama3.2:3b",
      decision: "accepted",
      decided_at: "2026-04-15T10:00:00Z",
      confidence: 0.91,
    };
    const d = buildProvenanceDisplay(makeApp({ ai_provenance: aip }));
    const text = formatProvenanceText(d);
    expect(text).toContain(
      "AI: Code suggestion · ollama · llama3.2:3b · Accepted",
    );
    expect(text).toContain("AI confidence 0.91");
  });

  it("includes note section", () => {
    const d = buildProvenanceDisplay(makeApp({ note: "Multi\nline" }));
    const text = formatProvenanceText(d);
    expect(text).toContain("Note:");
    expect(text).toContain("Multi");
    expect(text).toContain("line");
  });

  it("renders extra provenance keys", () => {
    const d = buildProvenanceDisplay(
      makeApp({ provenance: { source: "human", x: "y" } }),
    );
    const text = formatProvenanceText(d);
    expect(text).toContain("x: y");
  });
});

// --------------------------------------------------------------------------- //
// formatProvenanceHtml
// --------------------------------------------------------------------------- //

describe("formatProvenanceHtml", () => {
  it("returns a well-formed div", () => {
    const d = buildProvenanceDisplay(makeApp());
    const html = formatProvenanceHtml(d);
    expect(html.startsWith('<div class="provenance-display">')).toBe(true);
    expect(html.endsWith("</div>")).toBe(true);
    expect(html).toContain("Human-coded");
  });

  it("escapes user-supplied note", () => {
    const d = buildProvenanceDisplay(
      makeApp({ note: "<script>alert(1)</script>" }),
    );
    const html = formatProvenanceHtml(d);
    expect(html).not.toContain("<script>alert(1)</script>");
    expect(html).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
  });

  it("includes drift section", () => {
    const original = makeCode({ definition: "A" });
    const version = makeVersion(original);
    const current = makeCode({ definition: "B" });
    const d = buildProvenanceDisplay(makeApp(), {
      code: current,
      codeVersion: version,
    });
    const html = formatProvenanceHtml(d);
    expect(html).toContain('class="provenance-drift"');
    expect(html).toContain("definition");
  });

  it("includes AI section", () => {
    const aip = {
      feature: "code_suggestion",
      backend: "ollama",
      generation_model: "llama3.2:3b",
      decision: "accepted",
      decided_at: "2026-04-15T10:00:00Z",
      confidence: 0.91,
    };
    const d = buildProvenanceDisplay(makeApp({ ai_provenance: aip }));
    const html = formatProvenanceHtml(d);
    expect(html).toContain('class="provenance-ai"');
    expect(html).toContain("ollama");
    expect(html).toContain("llama3.2:3b");
    expect(html).toContain("Accepted");
  });

  it("includes colour swatch when colour set", () => {
    const code = makeCode({ colour: "#abc123" });
    const version = makeVersion(code);
    const d = buildProvenanceDisplay(makeApp(), {
      code,
      codeVersion: version,
    });
    const html = formatProvenanceHtml(d);
    expect(html).toContain("provenance-swatch");
    expect(html).toContain("#abc123");
  });

  it("omits swatch when colour blank", () => {
    const code = makeCode({ colour: "" });
    const version = makeVersion(code);
    const d = buildProvenanceDisplay(makeApp(), {
      code,
      codeVersion: version,
    });
    const html = formatProvenanceHtml(d);
    expect(html).not.toContain("provenance-swatch");
  });

  it("renders extra provenance as dl", () => {
    const d = buildProvenanceDisplay(
      makeApp({ provenance: { source: "human", import_run: "r1" } }),
    );
    const html = formatProvenanceHtml(d);
    expect(html).toContain('class="provenance-extra"');
    expect(html).toContain("<dt>import_run</dt><dd>r1</dd>");
  });

  it("shows snapshot-missing warning", () => {
    const d = buildProvenanceDisplay(makeApp());
    const html = formatProvenanceHtml(d);
    expect(html).toContain('class="provenance-warn"');
  });
});

// --------------------------------------------------------------------------- //
// Vocabulary completeness
// --------------------------------------------------------------------------- //

describe("vocabulary completeness", () => {
  it("AI feature labels cover the closed set", () => {
    const features = [
      "code_suggestion",
      "new_code_suggestion",
      "quote_similarity",
      "transcript_review",
      "second_coder",
      "memo_draft",
      "other",
    ];
    for (const f of features) expect(AI_FEATURE_LABELS).toHaveProperty(f);
  });

  it("AI decision labels cover the closed set", () => {
    for (const d of ["pending", "accepted", "modified", "rejected"]) {
      expect(AI_DECISION_LABELS).toHaveProperty(d);
    }
  });

  it("provenance source labels cover the application vocabulary", () => {
    for (const s of [
      "human",
      "ai_accepted",
      "ai_modified",
      "imported",
      "other",
    ]) {
      expect(PROVENANCE_SOURCE_LABELS).toHaveProperty(s);
    }
  });
});
