// Tests for upload-page formatters from helpers.mjs.

import { describe, it, expect } from "vitest";
import {
  fmtBytes,
  fmtDuration,
  fmtBitrate,
  fmtRate,
  fmtFps,
  escapeHtml,
} from "../../scribe/static/js/helpers.mjs";

describe("fmtBytes", () => {
  it.each([
    [0, "0 B"],
    [10, "10 B"],
    [1024, "1.00 KB"],
    [1536, "1.50 KB"],
    [10 * 1024, "10.0 KB"],
    [100 * 1024, "100 KB"],
    [1024 * 1024, "1.00 MB"],
    [365456121, "349 MB"],
    [1024 ** 3, "1.00 GB"],
    [1024 ** 4, "1.00 TB"],
  ])("%d → %s", (n, expected) => {
    expect(fmtBytes(n)).toBe(expected);
  });

  it("returns dash for invalid input", () => {
    expect(fmtBytes(NaN)).toBe("—");
    expect(fmtBytes(-1)).toBe("—");
    expect(fmtBytes(Infinity)).toBe("—");
  });
});

describe("fmtDuration", () => {
  // No zero-pad for the minutes column when there's no hour (under 1 hour).
  // Hour column has no zero-pad either, but minutes/seconds within an hour
  // are zero-padded.
  it.each([
    [60, "1:00"],
    [125, "2:05"],
    [3599, "59:59"],
    [3600, "1:00:00"],
    [3661, "1:01:01"],
    [36000, "10:00:00"],
  ])("%d → %s", (s, expected) => {
    expect(fmtDuration(s)).toBe(expected);
  });

  it("dash for non-positive or invalid", () => {
    expect(fmtDuration(0)).toBe("—");
    expect(fmtDuration(-5)).toBe("—");
    expect(fmtDuration(NaN)).toBe("—");
  });
});

describe("fmtBitrate", () => {
  it.each([
    [128_000, "128 kbps"],
    [192_000, "192 kbps"],
    [999_000, "999 kbps"],
    [1_000_000, "1.0 Mbps"],
    [1_500_000, "1.5 Mbps"],
    [128_500, "129 kbps"],  // rounds to nearest kbps below 1 Mbps
  ])("%d → %s", (n, expected) => {
    expect(fmtBitrate(n)).toBe(expected);
  });

  it("dash for invalid", () => {
    expect(fmtBitrate(0)).toBe("—");
    expect(fmtBitrate(NaN)).toBe("—");
    expect(fmtBitrate(-1)).toBe("—");
  });
});

describe("fmtRate", () => {
  it.each([
    [16_000, "16.0 kHz"],
    [44_100, "44.1 kHz"],
    [48_000, "48.0 kHz"],
    [800, "800 Hz"],
    [999, "999 Hz"],
  ])("%d → %s", (n, expected) => {
    expect(fmtRate(n)).toBe(expected);
  });
});

describe("fmtFps", () => {
  it.each([
    [24, "24.00 fps"],
    [29.97, "29.97 fps"],
    [60, "60.00 fps"],
  ])("%d → %s", (n, expected) => {
    expect(fmtFps(n)).toBe(expected);
  });

  it("dash for non-positive", () => {
    expect(fmtFps(0)).toBe("—");
    expect(fmtFps(-1)).toBe("—");
    expect(fmtFps(NaN)).toBe("—");
  });
});

describe("escapeHtml", () => {
  it("escapes the canonical five characters", () => {
    expect(escapeHtml('<script>alert("xss")</script>'))
      .toBe("&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;");
  });

  it("escapes ampersands", () => {
    expect(escapeHtml("Tom & Jerry")).toBe("Tom &amp; Jerry");
  });

  it("escapes single quotes", () => {
    expect(escapeHtml("don't")).toBe("don&#39;t");
  });

  it("handles null/undefined as empty string", () => {
    expect(escapeHtml(null)).toBe("");
    expect(escapeHtml(undefined)).toBe("");
  });

  it("converts non-strings via String()", () => {
    expect(escapeHtml(42)).toBe("42");
    expect(escapeHtml(true)).toBe("true");
  });

  it("leaves safe text alone", () => {
    expect(escapeHtml("hello world")).toBe("hello world");
  });
});
