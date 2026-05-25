// Tests for the active-GPU-backend helpers used by the Recording details
// card on the upload page (G1.4).

import { describe, it, expect } from "vitest";
import {
  formatBackendLabel,
  backendStatTile,
} from "../../scribe/static/js/helpers.mjs";

describe("formatBackendLabel", () => {
  it.each([
    ["cuda", "CUDA"],
    ["rocm", "ROCm"],
    ["mps",  "MPS"],
    ["cpu",  "CPU"],
  ])("%s → %s", (backend, expected) => {
    expect(formatBackendLabel(backend)).toBe(expected);
  });

  it("is case-insensitive on input", () => {
    expect(formatBackendLabel("CUDA")).toBe("CUDA");
    expect(formatBackendLabel("RoCm")).toBe("ROCm");
    expect(formatBackendLabel(" Mps ")).toBe("MPS");
  });

  it("falls back to CPU on unknown / empty / null", () => {
    expect(formatBackendLabel(null)).toBe("CPU");
    expect(formatBackendLabel(undefined)).toBe("CPU");
    expect(formatBackendLabel("")).toBe("CPU");
    expect(formatBackendLabel("xpu")).toBe("CPU");
  });
});

describe("backendStatTile", () => {
  it("builds a CUDA tile with device + VRAM sub-line", () => {
    const tile = backendStatTile({
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
    });
    expect(tile).toEqual({
      label: "Backend",
      value: "CUDA",
      sub: "NVIDIA GeForce RTX 4090 · 24 GB VRAM",
    });
  });

  it("builds a ROCm tile with proper case (not ROCM)", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
    });
    expect(tile.value).toBe("ROCm");
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
  });

  it("builds a CPU tile with no sub-line when no device / VRAM", () => {
    const tile = backendStatTile({ backend: "cpu" });
    expect(tile).toEqual({
      label: "Backend",
      value: "CPU",
      sub: null,
    });
  });

  it("builds an MPS tile (Apple Silicon — no VRAM number reported)", () => {
    const tile = backendStatTile({
      backend: "mps",
      device_name: "Apple M2 Max",
      vram_gb: null,
    });
    expect(tile).toEqual({
      label: "Backend",
      value: "MPS",
      sub: "Apple M2 Max",
    });
  });

  it("omits VRAM when zero or non-finite", () => {
    expect(backendStatTile({ backend: "cuda", device_name: "X", vram_gb: 0 }).sub)
      .toBe("X");
    expect(backendStatTile({ backend: "cuda", device_name: "X", vram_gb: NaN }).sub)
      .toBe("X");
    expect(backendStatTile({ backend: "cuda", device_name: "X", vram_gb: Infinity }).sub)
      .toBe("X");
  });

  it("omits device name when missing", () => {
    expect(backendStatTile({ backend: "cuda", vram_gb: 8 }).sub).toBe("8 GB VRAM");
  });

  it("returns null when gpu payload is missing", () => {
    expect(backendStatTile(null)).toBeNull();
    expect(backendStatTile(undefined)).toBeNull();
    expect(backendStatTile("not-an-object")).toBeNull();
  });

  it("falls back to CPU label for unknown backend strings", () => {
    expect(backendStatTile({ backend: "tpu" }).value).toBe("CPU");
  });
});
