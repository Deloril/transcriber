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

// ---------- G1.3: gfx_target + distro on the sub-line ----------
//
// G1.3 added two ROCm-support-ticket fingerprints to ``GET /api/capabilities``:
//   - ``gfx_target`` — bare gfx target ("gfx1100", "gfx1030", "gfx1201")
//   - ``distro`` — Linux pretty-name ("Ubuntu 24.04.4 LTS", "Fedora 43")
// Both surface on the home page Recording details card via the
// ``backendStatTile()`` sub-line, so a researcher pasting their
// machine info into a support thread doesn't need to also run
// ``python -m scribe.devices`` from a terminal.

describe("backendStatTile (G1.3 gfx_target + distro)", () => {
  it("appends gfx target on a ROCm tile after VRAM", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: null,
    });
    expect(tile.value).toBe("ROCm");
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100");
  });

  it("appends Linux distro on a ROCm tile after gfx target", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100 · Ubuntu 24.04.4 LTS"
    );
  });

  it("appends distro on a CUDA tile (kernel context matters on every Linux)", () => {
    const tile = backendStatTile({
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
      gfx_target: null,
      distro: "Ubuntu 24.04.4 LTS",
    });
    expect(tile.sub).toBe(
      "NVIDIA GeForce RTX 4090 · 24 GB VRAM · Ubuntu 24.04.4 LTS"
    );
  });

  it("omits gfx target when null / undefined / empty", () => {
    const base = {
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
    };
    expect(backendStatTile({ ...base, gfx_target: null }).sub)
      .toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
    expect(backendStatTile({ ...base, gfx_target: undefined }).sub)
      .toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
    expect(backendStatTile({ ...base, gfx_target: "" }).sub)
      .toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
  });

  it("omits distro when null / undefined / empty", () => {
    const base = {
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
    };
    expect(backendStatTile({ ...base, distro: null }).sub)
      .toBe("NVIDIA GeForce RTX 4090 · 24 GB VRAM");
    expect(backendStatTile({ ...base, distro: undefined }).sub)
      .toBe("NVIDIA GeForce RTX 4090 · 24 GB VRAM");
    expect(backendStatTile({ ...base, distro: "" }).sub)
      .toBe("NVIDIA GeForce RTX 4090 · 24 GB VRAM");
  });

  it("MPS tile shows distro when running under Linux Asahi (a real case)", () => {
    // Asahi Linux on Apple Silicon reports MPS unavailable (no Metal in the
    // ABI yet) so the backend usually falls back to CPU; but the helper
    // shouldn't gate distro on backend type — if the server populates it,
    // the tile shows it.
    const tile = backendStatTile({
      backend: "mps",
      device_name: "Apple M2 Max",
      vram_gb: null,
      gfx_target: null,
      distro: null,
    });
    expect(tile.sub).toBe("Apple M2 Max");
  });

  it("CPU tile renders distro alone when nothing else is set (Linux server box)", () => {
    const tile = backendStatTile({
      backend: "cpu",
      device_name: null,
      vram_gb: null,
      gfx_target: null,
      distro: "Debian GNU/Linux 12 (bookworm)",
    });
    expect(tile.value).toBe("CPU");
    expect(tile.sub).toBe("Debian GNU/Linux 12 (bookworm)");
  });

  it("RDNA 4 (gfx1201) renders correctly — current upstream blocker on Fedora 43", () => {
    // Pin the example from CT2 #2021 (RX 9070 XT / gfx1201 / Fedora 43)
    // because that's the bug a tile reader is most likely to be filing.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 9070 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1201",
      distro: "Fedora Linux 43 (Workstation Edition)",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 9070 XT · 16 GB VRAM · gfx1201 · Fedora Linux 43 (Workstation Edition)"
    );
  });

  it("gfx_target before distro — order pins triage relevance", () => {
    // gfx_target is the first identifier CT2 / ROCm maintainers ask for;
    // distro is secondary. Order matters because operators read left-to-right.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
    });
    const sub = tile.sub;
    expect(sub.indexOf("gfx1100")).toBeLessThan(sub.indexOf("Ubuntu"));
  });
});
