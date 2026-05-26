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
      // G2.1: ``warning`` is part of the tile shape now; null on the
      // happy path. Pin it explicitly so a regression that drops the
      // field fails this test loudly.
      warning: null,
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
      warning: null,
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
      warning: null,
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

// ---------- G2.1: CT2 ROCm pin + drift on the tile ----------
//
// G2.1 added three fields to ``GET /api/capabilities``:
//   - ``ct2_rocm_pin``        — pinned CT2 ROCm wheel version ("4.7.2")
//   - ``ct2_installed``       — actually-installed ctranslate2 version
//   - ``ct2_drift_message``   — human-readable warning when they disagree
// All three are null on non-ROCm backends. The pin surfaces in the
// Backend tile sub-line ("CT2 v4.7.2") so the user can spot the wheel
// version at a glance; drift surfaces via the dedicated ``warning``
// field which the home page renders as an amber banner.

describe("backendStatTile (G2.1 CT2 ROCm pin + drift)", () => {
  it("appends CT2 pin to the sub-line on ROCm after distro", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100 · Ubuntu 24.04.4 LTS · CT2 v4.7.2"
    );
    expect(tile.warning).toBeNull();
  });

  it("emits warning when CT2 has drifted (installed ≠ pin)", () => {
    const drift =
      "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2 " +
      "(run ./setup.sh --rocm to realign)";
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.6.0",
      ct2_drift_message: drift,
    });
    expect(tile.warning).toBe(drift);
    // Sub-line still shows the pin so the user can see what should be
    // installed; the warning explains what's actually installed.
    expect(tile.sub).toContain("CT2 v4.7.2");
  });

  it("emits warning when ctranslate2 isn't installed at all", () => {
    const drift =
      "ctranslate2 not found; pinned ROCm wheel is v4.7.2 " +
      "(run ./setup.sh --rocm)";
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: null,
      ct2_drift_message: drift,
    });
    expect(tile.warning).toBe(drift);
  });

  it("omits CT2 pin from sub-line on CUDA (ROCm-only fingerprint)", () => {
    const tile = backendStatTile({
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
      gfx_target: null,
      distro: "Ubuntu 24.04.4 LTS",
      // The API guarantees these are null on non-ROCm — but pin the
      // helper's behaviour explicitly so a future server bug that
      // leaks a CUDA ctranslate2 build version doesn't render
      // misleading text in the tile.
      ct2_rocm_pin: null,
      ct2_installed: null,
      ct2_drift_message: null,
    });
    expect(tile.sub).toBe(
      "NVIDIA GeForce RTX 4090 · 24 GB VRAM · Ubuntu 24.04.4 LTS"
    );
    expect(tile.sub).not.toContain("CT2");
    expect(tile.warning).toBeNull();
  });

  it("omits CT2 pin on MPS / CPU (no ctranslate2 ROCm wheel involved)", () => {
    const mps = backendStatTile({
      backend: "mps",
      device_name: "Apple M2 Max",
      vram_gb: null,
      gfx_target: null,
      distro: null,
      ct2_rocm_pin: null,
      ct2_installed: null,
      ct2_drift_message: null,
    });
    expect(mps.sub).toBe("Apple M2 Max");
    expect(mps.warning).toBeNull();
    const cpu = backendStatTile({ backend: "cpu" });
    expect(cpu.sub).toBeNull();
    expect(cpu.warning).toBeNull();
  });

  it("treats empty / missing drift message as no-warning", () => {
    // An empty string is falsy in JS but not null — the helper must
    // treat both as "no drift" so the renderer doesn't pop an empty
    // banner.
    const base = {
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
    };
    expect(backendStatTile({ ...base, ct2_drift_message: null }).warning).toBeNull();
    expect(backendStatTile({ ...base, ct2_drift_message: "" }).warning).toBeNull();
    expect(backendStatTile({ ...base, ct2_drift_message: undefined }).warning).toBeNull();
  });

  it("CT2 v-prefix matches the CLI render exactly", () => {
    // ``python -m scribe.devices`` prints ``CT2 ROCm pin:     v4.7.2``;
    // the tile uses ``CT2 v4.7.2`` so support tickets that paste the
    // tile or the CLI output use the same canonical version string.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      ct2_rocm_pin: "4.7.2",
    });
    expect(tile.sub).toContain("CT2 v4.7.2");
    expect(tile.sub).not.toContain("CT2 4.7.2"); // missing v-prefix
    expect(tile.sub).not.toContain("CTv4.7.2"); // smashed
  });

  it("RDNA 4 + Fedora + drift — chain that fires CT2 issue #2021", () => {
    // The realistic worst-case ROCm tile: RX 9070 XT on Fedora 43,
    // CT2 has drifted because pip pulled an unrelated version,
    // and there's an upstream blocker (CT2 #2021). The tile must
    // make all four pieces visible.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 9070 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1201",
      distro: "Fedora Linux 43 (Workstation Edition)",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.6.0",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2 " +
        "(run ./setup.sh --rocm to realign)",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 9070 XT · 16 GB VRAM · gfx1201 · " +
      "Fedora Linux 43 (Workstation Edition) · CT2 v4.7.2"
    );
    expect(tile.warning).toContain("v4.6.0 installed");
    expect(tile.warning).toContain("v4.7.2");
    expect(tile.warning).toContain("./setup.sh --rocm");
  });
});

// ---------- G2.2: CT2 ROCm wheel fallback-mirror count on the sub-line ----------
//
// G2.2 added ``gpu.ct2_rocm_fallback_urls`` to the /api/capabilities
// payload — the user-configured ``SCRIBE_CT2_ROCM_FALLBACK_URLS`` list
// that ``setup.sh --rocm`` walks if the primary GitHub URL is
// unreachable (corporate firewall, GitHub outage, air-gapped box). The
// home page Recording details card surfaces the count via the
// ``backendStatTile()`` sub-line so an air-gapped researcher who set
// the env var can confirm the value survived the shell plumbing
// without dropping to ``python -m scribe.devices``.
//
// Contract:
//   - non-ROCm backends: ``ct2_rocm_fallback_urls`` is null
//     (the field doesn't apply); tile sub-line stays clean.
//   - ROCm + no mirrors:  ``ct2_rocm_fallback_urls`` is []; tile
//     sub-line stays clean (no point printing "+0 mirrors").
//   - ROCm + N mirrors:   ``ct2_rocm_fallback_urls`` is [...] with
//     length N; tile sub-line gains "+N mirror" / "+N mirrors".

describe("backendStatTile (G2.2 CT2 ROCm fallback mirrors)", () => {
  it("appends '+1 mirror' to ROCm sub-line when one mirror configured", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      ct2_rocm_fallback_urls: ["https://lab-mirror.internal/ct2-rocm.zip"],
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100 · " +
      "Ubuntu 24.04.4 LTS · CT2 v4.7.2 · +1 mirror"
    );
  });

  it("appends '+N mirrors' (plural) when multiple mirrors configured", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      ct2_rocm_fallback_urls: [
        "https://internal-mirror/a.zip",
        "https://backup-mirror/b.zip",
        "https://offsite-mirror/c.zip",
      ],
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100 · " +
      "Ubuntu 24.04.4 LTS · CT2 v4.7.2 · +3 mirrors"
    );
  });

  it("omits mirror segment when ROCm payload carries empty list", () => {
    // ROCm box, no env var set → API returns []; tile sub-line stays
    // clean (no "+0 mirrors" noise).
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      ct2_rocm_pin: "4.7.2",
      ct2_rocm_fallback_urls: [],
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · CT2 v4.7.2"
    );
    // Pin: no mirror substring at all.
    expect(tile.sub).not.toContain("mirror");
  });

  it("omits mirror segment when field is null (non-ROCm backend)", () => {
    // CUDA box → API returns null; tile sub-line stays clean.
    const tile = backendStatTile({
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
      ct2_rocm_pin: null,
      ct2_rocm_fallback_urls: null,
    });
    expect(tile.sub).toBe("NVIDIA GeForce RTX 4090 · 24 GB VRAM");
    expect(tile.sub).not.toContain("mirror");
  });

  it("omits mirror segment when field is missing entirely", () => {
    // Defensive: an older API version that doesn't carry the field
    // shouldn't make the tile blow up.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
    });
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
    expect(tile.sub).not.toContain("mirror");
  });

  it("does not crash when fallback field is malformed (not an array)", () => {
    // Defensive: a proxy munging the JSON shouldn't crash the home
    // page. The Array.isArray guard collapses non-arrays to "no
    // mirrors".
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      ct2_rocm_fallback_urls: "https://only/a.zip",  // wrong shape
    });
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
  });

  it("renders mirror segment alongside drift warning (worst-case ROCm)", () => {
    // The realistic worst-case ROCm tile: drift detected AND mirrors
    // configured (admin set fallbacks because primary GitHub URL is
    // blocked, then pip drifted CT2). The tile must surface both —
    // the warning in `warning`, the count in the sub-line.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.6.0",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2 " +
        "(run ./setup.sh --rocm to realign)",
      ct2_rocm_fallback_urls: [
        "https://lab-mirror.internal/a.zip",
        "https://backup-mirror.internal/b.zip",
      ],
    });
    expect(tile.sub).toContain("+2 mirrors");
    expect(tile.warning).toContain("./setup.sh --rocm");
  });

  it("preserves the GPU tile shape (label / value / sub / warning)", () => {
    // Pin the contract: G2.2 is additive — the four-key tile shape
    // is unchanged, only the sub-line composition gains a segment.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      ct2_rocm_fallback_urls: ["https://m1/a.zip"],
    });
    expect(Object.keys(tile).sort()).toEqual(
      ["label", "sub", "value", "warning"]
    );
  });
});
