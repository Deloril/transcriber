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

// ---------- G2.3: Linux distro support tier on the sub-line ----------
//
// G2.3 added ``gpu.distro_tier`` to the /api/capabilities payload —
// the AMD-official ROCm support classification for the active
// distro, one of "first-class" / "supported" / "best-effort" /
// "unknown" on ROCm; null on every non-ROCm backend. The home page
// Recording details card surfaces the tier next to the distro
// pretty-name so a researcher pasting their machine info into a
// support thread sees at a glance whether AMD officially supports
// their distro for ROCm.
//
// Contract:
//   - non-ROCm backends: ``distro_tier`` is null (the field doesn't
//     apply); tile sub-line stays clean.
//   - ROCm + classifier failed: ``distro_tier`` is null (defensive
//     fallback when /etc/os-release is missing); tile stays clean.
//   - ROCm + classified:        ``distro_tier`` is one of the four
//     tier strings; tile sub-line gains "<tier> distro" segment.
//
// Renders the tier with a trailing " distro" so the bare tier label
// alone (e.g. "first-class") doesn't read strangely in the sub-line.
// "first-class distro" is what a researcher wants to copy into a
// bug report.

describe("backendStatTile (G2.3 distro support tier)", () => {
  it("appends 'first-class distro' on a ROCm + Ubuntu LTS tile", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      ct2_rocm_fallback_urls: [],
      distro_tier: "first-class",
      distro_tier_explanation:
        "AMD officially supports this distro for ROCm; tested by Scribe",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100 · " +
      "Ubuntu 24.04.4 LTS · CT2 v4.7.2 · first-class distro"
    );
  });

  it("appends 'supported distro' on a ROCm + RHEL 9 tile", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon Pro W6800",
      vram_gb: 32.0,
      gfx_target: "gfx1030",
      distro: "Red Hat Enterprise Linux 9.7 (Plow)",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      distro_tier: "supported",
    });
    expect(tile.sub).toContain("supported distro");
    // RHEL is "supported" but not "first-class" — pin the distinction
    // so a future change can't quietly upgrade RHEL to first-class
    // without updating the matrix.
    expect(tile.sub).not.toContain("first-class");
  });

  it("appends 'best-effort distro' on a ROCm + Fedora tile", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 9070 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1201",
      distro: "Fedora Linux 41 (Workstation Edition)",
      distro_tier: "best-effort",
    });
    expect(tile.sub).toContain("best-effort distro");
  });

  it("appends 'unknown distro' when classifier couldn't identify the host", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      distro: "Some Unknown Linux 1.0",
      distro_tier: "unknown",
    });
    expect(tile.sub).toContain("unknown distro");
  });

  it("omits tier segment when distro_tier is null (non-ROCm backend)", () => {
    // CUDA box on Ubuntu 24.04 — distro is set, but tier is null
    // because the AMD matrix is irrelevant on NVIDIA.
    const tile = backendStatTile({
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
      distro: "Ubuntu 24.04.4 LTS",
      distro_tier: null,
    });
    expect(tile.sub).toBe(
      "NVIDIA GeForce RTX 4090 · 24 GB VRAM · Ubuntu 24.04.4 LTS"
    );
    expect(tile.sub).not.toContain("distro");  // the segment is omitted
  });

  it("omits tier segment when distro_tier field is missing entirely", () => {
    // Defensive: an older API version that doesn't carry the field
    // shouldn't make the tile blow up.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      distro: "Ubuntu 24.04.4 LTS",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · Ubuntu 24.04.4 LTS"
    );
    // No tier segment at all — the trailing " distro" suffix
    // shouldn't leak.
    expect(tile.sub.endsWith("distro")).toBe(false);
  });

  it("omits tier segment when distro_tier is empty string", () => {
    // Defensive: server returning an empty string for some reason
    // should not render an empty " distro" segment.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      distro_tier: "",
    });
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
  });

  it("renders tier alongside drift warning + mirror count (worst-case ROCm)", () => {
    // Realistic worst case: best-effort distro with ROCm wheel drift
    // and configured fallback mirrors. The tile must surface all
    // three pieces clearly.
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
      ct2_rocm_fallback_urls: ["https://lab-mirror.internal/a.zip"],
      distro_tier: "best-effort",
    });
    expect(tile.sub).toContain("Fedora Linux 43");
    expect(tile.sub).toContain("CT2 v4.7.2");
    expect(tile.sub).toContain("+1 mirror");
    expect(tile.sub).toContain("best-effort distro");
    expect(tile.warning).toContain("./setup.sh --rocm");
  });

  it("preserves the GPU tile shape (label / value / sub / warning)", () => {
    // Pin the contract: G2.3 is additive — the four-key tile shape
    // is unchanged, only the sub-line composition gains a segment.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      distro_tier: "first-class",
    });
    expect(Object.keys(tile).sort()).toEqual(
      ["label", "sub", "value", "warning"]
    );
  });

  it("tier appears after mirror count (last sub-line segment)", () => {
    // Order pins triage relevance: device → VRAM → gfx → distro →
    // CT2 pin → mirrors → tier. Tier is last because it's the
    // highest-level meta-classification; the more concrete
    // identifiers (gfx target, distro pretty-name) come first so a
    // copy-pasted line into a bug report front-loads the most
    // searchable strings.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_rocm_fallback_urls: ["https://m1/a.zip"],
      distro_tier: "first-class",
    });
    const segments = tile.sub.split(" · ");
    expect(segments[segments.length - 1]).toBe("first-class distro");
  });
});

// ---------- G3.1: pyannote LSTM dropout MIOpen workaround state ----------
//
// G3.1 added ``gpu.rocm_lstm_patch`` to the /api/capabilities payload —
// a boolean that is ``true`` on ROCm (the pyannote LSTM dropout patch
// will fire when diarization loads, working around pyannote-audio
// #1995) and ``null`` on every non-ROCm backend (the patch is a no-op
// outside ROCm so reporting it elsewhere would be misleading). The
// home page Recording details card surfaces the boolean as a
// "LSTM patched" sub-line segment so a researcher pasting their
// machine info into a pyannote-audio #1995 support thread can
// confirm the workaround is in their install.
//
// Contract:
//   - non-ROCm backends:   ``rocm_lstm_patch`` is null; tile sub-line
//                          omits the segment.
//   - ROCm + true:         segment renders as "LSTM patched".
//   - ROCm + false:        segment omitted (defensive — should never
//                          happen on ROCm in practice but the helper
//                          must not render "LSTM patched" if the
//                          server reports false).
//   - missing field:       segment omitted (older API version).

describe("backendStatTile (G3.1 LSTM dropout patch)", () => {
  it("appends 'LSTM patched' on a ROCm tile with patch active", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      ct2_rocm_fallback_urls: [],
      distro_tier: "first-class",
      rocm_lstm_patch: true,
      rocm_lstm_patch_explanation:
        "pyannote LSTM dropout forced to 0.0 to avoid MIOpen " +
        "missing-header bug (pyannote-audio #1995) on ROCm ≥ 6.1.1; " +
        "inference behaviour is unchanged",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100 · " +
      "Ubuntu 24.04.4 LTS · CT2 v4.7.2 · first-class distro · LSTM patched"
    );
  });

  it("omits LSTM segment when rocm_lstm_patch is null (non-ROCm backend)", () => {
    // CUDA box on Ubuntu 24.04 — patch field is null because the
    // pyannote-audio #1995 workaround is irrelevant on NVIDIA.
    const tile = backendStatTile({
      backend: "cuda",
      device_name: "NVIDIA GeForce RTX 4090",
      vram_gb: 24.0,
      distro: "Ubuntu 24.04.4 LTS",
      rocm_lstm_patch: null,
    });
    expect(tile.sub).toBe(
      "NVIDIA GeForce RTX 4090 · 24 GB VRAM · Ubuntu 24.04.4 LTS"
    );
    expect(tile.sub).not.toContain("LSTM");
  });

  it("omits LSTM segment when rocm_lstm_patch field is missing entirely", () => {
    // Defensive: an older API version that doesn't carry the field
    // shouldn't make the tile blow up.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
    });
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
    expect(tile.sub).not.toContain("LSTM");
  });

  it("omits LSTM segment when rocm_lstm_patch is false", () => {
    // Defensive: server reporting false should not render the segment.
    // In practice this can't happen on ROCm (the patch is always
    // present), but the helper must not render "LSTM patched" against
    // an explicit false.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      rocm_lstm_patch: false,
    });
    expect(tile.sub).toBe("AMD Radeon RX 7900 XTX · 24 GB VRAM");
    expect(tile.sub).not.toContain("LSTM");
  });

  it("omits LSTM segment for truthy non-true values (defensive)", () => {
    // The helper checks ``=== true`` so a stray string / number / object
    // doesn't accidentally render the segment. Belt-and-braces against
    // an external proxy munging the JSON shape.
    for (const stray of ["true", "yes", 1, {}, []]) {
      const tile = backendStatTile({
        backend: "rocm",
        device_name: "AMD Radeon RX 7900 XTX",
        rocm_lstm_patch: stray,
      });
      expect(tile.sub).not.toContain("LSTM");
    }
  });

  it("LSTM segment appears after distro tier (last sub-line segment)", () => {
    // Order pin: device → VRAM → gfx → distro → CT2 pin → mirrors →
    // tier → LSTM patch. LSTM patch is last because it's the most
    // specific-to-pyannote bit; the distro tier is the highest-level
    // meta-classification and stays as the second-to-last segment.
    // The tier was previously last (G2.3); G3.1 demotes it by one.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_rocm_fallback_urls: ["https://m1/a.zip"],
      distro_tier: "first-class",
      rocm_lstm_patch: true,
    });
    const segments = tile.sub.split(" · ");
    expect(segments[segments.length - 1]).toBe("LSTM patched");
    expect(segments[segments.length - 2]).toBe("first-class distro");
  });

  it("preserves the GPU tile shape (label / value / sub / warning)", () => {
    // Pin the contract: G3.1 is additive — the four-key tile shape
    // is unchanged, only the sub-line composition gains a segment.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      rocm_lstm_patch: true,
    });
    expect(Object.keys(tile).sort()).toEqual(
      ["label", "sub", "value", "warning"]
    );
  });

  it("renders LSTM segment alongside drift warning (worst-case ROCm)", () => {
    // Realistic worst case: RDNA 3 box with CT2 wheel drift + the
    // LSTM patch active. The tile must surface both clearly.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 9070 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1201",
      distro: "Fedora Linux 43 (Workstation Edition)",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.6.0",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2",
      ct2_rocm_fallback_urls: ["https://lab-mirror.internal/a.zip"],
      distro_tier: "best-effort",
      rocm_lstm_patch: true,
    });
    expect(tile.sub).toContain("LSTM patched");
    expect(tile.sub).toContain("CT2 v4.7.2");
    expect(tile.sub).toContain("best-effort distro");
    expect(tile.warning).toContain("v4.6.0");
  });

  it("LSTM segment is the literal string 'LSTM patched' (no localisation)", () => {
    // The string is what a researcher copy-pastes into a support
    // thread; it has to be greppable upstream. Pin the spelling so
    // a future helpers refactor can't quietly change it to
    // "LSTM dropout patched" or similar — that would break grep
    // continuity with the CLI line and the upstream issue tracker.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      rocm_lstm_patch: true,
    });
    expect(tile.sub.split(" · ")).toContain("LSTM patched");
  });
});

// ---------- G4.1: RDNA 2 cub_caching allocator workaround state ----------
//
// G4.1 added three fields to the /api/capabilities payload —
// ``gpu.rocm_allocator_state`` (one of ``"auto"`` /
// ``"user-overridden"`` / ``"unset"``), ``gpu.rocm_allocator_value``
// (the literal env var value when set), and
// ``gpu.rocm_allocator_explanation`` (one-line rationale). All three
// are ``null`` outside RDNA 2 ROCm — the cub_caching workaround only
// applies to RX 6000-series cards (and the RDNA 2 APUs); on RDNA 3 /
// CDNA / CUDA / MPS / CPU reporting the state would just be noise.
//
// The home page tile renders the state as a sub-line segment:
//   - "auto"            → "alloc cub_caching"
//   - "user-overridden" → "alloc <value>" (echoes whatever the user set)
//   - "unset"           → "alloc unset" + warning banner with the
//                         explanation (CT2 is about to crash; the
//                         banner makes this glanceable)
//   - null / missing    → segment omitted entirely
//
// When the unset state coincides with a CT2 wheel drift, both
// warnings concatenate so neither gets swallowed.

describe("backendStatTile (G4.1 cub_caching allocator)", () => {
  it("appends 'alloc cub_caching' on a ROCm RDNA 2 tile (auto state)", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1030",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      ct2_rocm_fallback_urls: [],
      distro_tier: "first-class",
      rocm_lstm_patch: true,
      rocm_lstm_patch_explanation: "...",
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
      rocm_allocator_explanation: "CT2_CUDA_ALLOCATOR=cub_caching applied automatically (CT2 #2012)",
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 6800 XT · 16 GB VRAM · gfx1030 · " +
      "Ubuntu 24.04.4 LTS · CT2 v4.7.2 · first-class distro · " +
      "LSTM patched · alloc cub_caching"
    );
    // Auto state must NOT trigger a warning — the workaround is in place.
    expect(tile.warning).toBeNull();
  });

  it("appends 'alloc <value>' on user-overridden state", () => {
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
      rocm_allocator_state: "user-overridden",
      rocm_allocator_value: "MallocAsync",
      rocm_allocator_explanation:
        "CT2_CUDA_ALLOCATOR=MallocAsync (user-set; not clobbered) (CT2 #2012)",
    });
    const segments = tile.sub.split(" · ");
    expect(segments).toContain("alloc MallocAsync");
    // User-overridden state quotes the user value rather than
    // emitting a generic label — a researcher pasting the line into a
    // support thread can see exactly what the runtime saw.
    // We don't surface this through the warning banner (the user knows
    // what they did); the segment is the only signal.
    expect(tile.warning).toBeNull();
  });

  it("appends 'alloc unset' AND surfaces a warning on unset state", () => {
    // The danger state: RDNA 2 box where CT2 is about to crash. The
    // tile must:
    //   1. carry an "alloc unset" sub-line segment (so quick-glance
    //      readers see it next to the gfx target)
    //   2. populate the ``warning`` field with the explanation (so
    //      the home page's warning banner pops it visibly)
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1030",
      rocm_allocator_state: "unset",
      rocm_allocator_value: null,
      rocm_allocator_explanation:
        "CT2_CUDA_ALLOCATOR unset on RDNA 2 — CT2 is about to crash " +
        "(CT2 #2012). Call apply_rocm_runtime_workarounds() before " +
        "the CT2 import, or export CT2_CUDA_ALLOCATOR=cub_caching",
    });
    expect(tile.sub.split(" · ")).toContain("alloc unset");
    expect(tile.warning).toContain("2012");
    expect(tile.warning).toContain("CT2_CUDA_ALLOCATOR");
  });

  it("omits alloc segment when rocm_allocator_state is null", () => {
    // Non-RDNA-2 ROCm box (e.g. RX 7900 XTX) — the workaround is
    // irrelevant so all three fields are null.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 7900 XTX",
      vram_gb: 24.0,
      gfx_target: "gfx1100",
      rocm_allocator_state: null,
      rocm_allocator_value: null,
      rocm_allocator_explanation: null,
    });
    expect(tile.sub).toBe(
      "AMD Radeon RX 7900 XTX · 24 GB VRAM · gfx1100"
    );
    expect(tile.sub).not.toContain("alloc");
    expect(tile.warning).toBeNull();
  });

  it("omits alloc segment when rocm_allocator_state is missing entirely", () => {
    // Defensive: an older API version that doesn't carry the field
    // shouldn't make the tile blow up.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
    });
    expect(tile.sub).toBe("AMD Radeon RX 6800 XT · 16 GB VRAM");
    expect(tile.sub).not.toContain("alloc");
    expect(tile.warning).toBeNull();
  });

  it("omits alloc segment on non-ROCm backends (CUDA / MPS / CPU)", () => {
    for (const backend of ["cuda", "mps", "cpu"]) {
      const tile = backendStatTile({
        backend,
        device_name: "Some Device",
        rocm_allocator_state: null,
      });
      expect(tile.sub == null || !tile.sub.includes("alloc")).toBe(true);
    }
  });

  it("omits alloc segment on unknown / stray state strings (defensive)", () => {
    // The helper switches on three documented states; a stray string
    // (from a bad proxy / older client / test pollution) must not
    // accidentally render a segment.
    for (const stray of ["yes", "true", "on", 1, {}, [], "AUTO"]) {
      const tile = backendStatTile({
        backend: "rocm",
        device_name: "AMD Radeon RX 6800 XT",
        rocm_allocator_state: stray,
      });
      const sub = tile.sub || "";
      expect(sub).not.toContain("alloc cub_caching");
      expect(sub).not.toContain("alloc unset");
    }
  });

  it("alloc segment appears after LSTM patched (last sub-line segment)", () => {
    // Order pin: device → VRAM → gfx → distro → CT2 pin → mirrors →
    // tier → LSTM patch → alloc. alloc is last because it's the
    // most-specific-to-RDNA-2 bit; the LSTM patch is global to ROCm
    // and stays as the second-to-last segment.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1030",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      distro_tier: "first-class",
      rocm_lstm_patch: true,
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
    });
    const segments = tile.sub.split(" · ");
    expect(segments[segments.length - 1]).toBe("alloc cub_caching");
    expect(segments[segments.length - 2]).toBe("LSTM patched");
  });

  it("preserves the GPU tile shape (label / value / sub / warning)", () => {
    // Pin the contract: G4.1 is additive — the four-key tile shape
    // is unchanged, only the sub-line composition gains a segment
    // and the warning composition gains an extra source.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
    });
    expect(Object.keys(tile).sort()).toEqual(
      ["label", "sub", "value", "warning"]
    );
  });

  it("concatenates drift + unset-allocator warnings (worst-case ROCm)", () => {
    // Realistic worst case: RDNA 2 box with both CT2 wheel drift
    // (G2.1) AND the cub_caching env var unset (G4.1). Both warnings
    // are independently actionable; the tile must surface them both
    // through the warning banner (separator " — " between).
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1030",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.6.0",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2",
      rocm_allocator_state: "unset",
      rocm_allocator_explanation:
        "CT2_CUDA_ALLOCATOR unset on RDNA 2 — CT2 is about to crash " +
        "(CT2 #2012). Call apply_rocm_runtime_workarounds() before " +
        "the CT2 import, or export CT2_CUDA_ALLOCATOR=cub_caching",
    });
    expect(tile.warning).toContain("v4.6.0");
    expect(tile.warning).toContain("CT2_CUDA_ALLOCATOR unset");
    // Both warnings are present — neither got swallowed by the other.
    expect(tile.warning).toContain("2012");
  });

  it("no warning concatenation on auto state + drift (drift only)", () => {
    // When the allocator state is auto / user-overridden, the warning
    // field carries only the drift message — the alloc state is
    // surfaced through the sub-line segment, not the banner.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2",
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
      rocm_allocator_explanation: "...",
    });
    expect(tile.warning).toContain("v4.6.0");
    expect(tile.warning).not.toContain("CT2_CUDA_ALLOCATOR");
  });

  it("alloc segment uses the literal label 'alloc' (no localisation)", () => {
    // The string is what a researcher copy-pastes into a support
    // thread; it has to be greppable upstream. Pin the spelling so
    // a future helpers refactor can't quietly change it to
    // "allocator cub_caching" or similar.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
    });
    expect(tile.sub.split(" · ")).toContain("alloc cub_caching");
  });

  it("falls back to 'alloc user-set' when override has no value (defensive)", () => {
    // If the API reports user-overridden but no value (shouldn't
    // happen — server pairs the two — but defensive), the helper
    // emits a generic label rather than "alloc undefined" or "alloc null".
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      rocm_allocator_state: "user-overridden",
      rocm_allocator_value: null,
    });
    expect(tile.sub.split(" · ")).toContain("alloc user-set");
    expect(tile.sub).not.toContain("alloc null");
    expect(tile.sub).not.toContain("alloc undefined");
  });
});

// ---------- G4.2: RDNA 2 HSA_OVERRIDE_GFX_VERSION workaround state ----------
//
// G4.2 added three fields to the /api/capabilities payload —
// ``gpu.rocm_hsa_override_state`` (one of ``"user-set"`` / ``"missing"``),
// ``gpu.rocm_hsa_override_value`` (the literal env var value when set),
// and ``gpu.rocm_hsa_override_explanation`` (one-line rationale). All
// three are ``null`` outside the workaround scope: non-ROCm backends,
// ROCm on gfx1030 itself (the one RDNA 2 die AMD ships kernels for),
// and ROCm on RDNA 3 / RDNA 4 / CDNA cards.
//
// The home page tile renders the state as a sub-line segment:
//   - "user-set" → "HSA <value>" (echoes whatever the user exported,
//                                  including non-recommended values)
//   - "missing"  → "HSA missing" + warning banner with the explanation
//                                  (HIP runtime won't load kernels;
//                                  user has to export the env var)
//   - null / missing → segment omitted entirely
//
// When the missing state coincides with G4.1 unset-allocator OR G2.1
// CT2 wheel drift, all warnings concatenate so none get swallowed.

describe("backendStatTile (G4.2 HSA_OVERRIDE_GFX_VERSION)", () => {
  it("appends 'HSA <value>' on user-set state (recommended value)", () => {
    // Researcher exported HSA_OVERRIDE_GFX_VERSION=10.3.0 by hand on a
    // non-gfx1030 RDNA 2 die. We never clobber a user value; the tile
    // echoes whatever they set so a support bundle shows the active config.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      vram_gb: 12.0,
      gfx_target: "gfx1031",
      distro: "Ubuntu 24.04.4 LTS",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: "10.3.0",
      rocm_hsa_override_explanation:
        "HSA_OVERRIDE_GFX_VERSION=10.3.0 (user-set; not clobbered)",
    });
    expect(tile.sub.split(" · ")).toContain("HSA 10.3.0");
    // User-set state must NOT trigger a warning — the user knows what
    // they did, and the override is in effect.
    expect(tile.warning).toBeNull();
  });

  it("echoes a non-recommended user value verbatim (never silently rewritten)", () => {
    // Researcher experimentally set the var to "11.0.0" — the tile
    // must show that literal string so support bundles never lie.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6600",
      gfx_target: "gfx1032",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: "11.0.0",
      rocm_hsa_override_explanation:
        "HSA_OVERRIDE_GFX_VERSION=11.0.0 (user-set; not clobbered)",
    });
    expect(tile.sub.split(" · ")).toContain("HSA 11.0.0");
    expect(tile.sub).not.toContain("HSA 10.3.0");
    expect(tile.warning).toBeNull();
  });

  it("appends 'HSA missing' AND surfaces a warning on missing state", () => {
    // The actionable state: RDNA 2 non-gfx1030 die without the env
    // var. The tile must:
    //   1. carry an "HSA missing" sub-line segment so quick-glance
    //      readers see it next to the gfx target
    //   2. populate the ``warning`` field with the explanation so the
    //      home page's warning banner pops it visibly
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      vram_gb: 12.0,
      gfx_target: "gfx1031",
      rocm_hsa_override_state: "missing",
      rocm_hsa_override_value: null,
      rocm_hsa_override_explanation:
        "HSA_OVERRIDE_GFX_VERSION unset on gfx1031 — AMD ROCm only " +
        "ships kernels for gfx1030. Export HSA_OVERRIDE_GFX_VERSION=10.3.0 " +
        "before running Scribe so HIP treats gfx1031 as gfx1030",
    });
    expect(tile.sub.split(" · ")).toContain("HSA missing");
    expect(tile.warning).toContain("HSA_OVERRIDE_GFX_VERSION");
    expect(tile.warning).toContain("10.3.0");
    expect(tile.warning).toContain("gfx1031");
  });

  it("omits HSA segment when rocm_hsa_override_state is null", () => {
    // gfx1030 — AMD ships kernels for it, so no override needed; the
    // helper returns null and the tile omits the segment.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
      gfx_target: "gfx1030",
      rocm_hsa_override_state: null,
      rocm_hsa_override_value: null,
      rocm_hsa_override_explanation: null,
    });
    expect(tile.sub).not.toContain("HSA");
    expect(tile.warning).toBeNull();
  });

  it("omits HSA segment when rocm_hsa_override_state is missing entirely", () => {
    // Defensive: an older API version that doesn't carry the field
    // shouldn't make the tile blow up.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6800 XT",
      vram_gb: 16.0,
    });
    expect(tile.sub).toBe("AMD Radeon RX 6800 XT · 16 GB VRAM");
    expect(tile.sub).not.toContain("HSA");
    expect(tile.warning).toBeNull();
  });

  it("omits HSA segment on non-ROCm backends (CUDA / MPS / CPU)", () => {
    // Even if the user has the env var set on a CUDA / MPS / CPU
    // box (deeply weird but possible), the API nulls the field and
    // the tile silently drops the segment — the variable is
    // meaningless without a HIP runtime.
    for (const backend of ["cuda", "mps", "cpu"]) {
      const tile = backendStatTile({
        backend,
        device_name: "Some Device",
        rocm_hsa_override_state: null,
      });
      const sub = tile.sub || "";
      expect(sub).not.toContain("HSA");
    }
  });

  it("omits HSA segment on unknown / stray state strings (defensive)", () => {
    // The helper switches on two documented states; a stray string
    // (from a bad proxy / older client / test pollution) must not
    // accidentally render a segment.
    for (const stray of ["yes", "true", "on", 1, {}, [], "MISSING", "unset"]) {
      const tile = backendStatTile({
        backend: "rocm",
        device_name: "AMD Radeon RX 6700 XT",
        rocm_hsa_override_state: stray,
      });
      const sub = tile.sub || "";
      expect(sub).not.toContain("HSA missing");
      expect(sub).not.toContain("HSA 10.3.0");
    }
  });

  it("HSA segment appears after alloc segment (order pinned)", () => {
    // Order pin: device → VRAM → gfx → distro → CT2 pin → mirrors →
    // tier → LSTM patch → alloc → HSA. HSA is last because it
    // applies to the most-specific subset (RDNA 2 *non-gfx1030*),
    // making it the strictest qualifier in the chain.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      vram_gb: 12.0,
      gfx_target: "gfx1031",
      distro: "Ubuntu 24.04.4 LTS",
      ct2_rocm_pin: "4.7.2",
      distro_tier: "first-class",
      rocm_lstm_patch: true,
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: "10.3.0",
    });
    const segments = tile.sub.split(" · ");
    expect(segments[segments.length - 1]).toBe("HSA 10.3.0");
    expect(segments[segments.length - 2]).toBe("alloc cub_caching");
  });

  it("preserves the GPU tile shape (label / value / sub / warning)", () => {
    // Pin the contract: G4.2 is additive — the four-key tile shape
    // is unchanged.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: "10.3.0",
    });
    expect(Object.keys(tile).sort()).toEqual(
      ["label", "sub", "value", "warning"]
    );
  });

  it("concatenates drift + alloc-unset + HSA-missing warnings (worst case)", () => {
    // Worst case for a freshly-installed RX 6700 XT user: CT2 wheel
    // drift (G2.1) + cub_caching unset (G4.1) + HSA override missing
    // (G4.2). All three are independently actionable; the tile must
    // surface them all through the warning banner (separator " — ").
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      vram_gb: 12.0,
      gfx_target: "gfx1031",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.6.0",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2",
      rocm_allocator_state: "unset",
      rocm_allocator_explanation:
        "CT2_CUDA_ALLOCATOR unset on RDNA 2 — CT2 is about to crash " +
        "(CT2 #2012). Call apply_rocm_runtime_workarounds() before " +
        "the CT2 import, or export CT2_CUDA_ALLOCATOR=cub_caching",
      rocm_hsa_override_state: "missing",
      rocm_hsa_override_explanation:
        "HSA_OVERRIDE_GFX_VERSION unset on gfx1031 — AMD ROCm only " +
        "ships kernels for gfx1030. Export HSA_OVERRIDE_GFX_VERSION=10.3.0 " +
        "before running Scribe so HIP treats gfx1031 as gfx1030",
    });
    // All three warning sources are present — none got swallowed.
    // (The explanations themselves can contain " — " in their body
    // copy, so we verify presence of the three signature substrings
    // and their composition order rather than counting separators.)
    expect(tile.warning).toContain("v4.6.0");
    expect(tile.warning).toContain("CT2_CUDA_ALLOCATOR unset");
    expect(tile.warning).toContain("HSA_OVERRIDE_GFX_VERSION");
    // The drift message comes first, then alloc, then HSA — pin the
    // composition order so a future helpers refactor that swaps the
    // order is caught.
    const driftIdx = tile.warning.indexOf("v4.6.0");
    const allocIdx = tile.warning.indexOf("CT2_CUDA_ALLOCATOR unset");
    const hsaIdx = tile.warning.indexOf("HSA_OVERRIDE_GFX_VERSION unset");
    expect(driftIdx).toBeGreaterThanOrEqual(0);
    expect(allocIdx).toBeGreaterThan(driftIdx);
    expect(hsaIdx).toBeGreaterThan(allocIdx);
  });

  it("no warning concatenation on user-set state + drift (drift only)", () => {
    // When the HSA state is user-set, the warning field carries only
    // the drift message — the user-set state is surfaced through the
    // sub-line segment, not the banner.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      ct2_drift_message:
        "ctranslate2 v4.6.0 installed; pinned ROCm wheel is v4.7.2",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: "10.3.0",
      rocm_hsa_override_explanation: "...",
    });
    expect(tile.warning).toContain("v4.6.0");
    expect(tile.warning).not.toContain("HSA_OVERRIDE_GFX_VERSION");
  });

  it("HSA-missing warning fires standalone (no other warnings)", () => {
    // Researcher with a clean install and matching CT2 — only the
    // HSA-missing warning fires.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      vram_gb: 12.0,
      gfx_target: "gfx1031",
      ct2_rocm_pin: "4.7.2",
      ct2_installed: "4.7.2",
      ct2_drift_message: null,
      rocm_allocator_state: "auto",
      rocm_allocator_value: "cub_caching",
      rocm_allocator_explanation: "...",
      rocm_hsa_override_state: "missing",
      rocm_hsa_override_value: null,
      rocm_hsa_override_explanation:
        "HSA_OVERRIDE_GFX_VERSION unset on gfx1031 — AMD ROCm only " +
        "ships kernels for gfx1030. Export HSA_OVERRIDE_GFX_VERSION=10.3.0 " +
        "before running Scribe so HIP treats gfx1031 as gfx1030",
    });
    expect(tile.warning).toContain("HSA_OVERRIDE_GFX_VERSION");
    expect(tile.warning).not.toContain("v4.6.0");
    expect(tile.warning).not.toContain("CT2_CUDA_ALLOCATOR");
    // Single warning — the warning is exactly the HSA explanation
    // verbatim (the explanation itself has " — " in its body but
    // there are no joiner separators added by the helper).
    expect(tile.warning).toBe(
      "HSA_OVERRIDE_GFX_VERSION unset on gfx1031 — AMD ROCm only " +
      "ships kernels for gfx1030. Export HSA_OVERRIDE_GFX_VERSION=10.3.0 " +
      "before running Scribe so HIP treats gfx1031 as gfx1030"
    );
  });

  it("HSA segment uses the literal label 'HSA' (no localisation)", () => {
    // The string is what a researcher copy-pastes into a support
    // thread; it has to be greppable upstream. Pin the spelling so
    // a future helpers refactor can't quietly change it to "hsa "
    // or "HSA_OVERRIDE_GFX_VERSION " or similar.
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: "10.3.0",
    });
    expect(tile.sub.split(" · ")).toContain("HSA 10.3.0");
  });

  it("falls back to 'HSA user-set' when value is missing (defensive)", () => {
    // If the API reports user-set but no value (shouldn't happen —
    // server pairs the two — but defensive), the helper emits a
    // generic label rather than "HSA undefined" or "HSA null".
    const tile = backendStatTile({
      backend: "rocm",
      device_name: "AMD Radeon RX 6700 XT",
      rocm_hsa_override_state: "user-set",
      rocm_hsa_override_value: null,
    });
    expect(tile.sub.split(" · ")).toContain("HSA user-set");
    expect(tile.sub).not.toContain("HSA null");
    expect(tile.sub).not.toContain("HSA undefined");
  });
});
