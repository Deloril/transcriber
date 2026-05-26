// Pure helpers shared between the in-page <script> blocks and the
// Vitest test suite. ANY change here must keep the function shape
// stable so tests/js/* keep working — these are the contract.
//
// Loaded at runtime via `<script type="module">` from the templates,
// and via direct import from tests.

// ---------- formatters ----------

export function fmtElapsed(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export function fmtBytes(n) {
  if (!isFinite(n) || n < 0) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : (n >= 10 ? 1 : 2))} ${u[i]}`;
}

export function fmtDuration(s) {
  if (!isFinite(s) || s <= 0) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
          : `${m}:${String(sec).padStart(2, "0")}`;
}

export function fmtBitrate(b) {
  if (!isFinite(b) || b <= 0) return "—";
  if (b >= 1_000_000) return `${(b / 1_000_000).toFixed(1)} Mbps`;
  return `${Math.round(b / 1000)} kbps`;
}

export function fmtRate(r) {
  if (!isFinite(r) || r <= 0) return "—";
  return r >= 1000 ? `${(r / 1000).toFixed(1)} kHz` : `${r} Hz`;
}

export function fmtFps(f) {
  return isFinite(f) && f > 0 ? `${f.toFixed(2)} fps` : "—";
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ---------- Active GPU backend label (G1.4) ----------
//
// The upload page shows a "Backend" tile on the Recording details card so
// users can see at a glance whether the next transcription will run on
// CUDA / ROCm / MPS / CPU. The data comes from ``GET /api/capabilities``
// (server side: ``scribe.engine.gpu_backend()``); the helpers below turn
// the raw payload into the tile shape the renderer consumes.
//
// Two-step: ``formatBackendLabel`` is the pure naming policy (so the
// editor can reuse it later if we surface the backend there too), and
// ``backendStatTile`` builds the full tile dict.

const _BACKEND_DISPLAY = {
  cuda: "CUDA",
  rocm: "ROCm",
  mps: "MPS",
  cpu: "CPU",
};

/**
 * Pretty-cased label for a 4-state backend identifier. Unknown or
 * empty inputs fall back to ``"CPU"`` — the safe default that matches
 * what the engine does when no GPU is detected.
 *
 * @param {string|null|undefined} backend — one of ``cuda``, ``rocm``,
 *   ``mps``, ``cpu`` (case-insensitive).
 * @returns {string}
 */
export function formatBackendLabel(backend) {
  const key = String(backend ?? "").trim().toLowerCase();
  return _BACKEND_DISPLAY[key] || "CPU";
}

/**
 * Build the {label, value, sub} stat-tile dict for the active GPU
 * backend. ``gpu`` is the ``capabilities.gpu`` object returned by the
 * server: ``{ backend, device_name, vram_gb }``. Returns ``null`` when
 * ``gpu`` is missing — the renderer should then skip the tile entirely
 * rather than show a placeholder.
 *
 * Sub-line composition:
 *   - device name (when present)
 *   - VRAM in GB (when reported; only CUDA / ROCm carry this)
 *
 * The sub line collapses to ``null`` when both components are missing
 * (e.g. CPU backend on a machine with no discrete GPU), so the
 * renderer doesn't print an empty " · ".
 *
 * @param {object|null|undefined} gpu
 * @returns {{label: string, value: string, sub: string|null}|null}
 */
export function backendStatTile(gpu) {
  if (!gpu || typeof gpu !== "object") return null;
  const value = formatBackendLabel(gpu.backend);
  const parts = [];
  if (gpu.device_name) parts.push(String(gpu.device_name));
  if (typeof gpu.vram_gb === "number" && isFinite(gpu.vram_gb) && gpu.vram_gb > 0) {
    parts.push(`${gpu.vram_gb} GB VRAM`);
  }
  return {
    label: "Backend",
    value,
    sub: parts.length ? parts.join(" · ") : null,
  };
}

// ---------- Parakeet visibility on the upload page (G5.1) ----------
//
// NVIDIA Parakeet (NeMo) is a CUDA-only model: NeMo has no AMD/ROCm
// support and there's no community fork that does. So when the active
// backend is AMD ROCm, the Parakeet optgroup is hidden from the model
// dropdown entirely — we don't tantalise the user with an option that
// will fail at load time. On other non-CUDA backends (MPS / CPU) the
// optgroup stays visible because Parakeet *can* still run on CPU
// (slowly) and the user might be on a machine with multiple
// pipelines; the server-side ``capabilities.parakeet.blocked_by_backend``
// flag and the in-page hint cover those cases.

/**
 * Decide whether to hide the Parakeet optgroup in the model dropdown.
 *
 * Returns ``true`` only on AMD ROCm. The reasoning is intentionally
 * narrow: on Apple Silicon (MPS) or CPU the user isn't blocked from
 * loading NeMo, just told "this will be slow / GPU recommended" via
 * the model hint. ROCm is the only backend where the model literally
 * cannot run.
 *
 * @param {string|null|undefined} backend - one of ``cuda``, ``rocm``,
 *   ``mps``, ``cpu`` (case-insensitive).
 * @returns {boolean}
 */
export function shouldHideParakeetOptgroup(backend) {
  return String(backend ?? "").trim().toLowerCase() === "rocm";
}

/**
 * Compute the model-hint state for the upload page based on the
 * currently-selected model and the server-reported capabilities.
 *
 * Returns a structured object the renderer turns into DOM:
 *   - ``kind`` — ``"blocked"`` (Parakeet selected on a backend that
 *     can't run NeMo), ``"missing"`` (Parakeet selected but NeMo isn't
 *     installed), ``"info"`` (Parakeet selected and OK — tells the
 *     user it's English-only and GPU-recommended), or ``"none"``
 *     (Whisper / non-Parakeet model — hide the hint entirely).
 *   - ``tone`` — ``"warn"`` for blocked/missing, ``"muted"`` for info,
 *     ``null`` for none.
 *   - ``html`` — the HTML body for the hint, with the backend label
 *     pre-escaped. ``null`` when ``kind === "none"``.
 *
 * The "blocked" branch is the G5.1 surface: it names the active
 * backend so the user understands why the option is unavailable, and
 * nudges them to a Whisper model.
 *
 * @param {object} args
 * @param {string|null|undefined} args.model     - the currently-selected
 *   model id (e.g. ``"large-v3"`` or ``"nvidia/parakeet-tdt-0.6b-v2"``).
 * @param {string|null|undefined} args.backend   - active backend.
 * @param {object|null|undefined} args.parakeet  - the server's
 *   ``capabilities.parakeet`` payload: ``{ available, installed,
 *   blocked_by_backend, error }``.
 * @returns {{kind: "blocked"|"missing"|"info"|"none",
 *           tone: "warn"|"muted"|null,
 *           html: string|null}}
 */
export function parakeetModelHint({ model, backend, parakeet } = {}) {
  const isParakeet = String(model ?? "").toLowerCase().includes("parakeet");
  if (!isParakeet) {
    return { kind: "none", tone: null, html: null };
  }
  const p = parakeet || {};
  if (p.blocked_by_backend) {
    const label = formatBackendLabel(backend);
    return {
      kind: "blocked",
      tone: "warn",
      html:
        "⚠ Parakeet (NVIDIA NeMo) doesn't run on the active <strong>" +
        escapeHtml(label) + "</strong> backend. Pick a Whisper model.",
    };
  }
  if (!p.available && !p.installed) {
    return {
      kind: "missing",
      tone: "warn",
      html:
        "⚠ NVIDIA NeMo isn't installed. Install it with " +
        "<code>pip install -r requirements-parakeet.txt</code> in your venv, then reload.",
    };
  }
  return {
    kind: "info",
    tone: "muted",
    html: "English only · CUDA GPU recommended · ~30× faster than Whisper.",
  };
}

// ---------- Library view (F10.1) ----------

/**
 * Case-insensitive substring filter that mirrors
 * ``scribe.library.matches_query`` server-side. The library page
 * fetches every job once and then filters locally as the user types,
 * so this helper has to behave identically — same fields, same
 * normalisation — to avoid surprises.
 *
 * Empty/whitespace queries match everything.
 *
 * @param {object} row - a row produced by ``GET /api/jobs``.
 * @param {string} q
 * @returns {boolean}
 */
export function matchesLibraryQuery(row, q) {
  const needle = String(q ?? "").trim().toLowerCase();
  if (!needle) return true;
  const speakers = Array.isArray(row.speakers) ? row.speakers.join(" ") : "";
  const haystack = [
    row.input_filename,
    row.status,
    row.mode,
    row.language,
    row.model,
    speakers,
  ].map(x => String(x ?? "")).join(" ").toLowerCase();
  return haystack.includes(needle);
}

/**
 * Filter library rows. Preserves input order so the caller
 * controls sort. Mirrors the server's ``filter_rows`` exactly.
 *
 * @param {Array<object>} rows
 * @param {string} q
 * @returns {Array<object>}
 */
export function searchLibraryRows(rows, q) {
  if (!Array.isArray(rows)) return [];
  return rows.filter(r => matchesLibraryQuery(r, q));
}

/**
 * Compare two library rows by a known sort key for use with
 * ``Array.prototype.sort``. The keys we currently support are:
 *
 *   - ``input_filename`` (string, case-insensitive)
 *   - ``mode`` (string)
 *   - ``language`` (string)
 *   - ``status`` (string)
 *   - ``duration_seconds`` (numeric, nulls last)
 *   - ``speaker_count`` (numeric)
 *   - ``created_at`` (ISO string; lexicographic order matches chronological)
 *
 * Unknown keys fall back to ``id`` so the sort is still
 * deterministic (the table won't shuffle rows around for no
 * reason). Direction must be ``"asc"`` or ``"desc"``.
 *
 * @param {object} a
 * @param {object} b
 * @param {string} key
 * @param {"asc"|"desc"} dir
 * @returns {number}
 */
export function compareLibraryRows(a, b, key, dir) {
  const sign = dir === "asc" ? 1 : -1;
  const va = a == null ? undefined : a[key];
  const vb = b == null ? undefined : b[key];
  // Nulls / undefineds always sink to the bottom regardless of direction.
  const aMissing = va === null || va === undefined || va === "";
  const bMissing = vb === null || vb === undefined || vb === "";
  if (aMissing && bMissing) {
    // Tie-break on id ascending so stable across re-sorts.
    return String(a?.id ?? "").localeCompare(String(b?.id ?? ""));
  }
  if (aMissing) return 1;
  if (bMissing) return -1;
  let cmp;
  if (typeof va === "number" && typeof vb === "number") {
    cmp = va - vb;
  } else {
    // Strings: case-insensitive natural-ish compare.
    cmp = String(va).toLowerCase().localeCompare(String(vb).toLowerCase());
  }
  if (cmp !== 0) return sign * cmp;
  // Tie-break on id (always ascending) for determinism.
  return String(a?.id ?? "").localeCompare(String(b?.id ?? ""));
}

/**
 * Format an ISO-8601 timestamp string (e.g. ``2026-05-25T14:30:00Z``)
 * for the library's "Created" column. Returns the empty string for
 * blank input and the raw input on parse failure so the user always
 * sees *something* — never just a silent gap.
 *
 * Output shape: ``YYYY-MM-DD HH:MM`` in the browser's local
 * timezone (no seconds, no timezone suffix). The full ISO string
 * is preserved on the row's ``title`` for hover.
 *
 * @param {string} iso
 * @returns {string}
 */
export function formatLibraryDate(iso) {
  if (iso == null || iso === "") return "";
  const s = String(iso);
  const d = new Date(s);
  if (isNaN(d.getTime())) return s;
  const pad = (n) => String(n).padStart(2, "0");
  const Y = d.getFullYear();
  const M = pad(d.getMonth() + 1);
  const D = pad(d.getDate());
  const h = pad(d.getHours());
  const m = pad(d.getMinutes());
  return `${Y}-${M}-${D} ${h}:${m}`;
}

/**
 * Render a list of speaker labels for the library row's "Speakers"
 * column. We always show at most two names; the rest collapse into
 * a "+N more" suffix so the column stays compact.
 *
 * @param {Array<string>} speakers
 * @returns {string}
 */
export function formatLibrarySpeakers(speakers) {
  if (!Array.isArray(speakers) || speakers.length === 0) return "";
  if (speakers.length <= 2) return speakers.join(", ");
  return `${speakers[0]}, ${speakers[1]} +${speakers.length - 2} more`;
}

// ---------- ETA math ----------

/**
 * Given a snapshot of job state, return the absolute predicted finish time
 * (epoch seconds), or null if there isn't enough signal yet.
 *
 * @param {object} state
 * @param {number} state.startedAt - epoch seconds when the job began
 * @param {number} state.lastProgress - fraction in [0, 1]
 * @param {number} state.lastProgressTime - epoch seconds when lastProgress was set
 * @returns {number|null}
 */
export function predictFinishTime({ startedAt, lastProgress, lastProgressTime }) {
  if (startedAt == null || lastProgressTime == null) return null;
  if (!(lastProgress > 0.05)) return null;
  const elapsedAtUpdate = lastProgressTime - startedAt;
  if (elapsedAtUpdate <= 0) return null;
  const total = elapsedAtUpdate / lastProgress;
  return startedAt + total;
}

// ---------- word highlighting ----------

/**
 * Find the index of the active word at time `t` given a sorted array of
 * `{start, end}` words.
 *
 * Returns -1 when `t` lies in a gap between words (silence) — UI should
 * clear the highlight rather than letting it stick on the previous word.
 *
 * `lastActive`, when supplied as the previously-returned index, lets the
 * function short-circuit common monotonic playback. Pass -1 to force a
 * full search.
 */
export function findActiveWord(wordSpans, t, lastActive = -1) {
  const GAP = 0.05;
  const n = wordSpans.length;
  if (n === 0) return -1;

  if (lastActive >= 0 && lastActive < n) {
    const w = wordSpans[lastActive];
    if (t >= w.start - 0.02 && t <= w.end + GAP) return lastActive;
    for (let i = lastActive + 1; i < Math.min(lastActive + 6, n); i++) {
      if (t < wordSpans[i].start) {
        const prev = wordSpans[i - 1];
        if (prev && t <= prev.end + GAP) return i - 1;
        return -1;
      }
      if (t <= wordSpans[i].end + GAP) return i;
    }
  }

  // binary search for the largest word with start <= t
  let lo = 0, hi = n - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (wordSpans[mid].start <= t) { best = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  if (best === -1) return -1;
  if (t > wordSpans[best].end + GAP) return -1;
  return best;
}

/**
 * Spread an array of word tokens evenly across the [start, end] interval,
 * producing word entries with synthesised timestamps. Used after the user
 * edits a segment's text outside the timed word boundaries.
 */
export function spreadTokensAcrossSpan(tokens, start, end, speaker) {
  if (!tokens.length) return [];
  const span = Math.max(0.05, end - start);
  const per = span / tokens.length;
  return tokens.map((tok, i) => ({
    text: tok,
    start: start + i * per,
    end: start + (i + 1) * per,
    speaker,
    score: null,
  }));
}

// ---------- search ----------

/**
 * Map a search hit (character range in the joined-with-spaces token text)
 * to an inclusive [firstWord, lastWord] range. Used by the editor's
 * search highlighter to mark whole word spans without mutating the
 * editable text node tree.
 *
 * Returns null when the needle isn't found, or `{firstWord, lastWord}`.
 */
export function rangeForMatch(tokens, needle) {
  const joined = tokens.join(" ").toLowerCase();
  const lneedle = needle.toLowerCase();
  const matches = [];
  let from = 0;
  while (true) {
    const pos = joined.indexOf(lneedle, from);
    if (pos < 0) break;
    const endPos = pos + lneedle.length;
    let cum = 0, firstWord = -1, lastWord = -1;
    for (let i = 0; i < tokens.length; i++) {
      const tokStart = cum;
      const tokEnd = cum + tokens[i].length;
      if (firstWord < 0 && tokEnd > pos) firstWord = i;
      if (tokStart < endPos) lastWord = i;
      cum = tokEnd + 1; // +1 for joining space
    }
    if (firstWord >= 0 && lastWord >= firstWord) {
      matches.push({ firstWord, lastWord });
    }
    from = endPos;
  }
  return matches;
}

/**
 * Apply a find-and-replace pass to a single segment's word array.
 *
 * The transcript editor's segments hold an ordered list of word
 * objects (`{text, start, end, speaker, score}`). This helper produces
 * a *new* word array with `needle` replaced by `replacement` everywhere
 * it occurs in the segment's text, preserving each surviving word's
 * timestamps and metadata so the player keeps working.
 *
 * Rules:
 * - Match is case-insensitive (mirrors the search highlighter).
 * - Single-word match: edit the matched word's text in place; replace
 *   the matched substring inside the word, keep the rest.
 * - Multi-word match: collapse the run of matched words into a single
 *   word holding the replacement text, with start = first word's
 *   start and end = last word's end. Confidence score is cleared
 *   (we no longer have a speech-to-text confidence for the synthetic
 *   token).
 * - Empty replacement: matched words are dropped entirely. If a
 *   single-word match leaves the word's text empty, drop the word.
 * - Empty needle: returns the input unchanged. Avoids an infinite loop.
 *
 * Returns `{words, replacements}` where `replacements` is the count
 * of needle occurrences replaced (so the caller can report
 * "replaced N occurrences" / decide whether to push undo).
 *
 * Pure: never touches the DOM, never mutates inputs.
 */
export function replaceInSegmentWords(words, needle, replacement) {
  if (!needle || !words || !words.length) {
    return { words: words ? words.slice() : [], replacements: 0 };
  }
  const lneedle = needle.toLowerCase();
  const tokens = words.map(w => (w.text || "").toLowerCase());
  const joined = tokens.join(" ");
  // Per-character → word index map. Build once so multi-word matches
  // are easy: any char position resolves to a single word.
  const charToWord = new Array(joined.length);
  let cum = 0;
  for (let i = 0; i < tokens.length; i++) {
    for (let k = 0; k < tokens[i].length; k++) {
      charToWord[cum + k] = i;
    }
    if (i < tokens.length - 1) {
      // The space separator belongs to the *previous* word for the
      // purpose of "which word does this char fall under." That keeps
      // a needle like "abc " (with a trailing space) from accidentally
      // including the next word.
      charToWord[cum + tokens[i].length] = i;
    }
    cum += tokens[i].length + 1; // +1 for the joining space
  }

  // Walk the joined text, find each match, record its range in word
  // indices. We don't apply edits during the walk because applying
  // them in place would shift indices.
  const ranges = [];
  let from = 0;
  while (true) {
    const pos = joined.indexOf(lneedle, from);
    if (pos < 0) break;
    const endPos = pos + lneedle.length - 1;
    const firstWord = charToWord[pos];
    const lastWord = charToWord[endPos];
    if (firstWord != null && lastWord != null && lastWord >= firstWord) {
      // Save char positions inside the first/last words so we can
      // surgically rewrite single-word and partial-word matches.
      let cumFirst = 0;
      for (let i = 0; i < firstWord; i++) cumFirst += tokens[i].length + 1;
      let cumLast = 0;
      for (let i = 0; i < lastWord; i++) cumLast += tokens[i].length + 1;
      ranges.push({
        firstWord,
        lastWord,
        offsetInFirst: pos - cumFirst,
        offsetEndInLast: endPos - cumLast + 1,
      });
    }
    // Always advance by needle length to avoid infinite loops on
    // overlapping matches; this matches the search highlighter.
    from = pos + Math.max(1, lneedle.length);
  }
  if (!ranges.length) return { words: words.slice(), replacements: 0 };

  // Two cases need different handling:
  //   - All ranges where firstWord === lastWord can be applied as a
  //     single per-word substring rewrite (regex) so multiple matches
  //     in the same word all land cleanly.
  //   - Ranges that span multiple words ("multi-word matches") have
  //     to splice the array. Multi-word matches that overlap each
  //     other can't really happen with the simple "advance past
  //     end" walk we just did, so right-to-left splicing is safe.
  // Group single-word and multi-word ranges separately.
  const singleWordIdx = new Set();
  const multiWord = [];
  for (const r of ranges) {
    if (r.firstWord === r.lastWord) singleWordIdx.add(r.firstWord);
    else multiWord.push(r);
  }

  let out = words.slice();

  // First pass: per-word regex rewrite for any word that has at
  // least one single-word match. This handles the "ababab" → "XXX"
  // case correctly because it's a single regex pass over the word.
  if (singleWordIdx.size) {
    // Escape the needle for regex use.
    const reNeedle = new RegExp(
      lneedle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
      "gi",
    );
    for (const idx of singleWordIdx) {
      // Skip words that are also covered by a multi-word match — those
      // get rewritten by the multi-word splice instead. Detect by
      // checking if any multiWord range includes this index.
      let covered = false;
      for (const m of multiWord) {
        if (idx >= m.firstWord && idx <= m.lastWord) { covered = true; break; }
      }
      if (covered) continue;
      const orig = words[idx];
      const next = (orig.text || "").replace(reNeedle, replacement);
      if (next.length === 0) {
        // Whole word emptied — mark for deletion via a sentinel; we'll
        // drop sentinels after the multi-word pass to keep indices
        // stable until then.
        out[idx] = { ...orig, text: "", _drop: true };
      } else {
        out[idx] = { ...orig, text: next };
      }
    }
  }

  // Second pass: multi-word splices, right-to-left.
  multiWord.sort((a, b) => a.firstWord - b.firstWord);
  for (let r = multiWord.length - 1; r >= 0; r--) {
    const range = multiWord[r];
    const first = words[range.firstWord];
    const last = words[range.lastWord];
    const before = (first.text || "").slice(0, range.offsetInFirst);
    const after = (last.text || "").slice(range.offsetEndInLast);
    const merged = before + replacement + after;
    if (merged.length === 0) {
      out.splice(range.firstWord, range.lastWord - range.firstWord + 1);
    } else {
      out.splice(range.firstWord, range.lastWord - range.firstWord + 1, {
        text: merged,
        start: first.start,
        end: last.end,
        speaker: first.speaker,
        score: null,
      });
    }
  }

  // Drop sentinel-marked empty words.
  out = out.filter(w => !w._drop);
  return { words: out, replacements: ranges.length };
}

/**
 * Recompute a segment's `text` field from its word list — the
 * canonical way the editor keeps `seg.text` and `seg.words` in sync
 * after a structural edit. Mirrors the editor's existing inline
 * normalisation but lifted here so find/replace can reuse it.
 */
export function rebuildSegmentText(words) {
  if (!words || !words.length) return "";
  return words.map(w => w.text || "").join(" ").replace(/\s+/g, " ").trim();
}

// ---------- gutter / margin layout for code applications (F4.3) ----------

/**
 * Parse a Scribe word id (`s<seg>w<word>`) into `[seg, word]` integers.
 * Returns null on malformed input — the caller can decide whether that's
 * an exception or a "skip this application" event. Mirrors
 * `scribe.applications.parse_word_id` (Python).
 */
export function parseWordId(wordId) {
  if (typeof wordId !== "string") return null;
  const m = /^s(\d+)w(\d+)$/.exec(wordId);
  if (!m) return null;
  return [parseInt(m[1], 10), parseInt(m[2], 10)];
}

/**
 * Compare two 3-tuples of `(seg, word, offset)` lexicographically, where
 * `offset` may be Number.POSITIVE_INFINITY (used as the "end of word"
 * sentinel — matches the Python `_END_OF_WORD = math.inf` convention).
 *
 * Returns -1, 0, or +1.
 */
function _cmpAnchor(a, b) {
  for (let i = 0; i < 3; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}

/**
 * Compute the "leftmost position" of an application's span as a
 * `[seg, word, startOffset]` tuple. Mirrors `_start_position` in
 * `scribe.application_spans`.
 *
 * `app` is expected to carry: anchorStartWordId, startCharOffset (or null).
 */
function _startPos(app) {
  const p = parseWordId(app.anchorStartWordId);
  if (!p) return null;
  const so = app.startCharOffset == null ? 0 : app.startCharOffset;
  return [p[0], p[1], so];
}

/**
 * Compute the "rightmost position" of an application's span as a
 * `[seg, word, endOffsetOrInf]` tuple.
 */
function _endPos(app) {
  const p = parseWordId(app.anchorEndWordId);
  if (!p) return null;
  const eo = app.endCharOffset == null
    ? Number.POSITIVE_INFINITY
    : app.endCharOffset;
  return [p[0], p[1], eo];
}

/**
 * Sort applications in document order, ties broken by application id.
 * Mirrors `sort_by_anchor` in `scribe.application_spans`.
 *
 * Pure: returns a new array; input is not mutated.
 */
export function sortApplicationsByAnchor(apps) {
  const decorated = apps.map((a) => ({
    a,
    start: _startPos(a),
    end: _endPos(a),
  }));
  decorated.sort((x, y) => {
    const c1 = _cmpAnchor(x.start, y.start);
    if (c1 !== 0) return c1;
    const c2 = _cmpAnchor(x.end, y.end);
    if (c2 !== 0) return c2;
    return x.a.id < y.a.id ? -1 : x.a.id > y.a.id ? 1 : 0;
  });
  return decorated.map((d) => d.a);
}

/**
 * Lay out a single source's applications into non-overlapping lanes for
 * the gutter renderer. Mirrors `assign_lanes` in
 * `scribe.application_gutter` — same algorithm, same lane numbering.
 *
 * Input: an array of objects with at least:
 *   - id (string)
 *   - sourceId (string) — must all match
 *   - anchorStartWordId / anchorEndWordId (string, `s<seg>w<word>`)
 *   - startCharOffset / endCharOffset (number or null)
 *
 * Returns:
 *   {
 *     sourceId,
 *     placements: [{ applicationId, lane, stackDepth }, ...],   // doc order
 *     laneCount,
 *     maxStackDepth,
 *   }
 *
 * Throws on mixed sources (matches the Python guard).
 */
export function assignLanes(apps) {
  if (!apps || apps.length === 0) {
    return { sourceId: "", placements: [], laneCount: 0, maxStackDepth: 0 };
  }
  const sourceIds = new Set(apps.map((a) => a.sourceId));
  if (sourceIds.size > 1) {
    throw new Error(
      "assignLanes requires single-source input; got " +
      sourceIds.size + " distinct sourceIds"
    );
  }
  const [sourceId] = sourceIds;

  const ordered = sortApplicationsByAnchor(apps);

  const laneEnds = []; // index → end-position tuple of last app on that lane
  const laneOf = Object.create(null);

  for (const app of ordered) {
    const start = _startPos(app);
    const end = _endPos(app);
    let chosen = -1;
    for (let i = 0; i < laneEnds.length; i++) {
      if (_cmpAnchor(laneEnds[i], start) <= 0) {
        chosen = i;
        break;
      }
    }
    if (chosen === -1) {
      chosen = laneEnds.length;
      laneEnds.push(end);
    } else {
      laneEnds[chosen] = end;
    }
    laneOf[app.id] = chosen;
  }

  // Stack depth: pairwise overlap (strict). O(n^2) like the Python side.
  const depths = Object.create(null);
  for (const a of ordered) depths[a.id] = 0;
  for (let i = 0; i < ordered.length; i++) {
    const aLo = _startPos(ordered[i]);
    const aHi = _endPos(ordered[i]);
    for (let j = 0; j < ordered.length; j++) {
      if (i === j) continue;
      const b = ordered[j];
      const bLo = _startPos(b);
      const bHi = _endPos(b);
      // [aLo, aHi] and [bLo, bHi] overlap iff aLo < bHi and bLo < aHi.
      if (_cmpAnchor(aLo, bHi) < 0 && _cmpAnchor(bLo, aHi) < 0) {
        depths[ordered[i].id] += 1;
      }
    }
  }

  const placements = ordered.map((a) => ({
    applicationId: a.id,
    lane: laneOf[a.id],
    stackDepth: depths[a.id],
  }));

  let maxDepth = 0;
  for (const v of Object.values(depths)) {
    if (v > maxDepth) maxDepth = v;
  }

  return {
    sourceId,
    placements,
    laneCount: laneEnds.length,
    maxStackDepth: maxDepth,
  };
}

/**
 * Bucket applications by sourceId and lay each source out independently.
 * Mirrors `assign_lanes_per_source`. Returns a plain object
 * `{ [sourceId]: layout }`.
 */
export function assignLanesPerSource(apps) {
  const buckets = Object.create(null);
  for (const a of apps || []) {
    if (!buckets[a.sourceId]) buckets[a.sourceId] = [];
    buckets[a.sourceId].push(a);
  }
  const out = Object.create(null);
  for (const [sid, list] of Object.entries(buckets)) {
    out[sid] = assignLanes(list);
  }
  return out;
}

// ---------- snap-to-word / sentence / paragraph (F4.4) ----------
//
// JS mirror of `scribe.selection_snap`. Same semantics, same outputs;
// the parallel suite in tests/js/selection-snap.test.mjs exercises
// the same fixtures as tests/test_selection_snap.py and the two
// must agree.

const _CLOSING_TRIM = `"')]}»›”’`;
const _SENTENCE_FINAL = ".?!";

/**
 * Return true iff `text` (a single token) ends a sentence.
 * Strips trailing closing-quote / bracket characters before
 * checking the final character. Mirrors `_is_sentence_final` in
 * scribe.selection_snap.
 */
export function isSentenceFinal(text) {
  if (typeof text !== "string") return false;
  let s = text.replace(/\s+$/u, "");
  while (s && _CLOSING_TRIM.indexOf(s[s.length - 1]) >= 0) {
    s = s.slice(0, -1);
  }
  if (!s) return false;
  return _SENTENCE_FINAL.indexOf(s[s.length - 1]) >= 0;
}

/**
 * Return inclusive [startWordIdx, endWordIdx] sentence ranges for a
 * segment's words. Mirrors `sentence_ranges_in_segment`.
 *
 * `words` is an array of `{text}` objects (extra fields ignored).
 * Returns `[]` for empty input. The output partitions [0, n) exactly.
 */
export function sentenceRangesInSegment(words) {
  if (!words || words.length === 0) return [];
  const ranges = [];
  let start = 0;
  for (let i = 0; i < words.length; i++) {
    const text = (words[i] && typeof words[i].text === "string") ? words[i].text : "";
    if (isSentenceFinal(text)) {
      ranges.push([start, i]);
      start = i + 1;
    }
  }
  if (start <= words.length - 1) {
    ranges.push([start, words.length - 1]);
  }
  return ranges;
}

/**
 * Return inclusive [startSegIdx, endSegIdx] paragraph ranges for the
 * transcript. Two consecutive segments share a paragraph iff they
 * share a non-null `speaker`; null/undefined speakers always start a
 * new paragraph. Mirrors `paragraph_ranges`.
 */
export function paragraphRanges(segments) {
  if (!segments || segments.length === 0) return [];
  const ranges = [];
  let start = 0;
  for (let i = 1; i < segments.length; i++) {
    const prev = segments[i - 1] ? segments[i - 1].speaker : null;
    const cur = segments[i] ? segments[i].speaker : null;
    if (prev == null || cur == null || prev !== cur) {
      ranges.push([start, i - 1]);
      start = i;
    }
  }
  ranges.push([start, segments.length - 1]);
  return ranges;
}

function _segmentWordCount(segments, segIdx) {
  if (segIdx < 0 || segIdx >= segments.length) {
    throw new Error(`segment index ${segIdx} out of range [0, ${segments.length})`);
  }
  const seg = segments[segIdx];
  if (!seg || typeof seg !== "object") {
    throw new Error(`segment ${segIdx} must be an object`);
  }
  const words = seg.words;
  if (!Array.isArray(words)) return 0;
  return words.length;
}

function _segmentWords(segments, segIdx) {
  const seg = segments[segIdx];
  if (!seg || typeof seg !== "object") return [];
  return Array.isArray(seg.words) ? seg.words : [];
}

function _validateWordRef(segments, wordId) {
  const parsed = parseWordId(wordId);
  if (!parsed) {
    throw new Error(`word id must match s<seg>w<word>; got ${JSON.stringify(wordId)}`);
  }
  const [segIdx, wordIdx] = parsed;
  const n = _segmentWordCount(segments, segIdx);
  if (n === 0) {
    throw new Error(`segment ${segIdx} has no words; cannot resolve ${wordId}`);
  }
  if (wordIdx >= n) {
    throw new Error(
      `word index ${wordIdx} out of range [0, ${n}) in segment ${segIdx}`
    );
  }
  return [segIdx, wordIdx];
}

function _sentenceFor(sentences, wordIdx) {
  for (const r of sentences) {
    if (r[0] <= wordIdx && wordIdx <= r[1]) return r;
  }
  throw new Error(`word index ${wordIdx} is not in any sentence range`);
}

function _paragraphFor(paragraphs, segIdx) {
  for (const r of paragraphs) {
    if (r[0] <= segIdx && segIdx <= r[1]) return r;
  }
  throw new Error(`segment index ${segIdx} is not in any paragraph range`);
}

/**
 * A selection record, mirroring `scribe.selection_snap.Selection`.
 * Fields are camelCase to match the rest of helpers.mjs:
 *   - startWordId, endWordId (string, `s<seg>w<word>`)
 *   - startCharOffset, endCharOffset (number or null)
 */
function _validateSelection(sel) {
  if (!sel || typeof sel !== "object") {
    throw new Error("selection must be an object");
  }
  if (typeof sel.startWordId !== "string" || typeof sel.endWordId !== "string") {
    throw new Error("selection.{startWordId,endWordId} must be strings");
  }
}

/**
 * Drop sub-word character offsets; return a whole-word selection.
 * Mirrors `snap_to_word`. Idempotent.
 */
export function snapToWord(sel) {
  _validateSelection(sel);
  if (!parseWordId(sel.startWordId)) {
    throw new Error(`startWordId malformed: ${sel.startWordId}`);
  }
  if (!parseWordId(sel.endWordId)) {
    throw new Error(`endWordId malformed: ${sel.endWordId}`);
  }
  if (sel.startCharOffset == null && sel.endCharOffset == null) {
    return sel;
  }
  return {
    ...sel,
    startCharOffset: null,
    endCharOffset: null,
  };
}

/**
 * Snap to sentence boundaries within each endpoint's segment.
 * Mirrors `snap_to_sentence`. Drops sub-word offsets, never narrows.
 */
export function snapToSentence(sel, segments) {
  _validateSelection(sel);
  if (!Array.isArray(segments)) {
    throw new Error("segments must be an array");
  }
  const [sSeg, sWord] = _validateWordRef(segments, sel.startWordId);
  const [eSeg, eWord] = _validateWordRef(segments, sel.endWordId);
  if (sSeg > eSeg || (sSeg === eSeg && sWord > eWord)) {
    throw new Error("selection start is after selection end; cannot snap");
  }
  const startSentences = sentenceRangesInSegment(_segmentWords(segments, sSeg));
  const endSentences = sentenceRangesInSegment(_segmentWords(segments, eSeg));
  const [newStartWord] = _sentenceFor(startSentences, sWord);
  const newEndWord = _sentenceFor(endSentences, eWord)[1];
  return {
    ...sel,
    startWordId: `s${sSeg}w${newStartWord}`,
    endWordId: `s${eSeg}w${newEndWord}`,
    startCharOffset: null,
    endCharOffset: null,
  };
}

/**
 * Snap to whole-paragraph boundaries (consecutive same-speaker
 * segments form one paragraph). Mirrors `snap_to_paragraph`.
 */
export function snapToParagraph(sel, segments) {
  _validateSelection(sel);
  if (!Array.isArray(segments)) {
    throw new Error("segments must be an array");
  }
  const [sSeg, sWord] = _validateWordRef(segments, sel.startWordId);
  const [eSeg, eWord] = _validateWordRef(segments, sel.endWordId);
  if (sSeg > eSeg || (sSeg === eSeg && sWord > eWord)) {
    throw new Error("selection start is after selection end; cannot snap");
  }
  const paragraphs = paragraphRanges(segments);
  const startPara = _paragraphFor(paragraphs, sSeg);
  const endPara = _paragraphFor(paragraphs, eSeg);
  const endSegIdx = endPara[1];
  const endWordCount = _segmentWordCount(segments, endSegIdx);
  if (endWordCount === 0) {
    throw new Error(
      `paragraph end segment ${endSegIdx} has no words; cannot snap`
    );
  }
  return {
    ...sel,
    startWordId: `s${startPara[0]}w0`,
    endWordId: `s${endSegIdx}w${endWordCount - 1}`,
    startCharOffset: null,
    endCharOffset: null,
  };
}

// ---------- playback ranges for coded segments (F4.6) ----------
//
// JS mirror of `scribe.application_playback`. Same semantics, same
// outputs; the parallel suite in tests/js/playback.test.mjs
// exercises the same fixtures as tests/test_application_playback.py
// and the two must agree.

/**
 * A timing record for a single word in the transcript. Fields match
 * `scribe.application_playback.WordTime`.
 *
 *   { start, end, text }
 */

/**
 * Coerce `value` to a finite Number or return null. Mirrors the
 * Python `_coerce_time` helper: bool / NaN / ±Infinity / non-numeric
 * inputs all become null. JS's typeof-bool catches `true` / `false`
 * (which are falsy/truthy but not Numbers in JS, so the typeof
 * check below already handles them).
 */
function _coerceTime(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "boolean") return null;
  if (typeof value !== "number") return null;
  if (!Number.isFinite(value)) return null;
  return value;
}

/**
 * Flatten a Scribe transcript's `segments[].words[]` into a
 * `{ "s<seg>w<word>": { start, end, text } }` dictionary.
 *
 * Words missing or with invalid timing are skipped — callers learn
 * by absence (`!(wordId in map)`). Mirrors `build_word_time_map` in
 * Python; insertion order is preserved.
 *
 * Throws on a non-array `segments` input. Empty `segments` returns
 * an empty object.
 */
export function buildWordTimeMap(segments) {
  if (!Array.isArray(segments)) {
    throw new Error("segments must be an array");
  }
  const out = Object.create(null);
  for (let segIdx = 0; segIdx < segments.length; segIdx++) {
    const seg = segments[segIdx];
    if (!seg || typeof seg !== "object") continue;
    const words = seg.words;
    if (!Array.isArray(words)) continue;
    for (let wordIdx = 0; wordIdx < words.length; wordIdx++) {
      const w = words[wordIdx];
      if (!w || typeof w !== "object") continue;
      const start = _coerceTime(w.start);
      const end = _coerceTime(w.end);
      if (start === null || end === null) continue;
      if (end < start) continue;
      const text = typeof w.text === "string"
        ? w.text
        : (w.text == null ? "" : String(w.text));
      out[`s${segIdx}w${wordIdx}`] = { start, end, text };
    }
  }
  return out;
}

function _segmentTime(segments, segIdx) {
  if (segIdx < 0 || segIdx >= segments.length) return [null, null];
  const seg = segments[segIdx];
  if (!seg || typeof seg !== "object") return [null, null];
  return [_coerceTime(seg.start), _coerceTime(seg.end)];
}

function _interpolateOffset(word, charOffset, side) {
  const textLen = word.text ? word.text.length : 0;
  if (textLen <= 0) return side === "start" ? word.start : word.end;
  const span = word.end - word.start;
  if (span <= 0) return word.start;
  const clamped = Math.max(0, Math.min(charOffset, textLen));
  return word.start + span * (clamped / textLen);
}

/**
 * Compute the wall-clock playback range `[start, end]` for one
 * application's anchor over `segments`. Mirrors
 * `playback_range_for_application` in Python.
 *
 * `app` carries the camelCase fields used elsewhere in helpers.mjs:
 *   - id, sourceId
 *   - anchorStartWordId, anchorEndWordId
 *   - startCharOffset, endCharOffset (number or null)
 *
 * Returns `null` when neither the anchor words nor their segments
 * have any usable timing — the caller should hide the play button.
 *
 * Returns
 *   { applicationId, sourceId, start, end }
 *
 * Throws on malformed anchor word ids and on anchors whose segment
 * index falls outside `segments` (matches the Python behaviour;
 * F4.5 is the right place to handle out-of-range anchors long-term).
 *
 * `wordTimeMap` may be passed by callers that already built one.
 */
export function playbackRangeForApplication(app, segments, wordTimeMap) {
  if (!app || typeof app !== "object") {
    throw new Error("app must be an object");
  }
  if (!Array.isArray(segments)) {
    throw new Error("segments must be an array");
  }
  const startParsed = parseWordId(app.anchorStartWordId);
  const endParsed = parseWordId(app.anchorEndWordId);
  if (!startParsed) {
    throw new Error(`anchorStartWordId malformed: ${app.anchorStartWordId}`);
  }
  if (!endParsed) {
    throw new Error(`anchorEndWordId malformed: ${app.anchorEndWordId}`);
  }
  const [saSeg] = startParsed;
  const [eaSeg] = endParsed;
  if (saSeg < 0 || saSeg >= segments.length) {
    throw new Error(
      `anchorStartWordId segment ${saSeg} out of range [0, ${segments.length})`
    );
  }
  if (eaSeg < 0 || eaSeg >= segments.length) {
    throw new Error(
      `anchorEndWordId segment ${eaSeg} out of range [0, ${segments.length})`
    );
  }

  const wmap = wordTimeMap || buildWordTimeMap(segments);
  const startWord = wmap[app.anchorStartWordId];
  const endWord = wmap[app.anchorEndWordId];
  const [segStart] = _segmentTime(segments, saSeg);
  const segEnd = _segmentTime(segments, eaSeg)[1];

  let startTime;
  if (startWord) {
    startTime = app.startCharOffset != null
      ? _interpolateOffset(startWord, app.startCharOffset, "start")
      : startWord.start;
  } else if (segStart != null) {
    startTime = segStart;
  } else {
    return null;
  }

  let endTime;
  if (endWord) {
    endTime = app.endCharOffset != null
      ? _interpolateOffset(endWord, app.endCharOffset, "end")
      : endWord.end;
  } else if (segEnd != null) {
    endTime = segEnd;
  } else {
    return null;
  }

  if (endTime < startTime) endTime = startTime;

  return {
    applicationId: app.id,
    sourceId: app.sourceId,
    start: startTime,
    end: endTime,
  };
}

/**
 * Bulk variant. `segmentsBySource` is `{ sourceId: segments }`.
 * Returns `{ applicationId: { applicationId, sourceId, start, end } }`,
 * skipping applications whose source isn't in the map and applications
 * whose individual lookup returns null.
 *
 * Builds the word-time map once per source. Mirrors
 * `playback_ranges_for_applications` in Python.
 */
export function playbackRangesForApplications(apps, segmentsBySource) {
  const cache = Object.create(null);
  const out = Object.create(null);
  for (const app of apps || []) {
    const segs = segmentsBySource ? segmentsBySource[app.sourceId] : undefined;
    if (!segs) continue;
    let wmap = cache[app.sourceId];
    if (!wmap) {
      wmap = buildWordTimeMap(segs);
      cache[app.sourceId] = wmap;
    }
    const r = playbackRangeForApplication(app, segs, wmap);
    if (r !== null) out[app.id] = r;
  }
  return out;
}

// ---------- F5.2 — right-click memo creation ----------
//
// Mirrors scribe/memo_context.py. The editor uses these to build a
// draft memo payload from any right-click context (a code chip, a
// quote highlight, a participant card, …) and POST it to
// /api/projects/<pid>/memos. The default-type mapping must match
// DEFAULT_MEMO_TYPE_BY_TARGET in Python so the same right-click on
// the same entity produces the same default regardless of who built
// the payload.
//
// Validation here is shape-only: full validation runs server-side via
// Memo.validate. We keep the JS strict enough to catch obvious
// caller bugs (typoed target type, missing fields) without
// duplicating MemoLink's regex library.

export const MEMO_LINK_TARGET_TYPES = Object.freeze([
  "code",
  "source",
  "application",
  "participant",
  "coder",
  "project",
  "memo",
]);

export const MEMO_TYPES = Object.freeze([
  "code",
  "theoretical",
  "methodological",
  "reflexive",
  "quote",
  "source",
  "project",
  "free",
]);

export const DEFAULT_MEMO_TYPE_BY_TARGET = Object.freeze({
  code: "code",
  application: "quote",
  source: "source",
  project: "project",
  coder: "methodological",
  memo: "theoretical",
  participant: "free",
});

const _TARGET_ID_RE = /^[a-f0-9]{12}$/;
const _ROLE_RE = /^[A-Za-z][\w \-]{0,63}$/;

export function defaultMemoTypeForTarget(targetType) {
  if (typeof targetType !== "string") {
    throw new Error("target_type must be a string");
  }
  return Object.prototype.hasOwnProperty.call(
    DEFAULT_MEMO_TYPE_BY_TARGET,
    targetType,
  )
    ? DEFAULT_MEMO_TYPE_BY_TARGET[targetType]
    : "free";
}

function _coerceLinkInput(raw) {
  // Accept the JS-camelCase shape that the editor uses internally
  // (`{ targetType, targetId, role }`) AND the Python-snake_case
  // shape that comes back from the server. Normalise to snake_case
  // since that's what the server endpoint expects on the wire.
  if (!raw || typeof raw !== "object") {
    throw new Error("link entries must be objects");
  }
  const targetType = raw.target_type ?? raw.targetType;
  const targetId = raw.target_id ?? raw.targetId;
  const role = raw.role ?? "";
  if (typeof targetType !== "string" || typeof targetId !== "string") {
    throw new Error("link entries need target_type and target_id strings");
  }
  if (!MEMO_LINK_TARGET_TYPES.includes(targetType)) {
    throw new Error(`unknown target_type: ${targetType}`);
  }
  if (!_TARGET_ID_RE.test(targetId)) {
    throw new Error(`target_id must be 12-char hex: ${targetId}`);
  }
  const out = { target_type: targetType, target_id: targetId };
  const r = String(role || "").trim();
  if (r) {
    if (!_ROLE_RE.test(r)) throw new Error(`invalid link role: ${r}`);
    out.role = r;
  }
  return out;
}

/**
 * Build the JSON body for `POST /api/projects/<pid>/memos` from a
 * right-click context. Returns a plain object — no fetch is made.
 *
 * @param {object} args
 * @param {string} args.targetType   one of MEMO_LINK_TARGET_TYPES
 * @param {string} args.targetId     12-char hex id
 * @param {string} [args.role]       optional link role
 * @param {string} [args.type]       memo type override (default = defaultMemoTypeForTarget)
 * @param {string} [args.title]
 * @param {string} [args.body]
 * @param {string} [args.bodyFormat] markdown / plain / html
 * @param {string} [args.authorCoderId]
 * @param {Array<object>} [args.extraLinks] additional links (de-duped against primary)
 * @param {Array<string>} [args.tags]
 * @param {object} [args.provenance] string→string
 *
 * The shape always includes `links: [primary, ...extras]`. The server
 * runs full validation; this is the editor's "what would I send" pure
 * helper.
 */
export function buildMemoDraftPayload({
  targetType,
  targetId,
  role = "",
  type = null,
  title = "",
  body = "",
  bodyFormat = "markdown",
  authorCoderId = null,
  extraLinks = null,
  tags = null,
  provenance = null,
} = {}) {
  if (typeof targetType !== "string" || !MEMO_LINK_TARGET_TYPES.includes(targetType)) {
    throw new Error(`unknown target_type: ${targetType}`);
  }
  if (typeof targetId !== "string" || !_TARGET_ID_RE.test(targetId)) {
    throw new Error(`target_id must be 12-char hex: ${targetId}`);
  }
  let chosenType;
  if (type == null) {
    chosenType = defaultMemoTypeForTarget(targetType);
  } else {
    if (!MEMO_TYPES.includes(type)) {
      throw new Error(`unknown memo type: ${type}`);
    }
    chosenType = type;
  }

  const primary = _coerceLinkInput({ target_type: targetType, target_id: targetId, role });

  const extras = [];
  for (const raw of extraLinks || []) {
    const link = _coerceLinkInput(raw);
    // Dedupe against primary by (type, id, role)
    if (
      link.target_type === primary.target_type &&
      link.target_id === primary.target_id &&
      (link.role || "") === (primary.role || "")
    ) {
      continue;
    }
    extras.push(link);
  }

  const payload = {
    type: chosenType,
    title: String(title || ""),
    body: String(body || ""),
    body_format: String(bodyFormat || "markdown"),
    links: [primary, ...extras],
  };
  if (authorCoderId) payload.author_coder_id = String(authorCoderId);
  if (Array.isArray(tags) && tags.length) {
    payload.tags = tags.map((t) => String(t));
  }
  if (provenance && typeof provenance === "object") {
    const p = {};
    for (const [k, v] of Object.entries(provenance)) p[String(k)] = String(v);
    if (Object.keys(p).length) payload.provenance = p;
  }
  return payload;
}

/**
 * Convenience: build the same payload but wrap the right-click
 * context inside a top-level `context` block. The server endpoint
 * accepts either form; this is the shape the right-click composer
 * sends when it doesn't yet know what extra fields the user wants.
 */
export function buildMemoContextPayload({
  targetType,
  targetId,
  role = "",
  type = null,
  title = "",
  body = "",
  bodyFormat = "markdown",
  authorCoderId = null,
  extraLinks = null,
  tags = null,
  provenance = null,
} = {}) {
  if (typeof targetType !== "string" || !MEMO_LINK_TARGET_TYPES.includes(targetType)) {
    throw new Error(`unknown target_type: ${targetType}`);
  }
  if (typeof targetId !== "string" || !_TARGET_ID_RE.test(targetId)) {
    throw new Error(`target_id must be 12-char hex: ${targetId}`);
  }
  const ctx = { target_type: targetType, target_id: targetId };
  const r = String(role || "").trim();
  if (r) {
    if (!_ROLE_RE.test(r)) throw new Error(`invalid link role: ${r}`);
    ctx.role = r;
  }
  const out = { context: ctx };
  if (type != null) {
    if (!MEMO_TYPES.includes(type)) {
      throw new Error(`unknown memo type: ${type}`);
    }
    out.type = type;
  }
  if (title) out.title = String(title);
  if (body) out.body = String(body);
  if (bodyFormat && bodyFormat !== "markdown") out.body_format = String(bodyFormat);
  if (authorCoderId) out.author_coder_id = String(authorCoderId);
  if (Array.isArray(extraLinks) && extraLinks.length) {
    out.extra_links = extraLinks.map(_coerceLinkInput);
  }
  if (Array.isArray(tags) && tags.length) {
    out.tags = tags.map((t) => String(t));
  }
  if (provenance && typeof provenance === "object") {
    const p = {};
    for (const [k, v] of Object.entries(provenance)) p[String(k)] = String(v);
    if (Object.keys(p).length) out.provenance = p;
  }
  return out;
}

// ---------- F5.3 — memo-sorting canvas ----------
//
// Mirrors scribe/memo_canvas.py: pure helpers for laying memo cards
// out on a 2D surface. Constants must match Python so the same drag
// operation produces the same canvas state regardless of which side
// computes the snap / clamp.
//
// What's here:
//
// * Bounds + grid helpers — clampToBounds, snapToGrid. Shared by the
//   drag handler and the keyboard-nudge accelerator.
// * hitTestCard — which memo card is under a (x, y) point.
// * buildAssignCardPayload / buildMoveCardPayload / buildAddCategoryPayload
//   — the JSON request bodies for the canvas endpoints. Each accepts
//   the editor's camelCase shape and emits the snake_case wire format
//   the server expects.

export const CANVAS_MAX_COORD = 1_000_000;
export const CANVAS_MAX_LABEL_LEN = 120;
const _CATEGORY_COLOR_RE = /^#[0-9A-Fa-f]{6}$/;

function _coerceCanvasCoord(value, name) {
  if (typeof value === "boolean") {
    throw new Error(`${name} must be a number, not bool`);
  }
  if (typeof value !== "number") {
    throw new Error(`${name} must be a number`);
  }
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite`);
  }
  if (Math.abs(value) > CANVAS_MAX_COORD) {
    throw new Error(`${name} out of range`);
  }
  return value;
}

/**
 * Clamp a coordinate pair to the given inclusive bounds. Returns
 * `[x, y]`. Defaults match the Python ±MAX_COORD bounds so unbound
 * usage still rejects NaN / inf without artificially restricting the
 * researcher's logical space.
 */
export function clampToBounds(
  x,
  y,
  {
    minX = -CANVAS_MAX_COORD,
    minY = -CANVAS_MAX_COORD,
    maxX = CANVAS_MAX_COORD,
    maxY = CANVAS_MAX_COORD,
  } = {},
) {
  const cx = _coerceCanvasCoord(x, "x");
  const cy = _coerceCanvasCoord(y, "y");
  if (minX > maxX || minY > maxY) {
    throw new Error("min bounds must be ≤ max bounds");
  }
  return [
    Math.max(minX, Math.min(maxX, cx)),
    Math.max(minY, Math.min(maxY, cy)),
  ];
}

/**
 * Snap a coordinate to the nearest multiple of `grid`. Default 1
 * is a no-op rounding; on-screen drag handlers typically pass 16.
 */
export function snapToGrid(x, y, { grid = 1 } = {}) {
  if (!(grid > 0)) {
    throw new Error("grid must be positive");
  }
  const cx = _coerceCanvasCoord(x, "x");
  const cy = _coerceCanvasCoord(y, "y");
  return [Math.round(cx / grid) * grid, Math.round(cy / grid) * grid];
}

/**
 * Return the memo_id of the topmost card under (x, y), or null.
 *
 * Cards is an iterable of `{ memo_id, x, y }` objects. "Topmost" =
 * later in iteration order, matching DOM stacking + the Python
 * implementation.
 */
export function hitTestCard(
  cards,
  x,
  y,
  { halfWidth = 80, halfHeight = 50 } = {},
) {
  const cx = _coerceCanvasCoord(x, "x");
  const cy = _coerceCanvasCoord(y, "y");
  if (!(halfWidth > 0) || !(halfHeight > 0)) {
    throw new Error("halfWidth and halfHeight must be positive");
  }
  let hit = null;
  for (const card of cards || []) {
    if (!card) continue;
    const card_x = card.x;
    const card_y = card.y;
    const memo_id = card.memo_id ?? card.memoId;
    if (typeof card_x !== "number" || typeof card_y !== "number") continue;
    if (Math.abs(card_x - cx) <= halfWidth && Math.abs(card_y - cy) <= halfHeight) {
      hit = memo_id;
    }
  }
  return hit;
}

/** Build the wire body for `PUT /api/projects/<pid>/canvas/cards/<memo_id>`. */
export function buildMoveCardPayload({ x, y } = {}) {
  const cx = _coerceCanvasCoord(x, "x");
  const cy = _coerceCanvasCoord(y, "y");
  return { x: cx, y: cy };
}

/** Build the wire body for `POST /api/projects/<pid>/canvas/categories`. */
export function buildAddCategoryPayload({
  label,
  color = "",
  x = 0,
  y = 0,
} = {}) {
  if (typeof label !== "string" || !label.trim()) {
    throw new Error("label must be a non-empty string");
  }
  const cleanLabel = label.trim();
  if (cleanLabel.length > CANVAS_MAX_LABEL_LEN) {
    throw new Error(`label must be ≤ ${CANVAS_MAX_LABEL_LEN} chars`);
  }
  const cx = _coerceCanvasCoord(x, "x");
  const cy = _coerceCanvasCoord(y, "y");
  const out = { label: cleanLabel, x: cx, y: cy };
  if (color) {
    if (typeof color !== "string" || !_CATEGORY_COLOR_RE.test(color)) {
      throw new Error(`color must be #rrggbb (got ${color})`);
    }
    out.color = color.toLowerCase();
  }
  return out;
}

/** Build the wire body for `PUT /api/projects/<pid>/canvas/categories/<cid>/members/<memo_id>`. */
export function buildAssignCardPayload() {
  // The endpoint takes path params for the ids; the body is empty.
  // Returning an explicit `{}` keeps caller code uniform across the
  // canvas API surface.
  return {};
}

// ---------- F5.5 — promote a memo into a code definition ----------
//
// Mirrors scribe/memo_promote.py. The editor uses these to build the
// wire body for `POST /api/projects/<pid>/memos/<mid>/promote-to-code`
// from a one-click "promote" action on a memo card. The server runs
// full validation; this helper is shape-only with the same closed-
// vocabulary checks the Python module performs (stage / status /
// reserved provenance keys) so an obvious caller bug surfaces in the
// browser before a round-trip.

const _CODE_STATUSES = Object.freeze(["active", "draft", "retired"]);
const _CODEBOOK_STAGES = Object.freeze([
  "initial",
  "focused",
  "axial",
  "theoretical",
  "locked",
]);
const _RESERVED_PROMOTE_PROVENANCE_KEYS = Object.freeze([
  "source",
  "memo_id",
]);
// Code colour: ``#RGB`` or ``#RRGGBB``. Mirrors CODE_COLOUR_RE.
const _CODE_COLOUR_RE = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;

export const PROMOTE_DEFAULTS = Object.freeze({
  stage: "initial",
  status: "active",
  recordBackLink: true,
  backLinkRole: "promoted_to",
});

/**
 * Build the JSON body for `POST /api/projects/<pid>/memos/<mid>/promote-to-code`.
 *
 * Returns a plain object — no fetch is made. All fields are optional;
 * the server fills defaults from scribe/memo_promote.py when keys are
 * absent. Pass `null` / leave undefined to defer to the server's
 * defaults; pass an explicit value to override.
 *
 * @param {object} [args]
 * @param {string} [args.name]              code name override (else server derives from memo)
 * @param {string} [args.definition]        code definition override (else memo.body)
 * @param {string} [args.inclusionCriteria]
 * @param {string} [args.exclusionCriteria]
 * @param {Array<string>} [args.exemplars]
 * @param {string} [args.parentCodeId]      parent in the codebook hierarchy
 * @param {Array<object>} [args.relatedCodes]  list of {code_id, relation_type}
 * @param {string} [args.theoreticalMemo]   override the theoretical memo seed
 * @param {string} [args.stage]             one of CODEBOOK_STAGES (default: 'initial')
 * @param {string} [args.colour]            #RGB / #RRGGBB
 * @param {string} [args.status]            one of CODE_STATUSES (default: 'active')
 * @param {object} [args.extraProvenance]   additional provenance keys (string→string)
 * @param {string} [args.codeId]            pin a specific code id (rare; tests/imports)
 * @param {string} [args.changeNote]        version-log note (default: server fills lineage)
 * @param {boolean} [args.recordBackLink]   default true; pass false to skip back-link
 * @param {string} [args.backLinkRole]      default 'promoted_to'
 *
 * Validation here is shape-only:
 *   - stage / status / colour are checked against their closed
 *     vocabularies / regex.
 *   - extraProvenance cannot contain the reserved keys (`source`,
 *     `memo_id`) — the server would refuse anyway, but failing fast
 *     in the browser keeps the audit trail safe.
 */
export function buildPromoteMemoPayload({
  name = null,
  definition = null,
  inclusionCriteria = null,
  exclusionCriteria = null,
  exemplars = null,
  parentCodeId = null,
  relatedCodes = null,
  theoreticalMemo = null,
  stage = null,
  colour = null,
  status = null,
  extraProvenance = null,
  codeId = null,
  changeNote = null,
  recordBackLink = null,
  backLinkRole = null,
} = {}) {
  const out = {};

  if (name != null) {
    if (typeof name !== "string") throw new Error("name must be a string");
    out.name = name;
  }
  if (definition != null) {
    if (typeof definition !== "string") {
      throw new Error("definition must be a string");
    }
    out.definition = definition;
  }
  if (inclusionCriteria != null) {
    if (typeof inclusionCriteria !== "string") {
      throw new Error("inclusion_criteria must be a string");
    }
    out.inclusion_criteria = inclusionCriteria;
  }
  if (exclusionCriteria != null) {
    if (typeof exclusionCriteria !== "string") {
      throw new Error("exclusion_criteria must be a string");
    }
    out.exclusion_criteria = exclusionCriteria;
  }
  if (exemplars != null) {
    if (!Array.isArray(exemplars)) {
      throw new Error("exemplars must be an array of strings");
    }
    out.exemplars = exemplars.map((e) => String(e));
  }
  if (parentCodeId != null) {
    if (typeof parentCodeId !== "string" || !_TARGET_ID_RE.test(parentCodeId)) {
      throw new Error(`parent_code_id must be 12-char hex (got ${parentCodeId})`);
    }
    out.parent_code_id = parentCodeId;
  }
  if (relatedCodes != null) {
    if (!Array.isArray(relatedCodes)) {
      throw new Error("related_codes must be an array");
    }
    out.related_codes = relatedCodes.map((r) => {
      if (!r || typeof r !== "object") {
        throw new Error("related_codes entries must be objects");
      }
      const code_id = r.code_id ?? r.codeId;
      const relation_type = r.relation_type ?? r.relationType;
      if (typeof code_id !== "string" || !_TARGET_ID_RE.test(code_id)) {
        throw new Error("related_codes.code_id must be 12-char hex");
      }
      if (typeof relation_type !== "string" || !relation_type) {
        throw new Error("related_codes.relation_type is required");
      }
      return { code_id, relation_type };
    });
  }
  if (theoreticalMemo != null) {
    if (typeof theoreticalMemo !== "string") {
      throw new Error("theoretical_memo must be a string");
    }
    out.theoretical_memo = theoreticalMemo;
  }
  if (stage != null) {
    if (!_CODEBOOK_STAGES.includes(stage)) {
      throw new Error(`unknown stage: ${stage}`);
    }
    out.stage = stage;
  }
  if (colour != null) {
    if (colour !== "" && !_CODE_COLOUR_RE.test(colour)) {
      throw new Error(`colour must be #RGB or #RRGGBB (got ${colour})`);
    }
    out.colour = colour;
  }
  if (status != null) {
    if (!_CODE_STATUSES.includes(status)) {
      throw new Error(`unknown status: ${status}`);
    }
    out.status = status;
  }
  if (extraProvenance != null) {
    if (typeof extraProvenance !== "object" || Array.isArray(extraProvenance)) {
      throw new Error("extra_provenance must be a plain object");
    }
    const cleaned = {};
    for (const [k, v] of Object.entries(extraProvenance)) {
      const key = String(k).trim();
      if (!key) continue;
      if (_RESERVED_PROMOTE_PROVENANCE_KEYS.includes(key)) {
        throw new Error(
          `extra_provenance cannot override reserved key: ${key}`,
        );
      }
      cleaned[key] = String(v);
    }
    if (Object.keys(cleaned).length) {
      out.extra_provenance = cleaned;
    }
  }
  if (codeId != null) {
    if (typeof codeId !== "string" || !_TARGET_ID_RE.test(codeId)) {
      throw new Error(`code_id must be 12-char hex (got ${codeId})`);
    }
    out.code_id = codeId;
  }
  if (changeNote != null) {
    if (typeof changeNote !== "string") {
      throw new Error("change_note must be a string");
    }
    out.change_note = changeNote;
  }
  if (recordBackLink != null) {
    out.record_back_link = Boolean(recordBackLink);
  }
  if (backLinkRole != null) {
    if (typeof backLinkRole !== "string") {
      throw new Error("back_link_role must be a string");
    }
    const role = backLinkRole.trim();
    if (role) {
      if (!_ROLE_RE.test(role)) {
        throw new Error(`invalid back_link_role: ${backLinkRole}`);
      }
      out.back_link_role = role;
    }
  }
  return out;
}

// ---------- F9.9 — per-application provenance display on hover ----------
//
// JS mirror of scribe/application_provenance_display.py. The editor's
// hover tooltip calls these helpers to turn an application + the
// related entities (code, code-version-at-apply, coder) into a
// structured display object plus formatters.
//
// Field set, vocabularies, and rendering MUST agree with the Python
// side — tests/js/application-provenance-display.test.mjs and
// tests/test_application_provenance_display.py share fixtures.

export const PROVENANCE_SOURCE_LABELS = Object.freeze({
  human: "Human-coded",
  ai_accepted: "AI-suggested · accepted",
  ai_modified: "AI-suggested · accepted with edits",
  imported: "Imported",
  other: "Other",
});

export const DEFAULT_PROVENANCE_SOURCE_LABEL = PROVENANCE_SOURCE_LABELS.human;

export const AI_FEATURE_LABELS = Object.freeze({
  code_suggestion: "Code suggestion",
  new_code_suggestion: "New code suggestion",
  quote_similarity: "Quote similarity",
  transcript_review: "Transcript review",
  second_coder: "AI second coder",
  memo_draft: "Memo draft",
  other: "Other AI",
});

export const AI_DECISION_LABELS = Object.freeze({
  pending: "Pending",
  accepted: "Accepted",
  modified: "Accepted with edits",
  rejected: "Rejected",
});

const _APPLICATION_PROVENANCE_SOURCES = new Set([
  "human",
  "ai_accepted",
  "ai_modified",
  "imported",
  "other",
]);

const _RESERVED_PROVENANCE_KEYS = new Set([
  "source",
  "model_id",
  "embedding_model",
  "suggestion_id",
  "accepted_at",
  "feature",
  "backend",
]);

// F2.2's DEFINITION_FIELDS — the closed set of code fields that
// trigger a new revision when changed.
const _DEFINITION_FIELDS = [
  "name",
  "definition",
  "inclusion_criteria",
  "exclusion_criteria",
  "exemplars",
  "theoretical_memo",
  "related_codes",
];

function _formatConfidence(value) {
  if (value === null || value === undefined || value === "") return "";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "";
  return n.toFixed(2);
}

function _formatAnchor(app) {
  const s = String(app.anchorStartWordId || app.anchor_start_word_id || "");
  const e = String(app.anchorEndWordId || app.anchor_end_word_id || "");
  const startOff = app.startCharOffset ?? app.start_char_offset ?? null;
  const endOff = app.endCharOffset ?? app.end_char_offset ?? null;
  if (s === e && startOff === null && endOff === null) return s;
  return `${s}–${e}`;
}

function _arraysEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x && typeof x === "object" && y && typeof y === "object") {
      if (JSON.stringify(x) !== JSON.stringify(y)) return false;
    } else if (x !== y) {
      return false;
    }
  }
  return true;
}

function _driftedFields(snapshot, current) {
  // Mirrors scribe.definition_at_apply.drifted_definition_fields:
  // Either side null/undefined → all fields drifted.
  if (!snapshot || !current) return _DEFINITION_FIELDS.slice();
  const out = [];
  for (const f of _DEFINITION_FIELDS) {
    let snap = snapshot[f];
    let cur = current[f];
    if (f === "exemplars" || f === "related_codes") {
      if (snap == null) snap = [];
      if (cur == null) cur = [];
      if (!_arraysEqual(snap, cur)) out.push(f);
    } else {
      // Treat undefined and "" the same as the Python ``or ""`` paths
      const a = snap == null ? "" : snap;
      const b = cur == null ? "" : cur;
      if (a !== b) out.push(f);
    }
  }
  return out;
}

function _coerceProvenance(p) {
  if (!p || typeof p !== "object") return {};
  const out = {};
  for (const [k, v] of Object.entries(p)) out[String(k)] = String(v);
  return out;
}

/**
 * Build a ProvenanceDisplay-equivalent object from an application + relations.
 *
 * Inputs accept either snake_case (matching the Python on-disk JSON) or
 * camelCase (matching the in-page JS state). All "related" args are
 * optional; missing ones surface as "(unknown)" / blank fields.
 *
 * @param {object} app   Application-shaped object
 * @param {object} [relations]
 * @param {object} [relations.code]            current Code-shaped object
 * @param {object} [relations.codeVersion]     CodeVersion-shaped object (the version-at-apply)
 * @param {object} [relations.coder]           Coder-shaped object (the application's coder)
 * @param {object} [relations.decidedByCoder]  Coder-shaped object (the AI-decision actor)
 * @param {string} [relations.sourceName]      display name for the source
 * @returns {object}
 */
export function buildProvenanceDisplay(app, relations = {}) {
  if (!app || typeof app !== "object") {
    throw new Error("buildProvenanceDisplay: app is required");
  }
  const {
    code = null,
    codeVersion = null,
    coder = null,
    decidedByCoder = null,
    sourceName = "",
  } = relations;

  const provenance = _coerceProvenance(app.provenance);

  const codeId = String(app.code_id ?? app.codeId ?? "");
  const sourceId = String(app.source_id ?? app.sourceId ?? "");
  const coderId = String(app.coder_id ?? app.coderId ?? "");
  const versionIdAtApply = String(
    app.definition_version_id_at_apply ?? app.definitionVersionIdAtApply ?? "",
  );

  // Code (current)
  let codeName, codeColour, codeStage;
  if (!code) {
    codeName = "(unknown)";
    codeColour = "";
    codeStage = "";
  } else {
    codeName = (code.name || "").trim() || "(unnamed)";
    codeColour = code.colour || "";
    codeStage = code.stage || "";
  }

  // Code version at apply
  const snapshotMissing = !codeVersion;
  let versionNumber = "";
  let versionRecordedAt = "";
  let versionChangeNote = "";
  let snapshot = {};
  let nameAtApply = "";
  if (codeVersion) {
    versionNumber = `v${parseInt(codeVersion.version, 10)}`;
    versionRecordedAt = String(codeVersion.created_at || codeVersion.createdAt || "");
    versionChangeNote = String(codeVersion.change_note || codeVersion.changeNote || "");
    snapshot =
      typeof codeVersion.snapshot === "object" && codeVersion.snapshot
        ? codeVersion.snapshot
        : {};
    nameAtApply = String(snapshot.name || "");
  }

  // Coder
  let coderName, coderRole;
  if (!coder) {
    coderName = "(unknown)";
    coderRole = "";
  } else {
    coderName = (coder.name || "").trim() || "(unnamed)";
    coderRole = (coder.role || "").trim();
  }

  // Provenance source
  const rawSource = (provenance.source || "").trim();
  let provenanceSource, provenanceSourceLabel;
  if (!_APPLICATION_PROVENANCE_SOURCES.has(rawSource)) {
    provenanceSource = "";
    provenanceSourceLabel = DEFAULT_PROVENANCE_SOURCE_LABEL;
  } else {
    provenanceSource = rawSource;
    provenanceSourceLabel =
      PROVENANCE_SOURCE_LABELS[rawSource] || DEFAULT_PROVENANCE_SOURCE_LABEL;
  }

  // AI provenance
  const aip = app.ai_provenance ?? app.aiProvenance ?? null;
  const aiPresent = !!aip;
  let aiFeature = "";
  let aiFeatureLabel = "";
  let aiBackend = "";
  let aiGenerationModel = "";
  let aiEmbeddingModel = "";
  let aiSuggestionId = "";
  let aiDecision = "";
  let aiDecisionLabel = "";
  let aiDecidedByCoderId = "";
  let aiDecidedByCoderName = "";
  let aiDecidedAt = "";
  let aiConfidence = "";
  let aiPromptHash = "";
  let aiNotes = "";
  if (aip) {
    aiFeature = String(aip.feature || "");
    aiFeatureLabel = AI_FEATURE_LABELS[aiFeature] || AI_FEATURE_LABELS.other;
    aiBackend = String(aip.backend || "");
    aiGenerationModel = String(aip.generation_model || aip.generationModel || "");
    aiEmbeddingModel = String(aip.embedding_model || aip.embeddingModel || "");
    aiSuggestionId = String(aip.suggestion_id || aip.suggestionId || "");
    aiDecision = String(aip.decision || "");
    aiDecisionLabel = AI_DECISION_LABELS[aiDecision] ?? aiDecision ?? "";
    aiDecidedByCoderId = String(
      aip.decided_by_coder_id || aip.decidedByCoderId || "",
    );
    if (decidedByCoder) {
      aiDecidedByCoderName =
        (decidedByCoder.name || "").trim() || "(unnamed)";
    } else {
      aiDecidedByCoderName = aiDecidedByCoderId ? "(unknown)" : "";
    }
    aiDecidedAt = String(aip.decided_at || aip.decidedAt || "");
    aiConfidence = _formatConfidence(aip.confidence);
    aiPromptHash = String(aip.prompt_hash || aip.promptHash || "");
    aiNotes = String(aip.notes || "");
  }

  // Drift relative to current Code
  const codeMissing = !code;
  let driftedFields = [];
  let definitionDrifted = false;
  if (code && codeVersion) {
    driftedFields = _driftedFields(snapshot, code);
    definitionDrifted = driftedFields.length > 0;
  }

  // Extra free-form provenance keys (sorted, reserved keys filtered out)
  const extraProvenance = [];
  const extraKeys = Object.keys(provenance)
    .filter((k) => !_RESERVED_PROVENANCE_KEYS.has(k))
    .sort();
  for (const k of extraKeys) {
    extraProvenance.push(`${k}: ${provenance[k]}`);
  }

  return {
    applicationId: String(app.id || ""),
    anchorLabel: _formatAnchor(app),
    createdAt: String(app.created_at || app.createdAt || ""),
    modifiedAt: String(app.modified_at || app.modifiedAt || ""),
    confidence: _formatConfidence(app.confidence),
    note: String(app.note || ""),
    codeId,
    codeName,
    codeColour,
    codeStage,
    versionIdAtApply,
    versionNumberAtApply: versionNumber,
    versionRecordedAt,
    versionChangeNote,
    snapshotMissing,
    nameAtApply,
    coderId,
    coderName,
    coderRole,
    sourceId,
    sourceName: String(sourceName || ""),
    provenanceSource,
    provenanceSourceLabel,
    aiPresent,
    aiFeature,
    aiFeatureLabel,
    aiBackend,
    aiGenerationModel,
    aiEmbeddingModel,
    aiSuggestionId,
    aiDecision,
    aiDecisionLabel,
    aiDecidedByCoderId,
    aiDecidedByCoderName,
    aiDecidedAt,
    aiConfidence,
    aiPromptHash,
    aiNotes,
    codeMissing,
    definitionDrifted,
    driftedFields,
    extraProvenance,
  };
}

/**
 * Compact one-line summary, e.g. "Alex · Human-coded · 2026-04-15".
 *
 * Skips empty fields so the order is stable regardless of which
 * relations the caller hydrated.
 */
export function provenanceSummaryLabel(d) {
  const parts = [];
  const name = (d.coderName || "").trim();
  if (name && name !== "(unknown)" && name !== "(unnamed)") {
    parts.push(name);
  }
  parts.push(d.provenanceSourceLabel);
  const date = (d.createdAt || "").slice(0, 10);
  if (date) parts.push(date);
  return parts.join(" · ");
}

/**
 * Multi-line plain-text rendering — suitable for an HTML title= attribute.
 */
export function formatProvenanceText(d) {
  const lines = [];
  let head = `${d.codeName} (${d.codeId})`;
  if (d.versionNumberAtApply) head += ` · ${d.versionNumberAtApply}`;
  lines.push(head);

  const meta = [d.provenanceSourceLabel];
  if (d.createdAt) meta.push(d.createdAt);
  if (d.confidence) meta.push(`confidence ${d.confidence}`);
  lines.push(meta.join(" · "));

  const anchor = [`anchor ${d.anchorLabel}`];
  if (d.sourceName) anchor.push(`source ${d.sourceName}`);
  else if (d.sourceId) anchor.push(`source ${d.sourceId}`);
  lines.push(anchor.join(" · "));

  const coder = [`by ${d.coderName || "(unknown)"}`];
  if (d.coderRole) coder.push(d.coderRole);
  lines.push(coder.join(" · "));

  if (d.snapshotMissing && d.versionIdAtApply) {
    lines.push("");
    lines.push("Definition snapshot at apply not found.");
  } else if (d.definitionDrifted && !d.codeMissing) {
    lines.push("");
    lines.push(
      `Definition has changed since apply (${d.driftedFields.join(", ")}).`,
    );
  }

  if (d.aiPresent) {
    lines.push("");
    const aiHead = [
      d.aiFeatureLabel,
      d.aiBackend,
      d.aiGenerationModel,
      d.aiDecisionLabel,
    ].filter(Boolean);
    lines.push("AI: " + aiHead.join(" · "));
    if (d.aiDecidedByCoderName && d.aiDecidedByCoderName !== "(unknown)") {
      const extra = [`decided by ${d.aiDecidedByCoderName}`];
      if (d.aiDecidedAt) extra.push(d.aiDecidedAt);
      lines.push(extra.join(" · "));
    } else if (d.aiDecidedAt) {
      lines.push(`decided at ${d.aiDecidedAt}`);
    }
    if (d.aiConfidence) lines.push(`AI confidence ${d.aiConfidence}`);
    if (d.aiPromptHash) lines.push(`prompt ${d.aiPromptHash}`);
  }

  if (d.extraProvenance && d.extraProvenance.length) {
    lines.push("");
    for (const line of d.extraProvenance) lines.push(line);
  }

  if (d.note) {
    lines.push("");
    lines.push("Note:");
    const noteLines = d.note.split(/\r?\n/);
    if (noteLines.length === 0) lines.push(d.note);
    else for (const ln of noteLines) lines.push(ln);
  }

  return lines.join("\n");
}

/**
 * Compact escaped HTML rendering for a hover popover.
 *
 * Returns a single ``<div class="provenance-display">…</div>``. All
 * user-supplied values are escaped via ``escapeHtml`` so the result
 * can be written safely into ``innerHTML``.
 */
export function formatProvenanceHtml(d) {
  const parts = [];
  parts.push('<div class="provenance-display">');

  let title = escapeHtml(d.codeName);
  if (d.versionNumberAtApply) {
    title +=
      ' <span class="provenance-version">' +
      escapeHtml(d.versionNumberAtApply) +
      "</span>";
  }
  if (d.codeColour) {
    title =
      '<span class="provenance-swatch" style="background:' +
      escapeHtml(d.codeColour) +
      '"></span>' +
      title;
  }
  parts.push('<div class="provenance-title">' + title + "</div>");

  parts.push(
    '<div class="provenance-source">' +
      escapeHtml(d.provenanceSourceLabel) +
      "</div>",
  );

  const rows = [];
  rows.push(["Anchor", escapeHtml(d.anchorLabel)]);
  if (d.sourceName) rows.push(["Source", escapeHtml(d.sourceName)]);
  else if (d.sourceId) rows.push(["Source", escapeHtml(d.sourceId)]);
  if (d.coderName) {
    let coderHtml = escapeHtml(d.coderName);
    if (d.coderRole) {
      coderHtml +=
        ' <span class="provenance-role">' +
        escapeHtml(d.coderRole) +
        "</span>";
    }
    rows.push(["By", coderHtml]);
  }
  if (d.createdAt) rows.push(["Applied", escapeHtml(d.createdAt)]);
  if (d.confidence) rows.push(["Confidence", escapeHtml(d.confidence)]);

  if (rows.length) {
    parts.push('<dl class="provenance-meta">');
    for (const [k, v] of rows) {
      parts.push("<dt>" + escapeHtml(k) + "</dt><dd>" + v + "</dd>");
    }
    parts.push("</dl>");
  }

  if (d.snapshotMissing && d.versionIdAtApply) {
    parts.push(
      '<div class="provenance-warn">' +
        escapeHtml("Definition snapshot at apply not found.") +
        "</div>",
    );
  } else if (d.definitionDrifted && !d.codeMissing) {
    parts.push(
      '<div class="provenance-drift">' +
        escapeHtml(
          "Definition has changed since apply: " +
            d.driftedFields.join(", "),
        ) +
        "</div>",
    );
  }

  if (d.aiPresent) {
    parts.push('<div class="provenance-ai">');
    parts.push(
      '<div class="provenance-ai-head">' +
        escapeHtml("AI: " + d.aiFeatureLabel) +
        "</div>",
    );
    const aiRows = [];
    if (d.aiBackend) aiRows.push(["Backend", escapeHtml(d.aiBackend)]);
    if (d.aiGenerationModel)
      aiRows.push(["Model", escapeHtml(d.aiGenerationModel)]);
    if (d.aiEmbeddingModel)
      aiRows.push(["Embeddings", escapeHtml(d.aiEmbeddingModel)]);
    if (d.aiDecisionLabel)
      aiRows.push(["Decision", escapeHtml(d.aiDecisionLabel)]);
    if (d.aiDecidedByCoderName && d.aiDecidedByCoderName !== "(unknown)") {
      aiRows.push(["Decided by", escapeHtml(d.aiDecidedByCoderName)]);
    }
    if (d.aiDecidedAt) aiRows.push(["Decided at", escapeHtml(d.aiDecidedAt)]);
    if (d.aiConfidence)
      aiRows.push(["AI confidence", escapeHtml(d.aiConfidence)]);
    if (d.aiPromptHash) aiRows.push(["Prompt", escapeHtml(d.aiPromptHash)]);
    if (aiRows.length) {
      parts.push('<dl class="provenance-meta">');
      for (const [k, v] of aiRows) {
        parts.push("<dt>" + escapeHtml(k) + "</dt><dd>" + v + "</dd>");
      }
      parts.push("</dl>");
    }
    if (d.aiNotes) {
      parts.push(
        '<div class="provenance-ai-notes">' +
          escapeHtml(d.aiNotes) +
          "</div>",
      );
    }
    parts.push("</div>");
  }

  if (d.extraProvenance && d.extraProvenance.length) {
    parts.push('<dl class="provenance-extra">');
    for (const line of d.extraProvenance) {
      const idx = line.indexOf(":");
      if (idx >= 0) {
        const k = line.slice(0, idx).trim();
        const v = line.slice(idx + 1).trim();
        parts.push(
          "<dt>" + escapeHtml(k) + "</dt><dd>" + escapeHtml(v) + "</dd>",
        );
      } else {
        parts.push("<dt></dt><dd>" + escapeHtml(line) + "</dd>");
      }
    }
    parts.push("</dl>");
  }

  if (d.note) {
    parts.push('<div class="provenance-note">');
    parts.push(
      '<div class="provenance-note-head">' + escapeHtml("Note") + "</div>",
    );
    const noteLines = d.note.split(/\r?\n/).map((ln) => escapeHtml(ln));
    parts.push(
      '<div class="provenance-note-body">' +
        (noteLines.length ? noteLines.join("<br>") : escapeHtml(d.note)) +
        "</div>",
    );
    parts.push("</div>");
  }

  parts.push("</div>");
  return parts.join("");
}


// ---------- Query builder (F3.5) ----------

// Map the query-builder form's selections to a `scribe.query.Query`
// payload. Lives in helpers (not inline in the template) so the
// translation layer is unit-testable and so any other surface
// (saved-queries page F3.7, matrix views F3.6) can build the same
// shape without re-implementing the boolean algebra.
//
// Inputs:
//   * projectId        — required, 12-hex.
//   * codeIds          — array of 12-hex code ids; multi-select.
//                        zero    → no codes filter.
//                        one     → leaf CodeExpr ({op: "code", code_id}).
//                        many    → "or" combinator over leaves.
//                        ("and" / "not" combinators belong to a future
//                        UI; this helper exposes only OR for now,
//                        matching the multi-select control's mental
//                        model.)
//   * sourceIds        — array of 12-hex source ids.
//   * speakerRole      — empty string or one of SPEAKER_ROLES.
//   * speakerLabels    — optional array; non-empty appends to
//                        the SpeakerFilter's labels list.
//   * speakerParticipantIds — optional array; non-empty appends
//                        to the SpeakerFilter's participant_ids list.
//   * proximity        — optional object {scope, requiredCodeIds, maxGap};
//                        omitted when scope is empty / requiredCodeIds is empty.
//
// Returns a plain object suitable for JSON.stringify into the body
// of `POST /api/projects/<pid>/queries/run` under the `query` key.
export function buildQueryPayload({
  projectId,
  codeIds = [],
  sourceIds = [],
  speakerRole = "",
  speakerLabels = [],
  speakerParticipantIds = [],
  proximity = null,
} = {}) {
  if (!projectId) {
    throw new Error("buildQueryPayload: projectId is required");
  }
  const out = { project_id: projectId };

  // Sources filter — only emitted when at least one id is selected.
  if (sourceIds && sourceIds.length) {
    out.sources = { source_ids: sourceIds.slice() };
  }

  // Speaker filter — combine role + labels + participant_ids into
  // one SpeakerFilter object. The pure executor matches a label /
  // role / participant_id with OR semantics across the three lists.
  const roles = speakerRole ? [speakerRole] : [];
  if (roles.length || speakerLabels.length || speakerParticipantIds.length) {
    out.speakers = {
      roles,
      labels: speakerLabels.slice(),
      participant_ids: speakerParticipantIds.slice(),
    };
  }

  // Codes filter — single id → leaf, many → OR combinator.
  if (codeIds && codeIds.length === 1) {
    out.codes = { expr: { op: "code", code_id: codeIds[0] } };
  } else if (codeIds && codeIds.length > 1) {
    out.codes = {
      expr: {
        op: "or",
        children: codeIds.map(cid => ({ op: "code", code_id: cid })),
      },
    };
  }

  // Proximity — emitted only when both scope + required_code_ids
  // are non-trivial. ProximityFilter.is_empty() returns True when
  // required_code_ids is empty, so we mirror that here.
  if (proximity
      && proximity.scope
      && Array.isArray(proximity.requiredCodeIds)
      && proximity.requiredCodeIds.length) {
    out.proximity = {
      scope: proximity.scope,
      required_code_ids: proximity.requiredCodeIds.slice(),
      max_gap: typeof proximity.maxGap === "number" ? proximity.maxGap : 0,
    };
  }

  return out;
}

// Group a flat list of applications by source_id. Used by the
// queries page to render results grouped by source. Returns a
// Map<sid, Array<app>> preserving the input order both within
// groups and across groups (first-seen source comes first).
export function groupApplicationsBySource(apps) {
  const out = new Map();
  if (!Array.isArray(apps)) return out;
  for (const a of apps) {
    if (!a || typeof a !== "object") continue;
    const sid = a.source_id || "";
    if (!out.has(sid)) out.set(sid, []);
    out.get(sid).push(a);
  }
  return out;
}


// ---------- F3.6 matrix payload builder ----------
//
// Mirror of `scribe.server`'s POST /api/projects/<pid>/matrices/run
// body shape. Pulled into helpers.mjs (rather than living in the
// queries.html template) so the JS-side translation is unit-tested
// against the same vocabulary the Python route accepts:
//
//   * kind            — required; one of "code-by-source",
//                       "code-by-code", "code-by-attribute".
//   * scope           — only emitted when kind === "code-by-code"
//                       AND scope is non-empty. Default on the server
//                       side is "source"; the form sends what the
//                       user picked so the round-trip is explicit.
//   * maxGap          — only emitted when kind === "code-by-code"
//                       AND scope === "paragraph" AND a number > 0.
//   * attributeKey    — required for kind === "code-by-attribute";
//                       trimmed.
//   * attributeKind   — only emitted when kind === "code-by-attribute";
//                       defaults to "source".
//   * includeMissing  — only emitted when kind === "code-by-attribute";
//                       defaults to true.
//   * compact         — defaults to true on the server; emitted only
//                       when explicitly false.
//   * query           — optional Query payload to pre-filter
//                       applications via the F3.5 executor.
//
// Returns a plain object suitable for JSON.stringify into the body of
// `POST /api/projects/<pid>/matrices/run`.
export function buildMatrixPayload({
  kind,
  scope = "",
  maxGap = 0,
  attributeKey = "",
  attributeKind = "source",
  includeMissing = true,
  compact = true,
  query = null,
} = {}) {
  if (!kind) {
    throw new Error("buildMatrixPayload: kind is required");
  }
  const allowed = new Set([
    "code-by-source", "code-by-code", "code-by-attribute",
  ]);
  if (!allowed.has(kind)) {
    throw new Error(`buildMatrixPayload: unknown kind ${kind}`);
  }

  const out = { kind };

  if (kind === "code-by-code") {
    if (scope) out.scope = scope;
    if (scope === "paragraph"
        && typeof maxGap === "number"
        && maxGap > 0
        && Number.isFinite(maxGap)) {
      out.max_gap = maxGap;
    }
  }

  if (kind === "code-by-attribute") {
    const k = String(attributeKey || "").trim();
    if (!k) {
      throw new Error(
        "buildMatrixPayload: attributeKey is required for code-by-attribute",
      );
    }
    out.attribute_key = k;
    out.attribute_kind = attributeKind || "source";
    out.include_missing = !!includeMissing;
  }

  if (compact === false) {
    out.compact = false;
  }

  if (query && typeof query === "object") {
    out.query = query;
  }

  return out;
}


// Build the GET URL for the F6.3 matrix-export endpoint
//   GET /api/projects/<projectId>/matrices/<kind>/export?format=...&...
//
// Mirrors `buildMatrixPayload` so the form on the queries page can
// drive both endpoints without re-deriving its state. The kind goes
// into the path; everything else is a query-string parameter so a
// browser can navigate to the URL directly (or assign it to an
// `<a download>` href).
//
// Inputs (all optional unless flagged):
//   * projectId      — REQUIRED.
//   * kind           — REQUIRED. One of code-by-source / code-by-code /
//                      code-by-attribute.
//   * format         — REQUIRED. csv | xlsx (case-insensitive; aliases
//                      excel / spreadsheet / xls accepted by the
//                      server but we don't pre-translate them here).
//   * scope          — only emitted when kind === "code-by-code".
//   * maxGap         — only emitted when kind === "code-by-code"
//                      AND scope === "paragraph" AND a finite > 0.
//   * attributeKey   — REQUIRED for kind === "code-by-attribute".
//   * attributeKind  — defaults to "source" for code-by-attribute.
//   * includeMissing — defaults to true for code-by-attribute.
//   * compact        — defaults to true on the server; emitted only
//                      when explicitly false.
//   * useTitles      — defaults to true on the server; emitted only
//                      when explicitly false.
//   * includeTotals  — defaults to true on the server; emitted only
//                      when explicitly false.
//
// Returns the full path-and-query URL as a string. Throws
// `Error` for missing or unknown kind / format / projectId, and for
// missing attributeKey when kind === "code-by-attribute".
export function buildMatrixExportUrl({
  projectId,
  kind,
  format,
  scope = "",
  maxGap = 0,
  attributeKey = "",
  attributeKind = "source",
  includeMissing = true,
  compact = true,
  useTitles = true,
  includeTotals = true,
} = {}) {
  if (!projectId || typeof projectId !== "string") {
    throw new Error("buildMatrixExportUrl: projectId is required");
  }
  if (!kind) {
    throw new Error("buildMatrixExportUrl: kind is required");
  }
  const allowedKinds = new Set([
    "code-by-source", "code-by-code", "code-by-attribute",
  ]);
  if (!allowedKinds.has(kind)) {
    throw new Error(`buildMatrixExportUrl: unknown kind ${kind}`);
  }
  const fmt = String(format || "").trim().toLowerCase();
  if (!fmt) {
    throw new Error("buildMatrixExportUrl: format is required");
  }
  // Mirror the server-side alias set in
  // scribe.matrix_export.normalise_format so the JS layer can fail
  // fast on a typo without round-tripping to the network.
  const allowedFormats = new Set([
    "csv", "xlsx", "xls", "excel", "spreadsheet",
  ]);
  if (!allowedFormats.has(fmt)) {
    throw new Error(`buildMatrixExportUrl: unknown format ${format}`);
  }

  const params = new URLSearchParams();
  params.set("format", fmt);

  if (kind === "code-by-code") {
    if (scope) params.set("scope", scope);
    if (scope === "paragraph"
        && typeof maxGap === "number"
        && maxGap > 0
        && Number.isFinite(maxGap)) {
      params.set("max_gap", String(maxGap));
    }
  }

  if (kind === "code-by-attribute") {
    const k = String(attributeKey || "").trim();
    if (!k) {
      throw new Error(
        "buildMatrixExportUrl: attributeKey is required for code-by-attribute",
      );
    }
    params.set("attribute_key", k);
    params.set("attribute_kind", attributeKind || "source");
    params.set("include_missing", includeMissing ? "1" : "0");
  }

  if (compact === false) {
    params.set("compact", "0");
  }
  if (useTitles === false) {
    params.set("use_titles", "0");
  }
  if (includeTotals === false) {
    params.set("include_totals", "0");
  }

  return (
    `/api/projects/${encodeURIComponent(projectId)}`
    + `/matrices/${encodeURIComponent(kind)}/export?`
    + params.toString()
  );
}


// Render a Matrix.to_dict() payload as a 2-D array suitable for the
// queries page's <table> render. Pure logic so the row/column shape
// is testable without a DOM.
//
// Returns:
//   {
//     header: [<top-left>, ...colTitles],
//     body:   [[<rowTitle>, ...cellValues], ...],
//     rowTotals: [<int>, ...],     // per row
//     colTotals: [<int>, ...],     // per col
//     grandTotal: <int>,
//   }
//
// `top-left` is the matrix's row_label (e.g. "Code"). Cell values
// fall back to 0 when missing — Matrix.to_dict() omits zero cells.
export function matrixToTable(payload) {
  if (!payload || typeof payload !== "object") {
    return { header: [], body: [], rowTotals: [], colTotals: [], grandTotal: 0 };
  }
  const rows = Array.isArray(payload.rows) ? payload.rows.slice() : [];
  const cols = Array.isArray(payload.cols) ? payload.cols.slice() : [];
  const rowTitles = payload.row_titles || {};
  const colTitles = payload.col_titles || {};
  const rawCells = Array.isArray(payload.cells) ? payload.cells : [];

  // Build a sparse lookup: "<r>::<c>" → value.
  const lookup = new Map();
  for (const triple of rawCells) {
    if (!Array.isArray(triple) || triple.length !== 3) continue;
    const [r, c, v] = triple;
    lookup.set(`${r}::${c}`, Number(v) || 0);
  }

  const cell = (r, c) => lookup.get(`${r}::${c}`) || 0;

  const header = [String(payload.row_label || ""),
                  ...cols.map(c => String(colTitles[c] || c))];

  const body = rows.map(r => {
    const rowKey = String(rowTitles[r] || r);
    return [rowKey, ...cols.map(c => cell(r, c))];
  });

  const rowTotals = rows.map(r => cols.reduce((s, c) => s + cell(r, c), 0));
  const colTotals = cols.map(c => rows.reduce((s, r) => s + cell(r, c), 0));
  const grandTotal = rowTotals.reduce((s, v) => s + v, 0);

  return { header, body, rowTotals, colTotals, grandTotal };
}


// ---------- F3.7 saved-query payload builders ----------
//
// The F3.7 saved-queries surface re-uses the F3.5 query payload
// (`buildQueryPayload` above) and wraps it with a name + optional
// description so the resulting POST body matches what
// `scribe.server.create_saved_query_endpoint` expects:
//
//   {
//     "query": <buildQueryPayload(...)>,
//     "name":  "Quotes about power",
//     "description": "Optional notes about this query"
//   }
//
// `buildSavedQueryPayload` is the create/PATCH shape; it validates
// the name (required, non-blank) before emitting. Pulled into
// helpers.mjs so the page-side wiring is unit-tested rather than
// crafted ad-hoc inside the template.
export function buildSavedQueryPayload({
  query,
  name = "",
  description = "",
} = {}) {
  if (!query || typeof query !== "object") {
    throw new Error("buildSavedQueryPayload: query is required");
  }
  const trimmed = String(name || "").trim();
  if (!trimmed) {
    throw new Error("buildSavedQueryPayload: name is required");
  }
  const out = {
    query: query,
    name: trimmed,
  };
  const desc = String(description || "");
  if (desc) {
    out.description = desc;
  }
  return out;
}


// Format a SavedQuery's run-tracking metadata into a single short
// string for the saved-queries list. Handles the never-run case
// ("never run") and the standard ISO timestamp ("last run … · 3 ×").
//
// `nowFn` is injectable so the test suite can pin `Date.now()` and
// assert "5 minutes ago" style strings without timing flake.
export function formatSavedQueryRunSummary(sq, { nowFn } = {}) {
  if (!sq || typeof sq !== "object") return "";
  const count = Number(sq.run_count || 0);
  const last = String(sq.last_run_at || "");
  if (!last) {
    return count > 0
      ? `${count} run${count === 1 ? "" : "s"}`
      : "never run";
  }
  // Best-effort parse — saved queries store ISO-8601 UTC strings.
  let when = last;
  try {
    const d = new Date(last);
    if (!isNaN(d.getTime())) {
      const now = (typeof nowFn === "function") ? nowFn() : Date.now();
      const dMs = now - d.getTime();
      const sec = Math.round(dMs / 1000);
      if (sec < 60) when = "just now";
      else if (sec < 3600) when = `${Math.round(sec / 60)} min ago`;
      else if (sec < 86400) when = `${Math.round(sec / 3600)} h ago`;
      else when = `${Math.round(sec / 86400)} d ago`;
    }
  } catch (_) { /* keep raw ISO */ }
  return `last run ${when} · ${count} ×`;
}


// ----------------------------------------------------------------------
// F6.7 — Anonymised export rule parser.
//
// The project-settings page (project_settings.html) lets researchers
// supply optional custom redaction rules alongside the participants'
// pseudonyms before POSTing to /api/projects/<pid>/qdpx/anonymised.
// Users type one rule per line in the format:
//
//   pattern => replacement           // literal substring rule
//   re:pattern => replacement        // regex rule
//
// This helper turns that newline-delimited text into the JSON shape
// the server endpoint accepts:
//
//   [{ pattern, replacement, regex? }]
//
// Returns ``{ rules, error }``. ``error`` is ``null`` on success and a
// human-readable string on failure (so the page can show it inline);
// ``rules`` is empty when ``error`` is non-null. Empty / whitespace-only
// lines are skipped without error. The same logic lives inline in the
// template's non-module script so the UI doesn't have to convert to
// ``type="module"``; this is the canonical, Vitest-covered source.
// ----------------------------------------------------------------------
export function parseAnonymisedRulesText(raw) {
  if (raw == null) return { rules: [], error: null };
  if (typeof raw !== "string") {
    return { rules: [], error: "input must be a string" };
  }
  const out = [];
  const lines = raw.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const idx = line.indexOf("=>");
    if (idx < 0) {
      return {
        rules: [],
        error: `line ${i + 1}: expected "pattern => replacement"`,
      };
    }
    let pattern = line.slice(0, idx).trim();
    const replacement = line.slice(idx + 2).trim();
    if (!pattern) {
      return { rules: [], error: `line ${i + 1}: empty pattern` };
    }
    let regex = false;
    if (pattern.startsWith("re:")) {
      regex = true;
      pattern = pattern.slice(3).trim();
      if (!pattern) {
        return {
          rules: [],
          error: `line ${i + 1}: empty regex pattern after "re:"`,
        };
      }
    }
    const rule = { pattern, replacement };
    if (regex) rule.regex = true;
    out.push(rule);
  }
  return { rules: out, error: null };
}
