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
