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
