# Constructivist Grounded Theory in Scribe — Research Report

This is the foundational deep-research pass that informed the academic coding engine section of `PLANNING.md`. Written 2026-05-25. Sources at the bottom.

## 1. Constructivist Grounded Theory (Charmaz) — Coding Stages and Workflow

Grounded theory (GT) was originated by Glaser & Strauss (1967), then split: Glaser stayed with a "discovery / emergence" stance; Strauss & Corbin (1990, 1998) added more structure (the famous "axial coding paradigm" of conditions/actions/consequences); Charmaz (2006, 2nd ed. 2014, *Constructing Grounded Theory*) reframed the whole enterprise on **constructivist / pragmatist epistemology**. For Charmaz, codes and theory are **constructed** through interaction between researcher and data, not "discovered" lying inside it. A few methodological consequences flow from that, and they all matter for tooling:

- The researcher's interpretive voice is *part of the analysis*, not noise to be filtered out. So the tool must surface **who** coded what and **why** (memos), not just final labels.
- Codes are **provisional and revisable** at every stage. The tool must treat code identity, code definition, and code application as three independently versioned things.
- Reflexivity matters: memos about the researcher's own assumptions sit alongside memos about the data.

### Charmaz's coding stages

**1. Initial coding (a.k.a. open coding)**
- *Line-by-line* coding for early transcripts: the researcher attaches a short code (often gerund-form — "managing pain," "avoiding the topic") to each line or each meaningful unit. Charmaz strongly favours gerunds because they keep codes close to action and process rather than topic.
- *Incident-by-incident* coding for field notes or observation data, where lines are not the natural unit.
- Codes at this stage are **provisional, fragmentary, in-vivo-friendly** (sometimes a code is literally a participant phrase in quotes). Many will never survive.
- Constant comparison begins immediately: data ↔ data, data ↔ code.
- **Artifacts produced:** a long, messy list of dozens-to-hundreds of initial codes per transcript; coded text segments; early memos noting puzzles ("why does she keep apologising before disagreeing?").

**2. Focused coding**
- The researcher takes the most **frequent, salient, or analytically promising** initial codes and uses them to re-code larger chunks of data across transcripts.
- Codes are consolidated, renamed, sharpened. Some initial codes get merged, others promoted to **categories**.
- This is where the codebook stops being a dump and starts being a structured taxonomy.
- **Artifacts:** a smaller set of focused codes / candidate categories; expanded code definitions; comparison memos; a tentative hierarchy.

**3. Axial coding — Charmaz's specific stance**
- Strauss & Corbin made axial coding a **mandatory** stage with a rigid paradigm (causal conditions, context, intervening conditions, action/interaction, consequences). Glaser called this "forcing" the data.
- Charmaz takes a middle position: she treats axial coding as **optional and informal**. She'll relate categories to subcategories and look for properties and dimensions, but she explicitly resists the S/C paradigm template. *Constructing Grounded Theory* says some researchers will benefit from axial coding's structure; others will find it constraining.
- **Tooling implication:** don't build the S/C paradigm as a hard schema. Support relating codes to codes (parent/child, sibling, "is property of," "is dimension of") *optionally*.

**4. Theoretical coding**
- Drawn from Glaser: the researcher specifies how the substantive codes/categories relate to each other to form a coherent theoretical story. Glaser provides "coding families" (e.g. the Six Cs: Causes, Contexts, Contingencies, Consequences, Covariances, Conditions; process families; type families; etc.).
- Charmaz uses theoretical codes more loosely — as *one* sensitising resource, not a checklist.
- **Artifacts:** a small set of core categories; a theoretical narrative / model; conceptual diagrams; integrative memos.

**5. Memo writing — runs through every stage**
- Charmaz treats memo-writing as the engine of the method, not an accessory. There are several memo types in practice:
  - **Code memos** (a.k.a. operational memos): definition, when to apply, when *not* to apply, examples — basically the codebook entry for one code.
  - **Theoretical memos**: ideas about how codes relate, possible categories, hunches.
  - **Reflexive memos**: the researcher's own assumptions, emotional reactions, biases.
  - **Methodological / decision memos**: "I split code X into X1 and X2 on 14 March because…"
- Memos are sorted, clustered, and integrated late in the process to scaffold the written-up theory.

**6. Theoretical sampling**
- Once categories start forming, the researcher decides *who or what to interview next* in order to fill out the properties of those categories — not for representativeness. E.g. if "managing disclosure" emerges from younger participants, sample older ones to test the category's range.
- **Tooling implication:** the project must connect transcripts to participant metadata, and let the researcher annotate why a transcript was included.

**7. Constant comparison**
- At every stage: data ↔ data, data ↔ code, code ↔ code, category ↔ category. The researcher continually asks "is this the same phenomenon as that?" "What's different?"
- **Tooling implication:** "show me every quote on code X side-by-side" is the workhorse query and must be one click.

**8. Theoretical saturation**
- Sampling and coding stop when new data stops generating new properties of categories — a judgement, not a number. The researcher needs evidence-of-saturation views: per-code growth curves, recent-vs-earlier transcripts, etc.

### Where Charmaz differs from Glaser and Strauss/Corbin (summary)

| Dimension | Glaser | Strauss/Corbin | **Charmaz** |
|---|---|---|---|
| Reality | Objective, discoverable | Objective, structured | Constructed, multiple perspectives ("obdurate reality") |
| Pre-study lit review | Avoid | Encouraged | Permitted, used reflexively |
| Axial coding | No (theoretical coding instead) | Mandatory paradigm | Optional, informal |
| Researcher voice | Minimised | Procedural | Foregrounded, reflexive |
| Coding language | Substantive | Conditional/causal | Gerunds, action-oriented |
| Memos | Core | Core | Core, *and* reflexive |

---

## 2. Codebooks — Structure, Evolution, Reliability

### What a codebook entry actually contains

A robust QDA codebook is **not** a name + count. The fields below are what NVivo, Atlas.ti, MAXQDA, Dedoose and the REFI-QDA Codebook Exchange standard cover, in various combinations. For Scribe, treat this as the canonical entry schema:

- **Code ID** (stable internal UUID — survives renames)
- **Name / label** (short, human-readable; gerund-form encouraged for Charmaz)
- **Definition** (1–3 sentences; the *operational* meaning)
- **Inclusion criteria** ("apply when…")
- **Exclusion criteria** ("do not apply when… use code X instead")
- **Exemplar quote(s)** with provenance (transcript ID + span)
- **Counter-example** (negative case, particularly important for grounded theory)
- **Parent code** (for hierarchies / taxonomies)
- **Related codes** (typed: see-also, contrasts-with, is-property-of, is-dimension-of, co-occurs-with)
- **Theoretical memo** (free-text rationale, link to broader category)
- **Stage** (initial / focused / axial / theoretical / retired)
- **Colour / icon** (for visual coding — heavy use in Quirkos and Atlas.ti)
- **Created by / created at**
- **Last modified by / last modified at**
- **Version / revision history** (definitions change; quotes coded under "old" definition need to remain auditable)
- **Frequency** (count of applications, derived not stored)
- **Sources count** (how many transcripts use it)
- **Status** (active / merged-into / split-into / retired / locked)
- **Provenance** (manually created, AI-suggested-and-accepted, imported from prior project)

### How codebooks evolve across stages

- **Initial coding** → flat list, hundreds of codes, most barely defined, many in-vivo. Codebook is closer to a junk drawer.
- **Focused coding** → merges and renames begin. The tool must preserve the link from old code → new code so old applications still work and the audit trail stays intact.
- **Axial / theoretical** → hierarchy emerges. Categories become parent codes; properties and dimensions become child codes; relationships between categories are explicit links.
- **Locked codebook** is common late in projects, especially for inter-coder reliability (ICR) work: "from this date, no new codes; we test agreement on the existing set."

### Code lifecycle operations

- **Merge** — combine code A into code B. All segments coded A become coded B (or both, until reconciled). Both code A's definition and history are preserved as a "merged-into" pointer.
- **Split** — split code A into A1 and A2 by re-reviewing each segment and assigning to one (or both, or neither). The tool must walk the researcher through every existing application.
- **Rename** — display name changes; ID does not. History records the old name.
- **Retire** — code is hidden from active use but applications remain; can be revived.
- **Promote / demote** in hierarchy — affects parent links only.

### Inter-coder reliability and reconciliation

- **Cohen's kappa** is the standard agreement statistic for two coders on a fixed set of segments and codes. Krippendorff's alpha is preferred for ≥3 coders or interval data.
- **Process:** divide a sample of transcripts → each coder codes independently using the locked codebook → tool computes per-code and overall agreement → coders meet to reconcile disagreements → either codebook gets clarified (definitions sharpened), the disagreement is resolved by consensus, or split.
- Charmaz herself is sceptical of ICR as a goodness-of-fit measure — she sees coding as interpretive, so disagreement is *informative*, not failure. Tools should support both quantitative ICR *and* qualitative reconciliation memos.

---

## 3. Multi-Transcript Projects with a Shared Codebook

A QDA project is not a transcript with codes — it's a **corpus** with shared analytic structure on top.

### Project-level metadata to model

- **Research question(s)** (free text; can evolve; versioned)
- **Theoretical sensitising concepts** (Blumer's term Charmaz uses heavily — concepts the researcher brings in that *suggest* directions but do not predetermine codes; should be visible distinctly from codes)
- **Methodology declaration** (e.g. "constructivist GT, Charmaz 2014") — drives UI defaults like gerund-encouragement.
- **Participants / sources table** with arbitrary demographic and contextual columns (age, gender, role, recruitment site, interview date, language, mode (in-person / video / phone)).
- **Source types**: interview transcript, focus group transcript (multi-speaker), field notes, document/artifact, image (out of scope for Scribe v1, probably).
- **Sampling log**: which sources were added when, why (theoretical-sampling justification), what category they were meant to fill.
- **Coding sessions**: bounded coding episodes with date, coder, codes used.

### Cross-transcript queries researchers actually run

This list is from MAXQDA / NVivo / Atlas.ti documentation and what active researchers post about needing:

1. *"Show every quote coded with X."* (the bread-and-butter "coding report")
2. *"…across all participants who are female and over 50."* (filtered coding report)
3. *"…and grouped by participant."* (matrix view: code × source)
4. *"Show segments where X **and** Y co-occur."* (boolean; same span or overlapping span)
5. *"Show segments where X appears **near** Y within the same transcript."* (proximity, e.g. within N paragraphs)
6. *"Show segments coded X by participants who were **not** coded with Y anywhere in the transcript."* (case-level filter)
7. *"Compare frequency of code X across age groups."* (cross-tab / matrix coding query)
8. *"Show every quote where the speaker is the interviewee, coded X."* (speaker-aware, important for focus groups and interviews where the interviewer's words shouldn't pollute counts)
9. *"What codes co-occur most with X across the corpus?"* (co-occurrence matrix → drives axial coding)
10. *"Show the timeline of when X was applied."* (audit / saturation curve)

---

## 4. Highlighting and Selection

### Selection model

- **Granularity:** arbitrary character spans. Word, sentence, paragraph are useful presets ("snap to sentence") but the underlying model must be character-offset based or token-offset based. Researchers do code mid-sentence ("…and then he said *I felt small*…").
- **Multi-segment code application:** one code applied to N non-contiguous spans within or across transcripts. Each application is an instance with its own location.
- **Overlapping coding:** multiple codes on the same span is the **norm**, not an edge case. A 30-word participant utterance routinely picks up 4–6 codes (topic, emotion, action, in-vivo phrase, reflexive note).
- **Nested coding:** code A spans 100 words; code B spans 20 words inside A. Both need to be visible and editable.
- **Visual representation:** Atlas.ti and Quirkos use coloured bars in a margin gutter; NVivo uses highlights of the text itself with a sidebar. The gutter approach scales to many overlapping codes; in-text highlights stop being readable past ~3 layers.

### Anchoring — the hard problem

**Researchers edit transcripts after coding** (typo fixes, anonymisation, retranscription of unclear sections). Codes anchored by raw character offsets break when the underlying text shifts. Three approaches:

1. **Token-anchored** (preferred): assign stable IDs to tokens (or words) at import; selections refer to (start_token_id, end_token_id, optional left/right character offsets within those tokens). Edits within the span shift offsets but the anchor survives. Edits that delete the anchor explicitly mark the application as "orphaned, needs review."
2. **Fuzzy text-anchored:** store the selected text and ±N characters of context, re-locate by best match on edit. Cheap to implement, fragile under heavy editing.
3. **Versioned transcript with diff-rebase:** maintain transcript revisions, rebase code applications on each save. Most robust, most complex.

Scribe likely already has word-level objects from the ASR/Parakeet pipeline — those are the natural stable anchors. **Use word IDs as the canonical anchor for any code application.**

### Speaker / segment awareness

Code applications should know which speaker turn(s) they fall in. This is needed for the speaker-filtered queries above and for audio-back-link (jump to where this was said). Scribe's existing diarised segment IDs are the unit.

---

## 5. Annotations and Memos

### Memo taxonomy (lifted from Charmaz, Glaser, MAXQDA)

- **Code memo / operational memo** — *attached to a code*. Operational definition, application notes, history of decisions. Acts as the long-form codebook entry.
- **Theoretical memo** — *free-floating or attached to one or more codes/categories*. Hunches, emerging relationships, "this category looks like X-as-process and Y-as-outcome."
- **Methodological / decision memo** — *attached to a project event*. "Decided to merge A into B today because…"
- **Reflexive memo** — *researcher-attached, sometimes timestamped to a coding session*. Researcher's own emotional reactions, biases, assumptions surfacing.
- **Quote memo / margin annotation** — *attached to a specific transcript segment*. "Note her hesitation here." "Compare to P07."
- **Source memo** — *attached to a whole transcript or participant*. Interview context, recruitment notes, "she was tired."
- **Project memo** — *project-level*. The research question evolution, sensitising concepts, audit trail headlines.

### Linked vs free-floating

Researchers want both. Free-floating memos at the start of analysis ("I keep noticing apology behaviour") get later linked into a category once that category exists. The tool must support **promoting** a memo into a code definition or attaching it after the fact, not lock memos to one host at creation.

### Clustering memos into theory

Late in a Charmaz project the researcher does **memo sorting**: physically (or in software) arrange memos in space, group, find the storyline. This is what eventually becomes the discussion section of the paper. Tools support this with:

- Drag-arrangeable memo cards (Atlas.ti's "network views," MAXQDA's MAXMaps, Quirkos's bubble canvas).
- Memo→category and memo→memo links.
- Export of all memos in chronological or by-cluster order as a single Word/Markdown file.

---

## 6. Export Formats

Researchers export at every stage, for different audiences.

### What gets exported, and when

| Stage | Export | Audience |
|---|---|---|
| Initial coding | Codebook draft (CSV) | Self / supervisor |
| Initial coding | Per-code segment report | Self / co-coder |
| Focused coding | Codebook RTF/Word with definitions | Supervisor / advisor |
| Focused coding | Frequency matrix (code × source) | Self / co-author |
| Axial / theoretical | Co-occurrence matrix | Self |
| Axial / theoretical | Network diagram (PNG/SVG) | Paper figure |
| Final | All quotes per code, grouped by source | Appendix / paper |
| Final | REFI-QDA project | Archival, hand-off, journal supplemental |
| Any | Single transcript with margin codes | Print review |

### Format details

- **Codebook CSV** — one row per code, columns from §2's field list. Tab- and comma-delimited variants.
- **Codebook RTF / Word** — formatted document with code name as heading, definition, exemplar quote indented, criteria as bullets. This is what gets attached to a thesis appendix.
- **Coded segment report** ("retrieval report") — for each code (or filtered set), every applied segment with source ID, speaker, surrounding context (±1 paragraph), and the quote highlighted. Grouped by code or by source.
- **Frequency matrix** — code × source counts (CSV or XLSX); often pivoted in Excel by the researcher.
- **Co-occurrence matrix** — code × code counts (within span / within paragraph / within source), CSV or XLSX.
- **REFI-QDA / QDPX** — the open interchange format from the Rotterdam Exchange Format Initiative, adopted by Atlas.ti, QDA Miner, Quirkos, Transana initially, and later by Dedoose, MAXQDA, NVivo. **A QDPX file is a ZIP archive** containing:
  - `project.qde` — the main XML file with project metadata, sources, codes, code applications, memos, links, users, sets/cases.
  - `Sources/` — original source files (text, PDF, audio, video) plus plain-text representations.
  - The schema is XML-based; codes have GUID/UUIDs, hierarchies via parent references, code applications reference source by GUID and use plain-text character offsets.
  - Some data loss between tools is expected because no two tools are feature-equivalent.
- **Codebook Exchange (REFI-QDA Codebook)** — a smaller XML format for just the codebook (no applications), useful for sharing taxonomies between projects.
- Per-tool native formats (NVivo `.nvp`/`.nvpx`, Atlas.ti `.atlproj`, Quirkos `.qrx`, MAXQDA `.mx22`/`.mx24`, Dedoose project export `.dedoose`) — Scribe doesn't need to read/write these natively; **REFI-QDA is the lingua franca**.

---

## 7. Commercial QDA Landscape — Feature Mining

### Per-tool snapshot

**NVivo (QSR / Lumivero)** — Windows/macOS, the long-time market leader in academia.
- Strengths: deep query language, matrix coding queries, visualisations (concept maps, hierarchy charts, word trees), mixed-methods integration with quantitative attributes, classifications/cases for participant metadata.
- Weaknesses: heavy desktop app, expensive, steep learning curve, slow, file format lock-in until REFI-QDA. Frequent online complaints about crashes on large projects, painful collaboration model (server edition required), and macOS feature parity gaps.

**Atlas.ti** — early adopter of REFI-QDA, broad OS support including iOS/Android/Cloud.
- Strengths: network views (semantic graph of codes and quotations), strong multimedia (audio, video, PDF, images), AI coding ("Intentional AI Coding") that uses the *user's* codebook to suggest applications.
- Weaknesses: cloud version splits the experience; complex UI; expensive; some researchers complain the AI features push them toward inductive shortcutting.

**MAXQDA (VERBI)** — Windows/macOS, popular in Europe and mixed-methods.
- Strengths: Statistics module, extensive visualisations, strong document side-by-side comparison, "Creative Coding" memo-sorting canvas (MAXMaps), MAXDictio for word-level analysis.
- Weaknesses: license cost; some complain feature creep clutters the UI.

**Quirkos** — UK, designed deliberately to be friendly.
- Strengths: bubble visualisation of codes, low learning curve, good price, runs on Linux too. Now offers Quirkos Cloud and AI features.
- Weaknesses: text only (no video), less powerful query side, smaller user base.

**Dedoose** — fully web-based, subscription.
- Strengths: cheap, accessible from anywhere, built for collaborative teams, mixed-methods, good for student work and dissertations.
- Weaknesses: web-only is a privacy and reliability worry — outages have hit researchers hard; not suitable for sensitive data under most IRB regimes; limited offline use. Lots of complaints about export quality.

**Taguette** — open source (BSD), Python/Tornado-based, runs locally or on a server.
- Strengths: free, simple, fast, exports to CSV/HTML/DOCX, runs on a laptop, no lock-in.
- Weaknesses: text-only, no co-occurrence matrices, no visualisations, basic codebook (name + description), no memos beyond highlights, no AI features. **It's the closest existence-proof of what Scribe could be in v1, and a useful low bar to clear.**

### Table-stakes vs nice-to-have (synthesised from forums, reviews, training docs)

**Table stakes** (researchers will reject the tool without these):
- Reliable highlighting with overlapping multi-code support
- Codebook with definitions and quotes
- "Show me all quotes for code X" report
- CSV/Word export of codebook and coded segments
- Memos linked to codes and quotes
- Find/search across transcripts
- Audit trail of changes (especially for theses requiring methodological transparency)

**Strongly expected**:
- Code hierarchy / parent-child
- Code merge/rename without breaking history
- Frequency matrix code × source
- Multiple sources in one project, source-level metadata
- Speaker awareness for interview transcripts
- REFI-QDA export (post-2020 expectation)

**Nice-to-have**:
- Co-occurrence matrix
- Network / concept-map visualisation
- ICR statistics
- Multi-coder collaboration
- Audio/video alignment (Scribe already has this — it's a differentiator)
- AI suggestions
- Memo-sorting canvas

**Recurring complaints to avoid**:
- File-format lock-in (NVivo)
- Cloud-only with sensitive data (Dedoose)
- Slow on large projects (NVivo, Atlas.ti)
- Painful collaboration via shared files (everything except true cloud tools)
- Expensive licenses ($$$ NVivo, MAXQDA, Atlas.ti)
- AI features that feel like "category prediction" rather than "code suggestion"

**Scribe's natural angle**: local-first (privacy), audio-grounded, AI-as-suggester-not-oracle, REFI-QDA from day one, free-or-cheap.

---

## 8. The AI-Suggestion Angle

### What researchers actually want from AI coding suggestions

Surveys, blog posts and the published methodological discussion converge on a small list:

1. **Suggest existing codes from the codebook for a highlighted span.** This is by far the most-asked feature. The researcher has done the conceptual work; they want the AI to spot pattern-matches they'd otherwise miss because of fatigue, especially in transcript 14 of 30. Atlas.ti calls this "Intentional AI Coding."
2. **Suggest *new* codes for a span only when nothing fits.** Researchers are wary here: AI-named codes drift toward generic, topic-flavoured labels ("emotion," "family"), away from Charmaz-style action gerunds ("hiding from family"). They want the suggestion phrased *as a candidate* not applied silently.
3. **Cluster similar uncoded quotes** to surface candidates for new codes — bottom-up, like grounded theory itself.
4. **Find quotes similar to this one** across the whole corpus — the embedding-driven "more like this" feature. This is the most uncontroversial AI use because it doesn't *judge*, it just retrieves.
5. **Identify segments likely missed under code X** given the existing exemplars — a recall booster.
6. **Memo prompts** — given a code's exemplars, draft a candidate definition or pose questions ("does this code differ from Y? in what cases?"). The researcher rewrites; the AI just primes the pump.
7. **Second-coder simulation** for ICR sanity checks. Not a substitute for a human co-coder, but a flag for "you applied X here and not in this similar place — was that intentional?"

### The methodological tension

Charmaz herself, in *Constructing Grounded Theory* (2nd ed, 2014) and in her 2020 piece "With Constructivist Grounded Theory You Can't Hide," is **wary of automation**. The constructivist epistemology says coding *is* interpretation; outsourcing it removes the very mechanism by which the researcher comes to understand the data. Reading slowly, line-by-line, *is the analysis* — speeding it up changes what's produced.

The published QDA methods literature (Christou 2023, Morgan 2023, Davidson & di Gregorio's "Five-Level QDA"; recent SAGE *International Journal of Qualitative Methods* papers) circles a few positions:

- **AI-as-shortcut is methodologically incompatible with grounded theory.** If the AI codes for you, the theory is no longer grounded in *your* engagement with the data.
- **AI-as-augmentation can be acceptable** when the researcher (a) has already done substantial hand-coding, (b) reviews every suggestion, (c) records AI provenance for each application, (d) treats AI suggestions as *one voice among many*.
- **AI-for-retrieval is uncontroversial** — semantic search, "similar quotes" — because it doesn't impose categories.
- **AI-for-memo-prompting is generally welcomed** — it spurs reflection rather than replacing it.

The strongest version of the principle: **the AI can suggest, never apply.** Every code application must have a human author. Provenance must record whether a code was AI-suggested-and-accepted, AI-suggested-and-modified, or human-originated. The researcher's audit trail must let an external reader recompute who-did-what.

### Workflows that integrate AI without violating the methodology

1. **Hand-code the first N transcripts the slow way.** AI is *off* until the codebook has shape. This protects the inductive opening of grounded theory.
2. **Switch on suggestion mode for transcripts N+1 onward.** AI proposes codes from the existing codebook for each highlighted span; researcher accepts, modifies, rejects, or codes manually.
3. **Periodic AI review pass** of already-coded transcripts: AI re-codes blind from the codebook, the tool diffs against the human coding, the researcher reviews flagged disagreements. This is the closest analogue to a second human coder.
4. **AI clustering for new-code candidates.** When the researcher feels stuck, run a clustering pass on uncoded or unevenly-coded segments; AI proposes candidate categories with exemplars; researcher decides whether to add to the codebook.
5. **Memo prompts on demand.** "Draft a definition for this code given these 12 exemplars." Researcher edits.
6. **Locked-codebook ICR mode.** AI runs as a second blind coder on a held-out sample; tool reports kappa; researcher decides if the codebook needs sharpening.

### Models that actually run offline

The user wants **fully local** — no cloud. Constraints: consumer GPUs in the 6–24 GB VRAM range, occasionally CPU-only laptops. Two model classes are needed:

#### A. Generative LLMs (for code suggestion, memo drafting, clustering labels)

| Model family | Typical sizes | VRAM (4-bit quant) | Strengths | Notes |
|---|---|---|---|---|
| **Llama 3.1 / 3.2 / 3.3** (Meta) | 3B, 8B, 70B | ~3 GB / ~6 GB / ~40 GB | Strong general reasoning, long context (128k) | 8B fits comfortably on 8 GB GPU; 70B needs workstation-class hardware |
| **Llama 3.2** | 1B, 3B | ~1 GB / ~2.5 GB | Tiny, fast | Useful for instant suggestions on laptop CPU |
| **Mistral 7B / Mixtral 8x7B / Mistral Small / Nemo 12B** | 7B / 47B MoE / 22B / 12B | ~5 GB / ~28 GB / ~14 GB / ~8 GB | European, permissive licensing, strong on European languages | Mistral Nemo 12B is a sweet spot for 16 GB GPUs |
| **Phi-3 / Phi-3.5 / Phi-4** (Microsoft) | 3.8B / 14B | ~2.5 GB / ~9 GB | Surprisingly capable for size, reasoning-focused | Phi-4 14B is excellent on 16 GB consumer cards |
| **Gemma 2 / Gemma 3** (Google) | 2B / 9B / 27B | ~2 GB / ~6 GB / ~17 GB | Strong instruction-following, multilingual | Gemma 3 27B at 4-bit fits 24 GB cards |
| **Qwen 2.5 / Qwen 3** (Alibaba) | 0.5B–72B; 7B / 14B / 32B common | ~5 / ~9 / ~20 GB | Often best-in-class for reasoning at size | Qwen 2.5 14B and 32B are strong picks for QDA-like tasks |
| **DeepSeek R1 distills** | 7B / 14B / 32B | ~5 / ~9 / ~20 GB | Reasoning-tuned | Slower but better at justification |

For Scribe v1 the realistic recommendation is a **tiered model selection**:
- **Default tier (laptop, 8 GB GPU or CPU)**: Llama 3.2 3B or Phi-3.5 3.8B or Gemma 3 4B at 4-bit. Latency ~1–4 s per suggestion on GPU, ~5–15 s on CPU.
- **Mid tier (16 GB GPU)**: Llama 3.1 8B or Mistral Nemo 12B or Phi-4 14B at 4-bit. Latency ~2–6 s.
- **Workstation tier (24 GB GPU)**: Qwen 2.5 32B / Gemma 3 27B / Llama 3.3 70B at heavy quant. Latency ~5–15 s but substantially better category-naming and definition-drafting.

Run via **llama.cpp / Ollama / vLLM** (CPU+GPU, GGUF format). Ollama is the easiest distribution path and matches Scribe's "drop-in local" ethos.

#### B. Embedding models (for semantic similarity, clustering, "more like this")

| Model | Dim | Size | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 33 MB | Tiny, fast; the laptop default |
| `BAAI/bge-base-en-v1.5` | 768 | 110 MB | Solid mid-tier |
| `BAAI/bge-large-en-v1.5` | 1024 | 335 MB | Best of the BGE family |
| `BAAI/bge-m3` | 1024 | 569 MB | Multilingual, multi-granularity, strong default for non-English transcripts |
| `nomic-ai/nomic-embed-text-v1.5` | 768 (Matryoshka) | 137 MB | Truncatable dims, long context (8k tokens), Apache 2.0 |
| `intfloat/e5-large-v2` / `e5-mistral-7b-instruct` | 1024 / 4096 | 335 MB / ~14 GB | Strong, mistral version is heavy |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 335 MB | Excellent quantisation tolerance (96% with binary) |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | 1024 | 568 MB | Multilingual, top of MTEB recently |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` / `gte-large` | up to 8960 / 1024 | larger / 335 MB | Qwen-based, top MTEB scores |
| `Stella` family (`stella_en_1.5B_v5` etc.) | 1024 | ~6 GB | High-end MTEB performer |

Sensible default for Scribe: **`bge-m3`** (multilingual, strong, ~600 MB) or **`nomic-embed-text-v1.5`** (Matryoshka, dim-truncatable, lighter). Both run on CPU at decent speed for the corpus sizes researchers typically have (10–100 transcripts, 5k–50k segments).

### Practical compute cost (rough numbers)

Numbers below assume 4-bit quantised LLMs and fp16 embedding models, modern CPUs, and consumer GPUs (RTX 3060 12 GB / 4070 12 GB as "mid laptop GPU," 4090 24 GB as "workstation"):

- **Embedding the entire corpus once** (e.g. 50 transcripts, ~30k segments): CPU ~5–15 minutes; GPU <1 minute. One-time cost, cached.
- **"Find similar quotes" query**: instant (vector dot-product on cached embeddings).
- **"Suggest codes for this highlighted span"** (LLM):
  - 3B model on GPU: 0.5–2 s
  - 8B model on GPU: 1–4 s
  - 14B model on GPU: 3–8 s
  - 3B model on CPU: 5–15 s
  - 8B model on CPU: 15–60 s — borderline acceptable for "review one suggestion at a time"
- **Whole-transcript pass** (~10k tokens through the LLM, asking for code suggestions per paragraph): minutes, not seconds. Run as a background job with a progress indicator, not interactively.
- **Memo drafting** (LLM, ~500 token output): 5–30 s depending on model and hardware.
- **Clustering** (k-means or HDBSCAN on cached embeddings + LLM-generated cluster labels): seconds for math + a few LLM calls for labels.

### Provenance / trust requirements that flow from AI integration

Every code application must record:
- Author (human user ID, or "AI-suggested-by:<model_name@version>")
- Acceptance event (timestamp + accepting human user) — even AI suggestions only become applications via a human accept
- Definition snapshot of the code at the moment of application (so that later edits to the definition don't invalidate the application)
- Optional confidence score from the model (0–1, displayed as a hint not a gate)

---

## 9. Trust, Reproducibility, and the Audit Trail

Researchers — especially graduate students whose theses depend on methodological defensibility — need to answer questions like:

- "Show me every change to code X's definition."
- "What did code X mean on 14 March, when I applied it to this quote?"
- "Who applied this code to this quote, and when?"
- "Was this code AI-suggested, and which model version?"
- "When did we lock the codebook for ICR? What was the codebook state at that moment?"
- "Reconstruct the project as it was on date Y."

### Audit trail requirements

- **Append-only event log.** Every operation (code create, definition edit, application create, application delete, merge, split, retire, AI invocation) becomes an event with timestamp, actor, and full payload diff.
- **Code definition versioning.** Each definition edit creates a new version; the application records which version was current at apply-time.
- **Codebook snapshots.** Named, immutable snapshots of the entire codebook at points in time ("Initial coding done 2026-04-12," "Locked for ICR 2026-05-01"). Reports can be regenerated against any snapshot.
- **Project-level checkpoints / versions.** Full project state snapshot — useful for "I'm trying axial coding; let me save first so I can roll back if it dead-ends." This maps cleanly to git-like semantics under the hood.
- **AI invocation log.** Every AI call (model name, model version, prompt template, input hash, raw output, accept/reject decision). This is what makes AI use reproducible and reviewable.
- **Reflexive memos as part of the trail.** Charmaz wants the *why*, not just the *what*. The audit view should interleave memos with mechanical events.

### Locked codebook stages

- A codebook can be **locked** (read-only for new codes; existing codes can be applied but not edited).
- A locked stage usually gates ICR: locked codebook → multiple coders apply blind → reconcile.
- Unlocking requires a recorded reason (a methodological memo).

### AI provenance specifically

- Each AI suggestion is logged whether accepted or not (rejected suggestions are valuable evidence — "the AI suggested 'family conflict' here and I rejected because the participant explicitly framed it as collaboration").
- Model version is part of the record: if `llama-3.1-8b-instruct@q4_K_M` is replaced by `@q5_K_M`, downstream reproducibility depends on knowing which.
- Prompts and prompt templates are versioned alongside the model.

### Export of audit trail

- Researchers writing up methodologically transparent papers will want to export the audit log as a Word or Markdown appendix: "Coding decisions log."
- Some journals (esp. *Qualitative Health Research*, *International Journal of Qualitative Methods*) increasingly request this.

---

## Strategic shape for Scribe specifically

Three convictions to anchor the planning doc:

1. **Word-level audio anchoring is Scribe's unique advantage.** No major QDA tool has a fully local, tightly aligned, audio-back-linkable coding surface. Lean into it: every quote is one click from playback; every coded segment can include speaker turn metadata for free; "find similar audio moments" is plausible later via embeddings on the audio itself.

2. **Local-AI-as-suggester is the right ethical and methodological frame.** Match Charmaz's epistemology, not a 2024 autocoding hype cycle. AI never applies a code. AI suggestions always have provenance. AI is off by default until the researcher has hand-coded enough to have a codebook with shape. This is also a genuine differentiator — most AI-QDA features being shipped right now do *not* take this stance.

3. **REFI-QDA from day one (export, then import).** It's the only credible answer to the "what if I outgrow Scribe" question, and shipping it removes the biggest objection a methodologically literate user will raise. It also forces a clean internal data model — anything that can round-trip through QDPX is well-shaped.

---

### Key sources used in this report

- Charmaz, K. (2014) *Constructing Grounded Theory* (2nd ed.) SAGE.
- Glaser, B. & Strauss, A. (1967) *The Discovery of Grounded Theory*.
- Strauss, A. & Corbin, J. (1990, 1998) *Basics of Qualitative Research*.
- REFI-QDA / QDPX standard documentation, qdasoftware.org.
- Wikipedia articles on Grounded Theory, Kathy Charmaz, CAQDAS.
- HuggingFace embedding-quantization comparison and MTEB leaderboard.
- Public feature documentation for NVivo, Atlas.ti, MAXQDA, Quirkos, Dedoose, Taguette.
- Methodological discussion on AI-in-QDA (Atlas.ti "Intentional AI Coding," recent SAGE *International Journal of Qualitative Methods* discourse, Christou 2023, Morgan 2023).
