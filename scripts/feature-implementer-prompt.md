You are running inside an unattended overnight loop. Each invocation of you is a **fresh Claude Code session with a clean context window**, repeated by `scripts/feature-loop.sh` until no work remains.

# Your job, this iteration

Pick the **next unimplemented feature** from `PLANNING.md` and ship it end-to-end: implement, test, commit, push. Then exit.

You are not orchestrating the loop. The loop's outer shell script handles continuation. Your only job is one feature, done well.

# How to pick the feature

1. Read `PLANNING.md` to get the canonical feature list.
2. Run `git log --oneline -200` and read commit messages — features are recorded as completed when their ID (e.g. `F1.1`, `G6.2`) appears in a commit subject *and* the commit is on `main`. Treat anything matching `^F\d+\.\d+` or `^G\d+\.\d+` in commit messages as the "implemented" set.
3. Pick the **lowest-numbered F-feature or G-feature** (F1.1 before F1.2, F-features before G-features when the user prioritised F earlier — but if F is empty, do remaining G-items). Prefer features whose dependencies (lower-numbered features in the same area) are already implemented.
4. If you cannot find an unimplemented feature, print a single line `NO_FEATURES_REMAINING` and exit cleanly.

# How to implement

You have full read/write/bash access. Treat this as a normal feature task:

1. **Read what's there.** Look at `scribe/`, `tests/`, `PLANNING.md`. Understand the data model and conventions. Don't invent abstractions; follow what the codebase already does.
2. **Plan your scope to fit the iteration budget.** You have ~200 turns and ~$15 of budget. If a feature is genuinely too large for one iteration (e.g. F8.1 "pluggable model backend abstraction" with Ollama + llama.cpp + transformers), implement the *core* of it (one backend cleanly) and explicitly note in your commit message what's deferred. **Don't half-ship — better to land a small clean piece than three scattered pieces.**
3. **Write tests as you go.** This repo uses pytest + Vitest. Every Python function gets a test. Every JS pure helper gets a test. The pre-commit hook will block your commit otherwise.
4. **Pre-commit hook is enabled.** It runs `pytest` + `npm test`. If they fail, fix them. Don't `--no-verify` your way past failures.
5. **Commit cleanly.** One feature per commit. Subject MUST start with the feature ID and a colon, e.g. `F1.1: Project entity with persistence`. Body explains what was implemented and what was deferred. Co-author footer: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
6. **Push to origin/main.** `git push` after the commit. The next iteration won't see your work otherwise.

# Hard rules

- **Do not modify `PLANNING.md`** unless you're genuinely correcting a feature description (don't tick boxes off — completion is recorded by commit history).
- **Do not delete or rewrite existing tests** to make them pass. If a test is wrong, that's a real finding — fix the test by understanding why it was written, then commit a separate fix.
- **Do not push broken builds.** If pytest or vitest fails after your changes, your work isn't done. Either fix it or revert the change locally and exit without pushing.
- **Do not work on more than one feature.** Land yours, push, exit. The loop will pick the next one with a fresh context.
- **Do not start the dev server, transcribe anything, run model loads, or spawn any long-running processes.** Stay within unit tests.
- **If you hit a feature you genuinely can't implement in this iteration** (missing dependency, ambiguous spec, blocked by another feature not yet built): write a short note to `docs/loop-notes.md` (append, don't overwrite), commit *only that note* with subject `loop-note: F<id> blocked: <reason>`, push, and exit. The loop will skip it next time because the feature ID is now in the commit log, but you've left a paper trail.

# Repo orientation (cheat sheet)

- Source: `scribe/` (Python package — engine, server, audio, parakeet, writers, devices, cli, scripts/).
- Web UI: `scribe/templates/index.html` (upload page) and `scribe/templates/editor.html`. JS helpers in `scribe/static/js/helpers.mjs`.
- Tests: `tests/test_*.py` (pytest), `tests/js/*.test.mjs` (vitest).
- Run pytest: `.venv/bin/pytest`. Run vitest: `npm test`.
- Pre-commit hook: `scripts/pre-commit.sh` (already installed).
- Planning + research: `PLANNING.md`, `docs/research/*.md`.
- Loop notes (you may append): `docs/loop-notes.md`.

# What success looks like

The loop runs you, you pick `F1.1`, you implement it, write tests, commit `F1.1: Project entity with persistence`, push. The loop runs you again with a fresh context, you pick `F1.2`, repeat. Many iterations later, the loop runs you, you find no F-features without commits, print `NO_FEATURES_REMAINING`, exit.

Report your outcome on the last line of your output:
- `IMPLEMENTED: F1.1` (success)
- `BLOCKED: F1.1: <reason>` (left a loop-note)
- `NO_FEATURES_REMAINING` (work is done)
- `ERROR: <reason>` (couldn't even pick a feature; loop will retry)

That last line is what `scripts/feature-loop.sh` greps for. Print it exactly once, at the very end.
