You are running inside an unattended overnight loop. Each invocation of you is a **fresh Claude Code session with a clean context window**, repeated by `scripts/feature-loop.sh` until no work remains.

# Your job, this iteration

Pick the **next unimplemented feature** from `PLANNING.md` and ship it end-to-end: implement, test, **prove it works through the user-facing surface**, commit, push. Then exit.

You are not orchestrating the loop. The loop's outer shell script handles continuation. Your only job is one feature, done well.

# What "done" means (read this before anything else)

A previous version of this loop used a regex over commit subjects to decide what counted as done. That detector marked F8.6 as shipped because a commit subject said `F8.6:`, even though only the engine module had landed and the user had no UI to invoke it. The user could not reach the feature. **That is not done.** This prompt now defines done structurally:

A feature is **done** if and only if all three are true:

1. **Pure logic exists with passing unit tests.** Module under `scribe/`, tests under `tests/`.
2. **It is reachable from the user-facing surface.** For features that produce user-visible behaviour, this means a FastAPI route exercised by a `TestClient` integration test in `tests/test_server*.py`, AND (where the feature has UI) a template render check that asserts the new control / page / panel is in the response HTML. Pure-internal features (e.g. data-model classes, helper modules consumed by other modules) are exempt — but state explicitly in your commit body why no user surface is needed.
3. **The commit subject starts with the feature ID and a colon**, e.g. `F1.1: Project entity with persistence`. The body MUST contain a `Reachable-via:` line listing the user-facing surface (route + UI element) OR the line `Reachable-via: pure-internal — consumed by <module>` if no UI is needed. The loop's "done" detector parses this line; if it's missing the feature is treated as not done and the next iteration will revisit it.

If you find a feature whose ID is in the commit log but whose `Reachable-via:` line is missing or points to something that doesn't exist, treat that feature as **not done** and finish the wiring. Do not assume "the previous iteration handled it."

# How to pick the feature

1. Read `PLANNING.md` to get the canonical feature list.
2. Run `git log --oneline -300` and read commit subjects + bodies.
3. For each feature ID found in a subject (`^F\d+\.\d+:` or `^G\d+\.\d+:`), check whether the body contains a `Reachable-via:` line. If it does, the feature is done. If it doesn't, the feature is **incomplete** and is a valid candidate for this iteration — but you must finish wiring it, not start over.
4. Pick the **lowest-numbered** unfinished or incomplete feature. F-features take priority over G-features unless all F-features are done. Prefer features whose dependencies are already complete.
5. If everything in `PLANNING.md` has a passing `Reachable-via:` line, print `NO_FEATURES_REMAINING` and exit.

# How to implement

You have full read/write/bash access. Treat this as a normal feature task:

1. **Read what's there.** Look at `scribe/`, `tests/`, `PLANNING.md`, `CLAUDE.md`. Understand the data model and conventions. Don't invent abstractions; follow what the codebase already does.
2. **Decide the scope.** You have ~200 turns and unbounded budget per iteration. If a feature is too large for one iteration, implement the *core slice that is reachable end-to-end* (engine + route + minimal UI + integration test) and explicitly note what's deferred in the commit body. **Don't ship a module without a route. Don't ship a route without a UI button. Don't ship a UI button without a `TestClient` test that hits it.** Half-shipped features are the failure mode this prompt is designed to prevent.
3. **Write tests as you go.** Pytest + Vitest. Every Python function gets a unit test. Every JS pure helper gets a test. **At least one integration test** must exercise the new route via `fastapi.testclient.TestClient`, AND assert that the relevant template (if any) renders the new control. The pre-commit hook will block your commit if tests fail.
4. **Pre-commit hook is enabled** (`scripts/pre-commit.sh` runs pytest + vitest). If it fails, fix the underlying issue. Never `--no-verify`.
5. **Commit with the required `Reachable-via:` line.** Example body:

   ```
   F8.6: Whole-transcript AI review pass + UI

   <body explaining what was implemented and any deferrals>

   Reachable-via:
     route: POST /api/projects/<pid>/sources/<sid>/review
     ui: scribe/templates/source_coding.html "Review whole transcript" button
     tests: tests/test_server_review.py::TestReviewPass::test_button_renders
            tests/test_server_review.py::TestReviewPass::test_post_starts_pass

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   ```

   For pure-internal features (no user surface):

   ```
   Reachable-via: pure-internal — consumed by scribe/<module>.py
   ```

6. **Push to origin/main.** Without `git push`, the next iteration won't see your work.

# Hard rules

- **The `Reachable-via:` line is mandatory.** If you commit without it, the next iteration will pick up your feature ID as incomplete and rebuild it. You will have wasted budget.
- **Do not stub the integration test.** A test that imports the module but never makes an HTTP call doesn't prove reachability. The integration test must instantiate `TestClient(scribe.server.app)` and exercise the new route.
- **Do not modify `PLANNING.md`** unless you're correcting a feature description. Completion is recorded in commit bodies via `Reachable-via:`, not in PLANNING.
- **Do not delete or rewrite existing tests** to make them pass. If a test is wrong, that's a real finding — fix the test in a separate commit.
- **Do not push broken builds.** If pytest or vitest fails after your changes, you are not done. Either fix it or `git reset --hard HEAD` and exit without pushing.
- **Do not work on more than one feature.** Land yours, push, exit.
- **Do not start the dev server, transcribe, run model loads, or spawn long-running processes.** Stay within unit + integration tests.
- **If you hit a feature you genuinely can't ship in this iteration** (missing dependency, ambiguous spec, requires a model running): append a short note to `docs/loop-notes.md` and commit *only that note* with subject `loop-note: F<id> blocked: <reason>`. Include a `Reachable-via:` line pointing to the note itself, e.g. `Reachable-via: blocked — see docs/loop-notes.md`. The next iteration will skip the feature because it's recorded as blocked.

# Repo orientation (cheat sheet)

- Source: `scribe/` (Python package — engine, server, audio, writers, devices, cli, ai_*, transcript_*, code_* etc).
- Web UI: `scribe/templates/index.html`, `scribe/templates/editor.html`, `scribe/templates/source_coding.html`, `scribe/templates/codebook_editor.html`, `scribe/templates/_shell.html`. JS helpers in `scribe/static/js/helpers.mjs`.
- Tests: `tests/test_*.py` (pytest), `tests/js/*.test.mjs` (vitest). Endpoint tests live in `tests/test_server*.py` — follow that file's `TestClient` fixture pattern.
- Run pytest: `.venv/bin/pytest -m "not slow and not gpu" -q`. Run vitest: `npm test`.
- Pre-commit hook: `scripts/pre-commit.sh` (already installed).
- Planning + research: `PLANNING.md`, `CLAUDE.md`, `docs/research/*.md`.
- Loop notes (you may append): `docs/loop-notes.md`.
- Feature flag pattern for AI: features that gate on F8.10 use `scribe.ai_gate.evaluate_project_ai_gate()`; features that bypass it (transcript editor's grammar bot is the precedent) say so in the commit body.

# What success looks like

The loop runs you, you pick `F8.6`, you find an existing engine module + an `F8.6:` commit *without* a `Reachable-via:` line. You wire the FastAPI route, the source_coding.html button, the project AI page that lists pending suggestions, and an integration test that POSTs to the route and asserts the button renders. You commit:

```
F8.6: Wire whole-transcript AI review pass to UI

scribe/transcript_review.py was already implemented in 167b8c6
but had no route and no UI surface. This commit adds:
  - POST /api/projects/<pid>/sources/<sid>/review (start pass)
  - GET  /api/projects/<pid>/review-passes/<rid> (poll status)
  - "✨ Review whole transcript" button in source_coding.html
  - /projects/<pid>/ai page lists pending review suggestions
    with bulk accept/reject

Reachable-via:
  route: POST /api/projects/.../review, GET /api/projects/.../review-passes/...
  ui: source_coding.html "Review whole transcript" button;
      project_subpage.html ai page (pending suggestions list)
  tests: tests/test_server_review.py (5 cases)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

You push, you exit. The loop runs you again with a fresh context, you pick the next unfinished feature, repeat.

Report your outcome on the last line of your output:
- `IMPLEMENTED: F8.6 (Reachable-via verified)` — feature shipped, `Reachable-via:` line in body
- `WIRED: F8.6 (was incomplete, finished UI)` — found a previous half-ship, finished it
- `BLOCKED: F8.6: <reason>` — loop-note committed
- `NO_FEATURES_REMAINING`
- `ERROR: <reason>` — couldn't even pick a feature

The shell script greps for the prefix on this last line. Print it exactly once, at the very end.
