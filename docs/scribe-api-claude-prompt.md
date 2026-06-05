# Talking to Scribe over the API

Drop this file into the repo / directory you start Claude Code from
on a *consumer* machine — i.e. anything that wants to read a Scribe
instance running elsewhere on the network. Claude Code auto-loads
`CLAUDE.md` at the working directory, so the simplest setup is:

```bash
mkdir -p ~/scribe-client && cd ~/scribe-client
curl -sH "Authorization: Bearer $SCRIBE_API_KEY" \
     $SCRIBE_HOST/docs/api-claude-prompt > CLAUDE.md
claude
```

(or just save *this* file there as `CLAUDE.md`).

The rest of this document is what a Claude session needs to know to
talk to a running Scribe. It is deliberately short — the API is
self-describing.

---

## Connection

Scribe runs locally on another laptop and exposes a **read-only**
HTTP API at `/api/v1/`. Two things you need:

- **Base URL**: `http://<scribe-host-lan-ip>:8765`. On the Scribe
  host, run `hostname -I` (Linux) or `ipconfig getifaddr en0`
  (macOS). Should look like `192.168.x.x` or `10.x.x.x`.
- **API key**: a bearer token of shape `sk_scribe_...`. Mint one
  on the Scribe host: **Settings → API keys → label → Mint**, or
  via CLI: `python -m scribe.scripts.api_keys mint <label>`.
  The plaintext is shown once; copy it immediately.

Set these as environment variables for the session so curl calls
stay tidy:

```bash
export SCRIBE_HOST=http://192.168.1.42:8765
export SCRIBE_API_KEY=sk_scribe_...
```

Sanity check:

```bash
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" $SCRIBE_HOST/api/v1/ | jq
```

If you see a JSON document with `"version": "v1"` and an `endpoints`
list — you're in.

If you get **401**, the key is wrong or expired. If you get **no
response / connection refused**, the Scribe host isn't reachable on
the LAN — check the URL and whether the server's running.

---

## Workflow guide

### Discover what's available

Always start here. The response lists every endpoint with a
one-line summary, so you don't need this document past the first
turn:

```bash
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" $SCRIBE_HOST/api/v1/
```

### Pull a single transcript's plain text

Cheapest way to get one transcript's content as prose:

```bash
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" \
     "$SCRIBE_HOST/api/v1/transcripts/<id>/text"
```

The `text` field has speaker labels by default (`Maria: I find it
hard to ask for help.`). Add `?include_timestamps=true` for `[mm:ss]`
markers, `?include_speakers=false` to drop the labels.

### Read every transcript on the box

```bash
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" \
     "$SCRIBE_HOST/api/v1/transcripts" | jq '.transcripts[].id'
```

Then loop those ids through `/transcripts/<id>/text`. Filter to a
project by adding `?project_id=<pid>`.

### Search across transcripts (substring)

Useful when the user asks "where did anyone say X?":

```bash
curl -sG -H "Authorization: Bearer $SCRIBE_API_KEY" \
     --data-urlencode "q=push back" \
     "$SCRIBE_HOST/api/v1/search"
```

Returns up to 200 matches with `transcript_id`, `segment_index`,
`speaker`, `start`, `end`, and the matching `text`.

### Ask a project a grounded question

The chat-like endpoint. Scribe's server retrieves the most relevant
snippets from the project's embedding index, builds a grounded
prompt, calls the project's configured LLM, and returns the answer
plus citations. **Stateless** — no conversation is persisted on the
server, so you manage the conversation history on this side if
multi-turn is wanted.

```bash
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"question":"What recurring tensions appear across these interviews?"}' \
     "$SCRIBE_HOST/api/v1/projects/<pid>/ask"
```

Restrict to specific sources by adding `"source_ids": ["<sid>", ...]`.

### Project + codebook

```bash
# List projects
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" $SCRIBE_HOST/api/v1/projects

# One project + its sources + code count
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" \
     "$SCRIBE_HOST/api/v1/projects/<pid>"

# Full codebook
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" \
     "$SCRIBE_HOST/api/v1/projects/<pid>/codes"

# Coded segments (which code applied to which span)
curl -s -H "Authorization: Bearer $SCRIBE_API_KEY" \
     "$SCRIBE_HOST/api/v1/projects/<pid>/applications"
```

---

## House rules

- **The API is read-only.** No POST/PUT/DELETE that mutates state.
  This means you can experiment freely; nothing you do can damage
  the user's data.
- **Don't try to write through the API.** If the user asks for a
  change, surface it back to them as a Scribe-side action they
  should perform from the UI on the host machine.
- **Pull only what you need.** Some installs have hundreds of
  transcripts. Use `?q=`, `?project_id=`, and the `search`
  endpoint before reaching for "every transcript's full text."
- **Cite by transcript_id + segment_index.** When summarising,
  always anchor claims to specific transcripts so the user can
  click through to the editor on the host machine to verify.
- **`/ask` already knows about the project.** Prefer it over
  pulling every transcript and reasoning from scratch when the
  user is asking a question about *one* project — the embedding
  retrieval is more accurate than your guess at what to load.

---

## Common pitfalls

- **401 Unauthorized** → the API key is missing, mistyped, or
  revoked. Re-mint and re-export `SCRIBE_API_KEY`.
- **404 on `/api/v1/transcripts/<id>`** → the id doesn't match
  any job. Use `GET /api/v1/transcripts` to find valid ids.
- **502 from `/ask`** → the project's AI backend (Ollama by
  default) isn't running on the Scribe host. The user has to
  start it there; this isn't something you can fix from the
  consumer side.
- **400 from `/ask`** → the project hasn't configured a
  `default_model`. The user needs to set one in the project's AI
  settings on the host.
- **Empty `citations` from `/ask`** → the project's embedding
  index is empty. The user can build it from
  `/projects/<pid>/chat` → "Build embedding index" button.
