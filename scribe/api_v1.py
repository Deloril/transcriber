"""Public, versioned, **read-only** API for machine clients.

This is the surface intended for an LLM (or any other automation)
to talk to a running Scribe instance over HTTP. Two principles
shape every endpoint:

1. **Read-only.** The router never writes to disk. Listing
   transcripts, fetching their text, searching across them,
   asking questions of a project — all queries. Even the chat
   endpoint, which calls the LLM backend, doesn't persist a
   conversation record like the in-app /chat does; the model's
   answer is computed on demand and returned, end of story.

2. **Stable + narrow.** Internal endpoints under ``/api/...`` are
   free to refactor; ``/api/v1/`` keeps a fixed shape per route
   so a client written today still works tomorrow. New behaviour
   lands as ``/api/v2/`` (or new endpoints under ``/api/v1/``,
   never breaking-shape changes to existing ones).

Auth: every endpoint requires ``Authorization: Bearer <api-key>``.
Mint a key via ``python -m scribe.scripts.api_keys mint <label>``.

The shape of every response is a JSON object — no bare arrays —
so future versions can add fields like ``next_cursor`` or
``warnings`` without breaking existing clients.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from . import api_auth


# Single APIRouter exported to the main app. The handlers below
# are intentionally compact; they pull from existing scribe.*
# modules via the helper imports near the top of each route so
# tests can monkeypatch a single name without re-stitching the
# whole router.
router = APIRouter(prefix="/api/v1", tags=["api-v1"])


# Apply the API-key dependency at the router level via a helper —
# every route below inherits it.
_AUTH = Depends(api_auth.require_api_key())


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@router.get("/", dependencies=[_AUTH])
def discovery() -> dict[str, Any]:
    """Self-describing index — first call an LLM should make.

    Returns the version, the available endpoints, and a one-line
    summary of each so a client can decide what to call without
    reading docs. This is a deliberate alternative to OpenAPI:
    a tiny hand-curated map is more useful for prompt context
    than the full auto-generated schema.
    """
    return {
        "name": "Scribe Read-Only API",
        "version": "v1",
        "description": (
            "Read-only access to transcripts and academic-coding "
            "projects. Designed for LLM / MCP clients that want to "
            "summarise, search, or reason over interview data."
        ),
        "auth": {
            "scheme": "bearer",
            "header": "Authorization: Bearer <api-key>",
            "mint_command": (
                "python -m scribe.scripts.api_keys mint <label>"
            ),
        },
        "endpoints": [
            {"method": "GET", "path": "/api/v1/",
             "summary": "This document."},
            {"method": "GET", "path": "/api/v1/transcripts",
             "summary": "List every transcription. Supports ?q= and "
                        "?project_id= filters; ?limit & ?offset paginate."},
            {"method": "GET", "path": "/api/v1/transcripts/{id}",
             "summary": "Full transcript: segments + speakers + language + duration."},
            {"method": "GET", "path": "/api/v1/transcripts/{id}/text",
             "summary": "Plain-text-only view of the transcript (saves tokens)."},
            {"method": "GET", "path": "/api/v1/projects",
             "summary": "List academic-coding projects."},
            {"method": "GET", "path": "/api/v1/projects/{id}",
             "summary": "Project + sources + codebook summary."},
            {"method": "GET", "path": "/api/v1/projects/{id}/codes",
             "summary": "Full codebook for a project."},
            {"method": "GET", "path": "/api/v1/projects/{id}/applications",
             "summary": "Coded segments (which code applied to which span)."},
            {"method": "GET", "path": "/api/v1/search",
             "summary": "Substring search across transcripts. ?q= required, "
                        "?project_id= optional, ?limit caps results."},
            {"method": "POST", "path": "/api/v1/projects/{id}/ask",
             "summary": "One-shot Q&A: ask a question of the project's "
                        "transcripts; the server retrieves grounded "
                        "snippets + calls the project's configured LLM. "
                        "Stateless — no conversation is persisted."},
        ],
        "limits": {
            "transcripts_list_default_limit": 100,
            "transcripts_list_max_limit": 500,
            "search_max_results": 200,
            "ask_max_question_chars": 8000,
        },
    }


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


@router.get("/transcripts", dependencies=[_AUTH])
def list_transcripts(
    q: str = Query("", description="Substring filter on filename or speakers."),
    project_id: str = Query("", description="Restrict to transcripts linked from this project."),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List transcriptions in newest-first order.

    Returns a row shape that mirrors the library page (id,
    display_name, input_filename, mode, language, duration,
    speakers, status, created_at). Searches against the same
    haystack the in-app library uses (display name, original
    filename, speakers, status, mode, language, model).

    ``project_id`` filters to transcripts that are wired up as a
    Source on that project. Useful for "give me everything in the
    Pilot Study project" without having to walk every job.
    """
    from . import library as _library
    from . import sources as _sources
    from . import server as _srv

    with _srv.JOBS_LOCK:
        snapshot = list(_srv.JOBS.values())
    rows = _library.summarise_jobs(snapshot)

    if project_id:
        try:
            srcs = _sources.list_sources(_srv._projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        wanted = {s.transcript_job_id for s in srcs if s.transcript_job_id}
        rows = [r for r in rows if r["id"] in wanted]

    if q:
        rows = _library.filter_rows(rows, q)
    total = len(rows)
    paged = rows[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "transcripts": paged,
    }


@router.get("/transcripts/{transcript_id}", dependencies=[_AUTH])
def get_transcript(transcript_id: str) -> dict[str, Any]:
    """Full transcript JSON: segments + words + speakers + language."""
    from . import server as _srv

    _srv._check_job_id(transcript_id)
    with _srv.JOBS_LOCK:
        job = _srv.JOBS.get(transcript_id)
    if job is None:
        raise HTTPException(404, "Transcript not found")
    payload = _srv._load_job_transcript(job)
    return {
        "id": transcript_id,
        "display_name": job.display_name or job.input_filename or "",
        "input_filename": job.input_filename,
        "language": payload.get("language"),
        "mode": payload.get("mode"),
        "speakers": payload.get("speakers") or [],
        "speaker_names": payload.get("speaker_names") or {},
        "segments": payload.get("segments") or [],
    }


@router.get("/transcripts/{transcript_id}/text", dependencies=[_AUTH])
def get_transcript_text(
    transcript_id: str,
    include_speakers: bool = Query(True),
    include_timestamps: bool = Query(False),
) -> dict[str, Any]:
    """Plain-text rendering — much cheaper for an LLM to consume.

    ``include_speakers=true`` (default) prefixes each segment with
    the speaker label (using the user's renames when present).
    ``include_timestamps=true`` adds ``[mm:ss]`` markers — off by
    default because they bloat token counts and aren't relevant
    for most reasoning tasks.
    """
    from . import server as _srv

    _srv._check_job_id(transcript_id)
    with _srv.JOBS_LOCK:
        job = _srv.JOBS.get(transcript_id)
    if job is None:
        raise HTTPException(404, "Transcript not found")
    payload = _srv._load_job_transcript(job)
    segments = payload.get("segments") or []
    speaker_names = payload.get("speaker_names") or {}
    parts: list[str] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        line = ""
        if include_timestamps:
            start = float(s.get("start") or 0.0)
            mm, ss = divmod(int(start), 60)
            hh, mm = divmod(mm, 60)
            line += f"[{hh:02d}:{mm:02d}:{ss:02d}] " if hh else f"[{mm:02d}:{ss:02d}] "
        if include_speakers:
            sp = s.get("speaker") or ""
            display = speaker_names.get(sp) or sp
            if display:
                line += f"{display}: "
        line += text
        parts.append(line)
    return {
        "id": transcript_id,
        "display_name": job.display_name or job.input_filename or "",
        "language": payload.get("language"),
        "text": "\n\n".join(parts),
    }


# --------------------------------------------------------------------------- #
# Projects + codebook
# --------------------------------------------------------------------------- #


@router.get("/projects", dependencies=[_AUTH])
def list_projects() -> dict[str, Any]:
    from . import projects as _projects
    from . import server as _srv

    items = _projects.list_projects(_srv._projects_root())
    return {
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "research_question": p.research_question,
                "methodology": p.methodology,
                "created_at": p.created_at,
                "modified_at": p.modified_at,
                "codebook_stage": p.codebook_stage,
            }
            for p in items
        ],
    }


@router.get("/projects/{project_id}", dependencies=[_AUTH])
def get_project(project_id: str) -> dict[str, Any]:
    from . import projects as _projects
    from . import sources as _sources
    from . import codes as _codes
    from . import server as _srv

    try:
        project = _projects.load_project(_srv._projects_root(), project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")
    srcs = _sources.list_sources(_srv._projects_root(), project_id)
    codes = _codes.list_codes(_srv._projects_root(), project_id)
    return {
        "id": project.id,
        "name": project.name,
        "research_question": project.research_question,
        "methodology": project.methodology,
        "sensitising_concepts": list(project.sensitising_concepts),
        "codebook_stage": project.codebook_stage,
        "created_at": project.created_at,
        "modified_at": project.modified_at,
        "sources": [
            {
                "id": s.id,
                "name": s.name,
                "source_type": s.source_type,
                "language": s.language,
                "transcript_job_id": s.transcript_job_id or "",
            }
            for s in srcs
        ],
        "code_count": len(codes),
    }


@router.get("/projects/{project_id}/codes", dependencies=[_AUTH])
def list_project_codes(project_id: str) -> dict[str, Any]:
    from . import codes as _codes
    from . import projects as _projects
    from . import server as _srv

    try:
        _projects.load_project(_srv._projects_root(), project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")
    codes = _codes.list_codes(_srv._projects_root(), project_id)
    return {
        "codes": [c.to_dict() for c in codes],
    }


@router.get("/projects/{project_id}/applications", dependencies=[_AUTH])
def list_project_applications(
    project_id: str,
    code_id: str = Query("", description="Filter to one code's applications."),
    source_id: str = Query("", description="Filter to one source's applications."),
) -> dict[str, Any]:
    from . import applications as _applications
    from . import projects as _projects
    from . import server as _srv

    try:
        _projects.load_project(_srv._projects_root(), project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")
    apps = _applications.list_applications(
        _srv._projects_root(), project_id,
        code_id=code_id or None,
        source_id=source_id or None,
    )
    return {
        "applications": [a.to_dict() for a in apps],
    }


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


@router.get("/search", dependencies=[_AUTH])
def search(
    q: str = Query(..., min_length=1, description="Query string."),
    project_id: str = Query("", description="Restrict to one project's transcripts."),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Substring search across transcript segments.

    Returns one match per segment. The match record carries the
    transcript id, segment index, speaker, start time, and the
    matching text — enough for a caller to build a citation back
    to the editor's segment view.

    Case-insensitive substring; this is deliberately not a fuzzy
    or stemmed search, because LLM clients tend to construct
    exact phrases from prior turns.
    """
    from . import server as _srv
    from . import sources as _sources

    needle = q.strip().lower()
    if not needle:
        raise HTTPException(400, "q is required")

    if project_id:
        try:
            srcs = _sources.list_sources(_srv._projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        scoped = {s.transcript_job_id for s in srcs if s.transcript_job_id}
    else:
        scoped = None

    matches: list[dict[str, Any]] = []
    with _srv.JOBS_LOCK:
        snapshot = list(_srv.JOBS.values())
    for job in snapshot:
        if scoped is not None and job.id not in scoped:
            continue
        try:
            payload = _srv._load_job_transcript(job)
        except Exception:
            continue
        segs = payload.get("segments") or []
        names = payload.get("speaker_names") or {}
        for i, seg in enumerate(segs):
            text = (seg.get("text") or "")
            if needle not in text.lower():
                continue
            sp = seg.get("speaker") or ""
            matches.append({
                "transcript_id": job.id,
                "transcript_name": job.display_name or job.input_filename or job.id,
                "segment_index": i,
                "speaker": names.get(sp) or sp,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": text,
            })
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break
    return {
        "query": q,
        "total": len(matches),
        "limit": limit,
        "matches": matches,
    }


# --------------------------------------------------------------------------- #
# One-shot ask
# --------------------------------------------------------------------------- #


@router.post("/projects/{project_id}/ask", dependencies=[_AUTH])
async def ask_project(
    project_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Ask a question of a project's transcripts.

    Body shape::

        {
          "question": "What surprised the participants?",
          "source_ids": ["<sid>", ...]    # optional; default = all sources
        }

    The server embeds the question, retrieves the top-k snippets
    from the project's embedding index (restricted to the named
    sources), builds the same prompt the in-app chat uses, calls
    the project's configured AI backend, and returns the answer +
    citations. Nothing is written to disk — the model's reply is
    computed on demand and discarded.

    This means the API client doesn't have to manage Scribe-side
    state; it can pass its own conversation history as part of the
    question text.
    """
    from . import ai_backend as _ai_backend
    from . import projects as _projects
    from . import project_chat as _project_chat
    from . import sources as _sources
    from . import server as _srv

    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected JSON object body")
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > _project_chat.MAX_MESSAGE_LEN:
        raise HTTPException(
            400, f"question exceeds {_project_chat.MAX_MESSAGE_LEN} chars",
        )

    raw_source_ids = payload.get("source_ids")
    if raw_source_ids is not None and not isinstance(raw_source_ids, list):
        raise HTTPException(400, "source_ids must be a list")

    try:
        project = _projects.load_project(_srv._projects_root(), project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")

    # Resolve source set: caller-supplied list (filtered to ones that
    # actually exist) or every source on the project.
    project_sources = _sources.list_sources(_srv._projects_root(), project_id)
    project_sids = {s.id for s in project_sources}
    if raw_source_ids:
        source_ids = [str(s) for s in raw_source_ids if str(s) in project_sids]
        if not source_ids:
            raise HTTPException(
                400,
                "source_ids matched no sources on this project. "
                "Omit the field to ask across every source.",
            )
    else:
        source_ids = list(project_sids)

    # Backend.
    try:
        cfg, backend = _srv._resolve_suggestion_backend(project)
    except _ai_backend.BackendValidationError as e:
        raise HTTPException(400, str(e))
    if not cfg.default_model:
        raise HTTPException(
            400,
            "Project AI backend has no default_model configured.",
        )
    try:
        embed_fn, generate_fn, _emb_model, gen_model = (
            _srv._make_embed_and_generate_fns(cfg, backend)
        )
    except _ai_backend.BackendValidationError as e:
        raise HTTPException(400, str(e))

    citations = _srv._retrieve_chat_snippets(
        project_id=project_id,
        source_ids=source_ids,
        query=question,
        embed_fn=embed_fn,
        top_k=_project_chat.DEFAULT_RETRIEVAL_TOP_K,
    )
    prompt = _project_chat.build_chat_prompt(
        user_question=question,
        snippets=citations,
        history=[],
    )
    try:
        answer = generate_fn(prompt)
    except _ai_backend.BackendUnavailable as e:
        raise HTTPException(502, f"AI backend unavailable: {e}")
    except _ai_backend.BackendError as e:
        raise HTTPException(500, str(e))

    return {
        "question": question,
        "answer": str(answer or ""),
        "model": gen_model,
        "citations": [c.to_dict() for c in citations],
        "source_ids": source_ids,
    }
