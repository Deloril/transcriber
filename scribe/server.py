"""FastAPI server — drag-and-drop UI for offline transcription, plus editor."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import markdown as md
from fastapi import Body, FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .audio import compute_waveform, probe_audio_streams, probe_media_info
from .engine import AdvancedOptions, Segment, TranscriptionResult, Word, transcribe
from .writers import write_all, write_json, write_srt, write_txt, write_vtt

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
ENV_PATH = ROOT / ".env"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _to_bool_persisted(value: Any) -> bool:
    """Tolerant bool coercion for fields read off disk.

    Handles the ``bool("false") is True`` footgun: an older or
    hand-edited ``job.json`` may carry the literal string ``"false"``
    for a boolean-shaped field, and Python's plain ``bool()`` would
    flip that to True. Treat the usual stringified-false vocabulary
    as False; everything else falls through to ``bool()``.

    Mirrors :func:`scribe.library._to_bool`. Duplicated rather than
    imported so the dataclass loader has no module-load dependency.
    """
    if isinstance(value, str):
        if value.strip().lower() in {"", "false", "no", "0", "off", "none", "null"}:
            return False
    return bool(value)


@dataclass
class Job:
    id: str
    input_path: Path
    output_dir: Path
    mode: str
    speakers: list[str] | None
    num_speakers: int | None
    language: str
    model: str
    created_at: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    audio_streams: int = 0
    input_filename: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 8
    # Wall-clock timestamps used for the UI's elapsed/ETA counters.
    # Stored as epoch seconds so the client can compute differences without
    # parsing ISO strings.
    started_at: float | None = None
    finished_at: float | None = None
    # F10.2 — when the user clicks "Discard source media" the
    # `uploads/<id>/` directory is removed but the transcript +
    # output sidecars are kept; this flag tells the editor to
    # gracefully degrade (hide the player, disable seek/play) and
    # tells the library row to render a small "media discarded" icon.
    media_discarded: bool = False

    def to_state(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_path"] = str(self.input_path)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Job":
        return cls(
            id=d["id"],
            input_path=Path(d["input_path"]),
            output_dir=Path(d["output_dir"]),
            mode=d["mode"],
            speakers=d.get("speakers"),
            num_speakers=d.get("num_speakers"),
            language=d.get("language", "en"),
            model=d.get("model", "large-v3"),
            created_at=d.get("created_at", ""),
            status=d.get("status", "done"),
            progress=d.get("progress", 1.0),
            message=d.get("message", ""),
            result=d.get("result"),
            error=d.get("error"),
            output_paths=d.get("output_paths", {}),
            audio_streams=d.get("audio_streams", 0),
            input_filename=d.get("input_filename", ""),
            options=d.get("options", {}) or {},
            batch_size=int(d.get("batch_size", 8) or 8),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            media_discarded=_to_bool_persisted(d.get("media_discarded", False)),
        )


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
EVENT_LOOP: asyncio.AbstractEventLoop | None = None


app = FastAPI(title="Scribe")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _job_state_path(output_dir: Path) -> Path:
    return output_dir / "job.json"


def _edited_path(output_dir: Path) -> Path:
    return output_dir / "edited.json"


def _persist_job(job: Job) -> None:
    try:
        _job_state_path(job.output_dir).write_text(
            json.dumps(job.to_state(), indent=2, ensure_ascii=False)
        )
    except Exception as e:  # noqa: BLE001
        # persistence is best-effort; don't crash the worker
        print(f"[scribe] could not persist job {job.id}: {e}")


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _validate_persisted_paths(job: Job) -> None:
    """Make sure persisted state hasn't been hand-edited to point at arbitrary files."""
    if not _is_under(job.input_path, UPLOAD_DIR):
        raise ValueError(f"input_path escapes UPLOAD_DIR: {job.input_path}")
    if not _is_under(job.output_dir, OUTPUT_DIR):
        raise ValueError(f"output_dir escapes OUTPUT_DIR: {job.output_dir}")
    for kind, rel in list(job.output_paths.items()):
        full = (ROOT / rel).resolve()
        if not _is_under(full, OUTPUT_DIR):
            raise ValueError(f"output_paths[{kind}] escapes OUTPUT_DIR: {rel}")


def _load_jobs_from_disk() -> None:
    if not OUTPUT_DIR.exists():
        return
    for d in sorted(OUTPUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        sp = _job_state_path(d)
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text())
            job = Job.from_state(data)
            _validate_persisted_paths(job)

            # Any job not in a terminal state was interrupted by a crash/restart.
            if job.status not in ("done", "error"):
                job.status = "error"
                job.error = f"Server restarted while job was {data.get('status', 'pending')}."
                job.message = "Interrupted"
                job.progress = 0.0
                _persist_job(job)

            JOBS[job.id] = job
        except Exception as e:  # noqa: BLE001
            print(f"[scribe] could not load job from {d}: {e}")


@app.on_event("startup")
async def _startup() -> None:
    global EVENT_LOOP
    EVENT_LOOP = asyncio.get_running_loop()
    _load_jobs_from_disk()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/edit/{job_id}", response_class=HTMLResponse)
async def editor_page(request: Request, job_id: str) -> HTMLResponse:
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != "done":
            raise HTTPException(409, f"Job not finished (status: {job.status})")
    return templates.TemplateResponse(
        request,
        "editor.html",
        {
            "job_id": job_id,
            "input_filename": job.input_filename or job.input_path.name,
        },
    )


# F10.1 — Library view. The home page is the upload form; ``/library``
# is a separate page that lists every persisted transcription so the
# user doesn't have to remember the per-job ``/edit/<id>`` URL.
@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "library.html", {})


# --------------------------------------------------------------------------- #
# Project / coding-engine UI shell
#
# These pages establish the new IA: top-level Library + Projects, with
# project-scoped subpages for sources, codebook, queries, memos, AI
# suggestions, and the audit timeline. Most are placeholder wireframes — they
# render correctly, link to each other, and gracefully degrade if the
# backing API isn't there yet. The data layer for a chunk of the F-features
# already exists; the UI graduations land per-feature as the loop reaches
# them.
# --------------------------------------------------------------------------- #


def _project_id_or_404(project_id: str) -> str:
    """
    Light validation for the URL parameter. Real lookups happen against
    /api/projects/{id} which validates strictly; this just keeps malformed
    paths from rendering a confusing wireframe. We don't 404 here — empty
    pages still render so the user can see the IA — but we do refuse
    obviously-malicious paths.
    """
    if not project_id or "/" in project_id or ".." in project_id or len(project_id) > 64:
        raise HTTPException(400, "Invalid project id")
    return project_id


@app.get("/projects", response_class=HTMLResponse)
async def projects_list_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "projects_list.html", {
        "page_title": "Projects",
        "subtitle": "Each project ties multiple transcripts to a shared codebook.",
    })


@app.get("/projects/new", response_class=HTMLResponse)
async def project_new_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "project_new.html", {
        "page_title": "New project",
        "subtitle": "Bundle transcripts under a shared codebook.",
    })


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_home_page(request: Request, project_id: str) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    # Best-effort lookup so we can show real metadata in the heading. If the
    # project doesn't exist yet we still render the wireframe — the page is
    # also useful as a "create one" landing.
    project = None
    try:
        from .projects import load_project  # type: ignore
        project = load_project(pid)
    except Exception:
        project = None
    return templates.TemplateResponse(request, "project_home.html", {
        "project_id": pid,
        "project_name": (project or {}).get("name") if isinstance(project, dict) else getattr(project, "name", None),
        "project_methodology": (project or {}).get("methodology") if isinstance(project, dict) else getattr(project, "methodology", None),
        "project_stage": (project or {}).get("codebook_stage") if isinstance(project, dict) else getattr(project, "codebook_stage", None),
    })


def _render_subpage(
    request: Request, project_id: str, *,
    page_kind: str, page_title: str, description: str,
    feature_refs: list[str], wireframe_blocks: list[dict[str, Any]],
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "project_subpage.html", {
        "project_id": pid,
        "page_kind": page_kind,
        "page_title": page_title,
        "description": description,
        "feature_refs": feature_refs,
        "wireframe_blocks": wireframe_blocks,
    })


@app.get("/projects/{project_id}/sources", response_class=HTMLResponse)
async def project_sources_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="sources",
        page_title="Sources",
        description="Transcripts and field notes attached to this project.",
        feature_refs=["F1.2", "F1.3", "F3.3", "F10.3"],
        wireframe_blocks=[
            {"heading": "Source list", "lines": [
                "<strong>Columns:</strong> filename · participant · duration · language · added · per-row action (open editor)",
                "<strong>Toolbar:</strong> search · filter by participant attribute (F3.2) · &quot;+ Add source&quot;",
            ]},
            {"heading": "Participants", "lines": [
                "Participant table with user-defined demographic columns (F1.3, F3.3).",
                "One participant ↔ many sources.",
            ]},
            {"heading": "Sampling log", "lines": [
                "Why each source was added, what category it was meant to fill (F1.4 — theoretical sampling).",
            ]},
        ],
    )


@app.get("/projects/{project_id}/sources/add", response_class=HTMLResponse)
async def project_source_add_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="sources",
        page_title="Add source",
        description="Pick an existing transcription from the library, upload a new recording, or import an existing transcript.",
        feature_refs=["F1.2", "F10.3"],
        wireframe_blocks=[
            {"heading": "From the library", "lines": [
                "Pick a completed transcription from <code>/library</code> and attach it to this project.",
            ]},
            {"heading": "Upload new audio/video", "lines": [
                "Drag a recording — same flow as <code>/</code>, but lands in this project on completion.",
            ]},
            {"heading": "Import existing transcript", "lines": [
                "Drop a <code>.txt / .srt / .vtt / .json</code> file. F10.3 covers the parsers.",
            ]},
        ],
    )


@app.get("/projects/{project_id}/sources/{source_id}", response_class=HTMLResponse)
async def project_source_coding_page(request: Request, project_id: str, source_id: str) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    sid = _project_id_or_404(source_id)
    return templates.TemplateResponse(request, "source_coding.html", {
        "project_id": pid,
        "source_id": sid,
        "page_title": "Coding",
    })


@app.get("/projects/{project_id}/codebook", response_class=HTMLResponse)
async def project_codebook_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="codebook",
        page_title="Codebook",
        description="Codes, definitions, hierarchy, and stage gates.",
        feature_refs=["F2.1", "F2.2", "F2.3", "F2.4", "F9.3"],
        wireframe_blocks=[
            {"heading": "Codes", "lines": [
                "<strong>Tree view:</strong> hierarchy of codes by parent · color swatch · application count · stage chip.",
                "<strong>Editor pane:</strong> definition, inclusion / exclusion criteria, exemplar quotes, related codes.",
                "<strong>Lifecycle ops:</strong> merge · split · rename · retire · promote (F2.3).",
            ]},
            {"heading": "Stages & snapshots", "lines": [
                "Current stage: initial / focused / axial / theoretical / locked.",
                "Named codebook snapshots (F9.3). Lock toggle (F2.4) requires a methodological memo on unlock.",
            ]},
            {"heading": "Export", "lines": [
                "CSV · Markdown · Word · REFI-QDA Codebook XML (F2.6).",
            ]},
            {"heading": "Inter-coder reliability (multi-coder)", "lines": [
                "Cohen's kappa per code · reconciliation queue (F2.5). Visible only when multi-coder mode is on.",
            ]},
        ],
    )


@app.get("/projects/{project_id}/queries", response_class=HTMLResponse)
async def project_queries_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="queries",
        page_title="Queries",
        description="Cross-corpus search and matrices.",
        feature_refs=["F3.4", "F3.5", "F3.6", "F3.7"],
        wireframe_blocks=[
            {"heading": "Query builder", "lines": [
                "Code filter · source filter · participant attribute · speaker filter · boolean combinator · proximity (F3.5).",
            ]},
            {"heading": "Saved queries", "lines": [
                "List of named, re-runnable queries (F3.7).",
            ]},
            {"heading": "Matrix views", "lines": [
                "code × source (frequency) · code × code (co-occurrence) · code × attribute (cross-tab). Export CSV/XLSX (F3.6, F6.3).",
            ]},
            {"heading": "Coded segment retrieval", "lines": [
                "&quot;Show all quotes for code X grouped by participant&quot; — the most-run query (F6.2).",
            ]},
        ],
    )


@app.get("/projects/{project_id}/memos", response_class=HTMLResponse)
async def project_memos_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="memos",
        page_title="Memos",
        description="Code · theoretical · methodological · reflexive · quote · source · project memos.",
        feature_refs=["F5.1", "F5.2", "F5.3", "F5.4", "F5.5"],
        wireframe_blocks=[
            {"heading": "Memo list", "lines": [
                "Filter by type · linked-to · author. Cards with body preview.",
            ]},
            {"heading": "Memo canvas", "lines": [
                "Drag-arrangeable canvas for sorting / clustering memos toward a theory (F5.3).",
            ]},
            {"heading": "Promote to code definition", "lines": [
                "One-click promotion of a memo into a code's operational definition (F5.5).",
            ]},
        ],
    )


@app.get("/projects/{project_id}/ai", response_class=HTMLResponse)
async def project_ai_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="ai",
        page_title="AI suggestions",
        description="Locally-running model proposes codes / similar quotes / memo drafts. Never auto-applies.",
        feature_refs=["F8.1", "F8.3", "F8.4", "F8.5", "F8.6", "F8.7", "F8.8", "F8.10"],
        wireframe_blocks=[
            {"heading": "Active model", "lines": [
                "Show backend (ollama / llama.cpp / off), model name, status. Link to settings.",
            ]},
            {"heading": "Suggestion queue", "lines": [
                "Pending whole-transcript review pass (F8.6) results · second-coder diff (F8.7) · per-span suggestions (F8.3, F8.4).",
                "Each suggestion: accept / modify / reject. Provenance recorded either way (F9.6).",
            ]},
            {"heading": "AI-off gate", "lines": [
                "F8.10 — AI suggestions disabled until the codebook has hand-coded shape (default ≥8 codes, ≥2 transcripts). Settings to override.",
            ]},
        ],
    )


@app.get("/projects/{project_id}/audit", response_class=HTMLResponse)
async def project_audit_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="audit",
        page_title="Audit timeline",
        description="Append-only event log: codes, applications, definitions, memos, AI invocations.",
        feature_refs=["F9.1", "F9.2", "F9.4", "F9.6", "F9.7", "F9.8"],
        wireframe_blocks=[
            {"heading": "Timeline", "lines": [
                "Chronological feed with event-type filter, actor filter, date range. Each row links to its target.",
            ]},
            {"heading": "Time-travel", "lines": [
                "&quot;Show project as it was on 2026-04-12&quot; — read-only snapshot view (F9.8).",
            ]},
            {"heading": "Checkpoints", "lines": [
                "Named project-wide checkpoints (F9.4). Restore (read-only).",
            ]},
            {"heading": "Export", "lines": [
                "CSV · Markdown · RTF audit log for thesis appendices (F9.7).",
            ]},
        ],
    )


@app.get("/projects/{project_id}/settings", response_class=HTMLResponse)
async def project_settings_page(request: Request, project_id: str) -> HTMLResponse:
    return _render_subpage(
        request, project_id,
        page_kind="settings",
        page_title="Project settings",
        description="Methodology, attribute schema, codebook stage, AI gating, danger zone.",
        feature_refs=["F1.1", "F2.4", "F3.2", "F8.10"],
        wireframe_blocks=[
            {"heading": "Methodology & sensitising concepts", "lines": [
                "Editable. Methodology choice drives gerund-encouragement and other UI nudges.",
            ]},
            {"heading": "Source attribute schema", "lines": [
                "User-defined columns on sources / participants (F3.2). Add / remove / reorder.",
            ]},
            {"heading": "Codebook stage", "lines": [
                "Stage selector. Lock toggle (F2.4) — unlocking requires a methodological memo with reason.",
            ]},
            {"heading": "AI gating", "lines": [
                "Override the F8.10 default (AI off until codebook has shape). Researcher discretion.",
            ]},
            {"heading": "Danger zone", "lines": [
                "Delete project. Export REFI-QDA / QDPX (F6.4). Anonymised export (F6.7).",
            ]},
        ],
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "settings.html", {
        "page_title": "Settings",
        "subtitle": "Host preferences and credentials.",
    })


_README_PATH = ROOT / "README.md"
_README_CACHE: dict[str, Any] = {"mtime": 0.0, "html": ""}


def _render_readme() -> str:
    if not _README_PATH.exists():
        return "<p>README.md not found.</p>"
    mtime = _README_PATH.stat().st_mtime
    if mtime != _README_CACHE["mtime"]:
        text = _README_PATH.read_text(encoding="utf-8")
        _README_CACHE["html"] = md.markdown(
            text,
            extensions=[
                "fenced_code",
                "tables",
                "toc",
                "sane_lists",
                "codehilite",
            ],
            extension_configs={
                "codehilite": {"guess_lang": False, "noclasses": True},
                "toc": {"permalink": False},
            },
            output_format="html5",
        )
        _README_CACHE["mtime"] = mtime
    return _README_CACHE["html"]


@app.get("/docs/readme", response_class=HTMLResponse)
async def readme(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "readme.html",
        {"content": _render_readme()},
    )


@app.get("/api/readme", response_class=HTMLResponse)
async def readme_fragment() -> HTMLResponse:
    return HTMLResponse(_render_readme())


# --------------------------------------------------------------------------- #
# Profiles — server-side defaults you can apply per recording
# --------------------------------------------------------------------------- #

PROFILES_PATH = ROOT / "profiles.json"
PROFILES_LOCK = threading.Lock()
_PROFILE_NAME_RE = re.compile(r"^[\w \-.()]{1,64}$")


def _load_profiles() -> list[dict[str, Any]]:
    if not PROFILES_PATH.exists():
        return []
    try:
        data = json.loads(PROFILES_PATH.read_text())
        if isinstance(data, list):
            return data
    except Exception as e:  # noqa: BLE001
        print(f"[scribe] could not load profiles.json: {e}")
    return []


def _save_profiles(profiles: list[dict[str, Any]]) -> None:
    tmp = PROFILES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profiles, indent=2, ensure_ascii=False))
    tmp.replace(PROFILES_PATH)


def _normalise_profile(p: dict[str, Any]) -> dict[str, Any]:
    """Pick out only the fields we recognise so users can't smuggle junk in."""
    name = str(p.get("name", "")).strip()
    if not _PROFILE_NAME_RE.match(name):
        raise HTTPException(400, "Profile name must be 1–64 chars (letters, digits, space, -._())")
    settings = p.get("settings") or {}
    if not isinstance(settings, dict):
        raise HTTPException(400, "settings must be an object")
    allowed = {
        "mode", "language", "model", "batch_size",
        "speakers", "num_speakers",
        # advanced options
        "beam_size", "best_of", "temperature",
        "no_speech_threshold", "compression_ratio_threshold", "condition_on_previous_text",
        "chunk_size", "vad_onset", "vad_offset",
        "initial_prompt", "hotwords",
    }
    cleaned = {k: v for k, v in settings.items() if k in allowed}
    return {
        "name": name,
        "description": str(p.get("description", "")).strip()[:300],
        "settings": cleaned,
    }


@app.get("/api/capabilities")
async def capabilities() -> JSONResponse:
    """Report which optional engines are installed and which GPU backend is
    active. The UI uses this to gate model options (e.g. Parakeet/NeMo
    isn't supported on AMD ROCm) and to surface backend info."""
    from .parakeet import nemo_available
    from .engine import gpu_backend, _gpu_device_name, _cuda_vram_gb
    parakeet_ok, parakeet_err = nemo_available()
    backend = gpu_backend()
    # Parakeet is NVIDIA-only at runtime even if NeMo imports successfully.
    parakeet_runtime_ok = parakeet_ok and backend in ("cuda", "cpu")
    return JSONResponse({
        "parakeet": {
            "available": parakeet_runtime_ok,
            "installed": parakeet_ok,
            "error": parakeet_err,
            "blocked_by_backend": parakeet_ok and not parakeet_runtime_ok,
        },
        "gpu": {
            "backend": backend,
            "device_name": _gpu_device_name() or None,
            "vram_gb": round(_cuda_vram_gb(), 1) if backend in ("cuda", "rocm") else None,
        },
    })


@app.get("/api/profiles")
async def list_profiles() -> JSONResponse:
    with PROFILES_LOCK:
        return JSONResponse({"profiles": _load_profiles()})


@app.put("/api/profiles/{name}")
async def upsert_profile(name: str, request: Request) -> JSONResponse:
    if not _PROFILE_NAME_RE.match(name):
        raise HTTPException(400, "Invalid profile name")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    body["name"] = name
    profile = _normalise_profile(body)
    with PROFILES_LOCK:
        profiles = _load_profiles()
        # replace if exists, else append
        for i, p in enumerate(profiles):
            if p.get("name") == name:
                profiles[i] = profile
                break
        else:
            profiles.append(profile)
        _save_profiles(profiles)
    return JSONResponse(profile)


@app.delete("/api/profiles/{name}")
async def delete_profile(name: str) -> JSONResponse:
    if not _PROFILE_NAME_RE.match(name):
        raise HTTPException(400, "Invalid profile name")
    with PROFILES_LOCK:
        profiles = _load_profiles()
        new_profiles = [p for p in profiles if p.get("name") != name]
        if len(new_profiles) == len(profiles):
            raise HTTPException(404, "Profile not found")
        _save_profiles(new_profiles)
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# Projects (F1.1) — academic-coding research projects
#
# A project is the parent of {sources, codebook, memos, audit trail}.
# F1.1 ships only the entity itself; F1.2 onwards layer in sources,
# codebook, etc. Storage lives under ``PROJECTS_DIR/<id>/project.json``.
# --------------------------------------------------------------------------- #

from . import projects as _projects  # noqa: E402  (after module-level state)

PROJECTS_DIR = ROOT / "projects"
PROJECTS_LOCK = threading.Lock()


def _projects_root() -> Path:
    """Resolve the projects root lazily so tests can monkeypatch
    PROJECTS_DIR at runtime, mirroring how UPLOAD_DIR/OUTPUT_DIR work.
    """
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def _check_project_id(project_id: str) -> None:
    if not _projects.PROJECT_ID_RE.match(project_id):
        raise HTTPException(400, "Invalid project id")


@app.get("/api/projects")
async def list_projects_endpoint() -> JSONResponse:
    with PROJECTS_LOCK:
        out = [p.to_dict() for p in _projects.list_projects(_projects_root())]
    return JSONResponse({"projects": out})


@app.post("/api/projects")
async def create_project_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    try:
        project = _projects.Project.new(
            name=str(body.get("name", "")),
            research_question=str(body.get("research_question", "") or ""),
            methodology=str(body.get("methodology", "") or ""),
            sensitising_concepts=body.get("sensitising_concepts") or [],
            description=str(body.get("description", "") or ""),
            codebook_stage=str(body.get("codebook_stage", "initial") or "initial"),
        )
    except _projects.ProjectValidationError as e:
        raise HTTPException(400, str(e))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid project payload: {e}")

    with PROJECTS_LOCK:
        _projects.save_project(_projects_root(), project)
    return JSONResponse(project.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}")
async def get_project_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
    return JSONResponse(project.to_dict())


@app.patch("/api/projects/{project_id}")
async def patch_project_endpoint(project_id: str, request: Request) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            project.apply_update(body)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _projects.save_project(_projects_root(), project)
    return JSONResponse(project.to_dict())


@app.delete("/api/projects/{project_id}")
async def delete_project_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        ok = _projects.delete_project(_projects_root(), project_id)
    if not ok:
        raise HTTPException(404, "Project not found")
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# Sources (F1.2) — primary-data items attached to a project
#
# A source is most commonly a Scribe transcript (linked via
# ``transcript_job_id``) but the schema is forward-compatible with field
# notes, documents, and images. Sources live under
# ``PROJECTS_DIR/<pid>/sources/<sid>.json``; deleting the parent project
# wipes them as a side-effect.
# --------------------------------------------------------------------------- #

from . import sources as _sources  # noqa: E402  (after module-level state)


def _check_source_id(source_id: str) -> None:
    if not _sources.SOURCE_ID_RE.match(source_id):
        raise HTTPException(400, "Invalid source id")


def _project_must_exist(project_id: str) -> None:
    """Raise 404 if the parent project doesn't have a project.json."""
    if not _projects.project_state_path(_projects_root(), project_id).exists():
        raise HTTPException(404, "Project not found")


@app.get("/api/projects/{project_id}/sources")
async def list_sources_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        out = [
            s.to_dict()
            for s in _sources.list_sources(_projects_root(), project_id)
        ]
    return JSONResponse({"sources": out})


@app.post("/api/projects/{project_id}/sources")
async def create_source_endpoint(project_id: str, request: Request) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.Source.new(
                project_id=project_id,
                name=str(body.get("name", "")),
                source_type=str(body.get("source_type", "transcript") or "transcript"),
                transcript_job_id=body.get("transcript_job_id") or None,
                language=str(body.get("language", "") or ""),
                recording_date=str(body.get("recording_date", "") or ""),
                notes=str(body.get("notes", "") or ""),
                custom_attributes=body.get("custom_attributes") or {},
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid source payload: {e}")
        _sources.save_source(_projects_root(), source)
    return JSONResponse(source.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}/sources/{source_id}")
async def get_source_endpoint(project_id: str, source_id: str) -> JSONResponse:
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
    return JSONResponse(source.to_dict())


@app.patch("/api/projects/{project_id}/sources/{source_id}")
async def patch_source_endpoint(
    project_id: str, source_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    _check_source_id(source_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        try:
            source.apply_update(body)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _sources.save_source(_projects_root(), source)
    return JSONResponse(source.to_dict())


@app.delete("/api/projects/{project_id}/sources/{source_id}")
async def delete_source_endpoint(project_id: str, source_id: str) -> JSONResponse:
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        ok = _sources.delete_source(_projects_root(), project_id, source_id)
    if not ok:
        raise HTTPException(404, "Source not found")
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# Participants (F1.3) — humans behind one or more sources
#
# A participant carries the project-defined demographic columns and a
# list of source IDs they appear in. Storage:
# ``PROJECTS_DIR/<pid>/participants/<part_id>.json``. Like sources,
# participants are wiped when the parent project is deleted because
# they live inside the project directory.
# --------------------------------------------------------------------------- #

from . import participants as _participants  # noqa: E402  (after module-level state)


def _check_participant_id(participant_id: str) -> None:
    if not _participants.PARTICIPANT_ID_RE.match(participant_id):
        raise HTTPException(400, "Invalid participant id")


@app.get("/api/projects/{project_id}/participants")
async def list_participants_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        out = [
            p.to_dict()
            for p in _participants.list_participants(_projects_root(), project_id)
        ]
    return JSONResponse({"participants": out})


@app.post("/api/projects/{project_id}/participants")
async def create_participant_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            participant = _participants.Participant.new(
                project_id=project_id,
                name=str(body.get("name", "")),
                pseudonym=str(body.get("pseudonym", "") or ""),
                demographics=body.get("demographics") or {},
                notes=str(body.get("notes", "") or ""),
                source_ids=body.get("source_ids") or [],
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid participant payload: {e}")
        _participants.save_participant(_projects_root(), participant)
    return JSONResponse(participant.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}/participants/{participant_id}")
async def get_participant_endpoint(
    project_id: str, participant_id: str
) -> JSONResponse:
    _check_project_id(project_id)
    _check_participant_id(participant_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            participant = _participants.load_participant(
                _projects_root(), project_id, participant_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Participant not found")
    return JSONResponse(participant.to_dict())


@app.patch("/api/projects/{project_id}/participants/{participant_id}")
async def patch_participant_endpoint(
    project_id: str, participant_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    _check_participant_id(participant_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            participant = _participants.load_participant(
                _projects_root(), project_id, participant_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Participant not found")
        try:
            participant.apply_update(body)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _participants.save_participant(_projects_root(), participant)
    return JSONResponse(participant.to_dict())


@app.delete("/api/projects/{project_id}/participants/{participant_id}")
async def delete_participant_endpoint(
    project_id: str, participant_id: str
) -> JSONResponse:
    _check_project_id(project_id)
    _check_participant_id(participant_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        ok = _participants.delete_participant(
            _projects_root(), project_id, participant_id
        )
    if not ok:
        raise HTTPException(404, "Participant not found")
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# Memos (F5.1) + right-click memo creation (F5.2)
#
# F5.1 added the on-disk Memo entity. F5.2 closes the loop with the
# right-click flow: the editor sends either
#
#   1. a flat memo body — same shape as Memo.to_dict, used when the
#      caller has already built the link list itself; or
#   2. a body containing a top-level ``"context": {target_type,
#      target_id, role?}`` block — the right-click composer's "I just
#      clicked on this thing, populate the link for me" payload.
#
# Both routes converge on the same persisted Memo. Type-defaulting
# and primary-link prepopulation live in scribe.memo_context so the
# JS helpers can mirror them exactly.
# --------------------------------------------------------------------------- #

from . import memos as _memos  # noqa: E402  (after module-level state)
from . import memo_context as _memo_context  # noqa: E402


@app.post("/api/projects/{project_id}/memos")
async def create_memo_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Create a memo, optionally from a right-click context block.

    Accepts either:

    * ``{"context": {"target_type": ..., "target_id": ..., "role": ...},
       "title": "...", "body": "...", ...}`` — F5.2 right-click flow.
       The primary link to the target is prepopulated; ``type``
       defaults to :data:`scribe.memo_context.DEFAULT_MEMO_TYPE_BY_TARGET`
       unless overridden. Additional links may be passed under
       ``extra_links``.
    * A flat memo body (``{"type": ..., "title": ..., "links": [...],
      ...}``) — direct Memo.new shape. Used for non-right-click
      creation paths and tests.

    Returns 201 with the persisted memo on success; 400 on validation;
    404 if the project is missing.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            if "context" in body and body.get("context") is not None:
                # Right-click flow: build via memo_context helper so
                # the type-defaulting and primary-link prepopulation
                # rules match the JS helper one-for-one.
                ctx = body["context"]
                fields: dict[str, Any] = {}
                if "type" in body and body["type"] is not None:
                    fields["type"] = str(body["type"])
                if "title" in body:
                    fields["title"] = str(body.get("title", "") or "")
                if "body" in body:
                    fields["body"] = str(body.get("body", "") or "")
                if "body_format" in body:
                    fields["body_format"] = str(
                        body.get("body_format", "markdown") or "markdown"
                    )
                if "author_coder_id" in body and body["author_coder_id"]:
                    fields["author_coder_id"] = str(body["author_coder_id"])
                if "extra_links" in body:
                    fields["extra_links"] = body.get("extra_links") or []
                if "tags" in body:
                    fields["tags"] = body.get("tags") or []
                if "provenance" in body:
                    fields["provenance"] = body.get("provenance") or {}
                memo = _memo_context.build_memo_draft_from_context(
                    project_id=project_id,
                    context=ctx,
                    **fields,
                )
            else:
                memo = _memos.Memo.new(
                    project_id=project_id,
                    type=str(body.get("type", "free") or "free"),
                    title=str(body.get("title", "") or ""),
                    body=str(body.get("body", "") or ""),
                    body_format=str(
                        body.get("body_format", "markdown") or "markdown"
                    ),
                    author_coder_id=body.get("author_coder_id") or None,
                    links=body.get("links") or [],
                    tags=body.get("tags") or [],
                    provenance=body.get("provenance") or {},
                )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid memo payload: {e}")
        _memos.save_memo(_projects_root(), memo)
    return JSONResponse(memo.to_dict(), status_code=201)


# --------------------------------------------------------------------------- #
# Memo-sorting canvas (F5.3)
#
# One canvas per project: cards (memo→x,y), categories (named clusters),
# and category memberships. Memo→memo links continue to live on the
# Memo entity (F5.1's MemoLink); the link helper here is a thin wrapper
# around scribe.memo_canvas.link_memos_on_canvas, kept on the canvas
# URL surface so the editor can wire memo→memo edges as drag-drop.
#
# Endpoints follow the same locking + project-must-exist contract as
# the other Phase A surfaces.
# --------------------------------------------------------------------------- #

from . import memo_canvas as _memo_canvas  # noqa: E402  (after module-level state)


def _check_category_id(category_id: str) -> None:
    if not _memo_canvas.CATEGORY_ID_RE.match(category_id):
        raise HTTPException(400, "Invalid category id")


def _check_memo_id(memo_id: str) -> None:
    if not _memos.MEMO_ID_RE.match(memo_id):
        raise HTTPException(400, "Invalid memo id")


@app.get("/api/projects/{project_id}/canvas")
async def get_canvas_endpoint(project_id: str) -> JSONResponse:
    """Return the project's memo-sorting canvas.

    Lazy: a project that has never used the canvas returns an empty
    one ({"cards": [], "categories": [], "category_members": {}}).
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
    return JSONResponse(canvas.to_dict())


@app.put("/api/projects/{project_id}/canvas/cards/{memo_id}")
async def put_canvas_card_endpoint(
    project_id: str, memo_id: str, request: Request
) -> JSONResponse:
    """Place / move a card on the canvas at (x, y).

    Idempotent: same coords twice is a no-op apart from the
    ``modified_at`` bump.
    """
    _check_project_id(project_id)
    _check_memo_id(memo_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    if "x" not in body or "y" not in body:
        raise HTTPException(400, "Body must include x and y")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        try:
            canvas.move_card(memo_id, body["x"], body["y"])
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse(canvas.to_dict())


@app.delete("/api/projects/{project_id}/canvas/cards/{memo_id}")
async def delete_canvas_card_endpoint(
    project_id: str, memo_id: str
) -> JSONResponse:
    """Remove a memo's card from the canvas (and any memberships).

    The memo entity itself is untouched. Returns 404 if the memo
    wasn't on the canvas — the operation is meaningless otherwise.
    """
    _check_project_id(project_id)
    _check_memo_id(memo_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        try:
            removed = canvas.remove_card(memo_id)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not removed:
            raise HTTPException(404, "Card not on canvas")
        _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse({"ok": True})


@app.post("/api/projects/{project_id}/canvas/categories")
async def add_canvas_category_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Create a new category on the canvas.

    Body: ``{"label": "...", "color": "#rrggbb"?, "x": float?, "y": float?}``.
    Labels must be unique within the canvas.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        try:
            cat = canvas.add_category(
                label=str(body.get("label", "") or ""),
                color=str(body.get("color", "") or ""),
                x=body.get("x", 0),
                y=body.get("y", 0),
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid category payload: {e}")
        _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse(cat.to_dict(), status_code=201)


@app.patch("/api/projects/{project_id}/canvas/categories/{category_id}")
async def patch_canvas_category_endpoint(
    project_id: str, category_id: str, request: Request
) -> JSONResponse:
    """Update one or more fields on a category."""
    _check_project_id(project_id)
    _check_category_id(category_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    fields: dict[str, Any] = {}
    if "label" in body:
        fields["label"] = str(body["label"] or "")
    if "color" in body:
        fields["color"] = str(body["color"] or "")
    if "x" in body:
        fields["x"] = body["x"]
    if "y" in body:
        fields["y"] = body["y"]
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        try:
            cat = canvas.update_category(category_id, **fields)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse(cat.to_dict())


@app.delete("/api/projects/{project_id}/canvas/categories/{category_id}")
async def delete_canvas_category_endpoint(
    project_id: str, category_id: str
) -> JSONResponse:
    """Remove a category. Cards in it remain on the canvas."""
    _check_project_id(project_id)
    _check_category_id(category_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        ok = canvas.remove_category(category_id)
        if not ok:
            raise HTTPException(404, "Category not found")
        _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse({"ok": True})


@app.put(
    "/api/projects/{project_id}/canvas/categories/{category_id}/members/{memo_id}"
)
async def assign_canvas_member_endpoint(
    project_id: str, category_id: str, memo_id: str
) -> JSONResponse:
    """Make ``memo_id`` a member of ``category_id``.

    The memo must already have a card on the canvas (place it via PUT
    /canvas/cards/<memo_id> first). Idempotent.
    """
    _check_project_id(project_id)
    _check_category_id(category_id)
    _check_memo_id(memo_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        try:
            canvas.assign_card_to_category(memo_id, category_id)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse({"ok": True})


@app.delete(
    "/api/projects/{project_id}/canvas/categories/{category_id}/members/{memo_id}"
)
async def unassign_canvas_member_endpoint(
    project_id: str, category_id: str, memo_id: str
) -> JSONResponse:
    """Remove ``memo_id`` from ``category_id``."""
    _check_project_id(project_id)
    _check_category_id(category_id)
    _check_memo_id(memo_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        canvas = _memo_canvas.load_canvas(_projects_root(), project_id)
        try:
            removed = canvas.unassign_card_from_category(memo_id, category_id)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if removed:
            _memo_canvas.save_canvas(_projects_root(), canvas)
    return JSONResponse({"ok": True, "removed": bool(removed)})


@app.post("/api/projects/{project_id}/canvas/links")
async def link_memos_on_canvas_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Add a memo→memo link from the canvas surface.

    Body: ``{"from_memo_id": "...", "to_memo_id": "...", "role": "..."?}``.
    Wraps :func:`scribe.memo_canvas.link_memos_on_canvas` so the editor
    can record memo↔memo edges from the drag-drop surface without
    issuing a memo PATCH itself. Idempotent on the (from, to, role)
    triple; raises 400 if the source memo doesn't exist.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    from_id = str(body.get("from_memo_id", "") or "")
    to_id = str(body.get("to_memo_id", "") or "")
    role = str(body.get("role", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            memo = _memo_canvas.link_memos_on_canvas(
                _projects_root(),
                project_id,
                from_memo_id=from_id,
                to_memo_id=to_id,
                role=role,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
    return JSONResponse(memo.to_dict())


# --------------------------------------------------------------------------- #
# Promote a memo into a code definition (F5.5)
#
# One-click endpoint: load the memo, mint a Code, persist v1 of its
# definition to the version log, and (by default) back-link the memo
# to the new code with role 'promoted_to'. Codebook lock (F2.4) is
# enforced at the boundary — a locked codebook refuses promotions
# the same way it refuses code edits, mirroring the methodological
# transparency F2.4 was designed for.
# --------------------------------------------------------------------------- #

from . import memo_promote as _memo_promote  # noqa: E402
from . import codebook_lock as _codebook_lock  # noqa: E402


@app.post("/api/projects/{project_id}/memos/{memo_id}/promote-to-code")
async def promote_memo_to_code_endpoint(
    project_id: str, memo_id: str, request: Request
) -> JSONResponse:
    """Promote a memo into a new Code (F5.5).

    Body (all keys optional; the server fills defaults from
    :mod:`scribe.memo_promote` when keys are absent)::

        {
          "name": "...",
          "definition": "...",
          "inclusion_criteria": "...",
          "exclusion_criteria": "...",
          "exemplars": ["..."],
          "parent_code_id": "...",
          "related_codes": [{"code_id": "...", "relation_type": "..."}],
          "theoretical_memo": "...",
          "stage": "initial" | "focused" | "axial" | "theoretical",
          "colour": "#abc",
          "status": "active" | "draft" | "retired",
          "extra_provenance": {"key": "value", ...},
          "code_id": "...",
          "change_note": "...",
          "record_back_link": true,
          "back_link_role": "promoted_to"
        }

    Returns 201 with ``{"code": ..., "version": ..., "memo": ...}`` on
    success. 400 on validation, 404 on missing project/memo, 409 if
    the codebook is locked (F2.4).
    """
    _check_project_id(project_id)
    _check_memo_id(memo_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    # Build the keyword payload for promote_memo_to_code from the body.
    # Only forward keys the user actually supplied; missing keys fall
    # through to the helper's defaults.
    fields: dict[str, Any] = {}
    for key in (
        "name",
        "definition",
        "inclusion_criteria",
        "exclusion_criteria",
        "theoretical_memo",
        "stage",
        "colour",
        "status",
        "code_id",
        "change_note",
        "back_link_role",
    ):
        if key in body and body[key] is not None:
            fields[key] = body[key]
    if "parent_code_id" in body:
        v = body.get("parent_code_id")
        fields["parent_code_id"] = str(v) if v else None
    if "exemplars" in body:
        fields["exemplars"] = body.get("exemplars") or []
    if "related_codes" in body:
        fields["related_codes"] = body.get("related_codes") or []
    if "extra_provenance" in body:
        fields["extra_provenance"] = body.get("extra_provenance") or {}
    if "record_back_link" in body and body["record_back_link"] is not None:
        fields["record_back_link"] = bool(body["record_back_link"])

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            _codebook_lock.assert_codebook_unlocked(
                _projects_root(), project_id
            )
        except _codebook_lock.LockedCodebookError as e:
            raise HTTPException(409, str(e))
        try:
            result = _memo_promote.promote_memo_to_code(
                _projects_root(), project_id, memo_id, **fields
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid promotion payload: {e}")

    return JSONResponse(
        {
            "code": result.code.to_dict(),
            "version": result.version.to_dict(),
            "memo": result.memo.to_dict(),
        },
        status_code=201,
    )


# --------------------------------------------------------------------------- #
# Codebook export (F6.1)
#
# Surface the F2.6 pure exporters (CSV / Markdown / RTF) as a download
# endpoint. F2.6 left the disk-write helper / HTTP surface explicitly
# deferred to F6.1; F6.5 will own the REFI-QDA XML button (kept off
# this endpoint to keep the format set on this URL stable).
#
# Empty codebooks are valid input — they produce a header-only CSV /
# placeholder Markdown / minimal RTF, never a 404. The browser
# downloads the file with a slugified filename derived from the
# project name (``my-pilot-codebook.csv``), per F6.1's "researcher
# pastes this into their thesis appendix" use case.
# --------------------------------------------------------------------------- #

from . import codes as _codes  # noqa: E402
from . import codebook_export as _codebook_export  # noqa: E402


@app.get("/api/projects/{project_id}/codebook/export")
async def export_codebook_endpoint(
    project_id: str, format: str = "csv"
) -> Response:
    """Download the project's codebook in CSV / Markdown / RTF (F6.1).

    Query string ``format``:

    * ``csv`` — RFC-4180 CSV (default).
    * ``markdown`` — structured CommonMark; alias ``md``.
    * ``rtf`` — minimal RTF 1.x (Word, LibreOffice, Pages all open
      it natively); aliases ``word`` / ``doc`` / ``docx``.

    Headers:

    * ``Content-Type`` matches the format (charset=utf-8 for CSV /
      Markdown).
    * ``Content-Disposition: attachment; filename="<slug>-codebook.<ext>"``
      so browsers prompt a save rather than rendering inline.

    Status codes: ``404`` if the project is missing; ``400`` for an
    unrecognised format; ``200`` otherwise (including empty codebooks).
    """
    _check_project_id(project_id)
    try:
        fmt = _codebook_export.normalise_format(format)
    except ValueError as e:
        raise HTTPException(400, str(e))
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        codes = _codes.list_codes(_projects_root(), project_id)
    text = _codebook_export.render_codebook(fmt, codes, project=project)
    spec = _codebook_export.EXPORT_FORMATS[fmt]
    filename = _codebook_export.slugify_codebook_filename(project, fmt)
    headers = {
        # Quote the filename so spaces / non-ASCII never break the
        # header. We slugify to ASCII upstream, so the simple quoted
        # form is sufficient — no need for RFC 5987 ``filename*=``
        # extension.
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=text,
        media_type=spec.media_type,
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# REFI-QDA Codebook XML export (F6.5)
#
# F6.1 (the CSV / Markdown / RTF download) intentionally rejects
# ``format=xml`` / ``format=refi-qda`` so the Codebook XML can grow its
# own surface here. The surface is a separate URL — same project-id
# guard, same atomic-attachment plumbing, but always rendered with
# project-archive metadata (the comment block on the root). REFI-QDA
# importers ignore comments so the XML stays schema-valid in every
# downstream tool.
#
# Errors: 400 for malformed project id (project-id-format check shared
# with the rest of the API); 404 for a project id that doesn't resolve
# under :data:`PROJECTS_DIR`; otherwise 200 with the file body.
# Empty codebooks return a minimal ``<CodeBook><Codes/></CodeBook>``
# (still 200 — the schema permits no-codes, and a downstream import
# would treat that as "merge nothing").
# --------------------------------------------------------------------------- #


@app.get("/api/projects/{project_id}/codebook/refi-qda-xml")
async def export_codebook_refi_qda_xml_endpoint(project_id: str) -> Response:
    """Download the project's codebook as REFI-QDA Codebook XML (F6.5).

    The body is the REFI-QDA Codebook 1.0 XML produced by
    :func:`scribe.codebook_export.render_refi_qda_codebook_xml`,
    which calls :func:`scribe.codebook_export.to_refi_qda_xml` with
    ``include_project_metadata=True`` so the file carries an XML
    comment block summarising the project's research question /
    methodology / sensitising concepts / codebook stage. Importers
    ignore comments — the file remains schema-conformant for any
    REFI-QDA-aware QDA tool.

    Headers:

      * ``Content-Type: application/xml; charset=utf-8``
      * ``Content-Disposition: attachment; filename="<slug>-codebook.refi-qda.xml"``

    Status codes:

      * ``200`` — codebook XML body (including the empty-codebook case;
        a bare ``<CodeBook><Codes/></CodeBook>`` is valid output).
      * ``400`` — malformed project id.
      * ``404`` — project id not found.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        codes = _codes.list_codes(_projects_root(), project_id)
    text = _codebook_export.render_refi_qda_codebook_xml(
        codes, project=project
    )
    filename = _codebook_export.slugify_refi_qda_codebook_xml_filename(project)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=text,
        media_type=_codebook_export.REFI_QDA_XML_MEDIA_TYPE,
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# REFI-QDA / QDPX project export (F6.4)
# --------------------------------------------------------------------------- #

from . import anonymise as _anonymise  # noqa: E402
from . import applications as _applications  # noqa: E402
from . import coders as _coders  # noqa: E402
from . import refi_qda_project as _refi_qda_project  # noqa: E402
from . import speaker_map as _speaker_map  # noqa: E402
# _sources and _memos are already imported earlier; re-importing is
# harmless but kept out for clarity.


def _load_segments_for_source_for_qdpx(
    source: "_sources.Source",
) -> list[dict] | None:
    """Resolve a source's transcript segments under the server's OUTPUT_DIR.

    Mirrors the discovery rules used by the CLI script: prefer
    ``edited.json`` (the editor's authoritative version), fall back to
    any ``*.json`` engine sidecar that has a ``segments`` array.
    Returns ``None`` if no transcript is available — the caller emits
    the source in the QDPX without selections.
    """
    if not getattr(source, "transcript_job_id", ""):
        return None
    job_dir = OUTPUT_DIR / source.transcript_job_id
    if not job_dir.is_dir():
        return None
    edited = job_dir / "edited.json"
    candidates: list[Path] = []
    if edited.is_file():
        candidates.append(edited)
    candidates.extend(
        sorted(p for p in job_dir.glob("*.json") if p.name != "edited.json")
    )
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return data["segments"]
    return None


@app.get("/api/projects/{project_id}/qdpx")
async def export_qdpx_endpoint(project_id: str) -> Response:
    """Download the project as a REFI-QDA / QDPX archive (F6.4).

    The QDPX is a zip with a single ``project.qde`` XML manifest plus
    ``Sources/<source_id>.txt`` plain-text representations of each
    source. Importable into any QDA tool that accepts the REFI-QDA
    interchange format (Atlas.ti, MAXQDA, NVivo, QDA Miner, Quirkos,
    Dedoose).

    Sources whose transcripts can't be found under ``outputs/`` still
    appear in the manifest, but their applications are skipped (we
    can't compute char-offsets without the transcript).

    Status codes: ``404`` if the project is missing; ``200`` otherwise
    (including an empty project — a bare manifest is valid output).
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        sources = _sources.list_sources(_projects_root(), project_id)
        codes = _codes.list_codes(_projects_root(), project_id)
        apps = _applications.list_applications(
            _projects_root(), project_id
        )
        memos = _memos.list_memos(_projects_root(), project_id)
        coders = _coders.list_coders(_projects_root(), project_id)

    rendered_sources: list[_refi_qda_project.RenderedSource] = []
    for s in sources:
        segs = _load_segments_for_source_for_qdpx(s)
        if segs is None:
            continue
        rendered_sources.append(
            _refi_qda_project.render_source_plain_text(s.id, segs)
        )

    archive = _refi_qda_project.to_qdpx(
        project=project,
        sources=sources,
        codes=codes,
        applications=apps,
        memos=memos,
        coders=coders,
        rendered_sources=rendered_sources,
    )

    filename = _refi_qda_project.slugify_qdpx_filename(project)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=archive,
        # QDPX is a zip; ``application/x-qdpx`` is the de facto vendor
        # type used by Atlas.ti et al. but ``application/zip`` is the
        # generic fallback most browsers map to a save dialog.
        media_type="application/x-qdpx",
        headers=headers,
    )


@app.post("/api/projects/{project_id}/qdpx/anonymised")
async def export_anonymised_qdpx_endpoint(
    project_id: str,
    payload: dict | None = Body(default=None),
) -> Response:
    """Download a redacted QDPX archive for the project (F6.7).

    The redaction pass uses every Participant's ``name → pseudonym``
    mapping, every speaker map's resolved pseudonym, and any custom
    rules supplied in the request body. Posts a JSON body of shape::

        {
          "rules": [
            {"pattern": "Mercy General", "replacement": "[hospital]"},
            {"pattern": "\\\\d{3}-\\\\d{4}", "replacement": "[phone]",
             "regex": true}
          ],
          "note": "Pre-publication anon pass"
        }

    Both keys are optional. POST with an empty body relies entirely on
    the participants' pseudonyms.

    The output zip contains the standard QDPX layout *plus* a
    ``Redactions/manifest.json`` listing replacements + match counts
    (never the original identifiers — that would defeat the purpose).

    Status codes:
      * ``200`` — bundle returned.
      * ``400`` — invalid rule payload (bad regex, malformed object).
      * ``404`` — project not found.
    """
    _check_project_id(project_id)
    body = payload or {}
    if not isinstance(body, dict):
        raise HTTPException(400, "Body must be a JSON object")

    raw_rules = body.get("rules", [])
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise HTTPException(400, "'rules' must be a list of rule objects")
    custom_rules: list[_anonymise.RedactionRule] = []
    for entry in raw_rules:
        if not isinstance(entry, dict):
            raise HTTPException(
                400, "each rule must be an object with 'pattern' + 'replacement'"
            )
        try:
            custom_rules.append(_anonymise.RedactionRule.from_dict(entry))
        except ValueError as exc:
            raise HTTPException(400, f"invalid rule: {exc}")

    note = body.get("note") or ""
    if not isinstance(note, str):
        raise HTTPException(400, "'note' must be a string")

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        sources = _sources.list_sources(_projects_root(), project_id)
        codes = _codes.list_codes(_projects_root(), project_id)
        apps = _applications.list_applications(
            _projects_root(), project_id
        )
        memos = _memos.list_memos(_projects_root(), project_id)
        coders = _coders.list_coders(_projects_root(), project_id)
        participants = _participants.list_participants(
            _projects_root(), project_id
        )
        speaker_maps: list[_speaker_map.SpeakerMap] = []
        for s in sources:
            try:
                speaker_maps.append(
                    _speaker_map.load_speaker_map(
                        _projects_root(), project_id, s.id
                    )
                )
            except FileNotFoundError:
                continue

    segments_by_source_id: dict[str, list[dict]] = {}
    for s in sources:
        segs = _load_segments_for_source_for_qdpx(s)
        if segs is None:
            continue
        segments_by_source_id[s.id] = segs

    try:
        bundle = _anonymise.build_anonymised_qdpx(
            project=project,
            sources=sources,
            codes=codes,
            applications=apps,
            memos=memos,
            coders=coders,
            participants=participants,
            speaker_maps=speaker_maps,
            segments_by_source_id=segments_by_source_id,
            custom_rules=custom_rules,
            note=note,
        )
    except ValueError as exc:
        # Bad regex pattern caught at compile time inside the builder.
        raise HTTPException(400, f"invalid rule: {exc}")

    base_filename = _refi_qda_project.slugify_qdpx_filename(project)
    # Insert "-anon" before the extension for a recognisable filename.
    if base_filename.endswith(".qdpx"):
        filename = base_filename[: -len(".qdpx")] + "-anon.qdpx"
    else:
        filename = base_filename + "-anon"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        # Surface the manifest summary in a header for callers that
        # don't want to unzip just to count substitutions.
        "X-Scribe-Anon-Substitutions": str(
            bundle.manifest.get("total_substitutions", 0)
        ),
        "X-Scribe-Anon-Rule-Count": str(
            bundle.manifest.get("rule_count", 0)
        ),
    }
    return Response(
        content=bundle.archive,
        media_type="application/x-qdpx",
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# AI model backend (F8.1)
#
# These endpoints expose the pluggable model-backend abstraction. They
# do *not* run any AI workload — F8.10 still gates that — but they let
# the UI:
#
#   * GET   /ai/backend          — read the saved config + the list of
#                                  registered providers (so the UI can
#                                  populate a dropdown without a probe).
#   * PUT   /ai/backend          — replace the saved config (provider,
#                                  base_url, default_model, etc.).
#   * GET   /ai/backend/health   — actively probe the backend daemon.
#   * GET   /ai/backend/models   — list models installed on the daemon.
#
# Tests use a stub transport (see ``tests/test_server_ai_backend.py``)
# so no real network calls fire.
# --------------------------------------------------------------------------- #

from . import ai_backend as _ai_backend  # noqa: E402


@app.get("/api/projects/{project_id}/ai/backend")
async def get_project_ai_backend_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            cfg = _ai_backend.load_backend_config(project)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))
    body: dict[str, Any] = cfg.to_dict()
    body["extra_headers"] = {k: v for k, v in cfg.extra_headers}
    body["available_providers"] = _ai_backend.list_backends()
    return JSONResponse(body)


@app.put("/api/projects/{project_id}/ai/backend")
async def put_project_ai_backend_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected JSON object")
    headers = payload.pop("extra_headers", None)
    try:
        cfg = _ai_backend.BackendConfig.from_dict(
            payload, extra_headers=headers
        )
    except _ai_backend.BackendValidationError as e:
        raise HTTPException(400, str(e))
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            _ai_backend.store_backend_config(project, cfg)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _projects.save_project(_projects_root(), project)
    body: dict[str, Any] = cfg.to_dict()
    body["extra_headers"] = {k: v for k, v in cfg.extra_headers}
    return JSONResponse(body)


@app.get("/api/projects/{project_id}/ai/backend/health")
async def get_project_ai_backend_health_endpoint(
    project_id: str,
) -> JSONResponse:
    """Probe the configured backend.

    Always returns 200 with a JSON body whose ``ok`` flag indicates
    reachability. We deliberately do *not* return 5xx for an
    unreachable daemon — the frontend wants to render a "backend down"
    banner, not throw a network error of its own.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            cfg = _ai_backend.load_backend_config(project)
            backend = _ai_backend.backend_for_config(cfg)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))
    transport = _ai_backend_transport_override or _ai_backend.urllib_transport
    health = backend.health_check(cfg, transport=transport)
    return JSONResponse(
        {
            "ok": health.ok,
            "provider": health.provider,
            "base_url": health.base_url,
            "detail": health.detail,
            "error": health.error,
        }
    )


@app.get("/api/projects/{project_id}/ai/backend/models")
async def get_project_ai_backend_models_endpoint(
    project_id: str,
) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            cfg = _ai_backend.load_backend_config(project)
            backend = _ai_backend.backend_for_config(cfg)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))
    transport = _ai_backend_transport_override or _ai_backend.urllib_transport
    try:
        models = backend.list_models(cfg, transport=transport)
    except _ai_backend.BackendUnavailable as e:
        # 502: upstream daemon unreachable. The UI surfaces this as
        # "Ollama not running"; doesn't indicate a bug in Scribe.
        raise HTTPException(502, f"Backend unavailable: {e}")
    except _ai_backend.BackendValidationError as e:
        raise HTTPException(400, str(e))
    except _ai_backend.BackendError as e:
        raise HTTPException(500, str(e))
    return JSONResponse({"models": [m.to_dict() for m in models]})


@app.post("/api/projects/{project_id}/ai/backend/pull")
async def post_project_ai_backend_pull_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Pull a model into the configured backend's local store (F8.11).

    Body shape: ``{"model": "<name>"}``. The backend's ``pull_model``
    is invoked synchronously; callers get the full event log in the
    response so the UI can show what each phase took even though we
    don't stream progressively yet (deferred to a later iteration —
    the parser already handles partial streams).
    """
    _check_project_id(project_id)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected JSON object")
    model = str(payload.get("model", "") or "").strip()
    if not model:
        raise HTTPException(400, "model is required")
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            cfg = _ai_backend.load_backend_config(project)
            backend = _ai_backend.backend_for_config(cfg)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))
    transport = _ai_backend_transport_override or _ai_backend.urllib_transport
    try:
        summary = backend.pull_model(cfg, model, transport=transport)
    except _ai_backend.BackendUnavailable as e:
        raise HTTPException(502, f"Backend unavailable: {e}")
    except _ai_backend.BackendValidationError as e:
        raise HTTPException(400, str(e))
    except _ai_backend.BackendError as e:
        raise HTTPException(500, str(e))
    body = summary.to_dict()
    # 200 even when the daemon reported a model-side error: the caller
    # wants the per-event log to render. ``success`` flag distinguishes.
    return JSONResponse(body)


# Test hook: when set, the AI backend endpoints use this transport
# instead of ``urllib_transport``. Tests assign a stub here; production
# leaves it ``None``.
_ai_backend_transport_override: _ai_backend.Transport | None = None


# --------------------------------------------------------------------------- #
# Model-tier picker (F8.11)
#
# The UI calls this endpoint *without a project context*: the tier
# picker reflects what hardware Scribe can see, which is the same
# regardless of which project happens to be open. Returns the
# canonical tier list + a per-tier fit verdict + the recommended
# tier id.
# --------------------------------------------------------------------------- #


from . import model_tiers as _model_tiers  # noqa: E402
from . import model_recommendations as _model_recs  # noqa: E402


@app.get("/api/system/model-tiers")
async def get_system_model_tiers_endpoint() -> JSONResponse:
    """Return tier definitions, hardware snapshot, and recommendation.

    Pure read; no project, no AI invocation. The hardware snapshot can
    be overridden in tests via ``_model_tiers_snapshot_override`` so
    we can pin specific VRAM / RAM combinations without touching torch.
    """
    snapshot = _model_tiers_snapshot_override or _model_tiers.detect_hardware()
    return JSONResponse(_model_tiers.summarise(snapshot))


@app.get("/api/system/model-recommendations")
async def get_system_model_recommendations_endpoint() -> JSONResponse:
    """Return tier picker + concrete model recommendations (F8.12).

    Same hardware-snapshot story as ``/api/system/model-tiers``: pure
    read, no project context, hardware override hook for tests. The
    response shape is the F8.11 summary plus ``recommended_models``
    inside each tier and a top-level ``embedding_models`` array. The
    UI uses this to populate the model picker with sensible defaults
    rather than asking the user to hand-type Ollama tags.
    """
    snapshot = _model_tiers_snapshot_override or _model_tiers.detect_hardware()
    return JSONResponse(_model_recs.summarise_recommendations(snapshot))


# Test hook: when set, the model-tiers endpoint uses this snapshot
# instead of probing real hardware. Tests assign a stub; production
# leaves it ``None``.
_model_tiers_snapshot_override: "_model_tiers.HardwareSnapshot | None" = None


# --------------------------------------------------------------------------- #
# AI gate (F8.10)
#
# Exposes the "first-N-transcripts AI-off" gate's status + config:
#
#   * GET /ai/gate           — current status (allowed / reason / counts /
#                              thresholds), optionally for a specific
#                              feature via ?feature=<id>.
#   * PUT /ai/gate           — replace the saved config (thresholds,
#                              override, exempt features, enabled flag).
#
# These endpoints don't *invoke* AI; they tell the UI / future AI
# endpoints whether AI is allowed yet. Future iterations wire each AI
# feature endpoint to consult :func:`evaluate_project_ai_gate` before
# running.
# --------------------------------------------------------------------------- #

from . import ai_gate as _ai_gate  # noqa: E402


@app.get("/api/projects/{project_id}/ai/gate")
async def get_project_ai_gate_endpoint(
    project_id: str, feature: str = ""
) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            status = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id, feature=feature
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        # Echo the saved config separately so the UI can edit it
        # without re-deriving from the status.
        project = _projects.load_project(_projects_root(), project_id)
        cfg = _ai_gate.load_ai_gate_config(project)
    return JSONResponse(
        {
            "status": status.to_dict(),
            "config": {
                **cfg.to_dict(),
                "exempt_features": list(cfg.exempt_features),
            },
            "available_overrides": list(_ai_gate.GATE_OVERRIDES),
            "available_reasons": list(_ai_gate.REASONS),
        }
    )


@app.put("/api/projects/{project_id}/ai/gate")
async def put_project_ai_gate_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid JSON: {e}")
    if not isinstance(body, dict):
        raise HTTPException(400, "Body must be a JSON object")
    exempt = body.get("exempt_features")
    if exempt is not None and not isinstance(exempt, list):
        raise HTTPException(400, "exempt_features must be a list")
    try:
        cfg = _ai_gate.AIGateConfig.from_dict(
            {
                k: v
                for k, v in body.items()
                if k != "exempt_features"
            },
            exempt_features=exempt,
        )
    except _projects.ProjectValidationError as e:
        raise HTTPException(400, str(e))
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            _ai_gate.store_ai_gate_config(project, cfg)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _projects.save_project(_projects_root(), project)
        # Recompute the status so the response reflects the new config.
        status = _ai_gate.evaluate_project_ai_gate(
            _projects_root(), project_id
        )
    return JSONResponse(
        {
            "status": status.to_dict(),
            "config": {
                **cfg.to_dict(),
                "exempt_features": list(cfg.exempt_features),
            },
        }
    )


# --------------------------------------------------------------------------- #
# Upload + transcription job lifecycle
# --------------------------------------------------------------------------- #


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    speakers: str = Form(""),
    num_speakers: str = Form(""),
    language: str = Form("en"),
    model: str = Form("large-v3"),
    batch_size: str = Form("8"),
    options: str = Form("{}"),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    safe_name = Path(file.filename or "upload.bin").name
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / safe_name

    with input_path.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    speakers_list = [s.strip() for s in speakers.split(",") if s.strip()] or None
    nspk = int(num_speakers) if num_speakers.strip().isdigit() else None

    try:
        opts_dict = json.loads(options) if options else {}
        if not isinstance(opts_dict, dict):
            raise ValueError("options must be a JSON object")
    except Exception as e:
        raise HTTPException(400, f"Invalid options JSON: {e}")

    try:
        bs = max(1, min(64, int(batch_size)))
    except ValueError:
        bs = 8

    try:
        streams = probe_audio_streams(input_path)
    except Exception as e:
        raise HTTPException(400, f"Could not read audio streams: {e}")

    if not streams:
        raise HTTPException(400, "No audio streams found in this file.")

    try:
        media_info = probe_media_info(input_path)
    except Exception:
        media_info = None

    job = Job(
        id=job_id,
        input_path=input_path,
        output_dir=output_dir,
        mode=mode,
        speakers=speakers_list,
        num_speakers=nspk,
        language=language,
        model=model,
        created_at=datetime.utcnow().isoformat() + "Z",
        audio_streams=len(streams),
        input_filename=safe_name,
        options=opts_dict,
        batch_size=bs,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    _persist_job(job)

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return JSONResponse(
        {
            "job_id": job_id,
            "audio_streams": len(streams),
            "stream_titles": [s.title or f"track {i+1}" for i, s in enumerate(streams)],
            "media_info": media_info,
        }
    )


def _set_progress(job_id: str, msg: str, frac: float) -> None:
    import time as _time
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.message = msg
        job.progress = max(0.0, min(1.0, frac))
        if job.status == "queued":
            job.status = "running"
        if job.started_at is None:
            job.started_at = _time.time()


def _run_job(job_id: str) -> None:
    import time as _time
    with JOBS_LOCK:
        job = JOBS[job_id]
        if job.started_at is None:
            job.started_at = _time.time()
    try:
        result = transcribe(
            job.input_path,
            work_dir=job.output_dir / "work",
            mode=job.mode,  # type: ignore[arg-type]
            speaker_labels=job.speakers,
            num_speakers=job.num_speakers,
            model_name=job.model,
            language=job.language,
            batch_size=job.batch_size,
            hf_token=os.environ.get("HF_TOKEN"),
            options=AdvancedOptions.from_dict(job.options),
            progress=lambda m, f: _set_progress(job_id, m, f),
        )
        base = job.output_dir / job.input_path.stem
        paths = write_all(result, base)
        shutil.rmtree(job.output_dir / "work", ignore_errors=True)

        with JOBS_LOCK:
            job.status = "done"
            job.progress = 1.0
            job.message = "Done"
            job.result = result.to_dict()
            job.output_paths = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
            job.finished_at = _time.time()
        _persist_job(job)
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        tb_text = _tb.format_exc()
        # Always print to stdout so the developer sees it without digging into
        # the persisted state.
        print(f"[scribe] job {job_id} failed:\n{tb_text}", flush=True)
        # Persist the full traceback for the UI to fetch via /api/job/{id}/error
        try:
            (job.output_dir / "error.log").write_text(tb_text)
        except Exception:
            pass
        with JOBS_LOCK:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.message = "Error"
            job.finished_at = _time.time()
        _persist_job(job)


@app.get("/api/job/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return JSONResponse(_job_dict(job))


def _job_dict(job: Job) -> dict[str, Any]:
    # Cross-check the persisted media_discarded flag against the
    # filesystem so a stale True (from a partial discard, hand-edit,
    # or old serialiser bug) doesn't mislead the editor into hiding
    # the player when the source media is actually still there.
    discarded = bool(job.media_discarded)
    try:
        upload_dir = job.input_path.parent
        if upload_dir.exists() and any(upload_dir.iterdir()):
            discarded = False
    except Exception:
        pass
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "mode": job.mode,
        "audio_streams": job.audio_streams,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "output_paths": job.output_paths,
        "result": job.result,
        "input_filename": job.input_filename,
        "media_discarded": discarded,
    }


# --------------------------------------------------------------------------- #
# F10.1 — Library: list + delete
#
# The home-page upload form is unchanged; ``/library`` is a separate
# page (rendered above) that calls ``GET /api/jobs`` to render every
# persisted transcription as a sortable, filterable row. Deletion
# wipes the per-job ``uploads/`` and ``outputs/`` trees in one shot
# so reclaiming disk space is a single action.
# --------------------------------------------------------------------------- #


from . import library as _library  # noqa: E402  (after module-level state)


@app.get("/api/jobs")
async def list_jobs_endpoint(q: str = "") -> JSONResponse:
    """Return summary rows for every persisted job, newest first.

    The optional ``q`` query string applies the server-side
    case-insensitive substring filter (filename / speakers / mode /
    status / language / model). The client keeps a copy of the full
    list so live typing into the search box doesn't round-trip.
    """
    with JOBS_LOCK:
        # Snapshot-copy the values so we can release the lock before
        # building the heavy summary dicts. Pair each Job with a
        # filesystem-truth fingerprint so we can correct any stale
        # ``media_discarded`` flags below without re-querying JOBS.
        jobs_snapshot = list(JOBS.values())
        fs_truth: dict[str, bool] = {}
        for j in jobs_snapshot:
            try:
                # The source media lives under uploads/<id>/. The flag is
                # True ⇔ that directory should be gone. If the path or its
                # parent still exists with files in it, the flag is stale.
                upload_dir = j.input_path.parent
                fs_truth[j.id] = (
                    not upload_dir.exists()
                    or not any(upload_dir.iterdir())
                )
            except Exception:
                # If we can't probe (permission error, parent vanished),
                # trust the persisted flag rather than guess.
                fs_truth[j.id] = bool(getattr(j, "media_discarded", False))
    rows = _library.summarise_jobs(jobs_snapshot)
    # Reconcile: a row claiming media_discarded=True while the upload
    # directory still has files means the persisted flag is stale —
    # likely a hand-edit, an older serialiser that wrote a stringified
    # false, or an interrupted discard that left the dir behind. Trust
    # the filesystem; correct the row in-place. The persisted job.json
    # gets corrected lazily next time the user runs an action that
    # triggers ``_persist_job``.
    for r in rows:
        actual_discarded = fs_truth.get(r.get("id"), r.get("media_discarded", False))
        r["media_discarded"] = bool(actual_discarded)
    if q:
        rows = _library.filter_rows(rows, q)
    return JSONResponse({"jobs": rows, "total": len(rows)})


# --------------------------------------------------------------------------- #
# F10.3 — Import an existing transcript
#
# Researchers often arrive with an already-finished transcript (their
# own .txt, an SRT, a Scribe export from another machine). We give
# them a way in without re-running the engine: the parser produces
# the same envelope the worker would, the writers run normally, and
# the resulting job lands in the library indistinguishable from a
# transcribed one — except ``media_discarded=True`` when no companion
# media file is uploaded.
# --------------------------------------------------------------------------- #


from . import transcript_import as _transcript_import  # noqa: E402


# Cap the size of an uploaded transcript file. Even a several-hour
# interview's text export is well under 1 MB; subtitle files are
# similar. 10 MB is paranoia, not a real ceiling.
_TRANSCRIPT_MAX_BYTES = 10 * 1024 * 1024
# Companion media can be anything ffmpeg reads; same cap as a normal
# upload (FastAPI doesn't impose one and we don't want a runaway
# file to fill the disk).
_IMPORT_MEDIA_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


@app.post("/api/import")
async def import_transcript_endpoint(
    transcript: UploadFile = File(...),
    media: UploadFile | None = File(None),
    fmt: str = Form(""),
    language: str = Form(""),
) -> JSONResponse:
    """Treat an uploaded transcript as a finished job.

    Accepts ``transcript`` as the required transcript file (TXT /
    SRT / VTT / Scribe JSON; sniffed automatically unless ``fmt`` is
    set), and an optional ``media`` companion file so the editor can
    play audio/video alongside the transcript. When ``media`` is
    omitted the resulting job is created with
    ``media_discarded=True`` (F10.2): the editor degrades to
    no-playback, but everything else (text, search, edit, export)
    works.

    Returns the same shape as ``POST /api/upload`` so the client UI
    can route to the editor with no special-casing.
    """
    raw = await transcript.read(_TRANSCRIPT_MAX_BYTES + 1)
    if len(raw) > _TRANSCRIPT_MAX_BYTES:
        raise HTTPException(413, "Transcript file is too large (>10 MB).")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(400, "Transcript must be UTF-8 text.")

    fname = Path(transcript.filename or "").name or "imported.txt"
    fmt_arg: str | None = fmt.strip() or None
    if fmt_arg and fmt_arg not in _transcript_import.KNOWN_FORMATS:
        raise HTTPException(400, f"Unknown transcript format: {fmt_arg!r}")

    try:
        envelope = _transcript_import.parse_transcript(
            fname, content, fmt=fmt_arg
        )
    except ValueError as e:
        raise HTTPException(400, f"Could not parse transcript: {e}")

    # Honour an explicit override of the detected language (the form
    # field comes from the upload UI's existing language picker).
    if language.strip():
        envelope["language"] = language.strip()

    job_id = uuid.uuid4().hex[:12]
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Stem is the transcript's filename without its extension; the
    # writers append the right extension per format.
    stem = Path(fname).stem or "transcript"

    # Pull in the companion media if present, otherwise note that the
    # job has no source media and the editor should degrade.
    media_discarded = True
    media_input_path: Path = upload_dir / fname  # placeholder for path field
    if media is not None and (media.filename or "").strip():
        media_name = Path(media.filename).name
        if not media_name:
            media_name = f"{stem}.bin"
        media_input_path = upload_dir / media_name
        bytes_written = 0
        with media_input_path.open("wb") as f:
            while chunk := await media.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > _IMPORT_MEDIA_MAX_BYTES:
                    f.close()
                    media_input_path.unlink(missing_ok=True)
                    raise HTTPException(413, "Companion media file too large.")
                f.write(chunk)
        media_discarded = False
        try:
            streams = probe_audio_streams(media_input_path)
        except Exception:
            streams = []
        try:
            media_info = probe_media_info(media_input_path)
        except Exception:
            media_info = None
        audio_streams = len(streams) if streams else 0
    else:
        # Drop the empty upload dir if there's no media — F10.2's
        # discard logic already handles "no uploads/<id>/" gracefully
        # but it's tidy to not leave it behind.
        try:
            upload_dir.rmdir()
        except OSError:
            pass
        media_info = None
        audio_streams = 0

    # Build the standard TranscriptionResult and write the sidecars
    # (.json, .txt, .srt, .vtt) so the rest of the system sees this
    # job as indistinguishable from a transcribed one.
    result = _result_from_payload(envelope, input_path=media_input_path)
    base = output_dir / stem
    paths = write_all(result, base)

    now_iso = datetime.utcnow().isoformat() + "Z"
    import time as _time
    now_epoch = _time.time()
    job = Job(
        id=job_id,
        input_path=media_input_path,
        output_dir=output_dir,
        mode=envelope.get("mode", "diarize"),
        speakers=envelope.get("speakers") or None,
        num_speakers=None,
        language=envelope.get("language", "en"),
        model="imported",
        created_at=now_iso,
        status="done",
        progress=1.0,
        message="Imported",
        result=result.to_dict(),
        output_paths={k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        audio_streams=audio_streams,
        input_filename=fname,
        options={},
        batch_size=8,
        started_at=now_epoch,
        finished_at=now_epoch,
        media_discarded=media_discarded,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    _persist_job(job)

    return JSONResponse(
        {
            "job_id": job_id,
            "audio_streams": audio_streams,
            "stream_titles": [],
            "media_info": media_info,
            "imported": True,
            "media_discarded": media_discarded,
            "format": fmt_arg or _transcript_import.sniff_format(fname, content),
            "segment_count": len(envelope.get("segments") or []),
        }
    )


@app.post("/api/job/{job_id}/discard-media")
async def discard_media_endpoint(job_id: str) -> JSONResponse:
    """F10.2 — drop the source media for a job, keep the transcript.

    Removes the per-job ``uploads/<id>/`` directory (which holds the
    original recording) and rewrites ``job.json`` with
    ``media_discarded: true``. The output sidecars
    (``.json/.txt/.srt/.vtt/edited.json/waveform_*.json``) are
    untouched so the editor can still load and edit the transcript.

    Idempotent: a second call on an already-discarded job is a 200 with
    ``already=true`` and no filesystem work, so a duplicate click from
    the UI doesn't error.

    Refuses (403) when the upload path escapes the configured
    ``UPLOAD_DIR``, matching the same containment guard as the delete
    endpoint.
    """
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        # In-progress jobs still need their source on disk for the
        # worker; refuse to discard until the engine has finished.
        if job.status not in ("done", "error"):
            raise HTTPException(
                409,
                f"Cannot discard media while job is {job.status}",
            )
        in_path = job.input_path.resolve()
        already = bool(job.media_discarded)
    upload_dir = in_path.parent
    if upload_dir != UPLOAD_DIR.resolve() and not _is_under(upload_dir, UPLOAD_DIR.resolve()):
        raise HTTPException(403, "input_path escapes UPLOAD_DIR")
    if already:
        # Defensive cleanup in case a partial earlier discard left
        # stragglers — best-effort, no error surfaced.
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        return JSONResponse({"ok": True, "id": job_id, "already": True})
    # Remove the upload directory (source media + anything else the
    # uploader cached there). Errors deleting individual files are
    # *not* surfaced because the user-facing semantic is "best-effort
    # reclaim disk space"; the persistent flag below is what the rest
    # of the system reads.
    shutil.rmtree(upload_dir, ignore_errors=True)
    with JOBS_LOCK:
        # Re-check — the job could have been deleted out from under
        # us between unlock and re-lock.
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "Job not found")
        j.media_discarded = True
    _persist_job(j)
    return JSONResponse({"ok": True, "id": job_id, "already": False})


@app.delete("/api/job/{job_id}")
async def delete_job_endpoint(job_id: str) -> JSONResponse:
    """Remove a job from the registry and wipe its on-disk artefacts.

    Both the upload directory (``uploads/<id>/``) and the output
    directory (``outputs/<id>/``) are removed atomically from the
    user's perspective: the in-memory job is dropped first, then the
    filesystem is cleaned up best-effort. Errors deleting individual
    files are *not* surfaced because an interrupted partial delete
    is preferable to a stale registry entry the user can't act on.

    Returns 404 if no such job is registered.
    """
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
        if job is None:
            raise HTTPException(404, "Job not found")
        out_dir = job.output_dir.resolve()
        in_path = job.input_path.resolve()
    # Both paths must live inside the configured roots — refuse to
    # touch anything outside them so a hand-edited job.json can't
    # rm-rf the developer's filesystem.
    if not _is_under(out_dir, OUTPUT_DIR.resolve()):
        # Re-instate the job to keep the registry consistent.
        with JOBS_LOCK:
            JOBS[job_id] = job
        raise HTTPException(403, "output_dir escapes OUTPUT_DIR")
    upload_dir = in_path.parent
    if upload_dir != UPLOAD_DIR.resolve() and not _is_under(upload_dir, UPLOAD_DIR.resolve()):
        with JOBS_LOCK:
            JOBS[job_id] = job
        raise HTTPException(403, "input_path escapes UPLOAD_DIR")
    # Delete the per-job upload directory (which holds the source
    # media file) and the entire outputs tree (sidecars, edits,
    # waveform cache, error log).
    shutil.rmtree(upload_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    return JSONResponse({"ok": True, "id": job_id})


@app.get("/api/job/{job_id}/error")
async def job_error_log(job_id: str) -> Response:
    """Return the persisted Python traceback from a failed job, if any."""
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        log_path = (job.output_dir / "error.log").resolve()
        out_dir = job.output_dir.resolve()
    if not _is_under(log_path, out_dir) or not log_path.exists():
        raise HTTPException(404, "No error log for this job")
    return Response(content=log_path.read_text(errors="replace"), media_type="text/plain")


@app.get("/api/job/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    _check_job_id(job_id)
    async def generator():
        last = None
        while True:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'error':'job not found'})}\n\n"
                    return
                payload = {
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "error": job.error,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                }
            if payload != last:
                yield f"data: {json.dumps(payload)}\n\n"
                last = payload
            if payload["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/job/{job_id}/download/{kind}")
async def download(job_id: str, kind: str) -> FileResponse:
    _check_job_id(job_id)
    if kind not in {"json", "txt", "srt", "vtt"}:
        raise HTTPException(400, "Invalid format")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if kind not in job.output_paths:
            raise HTTPException(404, f"Format '{kind}' not available")
        rel = job.output_paths[kind]
    full = (ROOT / rel).resolve()
    if not _is_under(full, OUTPUT_DIR):
        raise HTTPException(403, "Forbidden")
    return FileResponse(full, filename=full.name)


# --------------------------------------------------------------------------- #
# Settings — Hugging Face token
# --------------------------------------------------------------------------- #


_HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")


def _mask_token(token: str) -> str:
    if len(token) <= 8:
        return "•" * len(token)
    return f"{token[:4]}…{token[-4:]}"


def _read_env_file() -> dict[str, str]:
    """Parse .env into a dict. Only handles simple KEY=VALUE lines we write ourselves."""
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _write_env_file(values: dict[str, str]) -> None:
    """Rewrite .env preserving comments/blank lines, updating known keys in place."""
    seen: set[str] = set()
    new_lines: list[str] = []

    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                new_lines.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                new_lines.append(f"{key}={values[key]}")
                seen.add(key)
            else:
                new_lines.append(line)

    for key, val in values.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")

    ENV_PATH.write_text("\n".join(new_lines).rstrip() + "\n")
    try:
        ENV_PATH.chmod(0o600)
    except OSError:
        pass


@app.get("/api/settings/hf_token")
async def get_hf_token() -> JSONResponse:
    token = os.environ.get("HF_TOKEN", "")
    return JSONResponse({"set": bool(token), "masked": _mask_token(token) if token else ""})


@app.put("/api/settings/hf_token")
async def put_hf_token(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected JSON object")
    token = (payload.get("token") or "").strip()
    if token and not _HF_TOKEN_RE.match(token):
        raise HTTPException(400, "Token must look like 'hf_…' (alphanumeric, ≥20 chars after prefix)")

    values = _read_env_file()
    if token:
        values["HF_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    else:
        values["HF_TOKEN"] = ""
        os.environ.pop("HF_TOKEN", None)
    _write_env_file(values)

    return JSONResponse({"set": bool(token), "masked": _mask_token(token) if token else ""})


# --------------------------------------------------------------------------- #
# Editor — media + transcript get/save
# --------------------------------------------------------------------------- #


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _check_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job id")


@app.get("/api/job/{job_id}/info")
async def job_info(job_id: str) -> JSONResponse:
    """Return media info (duration, codecs, etc) for a job."""
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.media_discarded:
            raise HTTPException(410, "Source media discarded for this job")
        path = job.input_path.resolve()
    if not _is_under(path, UPLOAD_DIR):
        raise HTTPException(403, "Forbidden")
    if not path.exists():
        raise HTTPException(404, "Source file is missing on disk")
    try:
        return JSONResponse(probe_media_info(path))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Could not probe media: {e}")


@app.get("/api/job/{job_id}/waveform")
async def job_waveform(job_id: str, bins: int = 1000) -> JSONResponse:
    """
    Compute (and cache) a peak-amplitude waveform for the input audio.
    Cached on disk so the upload page can re-fetch instantly on reload.
    """
    _check_job_id(job_id)
    bins = max(50, min(4000, int(bins)))
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.media_discarded:
            raise HTTPException(410, "Source media discarded for this job")
        path = job.input_path.resolve()
        out_dir = job.output_dir.resolve()
    if not _is_under(path, UPLOAD_DIR) or not _is_under(out_dir, OUTPUT_DIR):
        raise HTTPException(403, "Forbidden")
    if not path.exists():
        raise HTTPException(404, "Source file is missing on disk")

    cache = out_dir / f"waveform_{bins}.json"
    if cache.exists():
        try:
            return JSONResponse(json.loads(cache.read_text()))
        except Exception:
            cache.unlink(missing_ok=True)

    try:
        peaks = compute_waveform(path, bins=bins)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Could not compute waveform: {e}")

    payload = {"bins": bins, "peaks": peaks}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
    except Exception:
        pass
    return JSONResponse(payload)


@app.get("/api/job/{job_id}/media")
async def media(job_id: str, request: Request) -> Response:
    """Serve the original recording with HTTP Range support so <video>/<audio> can seek."""
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.media_discarded:
            raise HTTPException(410, "Source media discarded for this job")
        path = job.input_path.resolve()
    if not _is_under(path, UPLOAD_DIR):
        raise HTTPException(403, "Forbidden")
    if not path.exists():
        raise HTTPException(404, "Source file is missing on disk")

    file_size = path.stat().st_size
    content_type, _ = mimetypes.guess_type(str(path))
    content_type = content_type or "application/octet-stream"

    range_header = request.headers.get("range") or request.headers.get("Range")
    if not range_header:
        return FileResponse(path, media_type=content_type)

    m = _RANGE_RE.fullmatch(range_header.strip())
    if not m:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    if start_s == "":
        # Suffix range: "bytes=-N" → last N bytes.
        suffix = int(end_s)
        if suffix == 0:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
    if start > end or end >= file_size or start < 0:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    chunk_size = 1024 * 1024
    length = end - start + 1

    def iter_file():
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": content_type,
    }
    return StreamingResponse(iter_file(), status_code=206, headers=headers, media_type=content_type)


@app.get("/api/job/{job_id}/transcript")
async def get_transcript(job_id: str) -> JSONResponse:
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")

    edited = _edited_path(job.output_dir)
    if edited.exists():
        try:
            return JSONResponse(json.loads(edited.read_text()))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Could not read edited transcript: {e}")
    if job.result:
        return JSONResponse(job.result)
    raise HTTPException(404, "No transcript available")


@app.put("/api/job/{job_id}/transcript")
async def put_transcript(job_id: str, request: Request) -> JSONResponse:
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        out_dir = job.output_dir.resolve()
        input_stem = job.input_path.stem

    if not _is_under(out_dir, OUTPUT_DIR):
        raise HTTPException(403, "Forbidden")

    # Cap request body size — local app, but no reason to allow unbounded.
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > 50 * 1024 * 1024:
        raise HTTPException(413, "Transcript too large")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    if not isinstance(payload, dict) or "segments" not in payload:
        raise HTTPException(400, "Expected JSON object with 'segments'")

    # Persist the edited JSON
    edited = _edited_path(out_dir)
    edited.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Regenerate sidecar exports (TXT/SRT/VTT) from edited content
    try:
        result = _result_from_payload(payload, input_path=Path("dummy"))
        base = out_dir / input_stem
        write_txt(result, base.with_suffix(".txt"))
        write_srt(result, base.with_suffix(".srt"))
        write_vtt(result, base.with_suffix(".vtt"))
        write_json(result, base.with_suffix(".json"))

        with JOBS_LOCK:
            job.result = payload
            job.output_paths = {
                "json": str((base.with_suffix(".json")).relative_to(ROOT)),
                "txt": str((base.with_suffix(".txt")).relative_to(ROOT)),
                "srt": str((base.with_suffix(".srt")).relative_to(ROOT)),
                "vtt": str((base.with_suffix(".vtt")).relative_to(ROOT)),
            }
        _persist_job(job)
    except Exception as e:  # noqa: BLE001
        # The JSON was saved; the regeneration failed — surface that to the client.
        raise HTTPException(500, f"Saved JSON but failed to regenerate sidecars: {e}")

    return JSONResponse({"ok": True, "saved_at": datetime.utcnow().isoformat() + "Z"})


def _result_from_payload(payload: dict[str, Any], *, input_path: Path) -> TranscriptionResult:
    segs: list[Segment] = []
    speakers_seen: list[str] = []
    for s in payload.get("segments", []):
        speaker = s.get("speaker") or "SPEAKER_??"
        if speaker not in speakers_seen:
            speakers_seen.append(speaker)
        words = [
            Word(
                text=str(w.get("text", "")),
                start=float(w.get("start", s.get("start", 0))),
                end=float(w.get("end", s.get("end", 0))),
                speaker=str(w.get("speaker", speaker)),
                score=float(w["score"]) if w.get("score") is not None else None,
            )
            for w in s.get("words", [])
            if str(w.get("text", "")).strip()
        ]
        segs.append(
            Segment(
                text=str(s.get("text", "")),
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
                speaker=speaker,
                words=words,
            )
        )
    return TranscriptionResult(
        segments=segs,
        language=payload.get("language", "en"),
        mode=payload.get("mode", "diarize"),
        speaker_labels=payload.get("speakers", speakers_seen),
        audio_path=input_path,
    )
