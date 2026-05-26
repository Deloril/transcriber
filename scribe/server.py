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
from fastapi import Body, FastAPI, HTTPException, Query, Request, UploadFile, File, Form
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
    # also useful as a "create one" landing. (F1.1)
    project = None
    try:
        with PROJECTS_LOCK:
            project = _projects.load_project(_projects_root(), pid)
    except Exception:
        project = None
    return templates.TemplateResponse(request, "project_home.html", {
        "project_id": pid,
        "project_name": getattr(project, "name", None),
        "project_methodology": getattr(project, "methodology", None),
        "project_stage": getattr(project, "codebook_stage", None),
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
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "sources_list.html", {
        "project_id": pid,
        "page_title": "Sources",
    })


@app.get("/projects/{project_id}/sources/add", response_class=HTMLResponse)
async def project_source_add_page(request: Request, project_id: str) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "source_picker.html", {
        "project_id": pid,
        "page_title": "Add source",
    })


@app.get("/projects/{project_id}/sources/schema", response_class=HTMLResponse)
async def project_source_schema_page(
    request: Request, project_id: str
) -> HTMLResponse:
    """Source-attribute schema editor (F3.2).

    The pure data layer in ``scribe/source_schema.py`` lets a project
    declare typed columns (key / label / type / required / options /
    description) for each source's ``custom_attributes``. Without this
    UI the schema is unreachable: only the JSON PUT endpoint accepted
    it. This page renders the form that reads + writes the schema via
    ``GET / PUT /api/projects/<pid>/source_schema`` and the resulting
    columns are surfaced in the sources list (F1.2).
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "source_schema.html", {
        "project_id": pid,
        "page_title": "Source attribute schema",
        "subtitle": "Declare typed columns for the sources in this project.",
    })


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
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "codebook_editor.html", {
        "project_id": pid,
        "page_title": "Codebook",
    })


# --------------------------------------------------------------------------- #
# Participants (F1.3) — UI surface
#
# The data layer + REST API for participants shipped in caa3553. This block
# wires the user-facing pages so a researcher can list, create, and edit
# participants from the project shell. The API endpoints below in this
# file (POST/GET/PATCH/DELETE /api/projects/<pid>/participants[/<part_id>])
# are the same surface tests/test_server.py::TestParticipantsAPI covers.
# --------------------------------------------------------------------------- #


@app.get("/projects/{project_id}/participants", response_class=HTMLResponse)
async def project_participants_page(
    request: Request, project_id: str
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "participants_list.html", {
        "project_id": pid,
        "page_title": "Participants",
    })


@app.get("/projects/{project_id}/participants/new", response_class=HTMLResponse)
async def project_participant_new_page(
    request: Request, project_id: str
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "participant_new.html", {
        "project_id": pid,
        "page_title": "New participant",
    })


@app.get(
    "/projects/{project_id}/participants/{participant_id}",
    response_class=HTMLResponse,
)
async def project_participant_detail_page(
    request: Request, project_id: str, participant_id: str
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    # Reuse the same shape-check as the API (12-char hex). Don't 404 on
    # missing — the page shows a friendly error so the user can navigate
    # back without a hard server error.
    if not _participants.PARTICIPANT_ID_RE.match(participant_id):
        raise HTTPException(400, "Invalid participant id")
    return templates.TemplateResponse(request, "participant_detail.html", {
        "project_id": pid,
        "participant_id": participant_id,
        "page_title": "Participant",
    })

# --------------------------------------------------------------------------- #
# Coders (F2.5) — UI surface
#
# Pages backing the F2.5 multi-coder mode. The data layer (Coder
# entity + ICR statistics) shipped in cae5570 with no HTTP surface;
# the routes here let a researcher list, create, and edit coders, and
# the /icr page lets them compare two coders' applications via
# Cohen's kappa.
# --------------------------------------------------------------------------- #


@app.get("/projects/{project_id}/coders", response_class=HTMLResponse)
async def project_coders_page(
    request: Request, project_id: str
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "coders_list.html", {
        "project_id": pid,
        "page_title": "Coders",
    })


@app.get("/projects/{project_id}/coders/new", response_class=HTMLResponse)
async def project_coder_new_page(
    request: Request, project_id: str
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "coder_new.html", {
        "project_id": pid,
        "page_title": "New coder",
    })


@app.get(
    "/projects/{project_id}/coders/{coder_id}",
    response_class=HTMLResponse,
)
async def project_coder_detail_page(
    request: Request, project_id: str, coder_id: str
) -> HTMLResponse:
    pid = _project_id_or_404(project_id)
    # Same shape check as the API. Don't 404 on missing — the page
    # shows a friendly error so the user can navigate back.
    if not _coders.CODER_ID_RE.match(coder_id):
        raise HTTPException(400, "Invalid coder id")
    return templates.TemplateResponse(request, "coder_detail.html", {
        "project_id": pid,
        "coder_id": coder_id,
        "page_title": "Coder",
    })


@app.get("/projects/{project_id}/icr", response_class=HTMLResponse)
async def project_icr_page(
    request: Request, project_id: str
) -> HTMLResponse:
    """ICR comparison view (F2.5).

    Picks any two coders + an optional source filter; renders Cohen's
    kappa overall and per code, with the Landis & Koch interpretation
    label and per-code application counts. The page consumes
    ``GET /api/projects/<pid>/icr`` (above).
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "icr.html", {
        "project_id": pid,
        "page_title": "Inter-coder reliability",
    })


@app.get("/projects/{project_id}/sampling-log", response_class=HTMLResponse)
async def project_sampling_log_page(
    request: Request, project_id: str
) -> HTMLResponse:
    """List + append the project's theoretical-sampling log (F1.4).

    The log is project-wide append-only evidence; the page shows the
    chronological entries (newest first in the rendered table) and
    surfaces a small inline form to append a new entry. The form is
    on the same page rather than a separate /new route because the
    individual entries are tiny (rationale, optional ids, decision
    type) and the researcher mostly comes here to scan the trail.
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(
        request,
        "sampling_log.html",
        {
            "project_id": pid,
            "page_title": "Sampling log",
        },
    )


@app.get("/projects/{project_id}/queries", response_class=HTMLResponse)
async def project_queries_page(request: Request, project_id: str) -> HTMLResponse:
    """Query builder page (F3.5).

    Replaces the previous wireframe stub with a real page that lets
    a researcher build a query (code filter + source filter + speaker
    role filter), POST it to ``/api/projects/<pid>/queries/run``, and
    see the matching applications inline. Saved queries (F3.7) and
    matrix views (F3.6) live on the same page as future graduations;
    the minimum-viable F3.5 surface ships first so the pure
    ``scribe.query`` executor stops being unreachable from the UI.
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "queries.html", {
        "project_id": pid,
        "page_title": "Queries",
    })


@app.get("/projects/{project_id}/memos", response_class=HTMLResponse)
async def project_memos_page(request: Request, project_id: str) -> HTMLResponse:
    """Memos page (F5.1 user-facing surface).

    Replaces the wireframe stub. The pure module
    (:mod:`scribe.memos`) and the POST endpoint shipped earlier; this
    page is the list / create / edit / delete UI that finally makes
    F5.1 reachable from the project nav. The sorting-canvas (F5.3),
    "promote to code" affordance (F5.5), and "✨ draft with AI" (F8.8)
    layer onto this page as later graduations.
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "memos.html", {
        "project_id": pid,
        "page_title": "Memos",
    })


@app.get("/projects/{project_id}/memos/canvas", response_class=HTMLResponse)
async def project_memo_canvas_page(
    request: Request, project_id: str
) -> HTMLResponse:
    """Memo-sorting canvas page (F5.3 user-facing surface).

    The pure module (:mod:`scribe.memo_canvas`) and the API surface
    shipped in 9e1fd99 (cards / categories / memo↔memo links) but the
    drag-drop UI was deferred. This route ships the user-facing
    canvas: a 2D board where memos already in the project can be laid
    out, dragged, grouped into named categories ("emerging concept",
    "structural conditions", …), and linked memo→memo with a
    typed-relation role. Mirrors NVivo's concept-map / ATLAS.ti's
    network-view affordance.

    Reaches:
      * GET    /api/projects/<pid>/canvas
      * PUT    /api/projects/<pid>/canvas/cards/<memo_id>
      * DELETE /api/projects/<pid>/canvas/cards/<memo_id>
      * POST   /api/projects/<pid>/canvas/categories
      * PATCH  /api/projects/<pid>/canvas/categories/<cid>
      * DELETE /api/projects/<pid>/canvas/categories/<cid>
      * PUT    /api/projects/<pid>/canvas/categories/<cid>/members/<memo_id>
      * DELETE /api/projects/<pid>/canvas/categories/<cid>/members/<memo_id>
      * POST   /api/projects/<pid>/canvas/links
      * GET    /api/projects/<pid>/memos (to list memos that can be
        added to the canvas)
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "memo_canvas.html", {
        "project_id": pid,
        "page_title": "Memo canvas",
    })


@app.get("/projects/{project_id}/ai", response_class=HTMLResponse)
async def project_ai_page(request: Request, project_id: str) -> HTMLResponse:
    """AI dashboard / model picker (F8.1 user-facing surface).

    Graduates the wireframe stub: ``project_ai.html`` renders the
    Active-model card (F8.1 BackendConfig editor + Test connection +
    List installed models) plus a read-only F8.10 AI-gate status row
    and links to where each F8.x suggestion surface already lives.

    The F8.6 review queue and F8.7 second-coder diff panels graduate
    here in later iterations; the placeholders on this page link to
    where the feature lives or note "queue UI pending".
    """
    pid = _project_id_or_404(project_id)
    return templates.TemplateResponse(request, "project_ai.html", {
        "project_id": pid,
        "page_title": "AI suggestions",
    })


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
    """Project settings page (F3.1).

    F3.1 added project-level ``settings`` (a bounded key/value store on
    the Project entity) and pulled the codebook into the bundle.
    Without this UI the settings field is unreachable: only the JSON
    PATCH endpoint accepted it. This route renders the form that reads
    + writes those values via ``GET/PATCH /api/projects/<pid>``.
    """
    pid = _project_id_or_404(project_id)
    # Best-effort lookup so the heading shows the real project name even
    # if rendering happens before the JS load() resolves.
    project = None
    try:
        with PROJECTS_LOCK:
            project = _projects.load_project(_projects_root(), pid)
    except Exception:
        project = None
    return templates.TemplateResponse(request, "project_settings.html", {
        "project_id": pid,
        "page_title": "Project settings",
        "subtitle": "Metadata, preferences, and the project bundle download.",
        "project_name": getattr(project, "name", None),
    })


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
# Project archives (F1.5) — round-trippable .scribe.zip export / import
#
# F1.5 formalised the on-disk project layout into a versioned, archive-
# round-trippable file format (manifest.json + the F1.1–F1.4 sub-trees).
# The pure logic lives in scribe/project_format.py with a passing test
# suite. These endpoints surface that to the user:
#
#   GET  /api/projects/<pid>/archive  → download a .scribe.zip
#   POST /api/projects/import-archive → upload a .scribe.zip, restore it
#
# The export is a deterministic zip (sorted entries) so the same project
# state always produces the same archive bytes — useful for diffing two
# captures of a project. Optional ``include_outputs=1`` query param
# bundles the OUTPUT_DIR/<job_id>/ trees referenced by the project's
# sources, so the receiver gets a self-contained corpus.
# --------------------------------------------------------------------------- #

from . import project_format as _project_format  # noqa: E402

# Keep a single tmp directory for streaming archive bytes through. Each
# request gets its own file under it; we never reuse names.
_ARCHIVE_TMP_PREFIX = "scribe-archive-"


def _archive_query_flag(value: Any) -> bool:
    """Treat the usual truthy spellings ('1', 'true', 'yes', 'on') as True."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.get("/api/projects/{project_id}/archive")
async def export_project_archive_endpoint(
    project_id: str, request: Request
) -> Response:
    """Download the project as a Scribe-native ``.scribe.zip`` archive (F1.5).

    The archive's internal layout matches the on-disk layout, rooted at
    a single top-level ``<project_id>/`` directory:

      <project_id>/manifest.json
      <project_id>/project.json
      <project_id>/sources/<sid>.json
      <project_id>/participants/<pid>.json
      <project_id>/sampling_log.jsonl
      <project_id>/codes/<cid>.json
      <project_id>/outputs/<job_id>/...   (only if ?include_outputs=1)

    The receiver can drop this back through
    ``POST /api/projects/import-archive`` to round-trip it; the same
    bytes also feed REFI-QDA / QDPX (F6.4) interop and F9.4 project
    checkpoints.

    Status codes: ``404`` if the project is missing; ``400`` on a bad
    project id; ``200`` otherwise.
    """
    _check_project_id(project_id)
    _project_must_exist(project_id)

    include_outputs = _archive_query_flag(
        request.query_params.get("include_outputs")
    )

    # Stream the archive via a tmp file. project_format.export_project_archive
    # writes a real on-disk zip rather than holding everything in memory,
    # which keeps RAM bounded for projects with bundled outputs/.
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix=_ARCHIVE_TMP_PREFIX))
    archive_path = tmp_dir / f"{project_id}.scribe.zip"
    try:
        with PROJECTS_LOCK:
            try:
                project = _projects.load_project(
                    _projects_root(), project_id
                )
            except FileNotFoundError:
                raise HTTPException(404, "Project not found")
            _project_format.export_project_archive(
                _projects_root(),
                project_id,
                archive_path,
                outputs_root=OUTPUT_DIR if include_outputs else None,
                include_outputs=include_outputs,
            )
        archive_bytes = archive_path.read_bytes()
    finally:
        # Best-effort tmp cleanup; the OS reclaims it eventually anyway.
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:  # pragma: no cover — defensive
            pass

    filename = _project_format.slugify_archive_filename(project)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=archive_bytes,
        media_type="application/zip",
        headers=headers,
    )


@app.post("/api/projects/import-archive")
async def import_project_archive_endpoint(
    archive: UploadFile = File(...),
    overwrite: str | None = Form(default=None),
    include_outputs: str | None = Form(default=None),
) -> JSONResponse:
    """Restore a project from a ``.scribe.zip`` archive (F1.5).

    Accepts a multipart upload with a single ``archive`` file part.

    Optional form fields:
      * ``overwrite=1`` — replace an existing project of the same id
        (default: refuse with 409 to protect against accidental
        clobbers).
      * ``include_outputs=1`` — extract any ``outputs/<job_id>/`` trees
        bundled in the archive into the server's OUTPUT_DIR. Default
        is False so an import never silently overwrites the host's
        transcript corpus; the project still loads, it just won't
        have media playback for those sources until the user
        re-uploads or re-imports a transcript.

    Status codes:
      * ``201`` on a successful import. Body: ``{"project_id": "...",
        "name": "...", "redirect": "/projects/<id>"}``.
      * ``400`` on a malformed archive (zip-bomb, path traversal,
        missing manifest, etc.).
      * ``409`` if a project already exists at the same id and
        ``overwrite`` was not set.
      * ``413`` if the upload exceeds the F1.5 zip-bomb limits.
    """
    overwrite_flag = _archive_query_flag(overwrite)
    extract_outputs = _archive_query_flag(include_outputs)

    # Stream the upload into a tmp file rather than buffering the
    # whole archive in RAM — projects with bundled outputs/ can be
    # hundreds of megabytes.
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix=_ARCHIVE_TMP_PREFIX))
    upload_path = tmp_dir / "upload.scribe.zip"
    try:
        with upload_path.open("wb") as dst:
            while True:
                chunk = await archive.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        try:
            with PROJECTS_LOCK:
                bundle = _project_format.import_project_archive(
                    _projects_root(),
                    upload_path,
                    outputs_root=OUTPUT_DIR if extract_outputs else None,
                    overwrite=overwrite_flag,
                )
        except FileNotFoundError as e:
            raise HTTPException(400, f"Invalid archive: {e}")
        except _project_format.ProjectFormatError as e:
            msg = str(e)
            if "already exists" in msg:
                raise HTTPException(409, msg)
            if "exceeds limit" in msg or "too large" in msg:
                raise HTTPException(413, msg)
            raise HTTPException(400, msg)
        except Exception as e:  # zipfile.BadZipFile, json errors, …
            raise HTTPException(400, f"Invalid archive: {e}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:  # pragma: no cover — defensive
            pass

    return JSONResponse(
        {
            "project_id": bundle.project.id,
            "name": bundle.project.name,
            "redirect": f"/projects/{bundle.project.id}",
            "sources": len(bundle.sources),
            "participants": len(bundle.participants),
            "codes": len(bundle.codes),
            "sampling_entries": len(bundle.sampling_log),
        },
        status_code=201,
    )


# --------------------------------------------------------------------------- #
# REFI-QDA / QDPX project import (F6.6) — interoperable inverse of F6.4
#
# The pure builder shipped in 0426e44 (``scribe.refi_qda_import``) and a
# CLI wrapper in ``scribe.scripts.import_qdpx``. F6.6's reachability gate
# requires an HTTP surface — without it researchers can only import
# QDPX archives by typing a Python invocation, which is hostile UX
# given that REFI-QDA is meant to be Scribe's **no-lock-in** interchange
# format. This endpoint mirrors F1.5's ``import-archive`` shape:
#
#   * Multipart upload accepting a single ``archive`` file part.
#   * 201 on success with ``{"project_id": "...", "redirect": "...",
#     ...summary counts...}`` so the projects list JS can ``location =
#     redirect`` after upload.
#   * 400 on a malformed archive (not a zip / no project.qde / bad
#     XML).
#   * The endpoint always **mints a fresh project id** — the import
#     module does the same. Two installs sharing a workstation can't
#     silently collide, even on Scribe-origin QDPX.
# --------------------------------------------------------------------------- #


from . import refi_qda_import as _refi_qda_import  # noqa: E402
from .scripts import import_qdpx as _import_qdpx_script  # noqa: E402


@app.post("/api/projects/import-qdpx")
async def import_project_qdpx_endpoint(
    archive: UploadFile = File(...),
) -> JSONResponse:
    """Import a project from a REFI-QDA / QDPX (.qdpx) archive (F6.6).

    Accepts a multipart upload with a single ``archive`` file part.

    The archive is parsed with :func:`scribe.refi_qda_import.import_qdpx`
    and persisted via :func:`scribe.scripts.import_qdpx.persist_import_result`
    under ``PROJECTS_DIR/<new-project-id>/``. The new project id is
    always freshly minted (so two installs sharing a workstation can't
    collide); within the new project, individual entity ids are
    preserved when the archive was Scribe-origin (the F6.4 export pads
    ids into REFI-QDA GUIDs in a reversible way) and freshly minted
    otherwise.

    Status codes:
      * ``201`` — import succeeded. Body::

          {
            "project_id": "...",
            "name": "...",
            "redirect": "/projects/<id>",
            "sources": int,
            "codes": int,
            "coders": int,
            "memos": int,
            "applications": int,
            "warnings": [str, ...]   # truncated to 20 entries
          }

      * ``400`` — malformed archive (not a zip / no ``project.qde`` /
        unparseable XML).
      * ``413`` — upload exceeded the soft size limit (50 MB; QDPX
        archives in the wild are sub-megabyte, so this is a sanity
        bound rather than a per-tool quota).
    """
    # Soft cap: QDPX files are tiny in practice. Anything over 50 MB
    # is almost certainly a misclick (someone uploading raw audio).
    SOFT_LIMIT = 50 * 1024 * 1024

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="scribe-qdpx-import-"))
    upload_path = tmp_dir / "upload.qdpx"
    try:
        size = 0
        with upload_path.open("wb") as dst:
            while True:
                chunk = await archive.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > SOFT_LIMIT:
                    raise HTTPException(
                        413,
                        f"QDPX upload exceeds soft limit of "
                        f"{SOFT_LIMIT // (1024 * 1024)} MB",
                    )
                dst.write(chunk)

        try:
            with PROJECTS_LOCK:
                result = _refi_qda_import.import_qdpx(upload_path)
                projects_root = _projects_root()
                _import_qdpx_script.persist_import_result(
                    projects_root, result
                )
        except HTTPException:
            raise
        except (ValueError, TypeError) as exc:
            # ValueError: bad zip / no project.qde / bad XML.
            # TypeError: refi_qda_import rejected the input shape (only
            #     fires on programmer error here, but we surface 400
            #     so it doesn't leak as a 500).
            raise HTTPException(400, f"Invalid QDPX archive: {exc}")
        except Exception as exc:  # zipfile.BadZipFile, json errors, …
            raise HTTPException(400, f"Invalid QDPX archive: {exc}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:  # pragma: no cover — defensive
            pass

    # Truncate warnings to a UI-friendly cap; the CLI does the same.
    warnings_out = list(result.warnings[:20])
    return JSONResponse(
        {
            "project_id": result.project.id,
            "name": result.project.name,
            "redirect": f"/projects/{result.project.id}",
            "sources": len(result.sources),
            "codes": len(result.codes),
            "coders": len(result.coders),
            "memos": len(result.memos),
            "applications": len(result.applications),
            "warnings": warnings_out,
            "warnings_truncated": len(result.warnings) > len(warnings_out),
        },
        status_code=201,
    )


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
# Source attribute schema (F3.2) — user-defined columns per source.
#
# F1.2 already gave each Source a free-form ``custom_attributes`` slot.
# F3.2 layers a project-level schema on top: a typed list of attribute
# definitions (``key``, ``label``, ``type``, ``required``, ``options``,
# ``description``) so the eventual sources-table view renders consistent
# columns, validation catches typos, and exports get a uniform shape.
#
# Storage: a single file at ``PROJECTS_DIR/<pid>/source_schema.json``.
# GET returns the existing schema (or an empty schema for projects that
# never set one — the UI doesn't need to distinguish "missing" from
# "empty"). PUT replaces the schema atomically.
# --------------------------------------------------------------------------- #

from . import source_schema as _source_schema  # noqa: E402  (after module-level state)


@app.get("/api/projects/{project_id}/source_schema")
async def get_source_schema_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        schema = _source_schema.load_or_empty_source_schema(
            _projects_root(), project_id
        )
    return JSONResponse(schema.to_dict())


@app.put("/api/projects/{project_id}/source_schema")
async def put_source_schema_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    raw_attrs = body.get("attributes")
    if raw_attrs is None:
        raw_attrs = []
    if not isinstance(raw_attrs, list):
        raise HTTPException(400, "attributes must be a list")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            schema = _source_schema.SourceAttributeSchema.new(
                project_id=project_id,
                attributes=raw_attrs,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid schema payload: {e}")
        _source_schema.save_source_schema(_projects_root(), schema)
    return JSONResponse(schema.to_dict())


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
# Participant ↔ source mapping (F3.3) — inverse navigation + focus groups
#
# The pure module ``scribe/participant_sources.py`` shipped with passing
# unit tests in ``tests/test_participant_sources.py`` but had no HTTP
# surface — researchers couldn't link participants to sources or list
# the participants in a focus-group source from the UI.
#
# These endpoints close that gap:
#
# * ``GET    /api/projects/<pid>/sources/<sid>/participants`` — list
#   participants whose ``source_ids`` includes ``sid``. Inverse of the
#   forward mapping kept on the participant entity (F1.3); this is the
#   focus-group / multi-speaker view.
#
# * ``PUT    /api/projects/<pid>/sources/<sid>/participants`` — declare
#   the *exact* set of participants for a source in one call (the
#   focus-group editor pattern). Idempotent: only writes the
#   participants whose lists actually change.
#
# * ``POST   /api/projects/<pid>/sources/<sid>/participants/<part_id>``
#   — link one participant to a source. Returns ``added: bool`` so the
#   UI can show "already linked" without a confusing toast.
#
# * ``DELETE /api/projects/<pid>/sources/<sid>/participants/<part_id>``
#   — unlink one participant from a source.
#
# * ``GET    /api/projects/<pid>/orphan_participant_links`` — list
#   ``source_ids`` references that point at a missing source. The
#   audit / cleanup view; non-destructive.
# --------------------------------------------------------------------------- #

from . import participant_sources as _participant_sources  # noqa: E402


@app.get("/api/projects/{project_id}/sources/{source_id}/participants")
async def list_source_participants_endpoint(
    project_id: str, source_id: str
) -> JSONResponse:
    """Inverse navigation (F3.3): which participants are linked to this
    source? Always returns 200 with an empty list when nothing is
    linked. Validates ids strictly so a typo or path-traversal attempt
    fails loudly."""
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        # Confirm the source exists so the UI doesn't accidentally
        # render an empty roster for a deleted source. (Participants
        # that still reference it land in the orphan endpoint.)
        try:
            _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        try:
            parts = _participant_sources.list_participants_for_source(
                _projects_root(), project_id, source_id
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"participants": [p.to_dict() for p in parts]})


@app.put("/api/projects/{project_id}/sources/{source_id}/participants")
async def set_source_participants_endpoint(
    project_id: str, source_id: str, request: Request
) -> JSONResponse:
    """Focus-group editor (F3.3): declare exactly which participants
    are linked to ``source_id``. Body shape:

        {"participant_ids": ["p1", "p2", ...]}

    Returns the diff so the UI can render an audit-friendly toast
    ("added 1, removed 1, unchanged 2"). Idempotent — calling twice
    with the same desired set is a no-op on the second call."""
    _check_project_id(project_id)
    _check_source_id(source_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    raw_ids = body.get("participant_ids")
    if raw_ids is None:
        raw_ids = []
    if not isinstance(raw_ids, list):
        raise HTTPException(400, "participant_ids must be a list")
    pids = [str(x) for x in raw_ids]
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        try:
            change = _participant_sources.set_participants_for_source(
                _projects_root(),
                project_id,
                source_id,
                pids,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "source_id": change.source_id,
        "added": change.added,
        "removed": change.removed,
        "unchanged": change.unchanged,
        "changed": change.changed,
    })


@app.post(
    "/api/projects/{project_id}/sources/{source_id}"
    "/participants/{participant_id}"
)
async def link_participant_to_source_endpoint(
    project_id: str, source_id: str, participant_id: str
) -> JSONResponse:
    """Single-edge link (F3.3). Returns ``added`` so an "already
    linked" call doesn't read as a 4xx — it's a no-op success."""
    _check_project_id(project_id)
    _check_source_id(source_id)
    _check_participant_id(participant_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        try:
            added = _participant_sources.link_participant_to_source(
                _projects_root(),
                project_id,
                participant_id,
                source_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Participant not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "source_id": source_id,
        "participant_id": participant_id,
        "added": added,
    })


@app.delete(
    "/api/projects/{project_id}/sources/{source_id}"
    "/participants/{participant_id}"
)
async def unlink_participant_from_source_endpoint(
    project_id: str, source_id: str, participant_id: str
) -> JSONResponse:
    """Single-edge unlink (F3.3). Tolerates ``source_id`` not being on
    disk so a researcher can clean up after a deleted source — the
    common case for unlinking."""
    _check_project_id(project_id)
    _check_source_id(source_id)
    _check_participant_id(participant_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            removed = _participant_sources.unlink_participant_from_source(
                _projects_root(),
                project_id,
                participant_id,
                source_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Participant not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "source_id": source_id,
        "participant_id": participant_id,
        "removed": removed,
    })


@app.get("/api/projects/{project_id}/orphan_participant_links")
async def list_orphan_participant_links_endpoint(
    project_id: str,
) -> JSONResponse:
    """Audit view (F3.3). Reports every participant→source edge that
    points at a source not present on disk. Non-destructive — the UI
    decides whether each entry is a typo to fix or a legitimately
    deleted source whose link should be cleaned up."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        orphans = _participant_sources.find_orphan_links(
            _projects_root(), project_id
        )
    return JSONResponse({
        "orphans": [
            {"participant_id": o.participant_id, "source_id": o.source_id}
            for o in orphans
        ],
    })


@app.get("/api/projects/{project_id}/participant_source_map")
async def participant_source_map_endpoint(project_id: str) -> JSONResponse:
    """Whole-project inverse mapping (F3.3) — useful to the UI for
    rendering source-by-participant matrices and source-list participant
    counts in one fetch. Returns ``{source_id: [participant_id, …]}``
    including sources with no linked participants (empty list) so the
    caller can iterate every source."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        m = _participant_sources.participant_source_map(
            _projects_root(), project_id
        )
    return JSONResponse({"map": m})


# Speaker-map module is imported here (in addition to the QDPX block
# below) because the F3.4 endpoints below need it; the duplicate alias
# is harmless and keeps both feature blocks self-contained.
from . import speaker_map as _speaker_map  # noqa: E402  (F3.4)


# --------------------------------------------------------------------------- #
# Speaker map (F3.4) — speaker-awareness REST surface
#
# The pure data layer ``scribe/speaker_map.py`` shipped in 84039cb with
# 82 passing unit tests but had no HTTP surface — researchers couldn't
# tag a transcript label as ``interviewer`` vs ``interviewee`` from the
# UI, link a label to a Project participant, or run the role / participant
# distribution that powers the F3.6 matrix views. F3.4 closes that gap
# so "show me only the interviewee's words" stops being a curl-only
# operation.
#
# Endpoints:
#
#   GET    /api/projects/<pid>/sources/<sid>/speaker_map
#           Load the saved map (returns an empty map if none exists yet,
#           so the UI can branch on entries-vs-empty without a 404).
#           Always also returns ``transcript_labels`` — distinct labels
#           found in the source's transcript, in first-occurrence order
#           — so the UI can render a row per speaker even before the
#           researcher has saved anything. Hits the same edited.json
#           / *.json discovery as the QDPX exporter.
#
#   PUT    /api/projects/<pid>/sources/<sid>/speaker_map
#           Replace the map with the supplied entries (set-style save —
#           one Save click writes the whole roster). Body shape:
#               {"entries": [
#                  {"label": "SPEAKER_00",
#                   "role": "interviewer",
#                   "participant_id": null,
#                   "display_name": "",
#                   "notes": ""},
#                  ...
#               ]}
#           Validates each entry; the role must be in SPEAKER_ROLES and
#           any participant_id must shape-match a real participant on
#           disk. Returns the saved map.
#
#   POST   /api/projects/<pid>/sources/<sid>/speaker_map/seed
#           Build a fresh map from the source's transcript segments.
#           Pre-existing role / participant assignments are preserved
#           (uses the merge_segments_into_map helper). Body:
#               {"default_role": "unknown"}   (optional)
#           Returns the saved map and the list of newly-added labels.
#
#   GET    /api/projects/<pid>/sources/<sid>/speaker_map/distribution
#           Role + participant counts for the source's transcript,
#           computed from the current map. Powers the matrix-view
#           preview ("how much did each role talk?"). 200 with empty
#           dicts if the transcript can't be located.
# --------------------------------------------------------------------------- #


def _load_segments_for_source_speaker_map(
    source: "_sources.Source",
) -> list[dict] | None:
    """Resolve the segment list for a source — same discovery rules as
    the QDPX exporter (prefer ``edited.json``, fall back to any
    ``*.json`` engine sidecar with a ``segments`` key). Returns None
    if no transcript is available.

    Pulled out as a separate function so the speaker-map endpoints can
    use it without importing from the QDPX block (which lives further
    down the file and depends on different modules)."""
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


@app.get("/api/projects/{project_id}/sources/{source_id}/speaker_map")
async def get_source_speaker_map_endpoint(
    project_id: str, source_id: str
) -> JSONResponse:
    """Load the speaker map for a source plus the distinct transcript
    labels (F3.4).

    Returns 200 with an empty ``entries`` list when no map has been
    saved yet — the UI branches on ``entries.length`` rather than the
    HTTP status. Always returns ``transcript_labels`` (first-occurrence
    order) so the UI can show a row per speaker even before the
    researcher hits Save. ``transcript_labels`` is empty when no
    transcript can be located on disk.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.load_source(
                _projects_root(), project_id, source_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        speaker_map = _speaker_map.load_or_empty_speaker_map(
            _projects_root(), project_id, source_id
        )
    segments = _load_segments_for_source_speaker_map(source)
    if segments is None:
        labels: list[str] = []
    else:
        labels = _speaker_map.speaker_labels_in_segments(segments)
    return JSONResponse({
        "speaker_map": speaker_map.to_dict(),
        "transcript_labels": labels,
        "available_roles": list(_speaker_map.SPEAKER_ROLES),
    })


@app.put("/api/projects/{project_id}/sources/{source_id}/speaker_map")
async def put_source_speaker_map_endpoint(
    project_id: str, source_id: str, request: Request
) -> JSONResponse:
    """Save the entire speaker map for a source in one call (F3.4).

    Set-style save — the body lists every entry the researcher wants
    persisted; anything not in the list is dropped. Validates each
    entry (role vocabulary, participant_id shape, max field lengths).

    Body::

        {"entries": [
           {"label": "SPEAKER_00", "role": "interviewer"},
           {"label": "SPEAKER_01", "role": "interviewee",
            "participant_id": "abcdef012345"}
        ]}

    Status codes:
      * 200 — saved.
      * 400 — invalid payload / validation failure.
      * 404 — project or source not found.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    raw_entries = body.get("entries", [])
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        raise HTTPException(400, "'entries' must be a list")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        try:
            speaker_map = _speaker_map.SpeakerMap.new(
                project_id=project_id,
                source_id=source_id,
                entries=raw_entries,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        # Cross-check participant ids against on-disk participants to
        # catch typos and orphaned references — the entry validator
        # only checks shape.
        if speaker_map.entries:
            known_pids = {
                p.id
                for p in _participants.list_participants(
                    _projects_root(), project_id
                )
            }
            for e in speaker_map.entries:
                if e.participant_id and e.participant_id not in known_pids:
                    raise HTTPException(
                        400,
                        f"participant_id {e.participant_id!r} not in this "
                        "project",
                    )
        _speaker_map.save_speaker_map(_projects_root(), speaker_map)
    return JSONResponse({"speaker_map": speaker_map.to_dict()})


@app.post("/api/projects/{project_id}/sources/{source_id}/speaker_map/seed")
async def seed_source_speaker_map_endpoint(
    project_id: str, source_id: str, request: Request
) -> JSONResponse:
    """Seed the speaker map from the source's transcript (F3.4).

    Builds a row per distinct ``speaker`` label found in the segments,
    leaving any pre-existing rows untouched (so re-running after the
    researcher has set roles doesn't trample those choices). Returns
    the saved map plus the list of newly-added labels.

    Body::

        {"default_role": "unknown"}   (optional, defaults to ``"unknown"``)

    Status codes:
      * 200 — seeded (``added`` may be empty if nothing new was found).
      * 400 — invalid payload / role vocabulary error.
      * 404 — project / source not found.
      * 409 — no transcript available for the source.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    try:
        body = await request.json() if (await request.body()) else {}
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    default_role = (
        str((body or {}).get("default_role", "unknown") or "unknown")
    )

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.load_source(
                _projects_root(), project_id, source_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        segments = _load_segments_for_source_speaker_map(source)
        if segments is None:
            raise HTTPException(
                409, "No transcript available for this source yet"
            )
        speaker_map = _speaker_map.load_or_empty_speaker_map(
            _projects_root(), project_id, source_id
        )
        try:
            added = _speaker_map.merge_segments_into_map(
                speaker_map, segments, new_role=default_role
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _speaker_map.save_speaker_map(_projects_root(), speaker_map)
    return JSONResponse({
        "speaker_map": speaker_map.to_dict(),
        "added": added,
    })


@app.get(
    "/api/projects/{project_id}/sources/{source_id}/speaker_map/distribution"
)
async def get_source_speaker_map_distribution_endpoint(
    project_id: str, source_id: str
) -> JSONResponse:
    """Role + participant distribution for a source (F3.4).

    Counts segments per role and per linked participant, computed
    against the *currently saved* speaker map. The empty-string key in
    each dict is the catch-all bucket for unlabelled / unmapped
    segments. Returns 200 with empty dicts when no transcript is
    available — the UI shows "no transcript yet" rather than a 404.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.load_source(
                _projects_root(), project_id, source_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        speaker_map = _speaker_map.load_or_empty_speaker_map(
            _projects_root(), project_id, source_id
        )
    segments = _load_segments_for_source_speaker_map(source)
    if segments is None:
        return JSONResponse({
            "role_distribution": {},
            "participant_distribution": {},
            "has_transcript": False,
        })
    return JSONResponse({
        "role_distribution": _speaker_map.role_distribution(
            segments, speaker_map
        ),
        "participant_distribution": _speaker_map.participant_distribution(
            segments, speaker_map
        ),
        "has_transcript": True,
    })


# --------------------------------------------------------------------------- #
# Queries (F3.5) — REST surface
#
# scribe.query shipped the pure-Python query data model + executor
# in ae528c2 with full unit coverage; F3.5 had no HTTP route, so the
# only way to run a query was via curl or a Python REPL. This block
# wires the executor through to the user-facing /projects/<pid>/queries
# page via:
#
#   POST /api/projects/<pid>/queries/run
#     Body: {"query": <Query payload>}
#     Returns matched applications + per-source diagnostic.
#
# scribe.query_runtime owns the on-disk Application → query-shape dict
# adapter; this route is a thin wrapper that resolves the segments
# loader (the same edited.json / *.json discovery rules as the QDPX
# exporter and the speaker-map helpers above) and routes the result
# through the standard Application.to_dict serialiser.
# --------------------------------------------------------------------------- #

from . import query as _query  # noqa: E402  (after module-level state)
from . import query_runtime as _query_runtime  # noqa: E402


@app.post("/api/projects/{project_id}/queries/run")
async def run_project_query_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Execute a Query against this project's corpus (F3.5).

    Request body shape::

        {"query": {<scribe.query.Query payload>}}

    The wrapper expects the inner ``query.project_id`` to equal the
    URL ``project_id``; mismatch lands as 400 to keep saved queries
    from another project from silently executing here.

    Returns::

        {
          "applications": [<Application.to_dict()>, ...],
          "total_applications": <int>,
          "sources_missing_transcript": [<sid>, ...],
          "warnings": [<string>, ...]
        }

    Status codes:
      * 200 — query ran (even if zero matches).
      * 400 — invalid query payload, project_id mismatch, or any
        :class:`ProjectValidationError` from the executor / adapter.
      * 404 — project does not exist on disk.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    query_payload = body.get("query")
    if not isinstance(query_payload, dict):
        raise HTTPException(400, "Body must include a 'query' object")
    # Server-side: stamp project_id from the URL so a UI bug can't
    # accidentally execute a foreign-project query (the adapter
    # double-checks this, but a clean 400 is friendlier than the
    # adapter's mismatch error).
    query_payload = dict(query_payload)
    query_payload.setdefault("project_id", project_id)
    if query_payload.get("project_id") != project_id:
        raise HTTPException(
            400, "query.project_id must match the URL project_id"
        )
    try:
        q = _query.Query.from_dict(query_payload)
    except _projects.ProjectValidationError as e:
        raise HTTPException(400, f"Invalid query: {e}")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        sources_by_id: dict[str, _sources.Source] = {}

        def _segments_loader(sid: str):
            # Mirror the QDPX / speaker-map discovery rules. Cache
            # the source object so we hit disk once per source.
            src = sources_by_id.get(sid)
            if src is None:
                try:
                    src = _sources.load_source(_projects_root(), project_id, sid)
                except FileNotFoundError:
                    return None
                except _projects.ProjectValidationError:
                    return None
                sources_by_id[sid] = src
            return _load_segments_for_source_speaker_map(src)

        try:
            report = _query_runtime.run_query_against_project(
                _projects_root(), project_id, q,
                segments_loader=_segments_loader,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))

    return JSONResponse({
        "applications": [a.to_dict() for a in report.matches],
        "total_applications": report.total_applications,
        "sources_missing_transcript": list(report.sources_missing_transcript),
        "warnings": list(report.warnings),
    })


# --------------------------------------------------------------------------- #
# Matrix views (F3.6) — code × source / code × code / code × attribute.
#
# scribe.matrix shipped the pure builders in c206b8d with full unit
# coverage; the UI surface was deferred. This block wires those builders
# through to the queries page's matrix panel via:
#
#   POST /api/projects/<pid>/matrices/run
#     Body:
#       {
#         "kind": "code-by-source" | "code-by-code" | "code-by-attribute",
#         "scope": "source" | "segment" | "paragraph",   # code-by-code only
#         "max_gap": <number>,                            # code-by-code only
#         "attribute_key": "<key>",                       # code-by-attribute only
#         "attribute_kind": "source" | "participant",     # code-by-attribute only
#         "include_missing": <bool>,                      # code-by-attribute only
#         "compact": <bool>,                              # default True
#         "query": {<scribe.query.Query payload>}         # optional pre-filter
#       }
#
# When ``query`` is provided the matrix is computed over the matching
# applications only — the natural F3.5 → F3.6 pipeline. Without it,
# every application in the project participates.
# --------------------------------------------------------------------------- #

from . import matrix as _matrix  # noqa: E402  (after module-level state)


def _matrix_payload_bool(body: dict, key: str, default: bool) -> bool:
    """Coerce a boolean field that the JS client may have sent as a
    real bool, or as one of the usual truthy/falsy spellings.

    Mirrors :func:`_archive_query_flag` but with a configurable default
    (``compact`` defaults to True; ``include_missing`` to True; the
    rest aren't relevant here)."""
    if key not in body or body[key] is None:
        return default
    v = body[key]
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/projects/{project_id}/matrices/run")
async def run_project_matrix_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Compute a matrix view over the project's coded segments (F3.6).

    Three matrix kinds are supported (see :mod:`scribe.matrix`):

      * ``code-by-source`` — frequency: one cell per (code, source)
        counts how many applications attach that code to that source.
      * ``code-by-code``   — undirected co-occurrence within a chosen
        scope (``source`` / ``segment`` / ``paragraph``). The diagonal
        counts unordered pairs of distinct applications of the same
        code (4 in one source ⇒ C(4,2)=6 on the diagonal).
      * ``code-by-attribute`` — cross-tab against either a
        :mod:`scribe.source_schema` attribute key or a participant
        demographic key. Empty values bucket into a sentinel column
        when ``include_missing`` is true (default).

    The optional ``query`` body field re-uses the F3.5 executor: when
    provided, the matrix builds only over the matching applications.
    Without it the full corpus is used.

    Returns::

        {
          "kind": <str>,
          "matrix": <Matrix.to_dict()>,
          "total_applications": <int>,         # all applications in scope
          "matched_applications": <int>,       # applications that fed the matrix
          "sources_missing_transcript": [...], # only when query is in play
          "warnings": [...],
          "params": {<echoed kind / scope / attribute_key / ...>}
        }

    Status codes:
      * 200 — matrix computed (even when zero matches).
      * 400 — invalid kind / scope / attribute_kind / query payload, or
              any :class:`MatrixError` from the builder.
      * 404 — project does not exist on disk.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    kind_raw = body.get("kind")
    if not isinstance(kind_raw, str) or not kind_raw.strip():
        raise HTTPException(400, "'kind' is required")
    kind = kind_raw.strip().lower()
    if kind not in ("code-by-source", "code-by-code", "code-by-attribute"):
        raise HTTPException(
            400,
            "'kind' must be one of "
            "'code-by-source' / 'code-by-code' / 'code-by-attribute'",
        )

    # F3.6 matrix module's optional knobs — all defaulted server-side
    # so the caller can send a minimal `{kind: ...}` payload.
    scope = (body.get("scope") or "source")
    if not isinstance(scope, str):
        raise HTTPException(400, "'scope' must be a string")
    scope = scope.strip().lower() or "source"

    try:
        max_gap = float(body.get("max_gap", 0) or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "'max_gap' must be a number")

    attribute_key = body.get("attribute_key") or ""
    if not isinstance(attribute_key, str):
        raise HTTPException(400, "'attribute_key' must be a string")
    attribute_key = attribute_key.strip()

    attribute_kind = (body.get("attribute_kind") or "source")
    if not isinstance(attribute_kind, str):
        raise HTTPException(400, "'attribute_kind' must be a string")
    attribute_kind = attribute_kind.strip().lower() or "source"
    if attribute_kind not in ("source", "participant"):
        raise HTTPException(
            400, "'attribute_kind' must be 'source' or 'participant'"
        )

    include_missing = _matrix_payload_bool(body, "include_missing", True)
    compact = _matrix_payload_bool(body, "compact", True)

    if kind == "code-by-attribute" and not attribute_key:
        raise HTTPException(
            400, "'attribute_key' is required for 'code-by-attribute'"
        )

    # Optional pre-filter via F3.5 query.
    query_obj: _query.Query | None = None
    raw_query = body.get("query")
    if raw_query is not None:
        if not isinstance(raw_query, dict):
            raise HTTPException(400, "'query' must be an object")
        query_payload = dict(raw_query)
        query_payload.setdefault("project_id", project_id)
        if query_payload.get("project_id") != project_id:
            raise HTTPException(
                400, "query.project_id must match the URL project_id"
            )
        try:
            query_obj = _query.Query.from_dict(query_payload)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, f"Invalid query: {e}")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        codes = _codes.list_codes(_projects_root(), project_id)
        sources = _sources.list_sources(_projects_root(), project_id)
        all_apps = _applications.list_applications(
            _projects_root(), project_id
        )

        # Bind the same speaker-map / segments cache the query route
        # uses, so application_to_query_dict resolves speaker / start /
        # end consistently between the two pipelines.
        sources_by_id = {s.id: s for s in sources}
        smap_cache: dict[str, _speaker_map.SpeakerMap] = {}
        seg_cache: dict[str, "list[dict] | None"] = {}
        sources_missing_transcript: list[str] = []
        warnings: list[str] = []

        def _segments_loader(sid: str):
            if sid in seg_cache:
                return seg_cache[sid]
            src = sources_by_id.get(sid)
            if src is None:
                try:
                    src = _sources.load_source(
                        _projects_root(), project_id, sid
                    )
                    sources_by_id[sid] = src
                except FileNotFoundError:
                    seg_cache[sid] = None
                    return None
                except _projects.ProjectValidationError:
                    seg_cache[sid] = None
                    return None
            segs = _load_segments_for_source_speaker_map(src)
            seg_cache[sid] = segs
            if segs is None and sid not in sources_missing_transcript:
                sources_missing_transcript.append(sid)
            return segs

        def _smap_for(sid: str) -> _speaker_map.SpeakerMap:
            if sid not in smap_cache:
                try:
                    smap_cache[sid] = _speaker_map.load_or_empty_speaker_map(
                        _projects_root(), project_id, sid
                    )
                except Exception:
                    smap_cache[sid] = _speaker_map.SpeakerMap.new(
                        project_id=project_id, source_id=sid
                    )
            return smap_cache[sid]

        # If the caller provided a query, run it through the F3.5
        # runtime so the matrix is built over the matching subset only.
        # The runtime caches segments / speaker maps internally; we
        # re-load the survivors via application_to_query_dict so the
        # matrix sees the same start/end/speaker fields.
        total_applications = len(all_apps)
        if query_obj is not None:
            try:
                report = _query_runtime.run_query_against_project(
                    _projects_root(), project_id, query_obj,
                    segments_loader=_segments_loader,
                    applications=all_apps,
                )
            except _projects.ProjectValidationError as e:
                raise HTTPException(400, str(e))
            apps_for_matrix = list(report.matches)
            # Surface the runtime's diagnostics so the UI can mirror
            # the F3.5 page's "this source had no transcript" hint.
            for sid in report.sources_missing_transcript:
                if sid not in sources_missing_transcript:
                    sources_missing_transcript.append(sid)
            warnings.extend(report.warnings)
        else:
            apps_for_matrix = list(all_apps)

        # Project on-disk Application objects → the dict shape the
        # F3.6 builders accept (code_id, source_id, optional speaker /
        # start / end / participant_id). Doing this server-side keeps
        # the pure matrix module honest: it never touches disk.
        app_dicts: list[dict] = []
        participant_explicit_ids: dict[str, str] = {}
        for a in apps_for_matrix:
            segs = _segments_loader(a.source_id)
            d = _query_runtime.application_to_query_dict(a, segs)
            # For code-by-attribute participant resolution: the matrix
            # module accepts an optional ``participant_id`` field on
            # the application dict as a fallback when the speaker label
            # doesn't resolve via the speaker map. We inject it here
            # whenever the speaker_map maps the application's speaker
            # label to a participant — this lets the matrix builder
            # work even if the caller didn't pass speaker_maps.
            if attribute_kind == "participant":
                smap = _smap_for(a.source_id)
                pid = smap.participant_for(d.get("speaker", "") or "")
                if pid:
                    d["participant_id"] = pid
            app_dicts.append(d)

        try:
            if kind == "code-by-source":
                m = _matrix.code_by_source_matrix(
                    applications=app_dicts,
                    codes=codes,
                    sources=sources,
                )
            elif kind == "code-by-code":
                m = _matrix.code_by_code_matrix(
                    applications=app_dicts,
                    codes=codes,
                    scope=scope,
                    max_gap=max_gap,
                )
            else:  # code-by-attribute
                if attribute_kind == "source":
                    m = _matrix.code_by_attribute_matrix(
                        applications=app_dicts,
                        codes=codes,
                        attribute_key=attribute_key,
                        attribute_kind="source",
                        sources=sources,
                        include_missing=include_missing,
                    )
                else:
                    participants = _participants.list_participants(
                        _projects_root(), project_id
                    )
                    m = _matrix.code_by_attribute_matrix(
                        applications=app_dicts,
                        codes=codes,
                        attribute_key=attribute_key,
                        attribute_kind="participant",
                        participants=participants,
                        speaker_maps=smap_cache,
                        include_missing=include_missing,
                    )
        except _matrix.MatrixError as e:
            raise HTTPException(400, str(e))
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))

        if compact:
            m = m.compact()

    return JSONResponse({
        "kind": kind,
        "matrix": m.to_dict(),
        "total_applications": total_applications,
        "matched_applications": len(app_dicts),
        "sources_missing_transcript": sources_missing_transcript,
        "warnings": warnings,
        "params": {
            "kind": kind,
            "scope": scope if kind == "code-by-code" else None,
            "max_gap": max_gap if kind == "code-by-code" else None,
            "attribute_key": (
                attribute_key if kind == "code-by-attribute" else None
            ),
            "attribute_kind": (
                attribute_kind if kind == "code-by-attribute" else None
            ),
            "include_missing": (
                include_missing if kind == "code-by-attribute" else None
            ),
            "compact": compact,
        },
    })


# --------------------------------------------------------------------------- #
# Matrix exports (F6.3) — CSV / XLSX download surface.
#
# F3.6 ships the pure builders + the JSON /matrices/run endpoint above
# that powers the queries page's matrix panel. F6.3's pure module
# (:mod:`scribe.matrix_export`) ships CSV + XLSX renderers. This
# endpoint wires the two together as a download URL the browser can
# follow directly — assigned to the F6.3 download menu in
# ``queries.html``:
#
#   GET /api/projects/<pid>/matrices/<kind>/export?format=csv|xlsx&...
#
# Path: ``kind`` is one of ``code-by-source`` / ``code-by-code`` /
# ``code-by-attribute`` (aliases like ``frequency`` / ``cooccurrence`` /
# ``cross-tab`` are accepted via ``normalise_matrix_kind``).
#
# Query params:
#
#   format         csv | xlsx | xls | excel | spreadsheet  (default csv)
#   scope          source | segment | paragraph             (code-by-code)
#   max_gap        float                                    (code-by-code, paragraph)
#   attribute_key  str                                      (code-by-attribute, required)
#   attribute_kind source | participant                     (code-by-attribute)
#   include_missing 0 | 1 | true | false                    (code-by-attribute)
#   compact         0 | 1                                   (drop empty rows / cols)
#   use_titles      0 | 1                                   (use display titles)
#   include_totals  0 | 1                                   (footer / right-edge totals)
#
# This surface deliberately does **not** support the F3.5 ``query``
# pre-filter (the JSON endpoint above does). The reasoning: a download
# URL has to be navigable from a browser address bar / ``<a download>``
# anchor, and serialising a multi-clause Query into a query string is
# fiddly. Researchers who want a filtered export run the query first,
# then export the resulting Matrix from the panel. If we ever need
# server-side filtering for export we can add ``query_id`` referencing
# a saved query (F3.7).
# --------------------------------------------------------------------------- #


from . import matrix_export as _matrix_export  # noqa: E402


def _matrix_query_bool(value: object, default: bool) -> bool:
    """Coerce a query-string boolean (``"1"`` / ``"true"`` / ``"on"``
    / ``"yes"``) into a Python ``bool``.

    Treats ``None`` and empty strings as "use the default". Mirrors
    :func:`_matrix_payload_bool` but for the GET URL surface where
    everything arrives as a string.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if not s:
        return default
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _build_matrix_for_export(
    project_id: str,
    *,
    kind: str,
    scope: str,
    max_gap: float,
    attribute_key: str,
    attribute_kind: str,
    include_missing: bool,
    compact: bool,
):
    """Compute a matrix for the F6.3 export endpoint.

    Mirrors the project / codes / sources / applications hydration
    chain that ``run_project_matrix_endpoint`` uses, minus the
    optional F3.5 query pre-filter. Caller must hold ``PROJECTS_LOCK``.

    Returns the :class:`scribe.matrix.Matrix` (already compacted if
    requested). Raises :class:`HTTPException` 400 / 404 for the same
    failure modes the JSON endpoint exposes so the export surface
    speaks the same dialect.
    """
    _project_must_exist(project_id)
    codes = _codes.list_codes(_projects_root(), project_id)
    sources = _sources.list_sources(_projects_root(), project_id)
    all_apps = _applications.list_applications(
        _projects_root(), project_id
    )

    # Same speaker-map / segments cache the run endpoint binds, so
    # speaker / start / end resolve consistently across the two
    # paths.
    sources_by_id = {s.id: s for s in sources}
    smap_cache: dict[str, _speaker_map.SpeakerMap] = {}
    seg_cache: dict[str, "list[dict] | None"] = {}

    def _segments_loader(sid: str):
        if sid in seg_cache:
            return seg_cache[sid]
        src = sources_by_id.get(sid)
        if src is None:
            try:
                src = _sources.load_source(
                    _projects_root(), project_id, sid
                )
                sources_by_id[sid] = src
            except FileNotFoundError:
                seg_cache[sid] = None
                return None
            except _projects.ProjectValidationError:
                seg_cache[sid] = None
                return None
        segs = _load_segments_for_source_speaker_map(src)
        seg_cache[sid] = segs
        return segs

    def _smap_for(sid: str) -> _speaker_map.SpeakerMap:
        if sid not in smap_cache:
            try:
                smap_cache[sid] = _speaker_map.load_or_empty_speaker_map(
                    _projects_root(), project_id, sid
                )
            except Exception:
                smap_cache[sid] = _speaker_map.SpeakerMap.new(
                    project_id=project_id, source_id=sid
                )
        return smap_cache[sid]

    # Project Application objects → matrix-builder dict shape.
    app_dicts: list[dict] = []
    for a in all_apps:
        segs = _segments_loader(a.source_id)
        d = _query_runtime.application_to_query_dict(a, segs)
        if attribute_kind == "participant":
            smap = _smap_for(a.source_id)
            pid = smap.participant_for(d.get("speaker", "") or "")
            if pid:
                d["participant_id"] = pid
        app_dicts.append(d)

    try:
        if kind == "code-by-source":
            m = _matrix.code_by_source_matrix(
                applications=app_dicts,
                codes=codes,
                sources=sources,
            )
        elif kind == "code-by-code":
            m = _matrix.code_by_code_matrix(
                applications=app_dicts,
                codes=codes,
                scope=scope,
                max_gap=max_gap,
            )
        else:  # code-by-attribute
            if attribute_kind == "source":
                m = _matrix.code_by_attribute_matrix(
                    applications=app_dicts,
                    codes=codes,
                    attribute_key=attribute_key,
                    attribute_kind="source",
                    sources=sources,
                    include_missing=include_missing,
                )
            else:
                participants = _participants.list_participants(
                    _projects_root(), project_id
                )
                m = _matrix.code_by_attribute_matrix(
                    applications=app_dicts,
                    codes=codes,
                    attribute_key=attribute_key,
                    attribute_kind="participant",
                    participants=participants,
                    speaker_maps=smap_cache,
                    include_missing=include_missing,
                )
    except _matrix.MatrixError as e:
        raise HTTPException(400, str(e))
    except _projects.ProjectValidationError as e:
        raise HTTPException(400, str(e))

    if compact:
        m = m.compact()

    return m


@app.get(
    "/api/projects/{project_id}/matrices/{kind}/export"
)
async def export_project_matrix_endpoint(
    project_id: str,
    kind: str,
    format: str = "csv",
    scope: str = "source",
    max_gap: float = 0.0,
    attribute_key: str = "",
    attribute_kind: str = "source",
    include_missing: str | None = None,
    compact: str | None = None,
    use_titles: str | None = None,
    include_totals: str | None = None,
) -> Response:
    """Download a matrix view as CSV / XLSX (F6.3).

    See the section comment above for the full URL contract. Errors:

      * 400 — unknown ``kind`` / ``format`` / ``scope`` / ``attribute_kind``,
              missing ``attribute_key`` for ``code-by-attribute``, or
              any :class:`scribe.matrix.MatrixError` from the builder.
      * 404 — project does not exist.
      * 200 — matrix exported (including the empty-matrix case; the
              renderers produce a header-only CSV / a one-cell XLSX
              shell).
    """
    _check_project_id(project_id)
    try:
        canonical_kind = _matrix_export.normalise_matrix_kind(kind)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        fmt = _matrix_export.normalise_format(format)
    except ValueError as e:
        raise HTTPException(400, str(e))

    scope_norm = (scope or "source").strip().lower() or "source"
    attribute_kind_norm = (
        (attribute_kind or "source").strip().lower() or "source"
    )
    if attribute_kind_norm not in ("source", "participant"):
        raise HTTPException(
            400, "'attribute_kind' must be 'source' or 'participant'"
        )

    attribute_key_norm = (attribute_key or "").strip()
    if (
        canonical_kind == "code-by-attribute"
        and not attribute_key_norm
    ):
        raise HTTPException(
            400, "'attribute_key' is required for 'code-by-attribute'"
        )

    include_missing_b = _matrix_query_bool(include_missing, True)
    compact_b = _matrix_query_bool(compact, True)
    use_titles_b = _matrix_query_bool(use_titles, True)
    include_totals_b = _matrix_query_bool(include_totals, True)

    try:
        max_gap_f = float(max_gap or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "'max_gap' must be a number")

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(
                _projects_root(), project_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")

        m = _build_matrix_for_export(
            project_id,
            kind=canonical_kind,
            scope=scope_norm,
            max_gap=max_gap_f,
            attribute_key=attribute_key_norm,
            attribute_kind=attribute_kind_norm,
            include_missing=include_missing_b,
            compact=compact_b,
        )

    payload = _matrix_export.render_matrix(
        fmt, m,
        use_titles=use_titles_b,
        include_totals=include_totals_b,
    )
    spec = _matrix_export.EXPORT_FORMATS[fmt]
    filename = _matrix_export.slugify_matrix_filename(
        project, fmt, canonical_kind
    )
    headers = {
        # Quote the filename so spaces / non-ASCII never break the
        # header. We slugify to ASCII upstream, so the simple quoted
        # form is sufficient — same convention as the F6.1 codebook
        # export and the F6.2 retrieval report.
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=payload,
        media_type=spec.media_type,
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# Saved queries (F3.7) — named, re-runnable queries.
#
# scribe.saved_queries shipped the pure SavedQuery dataclass + on-disk
# CRUD + run-tracking in 6bb1947 with 53 unit tests, but had **no HTTP
# surface** — the F3.7 commit body explicitly deferred routes + UI
# ('to land alongside the rest of Phase B'). This block wires the
# saved-queries store through to the user-facing /projects/<pid>/queries
# page so a researcher can name a query, see the list of named queries,
# re-run one (recording the run + bumping run_count), and delete one
# they no longer want.
#
# Endpoints:
#
#   GET    /api/projects/<pid>/saved-queries
#           List every saved query in the project (sorted newest-edited
#           first to match scribe.saved_queries.list_saved_queries).
#
#   POST   /api/projects/<pid>/saved-queries
#           Create. Body: {query: <Query payload>, name?, description?}.
#           ``name`` / ``description`` shortcuts populate the wrapped
#           Query if the inner query payload didn't carry them. The
#           wrapped query.project_id must match the URL pid.
#
#   GET    /api/projects/<pid>/saved-queries/<sqid>
#           Fetch one. 404 if missing.
#
#   PATCH  /api/projects/<pid>/saved-queries/<sqid>
#           Partial update via SavedQuery.apply_update — accepts any
#           subset of {query, name, description}. ``project_id`` /
#           ``id`` / timestamps / run_count are managed by the entity.
#
#   DELETE /api/projects/<pid>/saved-queries/<sqid>
#           Remove. 404 if missing.
#
#   POST   /api/projects/<pid>/saved-queries/<sqid>/run
#           Load + execute via the F3.5 runtime; bump run_count and
#           stamp last_run_at. Returns the same shape as
#           POST /queries/run plus the post-run saved query (so the UI
#           can show the updated run_count without a follow-up GET).
# --------------------------------------------------------------------------- #

from . import saved_queries as _saved_queries  # noqa: E402


def _check_saved_query_id(saved_query_id: str) -> None:
    if not _saved_queries.SAVED_QUERY_ID_RE.match(saved_query_id):
        raise HTTPException(400, "Invalid saved query id")


def _saved_query_payload_to_query(
    project_id: str, body: dict
) -> "_query.Query":
    """Helper for POST/PATCH: lift the inner Query payload, stamp the
    URL project_id, and propagate name / description shortcuts so the
    UI can send {name, query} without re-typing the metadata into the
    inner Query.
    """
    raw_query = body.get("query")
    if not isinstance(raw_query, dict):
        raise HTTPException(
            400, "Body must include a 'query' object payload"
        )
    query_payload = dict(raw_query)
    query_payload.setdefault("project_id", project_id)
    if query_payload.get("project_id") != project_id:
        raise HTTPException(
            400, "query.project_id must match the URL project_id"
        )
    # Top-level name / description shortcuts override the inner ones
    # only when the caller actually supplied them — researchers can
    # rename a query by passing just {name: "..."} into PATCH without
    # re-sending the whole filter tree.
    if "name" in body and body["name"] is not None:
        query_payload["name"] = str(body["name"])
    if "description" in body and body["description"] is not None:
        query_payload["description"] = str(body["description"])
    try:
        q = _query.Query.from_dict(query_payload)
    except _projects.ProjectValidationError as e:
        raise HTTPException(400, f"Invalid query: {e}")
    return q


@app.get("/api/projects/{project_id}/saved-queries")
async def list_saved_queries_endpoint(project_id: str) -> JSONResponse:
    """List every saved query in the project (F3.7).

    Sorted newest-edited first; matches the underlying store's order
    so the UI can render the list without re-sorting.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        out = _saved_queries.list_saved_queries(_projects_root(), project_id)
    return JSONResponse(
        {"saved_queries": [sq.to_dict() for sq in out]}
    )


@app.post("/api/projects/{project_id}/saved-queries")
async def create_saved_query_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Create a new saved query (F3.7).

    Request body shape::

        {
          "query": {<scribe.query.Query payload>},
          "name": "<display name>",          # optional shortcut
          "description": "<optional>"        # optional shortcut
        }

    The wrapped query's ``project_id`` must equal the URL
    ``project_id`` (server-side: stamped from the URL when missing,
    400 on mismatch). The display name is required (a saved query
    needs a name for the "re-run X" UI to make sense); a 400 lands if
    the inner query and the top-level shortcut both omit it.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    q = _saved_query_payload_to_query(project_id, body)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            sq = _saved_queries.SavedQuery.new(
                project_id=project_id, query=q,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        try:
            _saved_queries.save_saved_query(_projects_root(), sq)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
    return JSONResponse(sq.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}/saved-queries/{saved_query_id}")
async def get_saved_query_endpoint(
    project_id: str, saved_query_id: str
) -> JSONResponse:
    """Fetch one saved query by id (F3.7). 404 if missing."""
    _check_project_id(project_id)
    _check_saved_query_id(saved_query_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            sq = _saved_queries.load_saved_query(
                _projects_root(), project_id, saved_query_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Saved query not found")
    return JSONResponse(sq.to_dict())


@app.patch("/api/projects/{project_id}/saved-queries/{saved_query_id}")
async def patch_saved_query_endpoint(
    project_id: str, saved_query_id: str, request: Request
) -> JSONResponse:
    """Partial update a saved query (F3.7).

    Accepts any subset of ``{query, name, description}``. Managed
    fields (``id``, ``project_id``, ``created_at``, ``modified_at``,
    ``last_run_at``, ``run_count``) are ignored if passed — the entity
    owns them.
    """
    _check_project_id(project_id)
    _check_saved_query_id(saved_query_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            sq = _saved_queries.load_saved_query(
                _projects_root(), project_id, saved_query_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Saved query not found")
        try:
            sq.apply_update(body)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid saved-query patch: {e}")
        _saved_queries.save_saved_query(_projects_root(), sq)
    return JSONResponse(sq.to_dict())


@app.delete("/api/projects/{project_id}/saved-queries/{saved_query_id}")
async def delete_saved_query_endpoint(
    project_id: str, saved_query_id: str
) -> JSONResponse:
    """Remove a saved query (F3.7). 404 if missing."""
    _check_project_id(project_id)
    _check_saved_query_id(saved_query_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        ok = _saved_queries.delete_saved_query(
            _projects_root(), project_id, saved_query_id
        )
    if not ok:
        raise HTTPException(404, "Saved query not found")
    return JSONResponse({"ok": True})


@app.post("/api/projects/{project_id}/saved-queries/{saved_query_id}/run")
async def run_saved_query_endpoint(
    project_id: str, saved_query_id: str
) -> JSONResponse:
    """Re-run a saved query (F3.7).

    Loads the saved query, executes it via the same F3.5 runtime the
    ad-hoc query route uses, and bumps the run-tracking fields
    (``run_count`` + ``last_run_at``). Returns the executor's output
    plus the post-run :class:`SavedQuery` so the UI can update the
    "last run" / "run count" indicator without a follow-up GET.

    Status codes:
      * 200 — query ran (even if zero matches).
      * 400 — invalid project / saved-query id, or any
        :class:`ProjectValidationError` from the executor.
      * 404 — project or saved query does not exist on disk.
    """
    _check_project_id(project_id)
    _check_saved_query_id(saved_query_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            sq = _saved_queries.load_saved_query(
                _projects_root(), project_id, saved_query_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Saved query not found")

        sources_by_id: dict[str, _sources.Source] = {}

        def _segments_loader(sid: str):
            src = sources_by_id.get(sid)
            if src is None:
                try:
                    src = _sources.load_source(
                        _projects_root(), project_id, sid
                    )
                except FileNotFoundError:
                    return None
                except _projects.ProjectValidationError:
                    return None
                sources_by_id[sid] = src
            return _load_segments_for_source_speaker_map(src)

        try:
            report = _query_runtime.run_query_against_project(
                _projects_root(), project_id, sq.query,
                segments_loader=_segments_loader,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))

        # Stamp run_count + last_run_at after a successful execution
        # (a 400 / 500 must not advance the audit trail). record_run
        # persists immediately.
        sq = _saved_queries.record_run(
            _projects_root(), project_id, saved_query_id,
        )

    return JSONResponse({
        "saved_query": sq.to_dict(),
        "applications": [a.to_dict() for a in report.matches],
        "total_applications": report.total_applications,
        "sources_missing_transcript": list(report.sources_missing_transcript),
        "warnings": list(report.warnings),
    })


# --------------------------------------------------------------------------- #
# Coders (F2.5, multi-coder mode) — REST surface
#
# The pure data layer (``scribe/coders.py``) and the ICR statistics
# (``scribe/icr.py``) shipped in cae5570 with 135 passing tests but had
# no HTTP surface — researchers couldn't add a second coder, set the
# active coder, or run an inter-coder reliability comparison from the
# UI.
#
# These endpoints close that gap: full CRUD on coders, plus an ICR
# computation route that returns Cohen's kappa per code (and overall)
# for any two coders. The ``/api/projects/<pid>/applications`` POST
# already accepts an optional ``coder_id`` field (see above) so the
# coding view can attribute new applications to whichever coder is
# active in the user's session.
# --------------------------------------------------------------------------- #

from . import coders as _coders  # noqa: E402
from . import applications as _applications  # noqa: E402
from . import application_spans as _application_spans  # noqa: E402  (F4.2)
from . import application_gutter as _application_gutter  # noqa: E402  (F4.3)
from . import codes as _codes  # noqa: E402


def _segment_word_counts_for_source(
    source: "_sources.Source",
) -> dict[int, int]:
    """Return ``{segment_index: word_count}`` for cross-segment
    adjacency detection (F4.2). Empty dict when no transcript is
    available — callers should treat that as "only within-segment
    adjacency is detectable", which matches the
    :func:`scribe.application_spans.applications_adjacent` contract.
    """
    segments = _load_segments_for_source_speaker_map(source)
    if not segments:
        return {}
    counts: dict[int, int] = {}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        words = seg.get("words")
        if isinstance(words, list):
            counts[i] = len(words)
    return counts


def _check_coder_id(coder_id: str) -> None:
    if not _coders.CODER_ID_RE.match(coder_id):
        raise HTTPException(400, "Invalid coder id")


@app.get("/api/projects/{project_id}/coders")
async def list_coders_endpoint(project_id: str) -> JSONResponse:
    """List all coders in a project (F2.5).

    Returns coders ordered by ``created_at`` ascending — the order in
    which the team was assembled. The default ``"You"`` coder created
    on first application is always present.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        out = [
            c.to_dict()
            for c in _coders.list_coders(_projects_root(), project_id)
        ]
    return JSONResponse({"coders": out})


@app.post("/api/projects/{project_id}/coders")
async def create_coder_endpoint(
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
            coder = _coders.Coder.new(
                project_id=project_id,
                name=str(body.get("name", "")),
                role=str(body.get("role", "researcher") or "researcher"),
                email=str(body.get("email", "") or ""),
                colour=str(body.get("colour", "") or ""),
                status=str(body.get("status", "active") or "active"),
                notes=str(body.get("notes", "") or ""),
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid coder payload: {e}")
        _coders.save_coder(_projects_root(), coder)
    return JSONResponse(coder.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}/coders/{coder_id}")
async def get_coder_endpoint(
    project_id: str, coder_id: str
) -> JSONResponse:
    _check_project_id(project_id)
    _check_coder_id(coder_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            coder = _coders.load_coder(
                _projects_root(), project_id, coder_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Coder not found")
    return JSONResponse(coder.to_dict())


@app.patch("/api/projects/{project_id}/coders/{coder_id}")
async def patch_coder_endpoint(
    project_id: str, coder_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    _check_coder_id(coder_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            coder = _coders.load_coder(
                _projects_root(), project_id, coder_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Coder not found")
        try:
            coder.apply_update(body)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _coders.save_coder(_projects_root(), coder)
    return JSONResponse(coder.to_dict())


@app.delete("/api/projects/{project_id}/coders/{coder_id}")
async def delete_coder_endpoint(
    project_id: str, coder_id: str
) -> JSONResponse:
    """Delete a coder.

    Per the F2.5 contract in :mod:`scribe.coders`, deleting a coder
    does **not** retroactively orphan their applications: the
    ``coder_id`` recorded on each application is a stable string
    reference, not a foreign key. The audit trail keeps the id even
    after the Coder record is gone.
    """
    _check_project_id(project_id)
    _check_coder_id(coder_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        ok = _coders.delete_coder(
            _projects_root(), project_id, coder_id
        )
    if not ok:
        raise HTTPException(404, "Coder not found")
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# F2.5 — Inter-coder reliability (Cohen's kappa) computation surface
#
# Given two coder ids the endpoint returns:
#   * overall Cohen's kappa across every (source, code) the two coders
#     touched between them,
#   * per-code kappa (binary "applied this code to item I?" decisions),
#   * per-code Landis & Koch interpretation labels,
#   * agreement counts (items both applied / only A / only B / neither).
# The ICR view template consumes this JSON to render the comparison
# table.
# --------------------------------------------------------------------------- #


from . import icr as _icr  # noqa: E402


def _icr_items_set(
    apps_a: "list[_applications.Application]",
    apps_b: "list[_applications.Application]",
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]],
           dict[tuple[str, str], set[str]]]:
    """Bucket two coders' applications into per-item code sets.

    The F2.5 unit of comparison is the smallest re-anchorable span
    that *either* coder marked. F4.1 anchors applications to
    ``(source_id, anchor_start_word_id, anchor_end_word_id)`` so we
    use the (source_id, anchor_start_word_id) tuple as the item id —
    exactly the shape :func:`scribe.icr.per_code_kappa` expects.
    """
    items: set[tuple[str, str]] = set()
    coder_a_map: dict[tuple[str, str], set[str]] = {}
    coder_b_map: dict[tuple[str, str], set[str]] = {}
    for a in apps_a:
        key = (a.source_id, a.anchor_start_word_id)
        items.add(key)
        coder_a_map.setdefault(key, set()).add(a.code_id)
    for a in apps_b:
        key = (a.source_id, a.anchor_start_word_id)
        items.add(key)
        coder_b_map.setdefault(key, set()).add(a.code_id)
    return items, coder_a_map, coder_b_map


@app.get("/api/projects/{project_id}/icr")
async def icr_endpoint(
    project_id: str,
    coder_a: str,
    coder_b: str,
    source_id: str = "",
) -> JSONResponse:
    """Compute Cohen's kappa for ``coder_a`` vs ``coder_b`` (F2.5).

    Optional ``source_id`` narrows the comparison to a single source.
    Returns:

        {
          "coder_a": {id, name},
          "coder_b": {id, name},
          "n_items": int,            # union of items either touched
          "n_both_applied_any": int, # both applied at least one code
          "overall_kappa": float,    # collapsing all codes per item to a "matches" boolean
          "overall_label": str,      # Landis & Koch
          "per_code": [
            {"code_id": ..., "code_name": ..., "kappa": ..., "label": ...,
             "n_a_applied": ..., "n_b_applied": ...},
            ...
          ],
        }

    400 on invalid coder ids; 404 if either coder doesn't exist (or
    the project doesn't); 200 on success including the empty case
    (n_items=0, kappa=1.0).
    """
    _check_project_id(project_id)
    _check_coder_id(coder_a)
    _check_coder_id(coder_b)

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            a_coder = _coders.load_coder(
                _projects_root(), project_id, coder_a
            )
        except FileNotFoundError:
            raise HTTPException(404, "Coder A not found")
        try:
            b_coder = _coders.load_coder(
                _projects_root(), project_id, coder_b
            )
        except FileNotFoundError:
            raise HTTPException(404, "Coder B not found")

        try:
            apps_a = _applications.list_applications(
                _projects_root(), project_id,
                source_id=source_id or None,
                coder_id=coder_a,
            )
            apps_b = _applications.list_applications(
                _projects_root(), project_id,
                source_id=source_id or None,
                coder_id=coder_b,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))

        codes = _codes.list_codes(_projects_root(), project_id)

    name_by_code_id = {c.id: c.name for c in codes}
    items, a_map, b_map = _icr_items_set(apps_a, apps_b)

    per_code = _icr.per_code_kappa(a_map, b_map, items=sorted(items))

    # Overall kappa: for each item, take the *set* of codes each coder
    # applied. Two coders agree on an item iff their code sets are
    # equal. Treat the equal/unequal flag as a single boolean label and
    # use Cohen's kappa over those booleans.
    a_labels = []
    b_labels = []
    for it in sorted(items):
        # Sort the code-set so the label is deterministic.
        a_labels.append(tuple(sorted(a_map.get(it, set()))))
        b_labels.append(tuple(sorted(b_map.get(it, set()))))
    overall_kappa = _icr.cohens_kappa(a_labels, b_labels)
    overall_label = _icr.interpret_kappa(overall_kappa)

    n_both_applied_any = sum(
        1 for it in items
        if a_map.get(it) and b_map.get(it)
    )

    per_code_out = []
    for code_id in sorted(per_code.keys()):
        kappa = per_code[code_id]
        n_a = sum(1 for it in items if code_id in a_map.get(it, set()))
        n_b = sum(1 for it in items if code_id in b_map.get(it, set()))
        per_code_out.append({
            "code_id": code_id,
            "code_name": name_by_code_id.get(code_id, code_id),
            "kappa": kappa,
            "label": _icr.interpret_kappa(kappa),
            "n_a_applied": n_a,
            "n_b_applied": n_b,
        })

    return JSONResponse({
        "coder_a": {"id": a_coder.id, "name": a_coder.name},
        "coder_b": {"id": b_coder.id, "name": b_coder.name},
        "source_id": source_id or None,
        "n_items": len(items),
        "n_both_applied_any": n_both_applied_any,
        "overall_kappa": overall_kappa,
        "overall_label": overall_label,
        "per_code": per_code_out,
    })


# --------------------------------------------------------------------------- #
# Sampling log (F1.4) — methodologically-transparent record of which
# source / participant was added (or planned, or removed, or just
# noted) at what time, under what sampling strategy, with what
# rationale, and aimed at filling which emerging category.
#
# The pure module + persistence shipped in f553954 (scribe/sampling_log.py)
# with 42 unit tests in tests/test_sampling_log.py. This block wires the
# user-facing surface so a researcher can record "why was this source
# added?" at attach-time and re-read the chronological log later — the
# core requirement of theoretical sampling for a credible GT audit
# trail (PLANNING.md W3.2).
#
# Append-only: the API is GET (list/count) + POST (append). Corrections
# are a fresh POST referencing the prior entry's id in ``notes``; we
# never expose PATCH/DELETE for individual entries. This mirrors the
# F9.1 event-log stance: the log is evidence, not editable state.
# --------------------------------------------------------------------------- #

from . import sampling_log as _sampling_log  # noqa: E402


def _check_sampling_entry_id(entry_id: str) -> None:
    if not _sampling_log.SAMPLING_ENTRY_ID_RE.match(entry_id):
        raise HTTPException(400, "Invalid sampling entry id")


@app.get("/api/projects/{project_id}/sampling_log")
async def list_sampling_log_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        entries = _sampling_log.read_sampling_log(_projects_root(), project_id)
    return JSONResponse(
        {
            "entries": [e.to_dict() for e in entries],
            "actions": list(_sampling_log.SAMPLING_ACTIONS),
            "decision_types": list(_sampling_log.SAMPLING_DECISION_TYPES),
        }
    )


@app.post("/api/projects/{project_id}/sampling_log")
async def append_sampling_log_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    # Optional source / participant references — empty strings are
    # treated as "not linked" because that's how HTML forms submit
    # blank fields. The dataclass validator rejects malformed shapes.
    source_id = body.get("source_id") or None
    if isinstance(source_id, str) and not source_id.strip():
        source_id = None
    participant_id = body.get("participant_id") or None
    if isinstance(participant_id, str) and not participant_id.strip():
        participant_id = None

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            entry = _sampling_log.SamplingEntry.new(
                project_id=project_id,
                action=str(body.get("action", "added") or "added"),
                decision_type=str(body.get("decision_type", "") or ""),
                source_id=source_id,
                participant_id=participant_id,
                target_category=str(body.get("target_category", "") or ""),
                rationale=str(body.get("rationale", "") or ""),
                notes=str(body.get("notes", "") or ""),
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid sampling-log payload: {e}")
        try:
            _sampling_log.append_sampling_entry(_projects_root(), entry)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
    return JSONResponse(entry.to_dict(), status_code=201)


# --------------------------------------------------------------------------- #
# Codes (F2.1) + applications (F4.1) — minimal CRUD wiring
#
# The data layer (scribe.codes / scribe.applications / scribe.code_versions)
# was built per-feature by the loop. This block exposes it through HTTP so
# the UI can actually create a code and apply it to a span. Scope is
# deliberately tight: list / create / delete codes; list / create / delete
# applications. Update goes through the lifecycle endpoints (F2.3) which
# the loop already wired separately.
#
# Single-user local tool, so every application needs a coder_id and we lazily
# create a "default" coder per project on first use rather than asking the
# UI to deal with multi-coder mode (F2.5).
# --------------------------------------------------------------------------- #

from . import codes as _codes  # noqa: E402
from . import applications as _applications  # noqa: E402
from . import code_versions as _code_versions  # noqa: E402
from . import code_lifecycle as _code_lifecycle  # noqa: E402
from . import coders as _coders  # noqa: E402
from . import codebook_lock as _codebook_lock  # noqa: E402


def _check_code_id(code_id: str) -> None:
    if not _codes.CODE_ID_RE.match(code_id):
        raise HTTPException(400, "Invalid code id")


def _check_application_id(application_id: str) -> None:
    if not _applications.APPLICATION_ID_RE.match(application_id):
        raise HTTPException(400, "Invalid application id")


def _ensure_default_coder(project_id: str) -> str:
    """Return a coder id for ``project_id``. Creates a "default" coder on
    first use so the single-user local flow doesn't have to surface coder
    management. Multi-coder mode (F2.5) writes additional coders via
    its own UI; this only ever creates one."""
    existing = _coders.list_coders(_projects_root(), project_id)
    if existing:
        # Use the oldest coder by created_at (fall back to id if missing).
        existing.sort(key=lambda c: (c.created_at, c.id))
        return existing[0].id
    coder = _coders.Coder.new(project_id=project_id, name="You", role="researcher")
    _coders.save_coder(_projects_root(), coder)
    return coder.id


@app.get("/api/projects/{project_id}/codes")
async def list_codes_endpoint(project_id: str) -> JSONResponse:
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        codes = _codes.list_codes(_projects_root(), project_id)
    return JSONResponse({"codes": [c.to_dict() for c in codes]})


@app.post("/api/projects/{project_id}/codes")
async def create_code_endpoint(project_id: str, request: Request) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        # F2.4: a locked codebook refuses new codes. The patch endpoint
        # already has the same guard; creating a brand-new code is just
        # another structural write that breaks the audit trail if it
        # silently lands while locked.
        try:
            _codebook_lock.assert_codebook_unlocked(
                _projects_root(), project_id
            )
        except _codebook_lock.LockedCodebookError as e:
            raise HTTPException(409, str(e))
        try:
            code = _codes.Code.new(
                project_id=project_id,
                name=str(body.get("name", "")),
                definition=str(body.get("definition", "") or ""),
                inclusion_criteria=str(body.get("inclusion_criteria", "") or ""),
                exclusion_criteria=str(body.get("exclusion_criteria", "") or ""),
                exemplars=body.get("exemplars") or [],
                parent_code_id=body.get("parent_code_id") or None,
                related_codes=body.get("related_codes") or [],
                theoretical_memo=str(body.get("theoretical_memo", "") or ""),
                stage=str(body.get("stage", "initial") or "initial"),
                colour=str(body.get("colour", "") or ""),
                status=str(body.get("status", "active") or "active"),
                provenance=body.get("provenance") or {},
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid code payload: {e}")
        _codes.save_code(_projects_root(), code)
        # Record the initial version so applications have something to
        # reference via definition_version_id_at_apply.
        _code_versions.record_code_version(_projects_root(), code, change_note="initial")
    return JSONResponse(code.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}/codes/{code_id}")
async def get_code_endpoint(project_id: str, code_id: str) -> JSONResponse:
    _check_project_id(project_id)
    _check_code_id(code_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            code = _codes.load_code(_projects_root(), project_id, code_id)
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
    return JSONResponse(code.to_dict())


@app.patch("/api/projects/{project_id}/codes/{code_id}")
async def patch_code_endpoint(
    project_id: str, code_id: str, request: Request
) -> JSONResponse:
    """Edit an existing code (F2.1's full field set + F2.2 versioning).

    Accepts any subset of the Code entity's writable fields:
    ``name``, ``definition``, ``inclusion_criteria``, ``exclusion_criteria``,
    ``exemplars``, ``parent_code_id``, ``related_codes``,
    ``theoretical_memo``, ``stage``, ``colour``, ``status``, ``provenance``.

    The optional ``change_note`` key annotates the new version that
    F2.2's revision log records when the definition actually changes.

    Refuses with 409 when the codebook is locked (F2.4); the UI should
    prompt for an unlock-with-reason memo before retrying.
    """
    _check_project_id(project_id)
    _check_code_id(code_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    change_note = str(body.get("change_note", "") or "")
    # Strip the metadata key from the patch dict so apply_update only
    # sees real Code fields.
    patch = {k: v for k, v in body.items() if k != "change_note"}

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        # F2.4: refuse edits to a locked codebook.
        try:
            _codebook_lock.assert_codebook_unlocked(
                _projects_root(), project_id
            )
        except _codebook_lock.LockedCodebookError as e:
            raise HTTPException(409, str(e))
        try:
            code = _codes.load_code(_projects_root(), project_id, code_id)
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        try:
            code.apply_update(patch)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid code patch: {e}")
        # save_code_with_version records a new revision when the
        # F2.2 DEFINITION_FIELDS actually changed; otherwise it just
        # writes the file and re-uses the latest version. Either way
        # callers (e.g. existing applications) keep a stable
        # definition_version_id_at_apply pointer.
        _code_versions.save_code_with_version(
            _projects_root(), code, change_note=change_note
        )
    return JSONResponse(code.to_dict())


@app.delete("/api/projects/{project_id}/codes/{code_id}")
async def delete_code_endpoint(project_id: str, code_id: str) -> JSONResponse:
    _check_project_id(project_id)
    _check_code_id(code_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        ok = _codes.delete_code(_projects_root(), project_id, code_id)
    if not ok:
        raise HTTPException(404, "Code not found")
    return JSONResponse({"ok": True})


@app.get("/api/projects/{project_id}/codes/{code_id}/versions")
async def list_code_versions_endpoint(
    project_id: str, code_id: str
) -> JSONResponse:
    """Return the F2.2 revision history for a code.

    Each entry exposes the version id, ordinal, timestamp, optional
    change note, and a diff-summary listing which DEFINITION_FIELDS
    changed relative to the previous version. The first version's diff
    is the full set of populated fields ("initial").

    The response is the surface the codebook editor's "Revision history"
    panel consumes; F9.2's definition-at-apply audit reports get the
    same data via :mod:`scribe.code_versions` directly.
    """
    _check_project_id(project_id)
    _check_code_id(code_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        # Confirm the code itself still exists; an existing version log
        # for a deleted code shouldn't be silently exposed.
        try:
            _codes.load_code(_projects_root(), project_id, code_id)
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        versions = _code_versions.read_code_versions(
            _projects_root(), project_id, code_id
        )

    # Build a diff summary per version: which DEFINITION_FIELDS differ
    # from the previous version. The first version's "diff" is the set
    # of populated fields, marking it as the initial snapshot.
    out: list[dict] = []
    prev_sig: dict | None = None
    for v in versions:
        sig = _code_versions.definition_signature(v.snapshot)
        if prev_sig is None:
            changed = sorted(
                f for f, val in sig.items()
                if val not in (None, "", [], {})
            )
        else:
            changed = sorted(
                f for f in _code_versions.DEFINITION_FIELDS
                if sig.get(f) != prev_sig.get(f)
            )
        out.append({
            "id": v.id,
            "version": v.version,
            "created_at": v.created_at,
            "change_note": v.change_note,
            "changed_fields": changed,
            "snapshot": v.snapshot,
        })
        prev_sig = sig

    return JSONResponse({"versions": out})


# --------------------------------------------------------------------------- #
# F2.3 — Code lifecycle ops (rename / retire / merge / split / hierarchy)
#
# These wrap :mod:`scribe.code_lifecycle` so the codebook editor's per-row
# ⋮ menu has somewhere to call. Every op respects F2.4's lock guard:
# locked codebooks 409 with the LockedCodebookError message.
# --------------------------------------------------------------------------- #


def _lifecycle_lock_guard(project_id: str) -> None:
    """409 if the codebook is locked. Caller must hold ``PROJECTS_LOCK``."""
    try:
        _codebook_lock.assert_codebook_unlocked(_projects_root(), project_id)
    except _codebook_lock.LockedCodebookError as e:
        raise HTTPException(409, str(e))


def _lifecycle_response(code: _codes.Code, version: _code_versions.CodeVersion | None) -> dict:
    return {
        "code": code.to_dict(),
        "version": version.to_dict() if version is not None else None,
    }


@app.post("/api/projects/{project_id}/codes/{code_id}/rename")
async def rename_code_endpoint(
    project_id: str, code_id: str, request: Request
) -> JSONResponse:
    """Rename a code (F2.3). ``name`` is a definition field, so the
    rename automatically records a new immutable version (F2.2)."""
    _check_project_id(project_id)
    _check_code_id(code_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    new_name = str(body.get("name", "") or "").strip()
    if not new_name:
        raise HTTPException(400, "name is required")
    change_note = str(body.get("change_note", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        _lifecycle_lock_guard(project_id)
        try:
            code, version = _code_lifecycle.rename_code(
                _projects_root(), project_id, code_id, new_name,
                change_note=change_note,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid rename payload: {e}")
    return JSONResponse(_lifecycle_response(code, version))


@app.post("/api/projects/{project_id}/codes/{code_id}/retire")
async def retire_code_endpoint(
    project_id: str, code_id: str, request: Request
) -> JSONResponse:
    """Mark a code as ``status='retired'`` (F2.3). Idempotent."""
    _check_project_id(project_id)
    _check_code_id(code_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    change_note = str(body.get("change_note", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        _lifecycle_lock_guard(project_id)
        try:
            code, version = _code_lifecycle.retire_code(
                _projects_root(), project_id, code_id,
                change_note=change_note,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse(_lifecycle_response(code, version))


@app.post("/api/projects/{project_id}/codes/{code_id}/parent")
async def set_code_parent_endpoint(
    project_id: str, code_id: str, request: Request
) -> JSONResponse:
    """Set / clear ``parent_code_id`` (F2.3). Cycles + missing-parent
    are rejected with 400. Pass ``parent_code_id: null`` (or the empty
    string) to detach a code to the top level."""
    _check_project_id(project_id)
    _check_code_id(code_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    raw_parent = body.get("parent_code_id")
    new_parent = None
    if raw_parent not in (None, ""):
        new_parent = str(raw_parent)
    change_note = str(body.get("change_note", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        _lifecycle_lock_guard(project_id)
        try:
            code, version = _code_lifecycle.set_code_parent(
                _projects_root(), project_id, code_id, new_parent,
                change_note=change_note,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse(_lifecycle_response(code, version))


@app.post("/api/projects/{project_id}/codes/{code_id}/promote")
async def promote_code_endpoint(
    project_id: str, code_id: str, request: Request
) -> JSONResponse:
    """Lift a code one level in the hierarchy (F2.3). Idempotent on
    root codes."""
    _check_project_id(project_id)
    _check_code_id(code_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    change_note = str(body.get("change_note", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        _lifecycle_lock_guard(project_id)
        try:
            code, version = _code_lifecycle.promote_code(
                _projects_root(), project_id, code_id,
                change_note=change_note,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse(_lifecycle_response(code, version))


@app.post("/api/projects/{project_id}/codes/merge")
async def merge_codes_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Merge one or more source codes into ``target_code_id`` (F2.3).

    Body shape::

        {
            "target_code_id": "...",
            "source_code_ids": ["...", "..."],
            "change_note": "optional"
        }
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    target_id = body.get("target_code_id") or ""
    source_ids = body.get("source_code_ids") or []
    if not target_id or not isinstance(target_id, str):
        raise HTTPException(400, "target_code_id is required")
    if not isinstance(source_ids, list) or not source_ids:
        raise HTTPException(400, "source_code_ids must be a non-empty list")
    _check_code_id(target_id)
    for sid in source_ids:
        if not isinstance(sid, str):
            raise HTTPException(400, "source_code_ids must be strings")
        _check_code_id(sid)
    change_note = str(body.get("change_note", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        _lifecycle_lock_guard(project_id)
        try:
            target, retired = _code_lifecycle.merge_codes(
                _projects_root(), project_id, source_ids, target_id,
                change_note=change_note,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "target": target.to_dict(),
        "retired": [c.to_dict() for c in retired],
    })


@app.post("/api/projects/{project_id}/codes/{code_id}/split")
async def split_code_endpoint(
    project_id: str, code_id: str, request: Request
) -> JSONResponse:
    """Split a code into two or more new codes (F2.3).

    Body shape::

        {
            "new_codes": [
                {"name": "...", "definition": "..." (optional), ...},
                {"name": "..."}
            ],
            "change_note": "optional"
        }

    The source code is retired with ``provenance['split_into']``;
    each new code carries ``provenance['split_from'] = <source_id>``.
    """
    _check_project_id(project_id)
    _check_code_id(code_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    new_codes_raw = body.get("new_codes")
    if not isinstance(new_codes_raw, list):
        raise HTTPException(400, "new_codes must be a list")
    if len(new_codes_raw) < 2:
        raise HTTPException(400, "Splitting requires at least two new codes")
    for spec in new_codes_raw:
        if not isinstance(spec, dict):
            raise HTTPException(400, "Each split entry must be an object")
        name = spec.get("name")
        if not name or not str(name).strip():
            raise HTTPException(400, "Each split entry must include a 'name'")
    change_note = str(body.get("change_note", "") or "")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        _lifecycle_lock_guard(project_id)
        try:
            source, new_codes = _code_lifecycle.split_code(
                _projects_root(), project_id, code_id,
                list(new_codes_raw),
                change_note=change_note,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid split payload: {e}")
    return JSONResponse({
        "source": source.to_dict(),
        "new_codes": [c.to_dict() for c in new_codes],
    })


# --------------------------------------------------------------------------- #
# F2.4 — Codebook lock / unlock toggle
#
# The pure module (`scribe.codebook_lock`) shipped in ed0cf5db with
# 57 tests; the lock guard is already wired into the codes /
# code-lifecycle / memo-promote endpoints. What was missing was a way
# for the user to *flip* the toggle: locking is a deliberate
# methodological move, and unlocking requires a written
# justification + memo. These three endpoints surface that workflow
# so the codebook editor can render a stage banner with a working
# "🔒 Lock codebook" / "🔓 Unlock with reason…" affordance.
#
# Shape:
#   GET  /api/projects/<pid>/codebook/lock       — current state
#                                                  ({locked, stage, log})
#   POST /api/projects/<pid>/codebook/lock       — body {"reason": "..."}
#   POST /api/projects/<pid>/codebook/unlock     — body
#       {"reason": "...", "methodological_memo": "...", "new_stage"?}
#
# Validation errors map to 400; "already locked" / "not locked" map to
# 409 (state conflict, not a bad request).
# --------------------------------------------------------------------------- #


@app.get("/api/projects/{project_id}/codebook/lock")
async def get_codebook_lock_state_endpoint(project_id: str) -> JSONResponse:
    """Return the current lock state of the codebook + the lock log.

    The codebook editor uses this to render the stage banner ("Stage:
    locked" / "Stage: focused") and the appropriate toggle button. The
    log is the audit trail of every lock / unlock event with the
    reason and (for unlocks) the methodological memo.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        project = _projects.load_project(_projects_root(), project_id)
        log = _codebook_lock.read_lock_log(_projects_root(), project_id)
    return JSONResponse({
        "project_id": project_id,
        "locked": project.codebook_stage == _codebook_lock.LOCKED_STAGE,
        "stage": project.codebook_stage,
        "log": [asdict(ev) for ev in log],
    })


@app.post("/api/projects/{project_id}/codebook/lock")
async def lock_codebook_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Lock the codebook (F2.4).

    Body: ``{"reason": "..."}``. ``reason`` is required and non-empty.
    Returns 409 if the codebook is already locked (no-op locks would
    pollute the audit log without changing state — re-locking is the
    "wait, did I already?" footgun the spec deliberately rejects).
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    reason = str(body.get("reason", "") or "").strip()
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            project, event = _codebook_lock.lock_codebook(
                _projects_root(), project_id, reason=reason
            )
        except _codebook_lock.LockedCodebookError as e:
            # Already-locked is a state conflict, not a bad request.
            raise HTTPException(409, str(e))
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "project": project.to_dict(),
        "event": asdict(event),
    })


@app.post("/api/projects/{project_id}/codebook/unlock")
async def unlock_codebook_endpoint(
    project_id: str, request: Request
) -> JSONResponse:
    """Unlock the codebook (F2.4).

    Body: ``{"reason": "...", "methodological_memo": "...",
    "new_stage": "..." (optional)}``. Both ``reason`` and
    ``methodological_memo`` are required and non-empty — the "breaking
    the seal" invariant from the spec.

    ``new_stage`` defaults to the most recent prior stage from the
    lock log (or ``"theoretical"`` if no history). Pass an explicit
    stage to land in a specific phase (initial / focused / axial /
    theoretical). Passing ``"locked"`` is rejected.

    Returns 409 if the codebook is not currently locked.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    reason = str(body.get("reason", "") or "").strip()
    memo = str(body.get("methodological_memo", "") or "").strip()
    raw_new_stage = body.get("new_stage")
    new_stage = (
        str(raw_new_stage).strip()
        if raw_new_stage is not None and str(raw_new_stage).strip()
        else None
    )
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        # The pure helper raises ProjectValidationError when the project
        # isn't currently locked. That's *state-shaped* (not the request
        # itself being invalid), so we sniff the message to map it to
        # 409 vs 400; clean separation matches the lock-already-set
        # branch above.
        try:
            project, event = _codebook_lock.unlock_codebook(
                _projects_root(), project_id,
                reason=reason,
                methodological_memo=memo,
                new_stage=new_stage,
            )
        except _projects.ProjectValidationError as e:
            msg = str(e)
            if "is not locked" in msg:
                raise HTTPException(409, msg)
            raise HTTPException(400, msg)
    return JSONResponse({
        "project": project.to_dict(),
        "event": asdict(event),
    })


@app.get("/api/projects/{project_id}/applications")
async def list_applications_endpoint(
    project_id: str, source_id: str = "", code_id: str = "",
    coder_id: str = "",
) -> JSONResponse:
    """List applications. Optional ``source_id`` / ``code_id`` /
    ``coder_id`` query parameters narrow the result. The source-coding
    view uses ``source_id`` to render only this source's applications;
    the F2.5 ICR view uses ``coder_id`` (and source filter) to compare
    two coders' work on the same items."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            all_apps = _applications.list_applications(
                _projects_root(), project_id,
                source_id=source_id or None,
                code_id=code_id or None,
                coder_id=coder_id or None,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"applications": [a.to_dict() for a in all_apps]})


@app.post("/api/projects/{project_id}/applications")
async def create_application_endpoint(project_id: str, request: Request) -> JSONResponse:
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    code_id = body.get("code_id")
    source_id = body.get("source_id")
    if not code_id:
        raise HTTPException(400, "code_id is required")
    if not source_id:
        raise HTTPException(400, "source_id is required")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        # Ensure the referenced code exists and grab the latest version id
        # so the application snapshot is methodologically correct (F4.1
        # requires definition_version_id_at_apply).
        try:
            code = _codes.load_code(_projects_root(), project_id, code_id)
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")
        latest = _code_versions.latest_code_version(_projects_root(), project_id, code_id)
        if latest is None:
            # Should be impossible because create_code records v1, but
            # belt-and-braces — record one now.
            latest = _code_versions.record_code_version(
                _projects_root(), code, change_note="initial-on-demand",
            )

        # F2.5 multi-coder mode: an explicit ``coder_id`` in the body
        # routes the application to a specific Coder. If omitted (the
        # single-coder default), fall back to the project's default
        # coder. Validate the supplied id shape *and* existence so a
        # caller can't smuggle a foreign coder onto an application.
        explicit_coder_id = body.get("coder_id")
        if explicit_coder_id:
            cid = str(explicit_coder_id)
            if not _coders.CODER_ID_RE.match(cid):
                raise HTTPException(400, "Invalid coder_id")
            try:
                _coders.load_coder(_projects_root(), project_id, cid)
            except FileNotFoundError:
                raise HTTPException(404, "Coder not found")
            coder_id = cid
        else:
            coder_id = _ensure_default_coder(project_id)

        try:
            app_obj = _applications.Application.new(
                project_id=project_id,
                code_id=code_id,
                source_id=str(source_id),
                coder_id=coder_id,
                anchor_start_word_id=str(body.get("anchor_start_word_id", "")),
                anchor_end_word_id=str(body.get("anchor_end_word_id", "")),
                definition_version_id_at_apply=latest.id,
                start_char_offset=body.get("start_char_offset"),
                end_char_offset=body.get("end_char_offset"),
                confidence=body.get("confidence"),
                provenance=body.get("provenance") or {},
                note=str(body.get("note", "") or ""),
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid application payload: {e}")
        _applications.save_application(_projects_root(), app_obj)
    return JSONResponse(app_obj.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}/applications/spans")
async def applications_spans_endpoint(
    project_id: str, source_id: str,
) -> JSONResponse:
    """Per-(code, source) span structure for a single source (F4.2).

    Required query param: ``source_id`` — the F4.2 headline question
    ("how many places in this source does this code apply?") is
    inherently scoped to one source, so we make the caller commit to
    one rather than pretending to summarise the whole project.

    Defined **before** ``GET /applications/{application_id}`` so the
    static ``spans`` segment wins over the parametric capture: FastAPI
    matches in registration order, and ``_check_application_id`` would
    otherwise reject the literal "spans" as a bad id.

    Response shape::

        {
          "source_id": "<sid>",
          "by_code": [
            {
              "code_id": "<cid>",
              "application_count": <int>,        # how many apps total
              "component_count": <int>,          # F4.2 headline:
                                                 # distinct non-contiguous
                                                 # places (1 = single span,
                                                 # 2+ = non-contiguous case)
              "components": [
                {
                  "start_word_id": "s<seg>w<word>",
                  "end_word_id":   "s<seg>w<word>",
                  "size": <int>,                 # apps in this component
                },
                ...
              ],
              "duplicate_anchor_count": <int>,   # diagnostic
            },
            ...
          ],
          "duplicate_anchor_groups": [
            {
              "code_id": "<cid>",
              "application_ids": ["<aid>", ...],
            },
            ...
          ]
        }

    Sorted: ``by_code`` follows the order each code first appears in
    document order on this source (so the side-panel can paint codes
    in reading order); within each code, components are anchor-sorted
    earliest first.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            source = _sources.load_source(
                _projects_root(), project_id, source_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        apps = _applications.list_applications(
            _projects_root(), project_id, source_id=source_id,
        )
    # Cross-segment adjacency: include word counts when the transcript
    # is reachable. When it isn't, only within-segment adjacency is
    # detected — which matches the application_spans contract.
    word_counts = _segment_word_counts_for_source(source)

    # Bucket by code_id so we get a single per-code summary even when
    # apps are interleaved across codes in the underlying list.
    code_ids_in_order: list[str] = []
    seen: set[str] = set()
    for a in _application_spans.sort_by_anchor(apps):
        if a.code_id not in seen:
            seen.add(a.code_id)
            code_ids_in_order.append(a.code_id)

    dup_groups_all = _application_spans.find_duplicate_anchors(apps)

    by_code: list[dict] = []
    for cid in code_ids_in_order:
        components = _application_spans.non_contiguous_components(
            apps, cid, source_id, segment_word_counts=word_counts or None,
        )
        comp_payload: list[dict] = []
        for comp in components:
            first = comp[0]
            last = comp[-1]
            comp_payload.append({
                "start_word_id": first.anchor_start_word_id,
                "end_word_id": last.anchor_end_word_id,
                "size": len(comp),
            })
        dup_groups_for_code = [
            g for g in dup_groups_all
            if g and g[0].code_id == cid and g[0].source_id == source_id
        ]
        duplicate_anchor_count = sum(len(g) for g in dup_groups_for_code)
        application_count = sum(
            1 for a in apps if a.code_id == cid and a.source_id == source_id
        )
        by_code.append({
            "code_id": cid,
            "application_count": application_count,
            "component_count": len(components),
            "components": comp_payload,
            "duplicate_anchor_count": duplicate_anchor_count,
        })

    duplicate_payload = [
        {
            "code_id": g[0].code_id,
            "application_ids": [a.id for a in g],
        }
        for g in dup_groups_all
        if g and g[0].source_id == source_id
    ]

    return JSONResponse({
        "source_id": source_id,
        "by_code": by_code,
        "duplicate_anchor_groups": duplicate_payload,
    })


@app.get("/api/projects/{project_id}/applications/gutter")
async def applications_gutter_endpoint(
    project_id: str, source_id: str,
) -> JSONResponse:
    """Lane-assignment for the gutter renderer (F4.3).

    Returns the deterministic lane layout for every application on
    ``source_id``. The renderer (``source_coding.html``) paints
    coloured bars in lane × anchor positions so a researcher can see
    overlapping codes that don't fit cleanly as in-text highlights
    (the F4.3 design point: "in-text highlights stop being readable
    past ~3 layers; the gutter scales to many overlapping codes").

    Defined **before** ``GET /applications/{application_id}`` so the
    static ``gutter`` segment wins over the parametric capture: same
    registration-order rule that protects ``/spans``.

    Response shape mirrors :func:`scribe.application_gutter.serialise_layout`::

        {
          "source_id": "<sid>",
          "lane_count": <int>,        # max simultaneous overlap on this source
          "max_stack_depth": <int>,   # largest pairwise-overlap clique - 1
          "placements": [
            {
              "application_id": "<aid>",
              "lane": <int>,          # zero-indexed; 0 is closest to the text
              "stack_depth": <int>,   # how many other apps overlap this one
            },
            ...
          ]
        }

    Empty source returns the empty layout
    (``lane_count == 0``, ``placements == []``) with the requested
    ``source_id`` echoed back — clients shouldn't have to special-case
    "no applications yet".

    The JS renderer in ``source_coding.html`` is the canonical
    consumer; ``scribe/static/js/helpers.mjs`` ships an in-browser
    mirror of the algorithm so the page can re-lay-out without a
    round-trip after a local apply / delete. Both implementations
    agree byte-for-byte on lane indices (a Python and JS test pair
    locks that down).
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            _sources.load_source(_projects_root(), project_id, source_id)
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        apps = _applications.list_applications(
            _projects_root(), project_id, source_id=source_id,
        )
    layout = _application_gutter.assign_lanes(apps) if apps else None
    if layout is None:
        return JSONResponse({
            "source_id": source_id,
            "lane_count": 0,
            "max_stack_depth": 0,
            "placements": [],
        })
    payload = _application_gutter.serialise_layout(layout)
    # The pure module returns ``source_id=""`` only on an empty input,
    # which we've already short-circuited above. Echo the requested
    # source_id either way to keep the contract simple.
    payload["source_id"] = source_id
    return JSONResponse(payload)


@app.get("/api/projects/{project_id}/applications/{application_id}")
async def get_application_endpoint(project_id: str, application_id: str) -> JSONResponse:
    _check_project_id(project_id)
    _check_application_id(application_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            a = _applications.load_application(
                _projects_root(), project_id, application_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Application not found")
    return JSONResponse(a.to_dict())


@app.delete("/api/projects/{project_id}/applications/{application_id}")
async def delete_application_endpoint(project_id: str, application_id: str) -> JSONResponse:
    _check_project_id(project_id)
    _check_application_id(application_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        ok = _applications.delete_application(
            _projects_root(), project_id, application_id,
        )
    if not ok:
        raise HTTPException(404, "Application not found")
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# F4.6 — One-click playback range for a coded segment.
#
# The pure module :mod:`scribe.application_playback` (shipped in d476da3,
# 34 unit tests) turns an Application's word-id anchor into a wall-clock
# ``[start, end]`` second interval by reusing the editor's word→time map.
# This endpoint exposes that lookup so the coding view can surface a
# play button per application without duplicating the time-mapping
# logic in JS — and so any external consumer (CLI, future REFI-QDA
# round-trip) can ask "where in the media does this application play?"
# through one URL.
#
# Returns 200 + the playback range when timing is available; 404 with a
# diagnostic ``reason`` when the application has no playable timing
# (untimed transcript, no source media, anchors that fell out of range
# after a transcript edit). The UI renders a disabled play button in
# those cases — never a silent seek-to-zero.
# --------------------------------------------------------------------------- #

from . import application_playback as _application_playback  # noqa: E402


@app.get("/api/projects/{project_id}/applications/{application_id}/playback")
async def application_playback_endpoint(
    project_id: str, application_id: str,
) -> JSONResponse:
    """Resolve the wall-clock playback range for one coded application (F4.6).

    Looks up the application, walks to its source's transcript via
    ``transcript_job_id``, builds a word→time map, and returns the
    ``[start, end]`` seconds the editor should seek through to play
    just that coded segment back.

    Response shape (mirrors :class:`scribe.application_playback.PlaybackRange`)::

        {
          "application_id": "<aid>",
          "source_id":      "<sid>",
          "transcript_job_id": "<job_id>",   # so the UI can fetch /api/job/<id>/media
          "start":          <float seconds>,
          "end":            <float seconds>
        }

    Errors:

    * 400 — bad project / application id format.
    * 404 — application not found, OR no playback range available
      (source missing transcript, anchors out of range, transcript has
      no usable timing). The body's ``reason`` field tells the UI which
      case it hit so the play button can be hidden / tooltipped
      sensibly rather than seeking to zero.
    """
    _check_project_id(project_id)
    _check_application_id(application_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            app_obj = _applications.load_application(
                _projects_root(), project_id, application_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Application not found")
        try:
            source = _sources.load_source(
                _projects_root(), project_id, app_obj.source_id,
            )
        except FileNotFoundError:
            raise HTTPException(
                404,
                {"detail": "Source not found", "reason": "no-source"},
            )

    job_id = getattr(source, "transcript_job_id", "") or ""
    if not job_id:
        raise HTTPException(
            404,
            {"detail": "Source has no transcript", "reason": "no-transcript"},
        )

    segments = _load_segments_for_source_speaker_map(source)
    if segments is None:
        raise HTTPException(
            404,
            {"detail": "Transcript file not found", "reason": "no-segments"},
        )

    try:
        rng = _application_playback.playback_range_for_application(
            app_obj, segments,
        )
    except _projects.ProjectValidationError:
        # Anchor's segment_index falls outside this transcript — F4.5
        # orphan condition. Surface as 404 + reason so the UI can route
        # the user to the orphan queue rather than seeking blindly.
        raise HTTPException(
            404,
            {"detail": "Application anchor out of range", "reason": "orphan"},
        )

    if rng is None:
        raise HTTPException(
            404,
            {"detail": "No playable timing for this application", "reason": "no-timing"},
        )

    return JSONResponse({
        "application_id": rng.application_id,
        "source_id": rng.source_id,
        "transcript_job_id": job_id,
        "start": rng.start,
        "end": rng.end,
    })


# --------------------------------------------------------------------------- #
# Re-anchor on transcript edit + orphan-application review queue (F4.5)
#
# F4.5 ships :mod:`scribe.application_reanchor` as a pure planner that
# diffs old/new transcripts and decides per-application whether the
# anchor is unchanged, can be re-anchored, or has been orphaned. The
# data layer was complete in 2ce8928 (83 unit tests) but had no HTTP
# surface. This block wires it up:
#
#   * ``put_transcript`` (PUT /api/job/<id>/transcript) now reads the
#     pre-edit transcript before overwriting, computes a reanchor plan
#     for every application across every project whose Source links to
#     the edited job, applies the unchanged/reanchored outcomes in
#     place, and queues orphans on the per-project orphan review queue.
#     The PUT response gains a ``reanchor`` summary field listing the
#     per-project counts so the editor can surface a toast.
#   * ``GET /api/projects/<pid>/orphan_applications`` returns the queue
#     (sorted oldest-first by ``detected_at``).
#   * ``DELETE /api/projects/<pid>/orphan_applications/<aid>`` removes
#     a queue entry — used after a researcher has triaged the orphan
#     by hand (deleting the application, re-applying it, or accepting
#     that it's truly gone).
#   * ``GET /projects/<pid>/orphans`` renders the orphan-review page
#     (``orphan_queue.html``).
#   * ``source_coding.html`` exposes a "🛟 Orphan queue (N)" link in
#     the page actions so a coder lands on the review page from the
#     same screen they were coding on.
#
# Scope notes:
#   * The reanchor pass is best-effort — a malformed application file
#     or a project whose disk state has gone weird is logged and
#     skipped, never crashes the transcript save. The user's edit is
#     more important than perfect bookkeeping.
#   * No UI re-prompt: the planner's outcomes are fully deterministic,
#     and the orphan queue is the human-triage surface for the cases
#     the planner couldn't resolve. We do not auto-delete orphaned
#     applications — they keep their old anchors until a human acts.
# --------------------------------------------------------------------------- #

from . import application_reanchor as _application_reanchor  # noqa: E402


def _reanchor_apps_for_job(
    job_id: str,
    old_segments: list[dict[str, Any]],
    new_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the F4.5 reanchor planner across every project that links to ``job_id``.

    For each project whose Source has ``transcript_job_id == job_id``:

    * compute a :class:`scribe.application_reanchor.ReanchorPlan`
      against the source's applications;
    * apply ``unchanged`` outcomes as no-ops, ``reanchored`` outcomes
      as in-place anchor updates (sub-word offsets dropped — the
      docstring promise);
    * persist orphaned outcomes to the project's
      ``orphan_queue.json`` via
      :func:`scribe.application_reanchor.append_orphan_entries`.

    Returns one summary dict per project that had at least one
    application checked. Each summary carries ``project_id``,
    ``source_id``, and the three counts the editor can surface as a
    toast. Errors per project are caught + logged so a single broken
    project doesn't block the transcript save.

    Pure file-system + lock-free; the caller already holds whatever
    coarse locks the request needs. Within :func:`put_transcript` we
    take :data:`PROJECTS_LOCK` around this call so a concurrent code-
    application POST doesn't see half-rewritten files.
    """
    summaries: list[dict[str, Any]] = []
    if not isinstance(old_segments, list) or not isinstance(new_segments, list):
        return summaries
    if not isinstance(job_id, str) or not job_id:
        return summaries

    try:
        projects = _projects.list_projects(_projects_root())
    except Exception:
        return summaries

    for project in projects:
        try:
            sources = _sources.list_sources(_projects_root(), project.id)
        except Exception:
            continue
        for src in sources:
            if (src.transcript_job_id or "") != job_id:
                continue
            try:
                apps = _applications.list_applications(
                    _projects_root(), project.id, source_id=src.id,
                )
            except Exception:
                continue
            if not apps:
                continue
            try:
                plan = _application_reanchor.reanchor_applications(
                    apps, old_segments, new_segments,
                )
            except Exception:
                continue

            updated_count = 0
            unchanged_count = len(plan.unchanged)
            orphaned_count = len(plan.orphaned)

            # Apply unchanged/reanchored outcomes. Save only the
            # reanchored ones; ``apply_reanchor_outcome`` returns the
            # input as-is for unchanged outcomes, so saving those would
            # only churn modified_at without a real change.
            apps_by_id = {a.id: a for a in apps}
            for outcome in plan.reanchored:
                app = apps_by_id.get(outcome.application_id)
                if app is None:
                    continue
                try:
                    new_app = _application_reanchor.apply_reanchor_outcome(
                        app, outcome,
                    )
                    _applications.save_application(_projects_root(), new_app)
                    updated_count += 1
                except Exception:
                    continue

            # Persist orphans (dedupes by application_id).
            try:
                _application_reanchor.record_orphans_from_plan(
                    _projects_root(), project.id, plan, apps_by_id,
                )
            except Exception:
                pass

            summaries.append({
                "project_id": project.id,
                "source_id": src.id,
                "checked": len(apps),
                "unchanged": unchanged_count,
                "reanchored": updated_count,
                "orphaned": orphaned_count,
            })
    return summaries


def _read_transcript_segments(out_dir: Path) -> list[dict[str, Any]]:
    """Best-effort read of the transcript currently on disk for a job.

    Prefers ``edited.json`` (the editor's authoritative version), falls
    back to any ``*.json`` sibling that isn't ``edited.json`` (the same
    rule the QDPX exporter and a few other readers use). Returns ``[]``
    if nothing is readable — F4.5's planner treats every application
    as orphaned in that case, which is the right thing: we have no
    "old" reference to match against.
    """
    if not out_dir.exists():
        return []
    edited = out_dir / "edited.json"
    candidates: list[Path] = []
    if edited.is_file():
        candidates.append(edited)
    candidates.extend(
        sorted(p for p in out_dir.glob("*.json") if p.name != "edited.json")
    )
    for p in candidates:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            segs = payload.get("segments")
            if isinstance(segs, list):
                return segs
    return []


@app.get("/api/projects/{project_id}/orphan_applications")
async def list_orphan_applications_endpoint(project_id: str) -> JSONResponse:
    """List the project's orphan-review queue (F4.5).

    Each entry carries enough context for a reviewer to relocate /
    delete / re-apply the orphaned application even after further
    edits: the original word-id range, a snapshot of the anchored
    text, the failure ``reason`` from the planner, and ``detected_at``.

    Sorted by ``detected_at`` ascending (oldest orphans first) so a
    long-running project's review queue is FIFO.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            entries = _application_reanchor.load_orphan_queue(
                _projects_root(), project_id,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "orphans": [e.to_dict() for e in entries],
        "count": len(entries),
    })


@app.delete(
    "/api/projects/{project_id}/orphan_applications/{application_id}"
)
async def remove_orphan_application_endpoint(
    project_id: str, application_id: str,
) -> JSONResponse:
    """Remove one entry from the project's orphan queue (F4.5).

    Used after a reviewer has triaged the orphan: deleting the
    underlying application, re-applying it manually, or accepting
    that the text is truly gone. 404 if the entry isn't there;
    otherwise 200 with ``{"ok": true}``.

    Note: this does *not* delete the underlying Application — that
    goes through the existing
    :func:`delete_application_endpoint`. The orphan queue is a
    separate, append-add review surface.
    """
    _check_project_id(project_id)
    _check_application_id(application_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            removed = _application_reanchor.remove_from_orphan_queue(
                _projects_root(), project_id, application_id,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    if not removed:
        raise HTTPException(404, "Orphan entry not found")
    return JSONResponse({"ok": True})


@app.get("/projects/{project_id}/orphans", response_class=HTMLResponse)
async def project_orphans_page(
    request: Request, project_id: str,
) -> HTMLResponse:
    """Orphan-review page (F4.5).

    Lists every orphan-queue entry with its anchored text snapshot,
    failure reason, and detection timestamp. Each row offers two
    actions:

    * **Delete application** — calls
      ``DELETE /api/projects/<pid>/applications/<aid>`` (the F4.1
      Remove route). If the application was already gone, the call
      404s; we still let the reviewer dismiss the queue entry.
    * **Dismiss from queue** — calls
      ``DELETE /api/projects/<pid>/orphan_applications/<aid>``. Used
      when the reviewer has re-applied the code by hand or accepts
      the gap.

    The page does **not** auto-delete; every action is explicit and
    audit-trail-friendly.
    """
    pid = _project_id_or_404(project_id)
    project = None
    try:
        with PROJECTS_LOCK:
            project = _projects.load_project(_projects_root(), pid)
    except Exception:
        project = None
    return templates.TemplateResponse(request, "orphan_queue.html", {
        "project_id": pid,
        "project_name": getattr(project, "name", None),
        "page_title": "Orphan applications",
        "subtitle": (
            "Code applications whose anchored text could no longer "
            "be located after a transcript edit. Triage one by one."
        ),
    })


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
# F5.1 reachability — list / read / update / delete memos
#
# The Memo entity has a POST endpoint above (right-click + flat-create
# flows for F5.1 / F5.2). The Memos page (/projects/<pid>/memos) needs
# to *list* memos; deleting and editing make the page functionally
# complete instead of a write-only inbox. These endpoints surface the
# corresponding helpers in :mod:`scribe.memos` (``list_memos``,
# ``load_memo``, ``apply_update``, ``delete_memo``).
# --------------------------------------------------------------------------- #


@app.get("/api/projects/{project_id}/memos")
async def list_memos_endpoint(
    project_id: str,
    type: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    author_coder_id: str | None = None,
    tag: str | None = None,
) -> JSONResponse:
    """List memos in a project, optionally filtered.

    Query parameters mirror :func:`scribe.memos.list_memos`:

    * ``type`` — restrict to a single :data:`scribe.memos.MEMO_TYPES` value.
    * ``target_type`` / ``target_id`` — restrict to memos linked to a
      specific entity. Pass both for an exact target; ``target_type``
      alone returns memos linked to *any* entity of that type.
    * ``author_coder_id`` — restrict to memos by a single author.
    * ``tag`` — restrict to memos carrying a tag (exact, case-sensitive).

    Returns ``{"memos": [...]}`` so the response shape mirrors the
    sources/codes/applications listing endpoints. Filter validation
    errors come back as 400; the project not existing is 404.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            memos = _memos.list_memos(
                _projects_root(),
                project_id,
                type=type,
                target_type=target_type,
                target_id=target_id,
                author_coder_id=author_coder_id,
                tag=tag,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"memos": [m.to_dict() for m in memos]})


# --------------------------------------------------------------------------- #
# Memo export (F5.4)
#
# Per PLANNING.md F5.4:
#
#   > "Export all memos" filtered by type / linked-to.
#
# The pure exporters (CSV / Markdown / RTF / JSONL) shipped in
# 3148000 inside :mod:`scribe.memo_export`; until this surface
# landed, those four formats could only be reached from a Python
# REPL. This endpoint surfaces all four behind one URL so the
# memos page (and any future CLI / button) dispatches through one
# code path — same shape as F6.1's codebook export endpoint.
#
# **Route ordering matters.** ``/memos/export`` must be registered
# *before* ``/memos/{memo_id}`` so FastAPI matches the literal
# ``export`` segment first; otherwise the GET would treat ``export``
# as a memo id and 400 with "Invalid memo id".
#
# Filter query parameters mirror :func:`scribe.memos.list_memos`
# *and* :func:`scribe.memo_export.filter_memos`. The summary line
# embedded in the Markdown / RTF header is built via
# :func:`scribe.memo_export.build_filter_summary` so the export file
# explains which filters produced it.
#
# Errors:
#   * 400 — malformed project id, unknown format, invalid filter
#     value (delegates to ``filter_memos`` validation).
#   * 404 — project id not found.
#   * 200 — file body (including the empty-memos case; CSV emits a
#     header-only file, Markdown a placeholder, RTF a minimal
#     document, JSONL the empty string).
# --------------------------------------------------------------------------- #

from . import memo_export as _memo_export  # noqa: E402


@app.get("/api/projects/{project_id}/memos/export")
async def export_memos_endpoint(
    project_id: str,
    format: str = "csv",
    type: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    author_coder_id: str | None = None,
    tag: str | None = None,
) -> Response:
    """Download the project's memos in CSV / Markdown / RTF / JSONL (F5.4).

    Query string ``format``:

    * ``csv`` — RFC-4180 CSV (default).
    * ``markdown`` — structured CommonMark; alias ``md``.
    * ``rtf`` — minimal RTF 1.x; aliases ``word`` / ``doc`` / ``docx``.
    * ``jsonl`` — newline-delimited JSON; aliases ``ndjson`` / ``json``.

    Filter query string (any subset, AND-combined; ``target_type`` and
    ``target_id`` together require both to match on the same link):

    * ``type`` — restrict to one :data:`scribe.memos.MEMO_TYPES` value.
    * ``target_type`` — restrict to memos linking to this entity type.
    * ``target_id`` — restrict to memos linking to this entity id.
    * ``author_coder_id`` — restrict to memos by one author.
    * ``tag`` — restrict to memos carrying a tag (exact match).

    Headers:

    * ``Content-Type`` matches the format (with ``charset=utf-8`` for
      the text formats).
    * ``Content-Disposition: attachment; filename="<slug>-memos.<ext>"``
      so browsers prompt a save rather than rendering inline.

    Status codes: ``404`` if the project is missing; ``400`` for an
    unrecognised format or filter; ``200`` otherwise.
    """
    _check_project_id(project_id)
    try:
        fmt = _memo_export.normalise_format(format)
    except ValueError as e:
        raise HTTPException(400, str(e))
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            memos = _memos.list_memos(
                _projects_root(),
                project_id,
                type=type,
                target_type=target_type,
                target_id=target_id,
                author_coder_id=author_coder_id,
                tag=tag,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        # Build target_names off whatever sibling lists we have
        # hydrated. Keeps the Markdown / RTF link bullets readable as
        # "Pacing (code:abcd…)" rather than the bare id pair.
        codes = _codes.list_codes(_projects_root(), project_id)
        sources = _sources.list_sources(_projects_root(), project_id)
        try:
            participants = _participants.list_participants(
                _projects_root(), project_id
            )
        except Exception:  # pragma: no cover — defensive on partial projects
            participants = []
        try:
            coders = _coders.list_coders(_projects_root(), project_id)
        except Exception:  # pragma: no cover
            coders = []
    target_names = _memo_export.build_target_names(
        codes=codes,
        sources=sources,
        participants=participants,
        coders=coders,
        memos=memos,
        project=project,
    )
    filter_summary = _memo_export.build_filter_summary(
        type=type,
        target_type=target_type,
        target_id=target_id,
        author_coder_id=author_coder_id,
        tag=tag,
    )
    text = _memo_export.render_memos(
        fmt,
        memos,
        project=project,
        target_names=target_names,
        filter_summary=filter_summary,
    )
    spec = _memo_export.EXPORT_FORMATS[fmt]
    filename = _memo_export.slugify_memos_filename(project, fmt)
    headers = {
        # Quote the filename so spaces / non-ASCII never break the
        # header. We slugify to ASCII upstream, so the simple quoted
        # form is sufficient — no need for RFC 5987 ``filename*=``.
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=text,
        media_type=spec.media_type,
        headers=headers,
    )


@app.get("/api/projects/{project_id}/memos/{memo_id}")
async def get_memo_endpoint(
    project_id: str, memo_id: str
) -> JSONResponse:
    """Return a single memo by id.

    Used by the memos page when a researcher clicks a row to load
    the memo into the edit form. 404 when the project or memo is
    missing.
    """
    _check_project_id(project_id)
    _check_memo_id(memo_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            memo = _memos.load_memo(
                _projects_root(), project_id, memo_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Memo not found")
    return JSONResponse(memo.to_dict())


@app.patch("/api/projects/{project_id}/memos/{memo_id}")
async def patch_memo_endpoint(
    project_id: str, memo_id: str, request: Request
) -> JSONResponse:
    """Apply a partial update to a memo (F5.1 reachability).

    Body shape matches :meth:`scribe.memos.Memo.apply_update`: any of
    ``type`` / ``title`` / ``body`` / ``body_format`` /
    ``author_coder_id`` / ``links`` / ``tags`` / ``provenance``.
    Updating ``links`` / ``tags`` replaces the whole list; the UI is
    expected to read-modify-write so each edit produces one event in
    the F9.1 log.
    """
    _check_project_id(project_id)
    _check_memo_id(memo_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            memo = _memos.load_memo(
                _projects_root(), project_id, memo_id
            )
        except FileNotFoundError:
            raise HTTPException(404, "Memo not found")
        try:
            memo.apply_update(body)
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid memo update: {e}")
        _memos.save_memo(_projects_root(), memo)
    return JSONResponse(memo.to_dict())


@app.delete("/api/projects/{project_id}/memos/{memo_id}")
async def delete_memo_endpoint(
    project_id: str, memo_id: str
) -> JSONResponse:
    """Hard-delete a memo. Returns 204-style ``{"ok": True}``.

    F9.1 will record the deletion in the event log when that wires
    in; the file itself goes away here. Returns 404 if the memo did
    not exist (so a redundant delete is visible to the UI rather
    than silent).
    """
    _check_project_id(project_id)
    _check_memo_id(memo_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        removed = _memos.delete_memo(
            _projects_root(), project_id, memo_id
        )
    if not removed:
        raise HTTPException(404, "Memo not found")
    return JSONResponse({"ok": True})


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
# Coded-segment retrieval report (F6.2)
#
# F6.2 phrasing in PLANNING.md:
#
#   > Coded-segment retrieval report (per code, filterable, grouped by
#   > source / participant).
#
# The pure renderer is :mod:`scribe.retrieval_report` (CSV / Markdown /
# RTF; group-by code / source / participant / none; per-code / per-
# source / per-coder / per-participant filters that AND-combine).
# This endpoint hydrates applications + codes + sources + coders +
# participants from the on-disk store, optionally pulls transcript
# text from ``outputs/<job_id>/edited.json`` (or the engine's JSON
# sidecar) so the rendered body carries the actual quoted spans, and
# streams the result back as an attachment.
#
# Query parameters mirror the CLI surface in
# :mod:`scribe.scripts.export_retrieval_report` so a researcher who
# learns one carries the muscle memory across:
#
#   ?format=csv|markdown|rtf       (default csv; aliases md / word /
#                                   doc / docx accepted)
#   ?group_by=code|source|participant|none
#                                  (default code; aliases codes /
#                                   sources / participants / flat)
#   ?code=<code_id>                (repeatable; AND-combined filter)
#   ?source=<source_id>            (repeatable)
#   ?coder=<coder_id>              (repeatable)
#   ?participant=<participant_id>  (repeatable)
#
# Filters default to "match every row" when absent; passing an empty
# repeated filter (e.g. ``?code=&code=``) is treated as the user
# stripping out empties. We intentionally don't expose the
# "match-nothing on empty list" branch through the URL because there's
# no way for a browser to tell the difference between "no filter" and
# "empty filter" — the F6.2 pure module honours that distinction in
# tests, the HTTP surface deliberately collapses it.
#
# Status codes: ``404`` if the project is missing; ``400`` for an
# unrecognised format / group_by; ``200`` otherwise (including empty
# projects — a header-only CSV / placeholder Markdown / minimal RTF
# is the right answer when there's nothing coded yet).
# --------------------------------------------------------------------------- #


from . import retrieval_report as _retrieval_report  # noqa: E402


def _load_segments_for_source_for_retrieval(
    source: "_sources.Source",
) -> list[dict] | None:
    """Resolve a source's transcript segments under the server's OUTPUT_DIR.

    Same discovery rules as the QDPX exporter
    (:func:`_load_segments_for_source_for_qdpx`): prefer
    ``edited.json`` (the editor's authoritative version), fall back
    to any ``*.json`` engine sidecar with a ``segments`` array.
    Returns ``None`` if no transcript is available — the caller
    leaves the row's ``text`` field empty rather than failing the
    whole report.
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


@app.get("/api/projects/{project_id}/retrieval-report")
async def export_retrieval_report_endpoint(
    project_id: str,
    format: str = "csv",
    group_by: str = "code",
    code: list[str] | None = Query(default=None),
    source: list[str] | None = Query(default=None),
    coder: list[str] | None = Query(default=None),
    participant: list[str] | None = Query(default=None),
) -> Response:
    """Download the project's coded-segment retrieval report (F6.2).

    Body is one of CSV / Markdown / RTF, rendered by
    :func:`scribe.retrieval_report.render_report`. The rows hydrate
    every application in the project with code / source / coder /
    participant names + the actual quoted text from the editor's
    transcript (when one is available). Filters are applied after
    hydration so a row only ever appears in the body if it matches
    every supplied filter (AND-combined).

    Headers:

    * ``Content-Type`` matches the format
      (text/csv / text/markdown / application/rtf, all charset=utf-8
      where appropriate).
    * ``Content-Disposition: attachment; filename="<slug>-coded-
      segments.<ext>"`` so browsers prompt a save rather than
      rendering inline.

    Status codes: ``404`` if the project is missing; ``400`` for an
    unrecognised format or group_by; ``200`` otherwise.
    """
    _check_project_id(project_id)
    try:
        fmt = _retrieval_report.normalise_format(format)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        gb = _retrieval_report.normalise_group_by(group_by)
    except ValueError as e:
        raise HTTPException(400, str(e))

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        apps = _applications.list_applications(_projects_root(), project_id)
        codes = _codes.list_codes(_projects_root(), project_id)
        sources = _sources.list_sources(_projects_root(), project_id)
        coders = _coders.list_coders(_projects_root(), project_id)
        parts = _participants.list_participants(
            _projects_root(), project_id
        )

    # Hydrate transcript segments per source so quoted text survives
    # into the rendered body. Sources without a discoverable
    # transcript get omitted from the map; build_retrieval_rows
    # leaves their ``text`` empty without raising.
    segments_by_source: dict[str, list[dict]] = {}
    for s in sources:
        segs = _load_segments_for_source_for_retrieval(s)
        if segs is not None:
            segments_by_source[s.id] = segs

    rows = _retrieval_report.build_retrieval_rows(
        applications=apps,
        codes=codes,
        sources=sources,
        coders=coders,
        participants=parts,
        segments_by_source=segments_by_source or None,
    )

    # Drop empty repeats and apply filters only when the user
    # supplied at least one non-empty value. The pure module's
    # "empty list = match nothing" branch is deliberately not
    # reachable through the URL — see the section comment above.
    def _clean(values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        kept = [v for v in values if v]
        return kept or None

    code_f = _clean(code)
    source_f = _clean(source)
    coder_f = _clean(coder)
    part_f = _clean(participant)
    if any(f is not None for f in (code_f, source_f, coder_f, part_f)):
        rows = _retrieval_report.filter_rows(
            rows,
            code_ids=code_f,
            source_ids=source_f,
            coder_ids=coder_f,
            participant_ids=part_f,
        )

    text = _retrieval_report.render_report(
        fmt, rows, project=project, group_by=gb
    )
    spec = _retrieval_report.EXPORT_FORMATS[fmt]
    filename = _retrieval_report.slugify_report_filename(project, fmt)
    headers = {
        # Quote the filename so spaces / non-ASCII never break the
        # header. We slugify to ASCII upstream, so the simple quoted
        # form is sufficient — same convention as the F6.1 codebook
        # export.
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
# AI code suggestions (F8.3 / F8.4)
#
# Two related endpoints expose the existing scribe.code_suggestions and
# scribe.new_code_suggestions logic to the coding UI:
#
#   POST .../ai/suggestions   — given a span + query_text + mode,
#                                returns ranked existing codes (or new-code
#                                proposals) and persists the suggestion as
#                                an audit record (F9.6).
#   POST .../ai/suggestions/{sid}/accept   — turn an accepted suggestion
#                                into an Application; record decision.
#   POST .../ai/suggestions/{sid}/reject   — record rejection (kept as
#                                evidence per F9.6).
#
# All three are guarded by the AI gate (F8.10): if the project hasn't met
# the hand-coded threshold and isn't overriding, we return 412 with the
# gate status so the UI can show "code more by hand first".
# --------------------------------------------------------------------------- #

from . import code_suggestions as _code_suggestions  # noqa: E402
from . import new_code_suggestions as _new_code_suggestions  # noqa: E402
from . import ai_provenance as _ai_provenance  # noqa: E402


# Optional per-process override so tests can swap in an in-memory backend
# without touching the real Ollama daemon. Production code never sets this.
_ai_suggest_backend_override: Any = None


def _make_embed_and_generate_fns(
    cfg: "_ai_backend.BackendConfig",
    backend: "_ai_backend.ModelBackend",
) -> tuple[Any, Any, str, str]:
    """Wrap the backend's embed/generate methods into the simple
    Callables the suggestion modules expect, plus return the model
    names so they get persisted into the suggestion record.

    The signatures the suggestion modules want are:
        embed_fn(texts: Sequence[str]) -> Sequence[Sequence[float]]
        generate_fn(prompt: str) -> str

    We bake the model names + transport into the closure so the
    suggestion modules don't have to know about BackendConfig.
    """
    embedding_model = cfg.default_embedding_model
    generation_model = cfg.default_model
    transport = _ai_backend_transport_override or _ai_backend.urllib_transport

    def embed_fn(texts):
        if not embedding_model:
            raise _ai_backend.BackendValidationError(
                "No default_embedding_model configured for this project"
            )
        req = _ai_backend.EmbeddingRequest(
            model=embedding_model, inputs=tuple(texts),
        )
        resp = backend.embed(cfg, req, transport=transport)
        return resp.vectors

    def generate_fn(prompt):
        if not generation_model:
            raise _ai_backend.BackendValidationError(
                "No default_model (generation) configured for this project"
            )
        req = _ai_backend.GenerationRequest(
            model=generation_model, prompt=prompt,
        )
        resp = backend.generate(cfg, req, transport=transport)
        return resp.text

    return embed_fn, generate_fn, embedding_model, generation_model


def _resolve_suggestion_backend(project: "_projects.Project"):
    """Load + dispatch the configured AI backend, surfacing a helpful
    HTTPException if the user hasn't picked one yet."""
    if _ai_suggest_backend_override is not None:
        cfg, backend = _ai_suggest_backend_override
        return cfg, backend
    cfg = _ai_backend.load_backend_config(project)
    backend = _ai_backend.backend_for_config(cfg)
    return cfg, backend


@app.post("/api/projects/{project_id}/ai/suggestions")
async def post_ai_suggestion_endpoint(project_id: str, request: Request) -> JSONResponse:
    """F8.3 / F8.4 — suggest codes for a highlighted span.

    Body shape:
        {
          "source_id": "<sid>",
          "anchor_start_word_id": "s0w0",
          "anchor_end_word_id": "s0w12",
          "query_text": "the highlighted text the user selected",
          "mode": "existing" | "new"     (default: "existing")
        }

    Returns the persisted CodeSuggestion or NewCodeSuggestion dict.
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    source_id = body.get("source_id")
    start_id = body.get("anchor_start_word_id")
    end_id = body.get("anchor_end_word_id")
    query_text = body.get("query_text") or ""
    mode = (body.get("mode") or "existing").lower()
    if not source_id or not start_id or not end_id:
        raise HTTPException(400, "source_id, anchor_start_word_id, anchor_end_word_id are required")
    if mode not in ("existing", "new"):
        raise HTTPException(400, "mode must be 'existing' or 'new'")

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")

        # AI gate (F8.10) — feature names live in scribe.ai_provenance.AI_FEATURES.
        gate_feature = (
            _ai_provenance.AI_FEATURE_NEW_CODE_SUGGESTION if mode == "new"
            else _ai_provenance.AI_FEATURE_CODE_SUGGESTION
        )
        try:
            gate = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id, feature=gate_feature,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not gate.allowed:
            raise HTTPException(412, {
                "detail": "AI gate not satisfied",
                "gate": gate.to_dict(),
            })

        # Backend
        try:
            cfg, backend = _resolve_suggestion_backend(project)
            embed_fn, generate_fn, emb_model, gen_model = _make_embed_and_generate_fns(cfg, backend)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))

        # Project state for the suggestion modules
        codes = _codes.list_codes(_projects_root(), project_id)
        applications = _applications.list_applications(
            _projects_root(), project_id,
        )

        try:
            if mode == "existing":
                suggestion = _code_suggestions.suggest_codes_for_span(
                    projects_root=_projects_root(),
                    project_id=project_id,
                    source_id=str(source_id),
                    anchor_start_word_id=str(start_id),
                    anchor_end_word_id=str(end_id),
                    query_text=str(query_text),
                    codes=codes,
                    applications=applications,
                    embed_fn=embed_fn,
                    generate_fn=generate_fn,
                    embedding_model=emb_model,
                    generation_model=gen_model,
                )
                _code_suggestions.save_suggestion(_projects_root(), suggestion)
                return JSONResponse({
                    "kind": "existing",
                    "suggestion": suggestion.to_dict(),
                })
            else:
                suggestion = _new_code_suggestions.suggest_new_codes_for_span(
                    project_id=project_id,
                    source_id=str(source_id),
                    anchor_start_word_id=str(start_id),
                    anchor_end_word_id=str(end_id),
                    query_text=str(query_text),
                    codes=codes,
                    embed_fn=embed_fn,
                    generate_fn=generate_fn,
                    embedding_model=emb_model,
                    generation_model=gen_model,
                )
                _new_code_suggestions.save_new_code_suggestion(_projects_root(), suggestion)
                return JSONResponse({
                    "kind": "new",
                    "suggestion": suggestion.to_dict(),
                })
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except _ai_backend.BackendUnavailable as e:
            raise HTTPException(502, f"Backend unavailable: {e}")
        except _ai_backend.BackendError as e:
            raise HTTPException(500, str(e))


@app.post("/api/projects/{project_id}/ai/suggestions/{suggestion_id}/accept")
async def accept_ai_suggestion_endpoint(
    project_id: str, suggestion_id: str, request: Request,
) -> JSONResponse:
    """Turn an existing-code suggestion into an Application.

    Body shape (all optional except code_id; if missing we use the top
    candidate from the suggestion):
        {
          "code_id": "<cid>",      # which candidate was accepted
          "anchor_start_word_id": "s0w0",   # may differ from suggestion
          "anchor_end_word_id": "s0w12",
          "modified": false,       # true if user changed the code/anchor
        }
    """
    _check_project_id(project_id)
    if not _code_suggestions.SUGGESTION_ID_RE.match(suggestion_id):
        raise HTTPException(400, "Invalid suggestion id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            suggestion = _code_suggestions.load_suggestion(
                _projects_root(), project_id, suggestion_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Suggestion not found")
        if suggestion.decision != _code_suggestions.SUGGESTION_DECISION_PENDING:
            raise HTTPException(409, f"Suggestion already {suggestion.decision}")

        # Pick which candidate the human actually accepted. Default to the
        # top one. If the body specifies a code_id, look it up among the
        # suggestion's candidates AND among the project's full codebook
        # (in case the user picked a code that the suggestion didn't list).
        code_id = body.get("code_id")
        if not code_id and suggestion.candidates:
            code_id = suggestion.candidates[0].code_id
        if not code_id:
            raise HTTPException(400, "No code_id supplied and the suggestion has no candidates")

        # Verify the chosen code exists.
        try:
            code = _codes.load_code(_projects_root(), project_id, str(code_id))
        except FileNotFoundError:
            raise HTTPException(404, "Code not found")

        # Anchors come from the suggestion unless overridden.
        start_id = body.get("anchor_start_word_id") or suggestion.anchor_start_word_id
        end_id = body.get("anchor_end_word_id") or suggestion.anchor_end_word_id

        # Latest version for the apply-record.
        latest = _code_versions.latest_code_version(
            _projects_root(), project_id, code.id,
        )
        if latest is None:
            latest = _code_versions.record_code_version(
                _projects_root(), code, change_note="initial-on-accept",
            )

        coder_id = _ensure_default_coder(project_id)

        # "modified" means the human deviated from the suggestion: a
        # different code, or a different span. The body can also assert
        # it explicitly when the UI knows the user fiddled with things
        # the server can't see.
        top_candidate_code_id = (
            suggestion.candidates[0].code_id if suggestion.candidates else None
        )
        deviated = (
            (top_candidate_code_id is not None and code.id != top_candidate_code_id)
            or str(start_id) != suggestion.anchor_start_word_id
            or str(end_id) != suggestion.anchor_end_word_id
        )
        modified = bool(body.get("modified")) or deviated

        decision = (
            _code_suggestions.SUGGESTION_DECISION_MODIFIED if modified
            else _code_suggestions.SUGGESTION_DECISION_ACCEPTED
        )

        # Pull the chosen candidate's confidence (if any) so the
        # Application's AIProvenance carries it forward (F8.9).
        chosen_confidence: float | None = None
        for cand in suggestion.candidates:
            if cand.code_id == code.id:
                chosen_confidence = float(cand.combined_score)
                break

        ai_prov = _applications.AIProvenance.new(
            feature=_ai_provenance.AI_FEATURE_CODE_SUGGESTION,
            generation_model=suggestion.generation_model,
            embedding_model=suggestion.embedding_model,
            suggestion_id=suggestion.id,
            decision=decision,
            decided_by_coder_id=coder_id,
            confidence=chosen_confidence,
        )

        try:
            app_obj = _applications.Application.new(
                project_id=project_id,
                code_id=code.id,
                source_id=suggestion.source_id,
                coder_id=coder_id,
                anchor_start_word_id=str(start_id),
                anchor_end_word_id=str(end_id),
                definition_version_id_at_apply=latest.id,
                confidence=chosen_confidence,
                ai_provenance=ai_prov,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _applications.save_application(_projects_root(), app_obj)

        try:
            _code_suggestions.record_decision(
                suggestion,
                decision=decision,
                coder_id=coder_id,
                accepted_code_id=code.id,
                accepted_application_id=app_obj.id,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _code_suggestions.save_suggestion(_projects_root(), suggestion)

    return JSONResponse({
        "suggestion": suggestion.to_dict(),
        "application": app_obj.to_dict(),
    })


@app.post("/api/projects/{project_id}/ai/suggestions/{suggestion_id}/reject")
async def reject_ai_suggestion_endpoint(
    project_id: str, suggestion_id: str, request: Request,
) -> JSONResponse:
    """Record a rejection (F9.6: rejected suggestions are evidence too)."""
    _check_project_id(project_id)
    if not _code_suggestions.SUGGESTION_ID_RE.match(suggestion_id):
        raise HTTPException(400, "Invalid suggestion id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    reason = (body or {}).get("reason", "") if isinstance(body, dict) else ""

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            suggestion = _code_suggestions.load_suggestion(
                _projects_root(), project_id, suggestion_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Suggestion not found")
        if suggestion.decision != _code_suggestions.SUGGESTION_DECISION_PENDING:
            raise HTTPException(409, f"Suggestion already {suggestion.decision}")
        try:
            _code_suggestions.record_decision(
                suggestion,
                decision=_code_suggestions.SUGGESTION_DECISION_REJECTED,
                coder_id=_ensure_default_coder(project_id),
                rejection_reason=str(reason or "")[:_code_suggestions.MAX_REJECTION_REASON_LEN],
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _code_suggestions.save_suggestion(_projects_root(), suggestion)

    return JSONResponse({"suggestion": suggestion.to_dict()})


@app.get("/api/projects/{project_id}/ai/suggestions")
async def list_ai_suggestions_endpoint(
    project_id: str, source_id: str = "", decision: str = "",
) -> JSONResponse:
    """List persisted suggestions for the project. Optional filters
    narrow by source or decision (e.g. ?decision=pending)."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        suggestions = _code_suggestions.list_suggestions(
            _projects_root(), project_id,
            source_id=source_id or None,
            decision=decision or None,
        )
    return JSONResponse({
        "suggestions": [s.to_dict() for s in suggestions],
    })


# --------------------------------------------------------------------------- #
# New-code suggestion lifecycle endpoints (F8.4)
#
# F8.3 (existing-codebook) and F8.4 (propose-a-new-code) share the
# /ai/suggestions POST entrypoint via the ``mode`` field, but the
# **decision lifecycle** for new-code suggestions is structurally
# different: accepting one means *creating a Code* using the proposal's
# name + definition (or the user's edited version of those), then
# stamping the resulting suggestion record with
# ``accepted_proposal_index`` + ``created_code_id`` via
# ``record_new_code_decision``. The F8.3 accept/reject endpoints don't
# know how to do that, so F8.4 needs its own pair of routes.
#
# Endpoints:
#
#   GET    /api/projects/<pid>/ai/new-code-suggestions
#                                   — list (filter by source / decision).
#   POST   /api/projects/<pid>/ai/new-code-suggestions/<sid>/accept
#                                   — create a Code from the chosen
#                                     proposal, stamp the audit record.
#                                     Body: {accepted_proposal_index,
#                                     name?, definition?,
#                                     inclusion_criteria?,
#                                     exclusion_criteria?, modified?,
#                                     apply?}.
#   POST   /api/projects/<pid>/ai/new-code-suggestions/<sid>/reject
#                                   — record rejection (F9.6 audit row).
#                                     Body: {reason?}.
#
# Acceptance creates an Application with AIProvenance
# (feature=new_code_suggestion) when the body sets ``apply: true``
# (default) — that way clicking a proposal in the coding view both
# adds the new code to the codebook AND marks the highlighted span
# with it, which is what the user expects.
# --------------------------------------------------------------------------- #


@app.get("/api/projects/{project_id}/ai/new-code-suggestions")
async def list_ai_new_code_suggestions_endpoint(
    project_id: str, source_id: str = "", decision: str = "",
) -> JSONResponse:
    """List persisted F8.4 new-code suggestions for the project."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            suggestions = _new_code_suggestions.list_new_code_suggestions(
                _projects_root(), project_id,
                source_id=source_id or None,
                decision=decision or None,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "suggestions": [s.to_dict() for s in suggestions],
    })


@app.post(
    "/api/projects/{project_id}/ai/new-code-suggestions/{suggestion_id}/accept"
)
async def accept_ai_new_code_suggestion_endpoint(
    project_id: str, suggestion_id: str, request: Request,
) -> JSONResponse:
    """Turn an F8.4 new-code proposal into a real :class:`Code`.

    Body (all keys optional except ``accepted_proposal_index``)::

        {
          "accepted_proposal_index": 0,    # which proposal the human picked
          "name": "...",                  # override the proposal's name
          "definition": "...",            # override the proposal's definition
          "inclusion_criteria": "...",
          "exclusion_criteria": "...",
          "modified": false,              # auto-set if name/def differ
          "apply": true                   # also create an Application
        }

    Returns ``{suggestion, code, application?}``. ``application`` is
    only present when ``apply`` is ``true``.
    """
    _check_project_id(project_id)
    if not _new_code_suggestions.NEW_CODE_SUGGESTION_ID_RE.match(suggestion_id):
        raise HTTPException(400, "Invalid new-code suggestion id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    body = body or {}

    raw_idx = body.get("accepted_proposal_index")
    if raw_idx is None:
        raise HTTPException(
            400, "accepted_proposal_index is required"
        )
    try:
        accepted_idx = int(raw_idx)
    except (TypeError, ValueError):
        raise HTTPException(
            400, "accepted_proposal_index must be an integer"
        )

    apply_flag = body.get("apply", True)
    if not isinstance(apply_flag, bool):
        apply_flag = bool(apply_flag)

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            suggestion = _new_code_suggestions.load_new_code_suggestion(
                _projects_root(), project_id, suggestion_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Suggestion not found")
        if suggestion.decision != _new_code_suggestions.NEW_CODE_DECISION_PENDING:
            raise HTTPException(
                409, f"Suggestion already {suggestion.decision}"
            )
        if accepted_idx < 0 or accepted_idx >= len(suggestion.proposals):
            raise HTTPException(
                400,
                "accepted_proposal_index "
                f"{accepted_idx} out of range for "
                f"{len(suggestion.proposals)} proposals",
            )
        proposal = suggestion.proposals[accepted_idx]

        # F2.4 — refuse new codes when the codebook is locked. The user
        # has to unlock with a methodological memo first; the suggestion
        # stays pending so they can revisit it after unlocking.
        try:
            _codebook_lock.assert_codebook_unlocked(
                _projects_root(), project_id
            )
        except _codebook_lock.LockedCodebookError as e:
            raise HTTPException(409, str(e))

        # Allow the body to override name / definition / criteria. If
        # the user edited the proposal in the UI before submitting, this
        # is how the modification reaches the new Code. We also use the
        # diff to auto-derive ``modified`` so the audit trail records
        # whether the human deviated from the AI's wording.
        name_in = body.get("name")
        def_in = body.get("definition")
        inc_in = body.get("inclusion_criteria")
        exc_in = body.get("exclusion_criteria")
        name = str(name_in) if name_in is not None else proposal.name
        definition = (
            str(def_in) if def_in is not None else proposal.definition
        )
        inclusion = str(inc_in) if inc_in is not None else ""
        exclusion = str(exc_in) if exc_in is not None else ""

        deviated = (
            name.strip() != (proposal.name or "").strip()
            or definition.strip() != (proposal.definition or "").strip()
            or bool(inclusion.strip())
            or bool(exclusion.strip())
        )
        modified_flag = bool(body.get("modified")) or deviated
        decision = (
            _new_code_suggestions.NEW_CODE_DECISION_MODIFIED if modified_flag
            else _new_code_suggestions.NEW_CODE_DECISION_ACCEPTED
        )

        try:
            code = _codes.Code.new(
                project_id=project_id,
                name=name,
                definition=definition,
                inclusion_criteria=inclusion,
                exclusion_criteria=exclusion,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid code payload: {e}")
        _codes.save_code(_projects_root(), code)
        version = _code_versions.record_code_version(
            _projects_root(), code,
            change_note=f"initial-from-new-code-suggestion:{suggestion.id}",
        )

        # Stamp the F8.4 suggestion's audit record with the decision +
        # which proposal was picked + which Code resulted.
        coder_id = _ensure_default_coder(project_id)
        try:
            _new_code_suggestions.record_new_code_decision(
                suggestion,
                decision=decision,
                coder_id=coder_id,
                accepted_proposal_index=accepted_idx,
                created_code_id=code.id,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _new_code_suggestions.save_new_code_suggestion(
            _projects_root(), suggestion,
        )

        application_dict: dict[str, Any] | None = None
        if apply_flag:
            ai_prov = _applications.AIProvenance.new(
                feature=_ai_provenance.AI_FEATURE_NEW_CODE_SUGGESTION,
                generation_model=suggestion.generation_model,
                embedding_model=suggestion.embedding_model,
                suggestion_id=suggestion.id,
                decision=decision,
                decided_by_coder_id=coder_id,
            )
            try:
                app_obj = _applications.Application.new(
                    project_id=project_id,
                    code_id=code.id,
                    source_id=suggestion.source_id,
                    coder_id=coder_id,
                    anchor_start_word_id=suggestion.anchor_start_word_id,
                    anchor_end_word_id=suggestion.anchor_end_word_id,
                    definition_version_id_at_apply=version.id,
                    ai_provenance=ai_prov,
                )
            except _projects.ProjectValidationError as e:
                raise HTTPException(400, str(e))
            _applications.save_application(_projects_root(), app_obj)
            application_dict = app_obj.to_dict()

    out: dict[str, Any] = {
        "suggestion": suggestion.to_dict(),
        "code": code.to_dict(),
    }
    if application_dict is not None:
        out["application"] = application_dict
    return JSONResponse(out)


@app.post(
    "/api/projects/{project_id}/ai/new-code-suggestions/{suggestion_id}/reject"
)
async def reject_ai_new_code_suggestion_endpoint(
    project_id: str, suggestion_id: str, request: Request,
) -> JSONResponse:
    """Record a rejection on an F8.4 suggestion (F9.6 audit row)."""
    _check_project_id(project_id)
    if not _new_code_suggestions.NEW_CODE_SUGGESTION_ID_RE.match(suggestion_id):
        raise HTTPException(400, "Invalid new-code suggestion id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    reason = (body or {}).get("reason", "") if isinstance(body, dict) else ""

    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            suggestion = _new_code_suggestions.load_new_code_suggestion(
                _projects_root(), project_id, suggestion_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Suggestion not found")
        if suggestion.decision != _new_code_suggestions.NEW_CODE_DECISION_PENDING:
            raise HTTPException(
                409, f"Suggestion already {suggestion.decision}"
            )
        try:
            _new_code_suggestions.record_new_code_decision(
                suggestion,
                decision=_new_code_suggestions.NEW_CODE_DECISION_REJECTED,
                coder_id=_ensure_default_coder(project_id),
                rejection_reason=str(reason or "")[
                    : _new_code_suggestions.MAX_REJECTION_REASON_LEN
                ],
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        _new_code_suggestions.save_new_code_suggestion(
            _projects_root(), suggestion,
        )

    return JSONResponse({"suggestion": suggestion.to_dict()})


# --------------------------------------------------------------------------- #
# Embedding index management (F8.2)
#
# F8.2 ships the storage + refresh layer underneath the F8.3 / F8.5 /
# F8.6 / F8.8 AI features. PLANNING.md describes its lifecycle as
# "built on import; refreshed on edit" — operationally, refresh is a
# project-wide pass: enumerate every coded segment and uncoded
# paragraph, embed the new/changed text, drop orphans. Until this
# block landed there was no FastAPI surface for invoking or inspecting
# the index — only the suggestion endpoints used the *search* helpers
# inside the engine modules. These three endpoints expose the index
# itself so the AI page can show "you have N spans embedded" and
# trigger a refresh after the user adds applications / edits transcripts.
# --------------------------------------------------------------------------- #

from . import embedding_index as _embedding_index  # noqa: E402


def _gather_segments_by_source(
    sources: "list[_sources.Source]",
) -> dict[str, list[dict[str, Any]]]:
    """Resolve segments for every source; missing transcripts are
    represented by an empty list so the refresh's desired-spans
    enumerator simply yields nothing for that source."""
    out: dict[str, list[dict[str, Any]]] = {}
    for src in sources:
        segs = _load_segments_for_source_speaker_map(src) or []
        out[src.id] = segs
    return out


@app.get("/api/projects/{project_id}/ai/embedding-index")
async def get_project_embedding_index_endpoint(project_id: str) -> JSONResponse:
    """Return summary stats for the project's embedding index (F8.2).

    Response shape::

        {
          "total": int,
          "by_kind": {"coded_segment": N, "uncoded_paragraph": M},
          "by_source": {"<sid>": K, ...},
          "models": ["bge-m3", ...],            # distinct model names
          "last_modified_at": "<ISO>" | null,
          "configured_embedding_model": "<str>" | null,
        }
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        entries = _embedding_index.list_embedding_entries(
            _projects_root(), project_id,
        )
        try:
            cfg = _ai_backend.load_backend_config(project)
            configured_embedding_model = cfg.default_embedding_model or None
        except _ai_backend.BackendValidationError:
            configured_embedding_model = None
    by_kind: dict[str, int] = {}
    by_source: dict[str, int] = {}
    models: set[str] = set()
    last_modified = ""
    for e in entries:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        by_source[e.source_id] = by_source.get(e.source_id, 0) + 1
        models.add(e.model_name)
        if e.modified_at and e.modified_at > last_modified:
            last_modified = e.modified_at
    return JSONResponse({
        "total": len(entries),
        "by_kind": by_kind,
        "by_source": by_source,
        "models": sorted(models),
        "last_modified_at": last_modified or None,
        "configured_embedding_model": configured_embedding_model,
    })


@app.post("/api/projects/{project_id}/ai/embedding-index/refresh")
async def post_project_embedding_index_refresh_endpoint(
    project_id: str,
) -> JSONResponse:
    """Refresh the embedding index in-band (F8.2).

    Walks every source's transcript, computes the desired set of spans,
    embeds new/changed spans via the project's configured F8.1 backend,
    and deletes orphans. Returns the F8.2 RefreshResult counts so the
    UI can render "added N, updated M, removed K, unchanged J".

    Errors:
      400 if no embedding model is configured or backend config invalid;
      404 if the project doesn't exist;
      502 / 500 on backend failures.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            cfg, backend = _resolve_suggestion_backend(project)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))
        if not cfg.default_embedding_model:
            raise HTTPException(
                400,
                "No default_embedding_model configured for this project",
            )
        embed_fn, _gen_fn, emb_model, _gen_model = _make_embed_and_generate_fns(
            cfg, backend,
        )

        sources = _sources.list_sources(_projects_root(), project_id)
        applications = _applications.list_applications(
            _projects_root(), project_id,
        )
        segments_by_source = _gather_segments_by_source(sources)
        try:
            result = _embedding_index.refresh_embedding_index(
                projects_root=_projects_root(),
                project_id=project_id,
                applications=applications,
                segments_by_source=segments_by_source,
                embed_fn=embed_fn,
                model_name=emb_model,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except _ai_backend.BackendUnavailable as e:
            raise HTTPException(502, f"Backend unavailable: {e}")
        except _ai_backend.BackendError as e:
            raise HTTPException(500, str(e))
    return JSONResponse({
        "added": result.added_count,
        "updated": result.updated_count,
        "removed": result.removed_count,
        "unchanged": result.unchanged_count,
        "model": emb_model,
        "source_count": len(sources),
        "application_count": len(applications),
    })


@app.delete("/api/projects/{project_id}/ai/embedding-index")
async def delete_project_embedding_index_endpoint(
    project_id: str,
) -> JSONResponse:
    """Clear the entire embedding index (F8.2).

    Used when the embedding model changes — the next refresh will then
    repopulate from scratch under the new model. Returns the number of
    entries removed.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        n = _embedding_index.clear_embedding_index(
            _projects_root(), project_id,
        )
    return JSONResponse({"removed": n})


# --------------------------------------------------------------------------- #
# Find similar quotes (F8.5)
#
# F8.5 is the safest AI affordance per PLANNING.md: pure embedding-space
# nearest-neighbour search over the F8.2 index. No LLM call, no category
# judgement, no decision lifecycle — just "show me other quotes that
# look like this one". The user starts a search either:
#
#   * from a coded segment (.app-row "🔎 Find similar quotes" button) —
#     the seed is the application_id; the engine reuses its index entry
#     so we don't burn an embed call on a vector we already have, OR
#   * from free-form text (future: a panel input). Today the surface is
#     application-only because that's the high-traffic case.
#
# The route is gated by F8.10 like the suggestion routes (412 if the
# project hasn't met the hand-coded threshold). The persisted
# QuoteSearch row is the audit record (F9.6) — even null-result
# searches stay on disk so "I asked the model X at time T" stays in
# the history.
# --------------------------------------------------------------------------- #

from . import quote_similarity as _quote_similarity  # noqa: E402
from . import ai_invocation_log as _ai_invocation_log  # noqa: E402


@app.post("/api/projects/{project_id}/ai/quote-searches")
async def post_quote_search_endpoint(
    project_id: str, request: Request,
) -> JSONResponse:
    """F8.5 — find quotes semantically similar to a span or text.

    Body shape (one of two modes):

        # application-mode: seed from an existing coded segment
        {
          "source_id": "<sid>",
          "application_id": "<aid>",   # 12-char hex
          "top_k": 10,                  # optional, 1..50
          "min_score": 0.0,             # optional, in [-1, 1]
          "kind_filter": "coded_segment" | "uncoded_paragraph" | null,
          "code_id_filter": "<cid>" | null,
          "exclude_source_ids": ["<sid>", ...],
          "exclude_code_ids": ["<cid>", ...],
          "exclude_seed": true,         # default true; drops the seed
        }

        # text-mode: free-form query
        {
          "query_text": "the participant describes coping",
          ...same filter knobs as above
        }

    Returns the persisted QuoteSearch dict (matches list included).
    """
    _check_project_id(project_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    # Decide mode: application_id present → application; else text.
    application_id = (body.get("application_id") or "").strip() or None
    source_id = (body.get("source_id") or "").strip() or None
    query_text = body.get("query_text") or ""
    if not application_id and not str(query_text).strip():
        raise HTTPException(
            400,
            "Provide either application_id (+ source_id) or query_text",
        )

    # Optional filters / knobs.
    top_k = body.get("top_k", _quote_similarity.DEFAULT_TOP_K)
    min_score = body.get("min_score", _quote_similarity.DEFAULT_MIN_SCORE)
    kind_filter = body.get("kind_filter") or None
    source_id_filter = body.get("source_id_filter") or None
    code_id_filter = body.get("code_id_filter") or None
    exclude_source_ids = body.get("exclude_source_ids") or ()
    exclude_code_ids = body.get("exclude_code_ids") or ()
    # exclude_seed defaults True for application mode (the natural ask),
    # False otherwise — matches the engine default.
    exclude_seed = bool(body.get("exclude_seed", True))
    notes = body.get("notes") or ""

    try:
        top_k_i = int(top_k)
        min_score_f = float(min_score)
    except (TypeError, ValueError):
        raise HTTPException(400, "top_k must be an int and min_score must be a float")
    if top_k_i < 1 or top_k_i > 100:
        raise HTTPException(400, "top_k must be in [1, 100]")
    if min_score_f < -1.0 or min_score_f > 1.0:
        raise HTTPException(400, "min_score must be in [-1, 1]")

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")

        # AI gate (F8.10). F8.5 is gateable like the other AI features.
        # Per PLANNING.md F8.10 a project can choose to exempt
        # quote_similarity (it's the safest feature) by listing it in
        # ai_gate.exempt_features; the gate evaluator already honours
        # that.
        try:
            gate = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id,
                feature=_ai_provenance.AI_FEATURE_QUOTE_SIMILARITY,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not gate.allowed:
            raise HTTPException(412, {
                "detail": "AI gate not satisfied",
                "gate": gate.to_dict(),
            })

        # Backend.
        try:
            cfg, backend = _resolve_suggestion_backend(project)
            embed_fn, _gen_fn, emb_model, _gen_model = (
                _make_embed_and_generate_fns(cfg, backend)
            )
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))

        # Project state needed for code-id resolution on coded matches.
        applications = _applications.list_applications(
            _projects_root(), project_id,
        )

        try:
            search = _quote_similarity.find_similar_quotes(
                projects_root=_projects_root(),
                project_id=project_id,
                embed_fn=embed_fn,
                applications=applications,
                query_text=str(query_text or ""),
                query_application_id=application_id,
                query_source_id=source_id,
                embedding_model=emb_model,
                top_k=top_k_i,
                min_score=min_score_f,
                kind_filter=kind_filter,
                source_id_filter=source_id_filter,
                exclude_source_ids=tuple(
                    str(x) for x in (exclude_source_ids or ())
                ),
                code_id_filter=code_id_filter,
                exclude_code_ids=tuple(
                    str(x) for x in (exclude_code_ids or ())
                ),
                exclude_seed=exclude_seed,
                notes=str(notes or ""),
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        except _ai_backend.BackendUnavailable as e:
            raise HTTPException(502, f"Backend unavailable: {e}")
        except _ai_backend.BackendError as e:
            raise HTTPException(500, str(e))

        _quote_similarity.save_quote_search(_projects_root(), search)

        # F9.6 audit — record the search invocation as an AIEvent
        # request. Quote searches have no decision lifecycle so we only
        # log the request side; the persisted QuoteSearch is the
        # canonical record of the matches.
        try:
            _ai_invocation_log.record_request_event_for_quote_search(
                _projects_root(),
                search,
                backend=getattr(backend, "name", "") or "",
            )
        except Exception:  # nosec - audit logging is best-effort
            pass

    return JSONResponse({"search": search.to_dict()})


@app.get("/api/projects/{project_id}/ai/quote-searches")
async def list_quote_searches_endpoint(
    project_id: str,
    query_kind: str | None = None,
    query_source_id: str | None = None,
) -> JSONResponse:
    """F8.5 — list past quote searches for a project.

    Optional filters:
      query_kind       — "text" | "application"
      query_source_id  — restrict to searches seeded from a given source

    Returns ``{"searches": [<dict>, ...]}`` sorted by created_at asc.
    """
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            searches = _quote_similarity.list_quote_searches(
                _projects_root(), project_id,
                query_kind=query_kind,
                query_source_id=query_source_id,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({
        "searches": [s.to_dict() for s in searches],
    })


@app.get("/api/projects/{project_id}/ai/quote-searches/{search_id}")
async def get_quote_search_endpoint(
    project_id: str, search_id: str,
) -> JSONResponse:
    """F8.5 — fetch a single persisted quote search by id."""
    _check_project_id(project_id)
    if not _quote_similarity.QUOTE_SEARCH_ID_RE.match(search_id):
        raise HTTPException(400, "Invalid quote-search id")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            search = _quote_similarity.load_quote_search(
                _projects_root(), project_id, search_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Quote search not found")
    return JSONResponse({"search": search.to_dict()})


# --------------------------------------------------------------------------- #
# Whole-transcript AI review pass (F8.6)
#
# The engine module ``scribe.transcript_review`` shipped in 167b8c6 with a
# full ReviewPass / ReviewItem data model and a step-at-a-time processor,
# but had no FastAPI surface, no UI button, no integration test — so a
# user could not invoke it. This block wires the route layer the engine
# was designed for, mirroring the F8.3 / F8.5 split (engine pure, server
# is a thin shell).
#
# Endpoints:
#
#   POST /api/projects/<pid>/sources/<sid>/review
#         Start a fresh pass on this source's transcript and (by
#         default) drive it forward up to ``max_steps`` items so the
#         user sees suggestions on the very first request. Returns the
#         persisted ReviewPass dict including items + status.
#         Body (all optional):
#           {
#             "granularity": "paragraph" | "sentence",  # default paragraph
#             "skip_already_coded": true,
#             "top_k": 5, "min_score": 0.0,
#             "max_steps": 5,    # how many items to process this call;
#                                # null/0 means start-and-leave-pending so
#                                # the client can drive run_pass itself.
#             "notes": "optional"
#           }
#
#   POST /api/projects/<pid>/review-passes/<rpid>/run
#         Resume an existing pass. Same ``max_steps`` knob; the route
#         drives that many items and returns the (possibly still-
#         running) pass. Calling this on a terminal pass is a 409.
#
#   POST /api/projects/<pid>/review-passes/<rpid>/cancel
#         Move a non-terminal pass to ``cancelled`` so the user can
#         abandon a long sweep. 409 if already terminal.
#
#   GET  /api/projects/<pid>/review-passes
#         List all passes in the project. Optional ``source_id`` and
#         ``status`` filters. Sorted by created_at asc.
#
#   GET  /api/projects/<pid>/review-passes/<rpid>
#         Single-pass fetch. 404 if missing, 400 if id is malformed.
#
# All write routes are gated by F8.10 (412 + structured gate body) like
# the existing suggestion endpoints. Each pass start is also recorded as
# an AIEvent of feature=transcript_review so the F9.6 audit log carries
# the request invocation.
# --------------------------------------------------------------------------- #

from . import transcript_review as _transcript_review  # noqa: E402


# Default cap on items processed per HTTP request. Keeping it small
# means a request never blocks longer than (max_steps * per-item-time).
# The client polls ``/run`` to drive the rest.
_REVIEW_DEFAULT_MAX_STEPS = 5
_REVIEW_HARD_MAX_STEPS = 100


def _coerce_review_max_steps(raw: Any) -> int:
    """Return a sane integer max_steps from an arbitrary body value.

    Falls back to the module default. A negative or zero value is
    treated as "process zero items this call" so the client can drive
    the loop itself; ``None``/missing produces the default.
    """
    if raw is None:
        return _REVIEW_DEFAULT_MAX_STEPS
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, "max_steps must be an integer")
    if n < 0:
        raise HTTPException(400, "max_steps must be ≥ 0")
    if n > _REVIEW_HARD_MAX_STEPS:
        raise HTTPException(
            400,
            f"max_steps must be ≤ {_REVIEW_HARD_MAX_STEPS}",
        )
    return n


def _drive_review_pass(
    pass_record: "_transcript_review.ReviewPass",
    *,
    project_id: str,
    cfg: "_ai_backend.BackendConfig",
    backend: "_ai_backend.ModelBackend",
    max_steps: int,
) -> "_transcript_review.ReviewPass":
    """Drive a pass forward up to ``max_steps`` items.

    Translates backend errors into HTTPExceptions so the caller doesn't
    have to repeat the boilerplate. Per-item failures are recorded by
    the engine as ``ReviewItem.error`` strings and don't raise; pass-
    level failures (backend unavailable from the start, validation
    error) DO raise — we mark the pass failed and re-raise so the
    client sees the structured error.
    """
    if max_steps <= 0:
        return pass_record
    if pass_record.status in _transcript_review.REVIEW_TERMINAL_STATUSES:
        return pass_record
    try:
        embed_fn, generate_fn, _emb, _gen = (
            _make_embed_and_generate_fns(cfg, backend)
        )
    except _ai_backend.BackendValidationError as e:
        # Couldn't even build the closure — mark the pass failed for
        # the audit trail, then re-raise as 400.
        try:
            _transcript_review.mark_review_pass_failed(
                pass_record,
                projects_root=_projects_root(),
                error_message=str(e)[:_transcript_review.MAX_ERROR_MESSAGE_LEN],
            )
        except _projects.ProjectValidationError:
            pass
        raise HTTPException(400, str(e))

    codes = _codes.list_codes(_projects_root(), project_id)
    applications = _applications.list_applications(
        _projects_root(), project_id,
    )
    try:
        _transcript_review.run_review_pass(
            pass_record,
            projects_root=_projects_root(),
            codes=codes,
            applications=applications,
            embed_fn=embed_fn,
            generate_fn=generate_fn,
            max_steps=max_steps,
        )
    except _projects.ProjectValidationError as e:
        # Validation error mid-loop is a genuine pass-level failure.
        try:
            _transcript_review.mark_review_pass_failed(
                pass_record,
                projects_root=_projects_root(),
                error_message=str(e)[:_transcript_review.MAX_ERROR_MESSAGE_LEN],
            )
        except _projects.ProjectValidationError:
            pass
        raise HTTPException(400, str(e))
    except _ai_backend.BackendUnavailable as e:
        raise HTTPException(502, f"Backend unavailable: {e}")
    except _ai_backend.BackendError as e:
        raise HTTPException(500, str(e))
    return pass_record


@app.post("/api/projects/{project_id}/sources/{source_id}/review")
async def start_review_pass_endpoint(
    project_id: str, source_id: str, request: Request,
) -> JSONResponse:
    """F8.6 — start a whole-transcript AI review pass on this source.

    Enumerates review items via :func:`transcript_review.start_review_pass`
    (paragraph by default), persists the new ReviewPass, and drives it
    forward up to ``max_steps`` items so the first poll returns useful
    suggestions. The pass is gated by F8.10.
    """
    _check_project_id(project_id)
    _check_source_id(source_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    body = body or {}

    granularity = (
        body.get("granularity") or _transcript_review.REVIEW_GRANULARITY_PARAGRAPH
    )
    if granularity not in _transcript_review.REVIEW_GRANULARITIES:
        raise HTTPException(
            400,
            f"granularity must be one of "
            f"{list(_transcript_review.REVIEW_GRANULARITIES)}",
        )
    skip_already_coded = bool(body.get("skip_already_coded", True))
    top_k = body.get("top_k", _code_suggestions.DEFAULT_TOP_K)
    min_score = body.get("min_score", 0.0)
    notes = body.get("notes") or ""
    max_steps = _coerce_review_max_steps(body.get("max_steps"))
    try:
        top_k_i = int(top_k)
        min_score_f = float(min_score)
    except (TypeError, ValueError):
        raise HTTPException(400, "top_k must be int and min_score must be float")
    if top_k_i < 1 or top_k_i > 50:
        raise HTTPException(400, "top_k must be in [1, 50]")
    if min_score_f < -1.0 or min_score_f > 1.0:
        raise HTTPException(400, "min_score must be in [-1, 1]")

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            source = _sources.load_source(
                _projects_root(), project_id, source_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")

        # F8.10 gate.
        try:
            gate = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id,
                feature=_ai_provenance.AI_FEATURE_TRANSCRIPT_REVIEW,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not gate.allowed:
            raise HTTPException(412, {
                "detail": "AI gate not satisfied",
                "gate": gate.to_dict(),
            })

        # Backend.
        try:
            cfg, backend = _resolve_suggestion_backend(project)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))

        # The engine needs the source's transcript segments + the
        # project's existing applications (so skip_already_coded works).
        segments = _load_segments_for_source_speaker_map(source) or []
        applications = _applications.list_applications(
            _projects_root(), project_id,
        )

        # Persist a fresh pass record. enumerate_review_items runs
        # inside start_review_pass and freezes the items onto disk.
        try:
            pass_record = _transcript_review.start_review_pass(
                projects_root=_projects_root(),
                project_id=project_id,
                source_id=source_id,
                segments=segments,
                applications=applications,
                granularity=granularity,
                skip_already_coded=skip_already_coded,
                embedding_model=cfg.default_embedding_model,
                generation_model=cfg.default_model,
                top_k=top_k_i,
                min_score=min_score_f,
                notes=str(notes or "")[:_transcript_review.MAX_NOTES_LEN],
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))

        # F9.6 audit — log the request side. Best-effort; a logging
        # failure must not break the user's workflow.
        try:
            _ai_invocation_log.record_request_event_for_review_pass(
                _projects_root(),
                pass_record,
                backend=getattr(backend, "name", "") or "",
            )
        except Exception:
            pass

        # Drive forward so the first request returns suggestions.
        _drive_review_pass(
            pass_record,
            project_id=project_id,
            cfg=cfg,
            backend=backend,
            max_steps=max_steps,
        )

    return JSONResponse({"pass": pass_record.to_dict()})


@app.post("/api/projects/{project_id}/review-passes/{pass_id}/run")
async def run_review_pass_endpoint(
    project_id: str, pass_id: str, request: Request,
) -> JSONResponse:
    """F8.6 — drive an existing pass forward by ``max_steps`` items."""
    _check_project_id(project_id)
    if not _transcript_review.REVIEW_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid review-pass id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    body = body or {}
    max_steps = _coerce_review_max_steps(body.get("max_steps"))

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            pass_record = _transcript_review.load_review_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Review pass not found")
        if pass_record.status in _transcript_review.REVIEW_TERMINAL_STATUSES:
            raise HTTPException(
                409,
                f"Pass already in terminal state: {pass_record.status}",
            )

        # F8.10 gate (revalidate on every request — the gate config
        # may have changed between start and resume).
        try:
            gate = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id,
                feature=_ai_provenance.AI_FEATURE_TRANSCRIPT_REVIEW,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not gate.allowed:
            raise HTTPException(412, {
                "detail": "AI gate not satisfied",
                "gate": gate.to_dict(),
            })

        try:
            cfg, backend = _resolve_suggestion_backend(project)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))

        _drive_review_pass(
            pass_record,
            project_id=project_id,
            cfg=cfg,
            backend=backend,
            max_steps=max_steps,
        )

    return JSONResponse({"pass": pass_record.to_dict()})


@app.post("/api/projects/{project_id}/review-passes/{pass_id}/cancel")
async def cancel_review_pass_endpoint(
    project_id: str, pass_id: str,
) -> JSONResponse:
    """F8.6 — abandon a non-terminal pass."""
    _check_project_id(project_id)
    if not _transcript_review.REVIEW_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid review-pass id")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            pass_record = _transcript_review.load_review_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Review pass not found")
        if pass_record.status == _transcript_review.REVIEW_STATUS_CANCELLED:
            return JSONResponse({"pass": pass_record.to_dict()})
        if pass_record.status in _transcript_review.REVIEW_TERMINAL_STATUSES:
            raise HTTPException(
                409,
                f"Pass already in terminal state: {pass_record.status}",
            )
        try:
            _transcript_review.cancel_review_pass(
                pass_record, projects_root=_projects_root(),
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"pass": pass_record.to_dict()})


@app.get("/api/projects/{project_id}/review-passes")
async def list_review_passes_endpoint(
    project_id: str,
    source_id: str = "",
    status: str = "",
) -> JSONResponse:
    """F8.6 — list past review passes for a project."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            passes = _transcript_review.list_review_passes(
                _projects_root(), project_id,
                source_id=source_id or None,
                status=status or None,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"passes": [p.to_dict() for p in passes]})


@app.get("/api/projects/{project_id}/review-passes/{pass_id}")
async def get_review_pass_endpoint(
    project_id: str, pass_id: str,
) -> JSONResponse:
    """F8.6 — fetch a single persisted review pass by id."""
    _check_project_id(project_id)
    if not _transcript_review.REVIEW_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid review-pass id")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            pass_record = _transcript_review.load_review_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Review pass not found")
    return JSONResponse({"pass": pass_record.to_dict()})


# --------------------------------------------------------------------------- #
# F8.7 — AI second-coder pass on a locked codebook (HTTP surface)
#
# The engine module ``scribe.ai_second_coder`` shipped in a3be790 with
# the full SecondCoderPass / SecondCoderDiff / SecondCoderICR data
# model + step-at-a-time processor + Cohen's-kappa computation, but
# had no FastAPI surface, no UI, and no integration test — so a user
# could not invoke it. This block wires the route layer the engine
# was designed for, mirroring the F8.6 split (engine pure, server is
# a thin shell).
#
# Endpoints:
#
#   POST /api/projects/<pid>/sources/<sid>/second-coder
#         Start a fresh second-coder pass on this source's transcript
#         and (by default) drive it forward up to ``max_steps`` items
#         so the user sees diff progress on the very first request.
#         Refuses on an unlocked codebook (409 + structured error).
#         Body (all optional except human_coder_id):
#           {
#             "human_coder_id": "<cid>",   # required
#             "granularity": "paragraph" | "sentence",
#             "top_n": 1,
#             "min_score": 0.0,
#             "embedding_model": "...",
#             "generation_model": "...",
#             "max_steps": 5,
#             "notes": "optional"
#           }
#
#   POST /api/projects/<pid>/second-coder-passes/<pid>/run
#         Resume an existing pass. Drives ``max_steps`` items.
#         Calling on a terminal pass returns 409.
#
#   POST /api/projects/<pid>/second-coder-passes/<pid>/cancel
#         Move a non-terminal pass to ``cancelled`` (also cancels
#         the inner review pass).
#
#   GET  /api/projects/<pid>/second-coder-passes
#         List passes in the project. Optional ``source_id`` /
#         ``human_coder_id`` / ``status`` filters.
#
#   GET  /api/projects/<pid>/second-coder-passes/<pid>
#         Single-pass fetch.
#
#   GET  /api/projects/<pid>/second-coder-passes/<pid>/diff
#         Live diff: walks the inner review pass + applications and
#         returns a SecondCoderDiff. Useful for the per-pass detail
#         view that highlights AI-only / human-only / agreement codes
#         per item.
#
# All write routes are gated by F8.10 (412 + structured gate body)
# like the existing suggestion endpoints. Each pass start is also
# recorded as an AIEvent of feature=second_coder so the F9.6 audit
# log carries the request invocation. Lock-required errors become
# 409 (state conflict) so the UI can render a helpful "lock the
# codebook first" notice rather than a bare 400.
# --------------------------------------------------------------------------- #

from . import ai_second_coder as _ai_second_coder  # noqa: E402


# Default cap on items processed per HTTP request. Same shape as the
# F8.6 review-pass cap. The client polls ``/run`` to drive the rest.
_SECOND_CODER_DEFAULT_MAX_STEPS = 5
_SECOND_CODER_HARD_MAX_STEPS = 100


def _coerce_second_coder_max_steps(raw: Any) -> int:
    """Same shape as ``_coerce_review_max_steps`` for the F8.7 routes."""
    if raw is None:
        return _SECOND_CODER_DEFAULT_MAX_STEPS
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, "max_steps must be an integer")
    if n < 0:
        raise HTTPException(400, "max_steps must be ≥ 0")
    if n > _SECOND_CODER_HARD_MAX_STEPS:
        raise HTTPException(
            400,
            f"max_steps must be ≤ {_SECOND_CODER_HARD_MAX_STEPS}",
        )
    return n


def _drive_second_coder_pass(
    pass_record: "_ai_second_coder.SecondCoderPass",
    *,
    project_id: str,
    cfg: "_ai_backend.BackendConfig",
    backend: "_ai_backend.ModelBackend",
    max_steps: int,
) -> "_ai_second_coder.SecondCoderPass":
    """Drive a second-coder pass up to ``max_steps`` items.

    Translates backend errors into HTTPExceptions. Per-item errors are
    recorded by the inner review pass; pass-level failures (backend
    refused entirely, validation error mid-loop) DO mark the pass
    failed and re-raise so the client sees a structured error.
    """
    if max_steps <= 0:
        return pass_record
    if pass_record.status in _ai_second_coder.SECOND_CODER_TERMINAL_STATUSES:
        return pass_record
    # Resolve the inner review pass first; if it's gone, the
    # second-coder pass can't make progress.
    try:
        review_pass = _transcript_review.load_review_pass(
            _projects_root(), project_id, pass_record.review_pass_id,
        )
    except FileNotFoundError:
        raise HTTPException(
            500,
            "Inner review pass for second-coder pass is missing",
        )
    try:
        embed_fn, generate_fn, _emb, _gen = (
            _make_embed_and_generate_fns(cfg, backend)
        )
    except _ai_backend.BackendValidationError as e:
        try:
            _ai_second_coder.mark_second_coder_pass_failed(
                pass_record,
                projects_root=_projects_root(),
                review_pass=review_pass,
                error_message=str(e)[:_ai_second_coder.MAX_ERROR_MESSAGE_LEN],
            )
        except _projects.ProjectValidationError:
            pass
        raise HTTPException(400, str(e))

    codes = _codes.list_codes(_projects_root(), project_id)
    applications = _applications.list_applications(
        _projects_root(), project_id,
    )
    try:
        _ai_second_coder.run_second_coder_pass(
            pass_record,
            projects_root=_projects_root(),
            review_pass=review_pass,
            codes=codes,
            applications=applications,
            embed_fn=embed_fn,
            generate_fn=generate_fn,
            max_steps=max_steps,
        )
    except _projects.ProjectValidationError as e:
        try:
            _ai_second_coder.mark_second_coder_pass_failed(
                pass_record,
                projects_root=_projects_root(),
                review_pass=review_pass,
                error_message=str(e)[:_ai_second_coder.MAX_ERROR_MESSAGE_LEN],
            )
        except _projects.ProjectValidationError:
            pass
        raise HTTPException(400, str(e))
    except _ai_backend.BackendUnavailable as e:
        raise HTTPException(502, f"Backend unavailable: {e}")
    except _ai_backend.BackendError as e:
        raise HTTPException(500, str(e))
    return pass_record


@app.post("/api/projects/{project_id}/sources/{source_id}/second-coder")
async def start_second_coder_pass_endpoint(
    project_id: str, source_id: str, request: Request,
) -> JSONResponse:
    """F8.7 — start an AI second-coder pass against a designated human coder."""
    _check_project_id(project_id)
    _check_source_id(source_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    body = body or {}

    human_coder_id = str(body.get("human_coder_id", "") or "").strip()
    if not human_coder_id:
        raise HTTPException(400, "human_coder_id is required")
    if not _coders.CODER_ID_RE.match(human_coder_id):
        raise HTTPException(400, "Invalid human_coder_id")

    granularity = (
        body.get("granularity") or _ai_second_coder.REVIEW_GRANULARITY_PARAGRAPH
    )
    if granularity not in _ai_second_coder.REVIEW_GRANULARITIES:
        raise HTTPException(
            400,
            f"granularity must be one of "
            f"{list(_ai_second_coder.REVIEW_GRANULARITIES)}",
        )

    raw_top_n = body.get("top_n", _ai_second_coder.DEFAULT_TOP_N)
    raw_min_score = body.get("min_score", _ai_second_coder.DEFAULT_MIN_SCORE)
    try:
        top_n = int(raw_top_n)
        min_score = float(raw_min_score)
    except (TypeError, ValueError):
        raise HTTPException(
            400, "top_n must be int and min_score must be numeric",
        )
    if top_n < 1 or top_n > 50:
        raise HTTPException(400, "top_n must be in [1, 50]")
    if min_score < -1.0 or min_score > 1.0:
        raise HTTPException(400, "min_score must be in [-1, 1]")
    notes = str(body.get("notes") or "")[:_ai_second_coder.MAX_NOTES_LEN]
    max_steps = _coerce_second_coder_max_steps(body.get("max_steps"))

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            source = _sources.load_source(
                _projects_root(), project_id, source_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Source not found")
        # Verify the human coder belongs to this project (404 mirrors
        # the rest of the API rather than a 400).
        try:
            _coders.load_coder(_projects_root(), project_id, human_coder_id)
        except FileNotFoundError:
            raise HTTPException(404, "Human coder not found")

        # F8.10 gate.
        try:
            gate = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id,
                feature=_ai_provenance.AI_FEATURE_SECOND_CODER,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not gate.allowed:
            raise HTTPException(412, {
                "detail": "AI gate not satisfied",
                "gate": gate.to_dict(),
            })

        # Backend.
        try:
            cfg, backend = _resolve_suggestion_backend(project)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))

        segments = _load_segments_for_source_speaker_map(source) or []
        applications = _applications.list_applications(
            _projects_root(), project_id,
        )

        # Persist a fresh pass record. start_second_coder_pass enforces
        # the lock guard internally and creates the inner review pass.
        try:
            pass_record = _ai_second_coder.start_second_coder_pass(
                projects_root=_projects_root(),
                project_id=project_id,
                source_id=source_id,
                human_coder_id=human_coder_id,
                segments=segments,
                applications=applications,
                granularity=granularity,
                embedding_model=cfg.default_embedding_model,
                generation_model=cfg.default_model,
                top_n=top_n,
                min_score=min_score,
                notes=notes,
            )
        except _ai_second_coder.CodebookNotLockedError as e:
            # Methodologically required: refuse to score against an
            # evolving codebook. Surface the message + a structured
            # marker the UI can branch on without parsing English.
            raise HTTPException(409, {
                "detail": str(e),
                "reason": "codebook_not_locked",
            })
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))

        # F9.6 audit — log the request side. Best-effort.
        try:
            _ai_invocation_log.record_request_event_for_second_coder_pass(
                _projects_root(),
                pass_record,
                backend=getattr(backend, "name", "") or "",
            )
        except Exception:
            pass

        # Drive forward so the first response carries useful state.
        _drive_second_coder_pass(
            pass_record,
            project_id=project_id,
            cfg=cfg,
            backend=backend,
            max_steps=max_steps,
        )

    return JSONResponse({"pass": pass_record.to_dict()})


@app.post("/api/projects/{project_id}/second-coder-passes/{pass_id}/run")
async def run_second_coder_pass_endpoint(
    project_id: str, pass_id: str, request: Request,
) -> JSONResponse:
    """F8.7 — drive an existing second-coder pass forward."""
    _check_project_id(project_id)
    if not _ai_second_coder.SECOND_CODER_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid second-coder pass id")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body and not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    body = body or {}
    max_steps = _coerce_second_coder_max_steps(body.get("max_steps"))

    with PROJECTS_LOCK:
        try:
            project = _projects.load_project(_projects_root(), project_id)
        except FileNotFoundError:
            raise HTTPException(404, "Project not found")
        try:
            pass_record = _ai_second_coder.load_second_coder_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Second-coder pass not found")
        if pass_record.status in _ai_second_coder.SECOND_CODER_TERMINAL_STATUSES:
            raise HTTPException(
                409,
                f"Pass already in terminal state: {pass_record.status}",
            )

        # F8.10 gate revalidation.
        try:
            gate = _ai_gate.evaluate_project_ai_gate(
                _projects_root(), project_id,
                feature=_ai_provenance.AI_FEATURE_SECOND_CODER,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        if not gate.allowed:
            raise HTTPException(412, {
                "detail": "AI gate not satisfied",
                "gate": gate.to_dict(),
            })

        try:
            cfg, backend = _resolve_suggestion_backend(project)
        except _ai_backend.BackendValidationError as e:
            raise HTTPException(400, str(e))

        _drive_second_coder_pass(
            pass_record,
            project_id=project_id,
            cfg=cfg,
            backend=backend,
            max_steps=max_steps,
        )

    return JSONResponse({"pass": pass_record.to_dict()})


@app.post("/api/projects/{project_id}/second-coder-passes/{pass_id}/cancel")
async def cancel_second_coder_pass_endpoint(
    project_id: str, pass_id: str,
) -> JSONResponse:
    """F8.7 — abandon a non-terminal second-coder pass."""
    _check_project_id(project_id)
    if not _ai_second_coder.SECOND_CODER_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid second-coder pass id")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            pass_record = _ai_second_coder.load_second_coder_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Second-coder pass not found")
        if pass_record.status == _ai_second_coder.SECOND_CODER_STATUS_CANCELLED:
            return JSONResponse({"pass": pass_record.to_dict()})
        if pass_record.status in _ai_second_coder.SECOND_CODER_TERMINAL_STATUSES:
            raise HTTPException(
                409,
                f"Pass already in terminal state: {pass_record.status}",
            )
        # Best-effort: cancel the inner review pass too if it loads.
        review_pass = None
        try:
            review_pass = _transcript_review.load_review_pass(
                _projects_root(), project_id, pass_record.review_pass_id,
            )
        except FileNotFoundError:
            review_pass = None
        try:
            _ai_second_coder.cancel_second_coder_pass(
                pass_record,
                projects_root=_projects_root(),
                review_pass=review_pass,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"pass": pass_record.to_dict()})


@app.get("/api/projects/{project_id}/second-coder-passes")
async def list_second_coder_passes_endpoint(
    project_id: str,
    source_id: str = "",
    human_coder_id: str = "",
    status: str = "",
) -> JSONResponse:
    """F8.7 — list all second-coder passes for a project."""
    _check_project_id(project_id)
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            passes = _ai_second_coder.list_second_coder_passes(
                _projects_root(), project_id,
                source_id=source_id or None,
                human_coder_id=human_coder_id or None,
                status=status or None,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
    return JSONResponse({"passes": [p.to_dict() for p in passes]})


@app.get("/api/projects/{project_id}/second-coder-passes/{pass_id}")
async def get_second_coder_pass_endpoint(
    project_id: str, pass_id: str,
) -> JSONResponse:
    """F8.7 — fetch a single persisted second-coder pass."""
    _check_project_id(project_id)
    if not _ai_second_coder.SECOND_CODER_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid second-coder pass id")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            pass_record = _ai_second_coder.load_second_coder_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Second-coder pass not found")
    return JSONResponse({"pass": pass_record.to_dict()})


@app.get("/api/projects/{project_id}/second-coder-passes/{pass_id}/diff")
async def get_second_coder_pass_diff_endpoint(
    project_id: str, pass_id: str,
) -> JSONResponse:
    """F8.7 — compute the live AI vs human diff for a pass.

    Returns the SecondCoderDiff (per-item AI / human / agreement /
    AI-only / human-only) AND the live SecondCoderICR. Diff is
    re-computed on every call (cheap; reads suggestion records +
    applications) so a researcher who edits applications mid-pass
    sees the kappa shift immediately.
    """
    _check_project_id(project_id)
    if not _ai_second_coder.SECOND_CODER_PASS_ID_RE.match(pass_id):
        raise HTTPException(400, "Invalid second-coder pass id")
    with PROJECTS_LOCK:
        _project_must_exist(project_id)
        try:
            pass_record = _ai_second_coder.load_second_coder_pass(
                _projects_root(), project_id, pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(404, "Second-coder pass not found")
        try:
            review_pass = _transcript_review.load_review_pass(
                _projects_root(), project_id, pass_record.review_pass_id,
            )
        except FileNotFoundError:
            raise HTTPException(
                500, "Inner review pass missing for this second-coder pass",
            )
        applications = _applications.list_applications(
            _projects_root(), project_id,
        )
        try:
            diff = _ai_second_coder.compute_second_coder_diff(
                projects_root=_projects_root(),
                pass_record=pass_record,
                review_pass=review_pass,
                applications=applications,
            )
        except _projects.ProjectValidationError as e:
            raise HTTPException(400, str(e))
        try:
            icr = _ai_second_coder.compute_second_coder_icr(diff)
        except _projects.ProjectValidationError as e:
            raise HTTPException(500, f"ICR computation failed: {e}")
    return JSONResponse({
        "pass": pass_record.to_dict(),
        "diff": diff.to_dict(),
        "icr": icr.to_dict(),
    })


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


@app.get("/api/job/{job_id}/projects")
async def job_projects(job_id: str) -> JSONResponse:
    """Return every project / source pair that links to this job, so the
    editor can offer "Open in coding view" links. A transcription can be
    attached to multiple projects; we return all matches sorted by
    project modified-time so the most recent is first."""
    _check_job_id(job_id)
    matches: list[dict[str, Any]] = []
    with PROJECTS_LOCK:
        for project in _projects.list_projects(_projects_root()):
            for source in _sources.list_sources(_projects_root(), project.id):
                if source.transcript_job_id == job_id:
                    matches.append({
                        "project_id": project.id,
                        "project_name": project.name,
                        "project_modified_at": project.modified_at,
                        "source_id": source.id,
                        "source_name": source.name,
                    })
    matches.sort(key=lambda m: m.get("project_modified_at") or "", reverse=True)
    return JSONResponse({"projects": matches})


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
        # Snapshot the existing transcript *before* we overwrite it.
        # F4.5 (re-anchor on transcript edit) needs a copy of the
        # pre-edit segments to diff against the new payload. Prefer
        # the in-memory ``job.result`` over disk because it's cheaper
        # and matches the editor's last view; fall back to disk when
        # the server has restarted between edits.
        previous_payload: dict[str, Any] | None = None
        if isinstance(job.result, dict):
            previous_payload = job.result

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

    # F4.5 — capture the *old* segments before we overwrite the file,
    # so we can re-anchor any code applications that point at this
    # transcript. The fall-back to disk runs only when JOBS doesn't
    # carry a ``result`` (cold start after restart, unfinished job).
    if previous_payload is None or not isinstance(
        previous_payload.get("segments"), list,
    ):
        previous_payload = {
            "segments": _read_transcript_segments(out_dir),
        }
    old_segments = previous_payload.get("segments")
    if not isinstance(old_segments, list):
        old_segments = []

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

    # F4.5 — fan out across every project whose Source links to this
    # job. Errors per project are caught in the helper so a transcript
    # save never fails because of an orphan-queue write.
    new_segments = payload.get("segments")
    if not isinstance(new_segments, list):
        new_segments = []
    with PROJECTS_LOCK:
        reanchor_summaries = _reanchor_apps_for_job(
            job_id, old_segments, new_segments,
        )

    return JSONResponse({
        "ok": True,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "reanchor": reanchor_summaries,
    })


# --------------------------------------------------------------------------- #
# Transcript tidy-up (grammar bot)
#
# Three endpoints that wrap :mod:`scribe.transcript_tidy` for the
# editor's "✨ Tidy speech with AI" feature:
#
#   GET  /api/job/{id}/tidy/runs                  — list candidate runs
#   POST /api/job/{id}/tidy/preview               — call LLM, return proposal
#   POST /api/job/{id}/tidy/apply                 — splice accepted text
#
# The grammar bot uses the *global* AI backend config (see
# :mod:`scribe.global_ai_backend`) rather than a per-project config,
# because the editor isn't bound to a project. The F8.10 AI gate also
# does not apply here — that gate is about hand-coding before AI
# coding, irrelevant to transcript cleanup.
# --------------------------------------------------------------------------- #

from . import transcript_tidy as _transcript_tidy  # noqa: E402
from . import global_ai_backend as _global_ai_backend  # noqa: E402


# Test injection point — when set, takes precedence over the global
# config + real backend. Production code never sets this.
_tidy_backend_override: Any = None


def _resolve_tidy_backend() -> tuple["_ai_backend.BackendConfig", Any]:
    """Pick the AI backend the tidy endpoints will talk to."""
    if _tidy_backend_override is not None:
        return _tidy_backend_override
    cfg = _global_ai_backend.load_global_config()
    backend = _ai_backend.backend_for_config(cfg)
    return cfg, backend


def _persist_edited_transcript(job: "Job", payload: dict[str, Any]) -> None:
    """Save the edited transcript to disk and regenerate sidecars.

    Same write path as :func:`put_transcript`; factored out so the
    tidy-apply endpoint can reuse it without duplicating the sidecar
    + ``_persist_job`` dance.
    """
    out_dir = job.output_dir.resolve()
    input_stem = job.input_path.stem
    if not _is_under(out_dir, OUTPUT_DIR):
        raise HTTPException(403, "Forbidden")
    edited = _edited_path(out_dir)
    edited.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
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
        raise HTTPException(500, f"Saved JSON but failed to regenerate sidecars: {e}")


def _load_job_transcript(job: "Job") -> dict[str, Any]:
    """Return the latest transcript payload for ``job`` (edited
    overrides the original result if it exists)."""
    edited = _edited_path(job.output_dir)
    if edited.exists():
        try:
            return json.loads(edited.read_text())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Could not read edited transcript: {e}")
    if job.result:
        return job.result
    raise HTTPException(404, "No transcript available")


@app.get("/api/job/{job_id}/tidy/runs")
async def list_tidy_runs_endpoint(job_id: str) -> JSONResponse:
    """List candidate runs the grammar bot can act on.

    Each run is a maximal block of consecutive same-speaker segments
    that's long enough to merit cleanup (see ``MIN_RUN_SEGMENTS``)
    and short enough not to blow the model's context window.
    """
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
    payload = _load_job_transcript(job)
    runs = _transcript_tidy.group_runs(payload.get("segments") or [])
    return JSONResponse({"runs": [r.to_dict() for r in runs]})


@app.post("/api/job/{job_id}/tidy/preview")
async def preview_tidy_run_endpoint(job_id: str, request: Request) -> JSONResponse:
    """Run the LLM on one run; return proposed paragraphs + segments.

    Body:
        {"segment_indices": [int, ...]}

    The indices must match a run :func:`group_runs` returned (we
    don't trust an arbitrary range — guarantees the speaker is
    consistent and the wall-clock window is sane).

    Returns:
        {
          "raw_text": "...",                    # the run as one string
          "paragraphs": ["para1", "para2", ...],
          "segments": [TidiedSegment.to_dict, ...],
          "model": "<model name>",
        }
    """
    _check_job_id(job_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    raw_indices = body.get("segment_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise HTTPException(400, "segment_indices must be a non-empty list")
    try:
        indices = tuple(int(i) for i in raw_indices)
    except (TypeError, ValueError):
        raise HTTPException(400, "segment_indices entries must be integers")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
    payload = _load_job_transcript(job)
    segments = payload.get("segments") or []
    runs = _transcript_tidy.group_runs(segments)
    matching = next((r for r in runs if r.segment_indices == indices), None)
    if matching is None:
        raise HTTPException(
            400,
            "segment_indices does not match any candidate run; "
            "fetch /tidy/runs first.",
        )

    # Resolve the backend.
    try:
        cfg, backend = _resolve_tidy_backend()
    except _ai_backend.BackendValidationError as e:
        raise HTTPException(400, str(e))
    if not cfg.default_model:
        raise HTTPException(
            400,
            "Global AI backend has no default_model configured. "
            "Set one in ~/.scribe/ai_backend.json.",
        )

    transport = _ai_backend_transport_override or _ai_backend.urllib_transport
    prompt = _transcript_tidy.build_tidy_prompt(matching.text)
    req = _ai_backend.GenerationRequest(model=cfg.default_model, prompt=prompt)
    try:
        resp = backend.generate(cfg, req, transport=transport)
    except _ai_backend.BackendUnavailable as e:
        raise HTTPException(502, f"AI backend unavailable: {e}")
    except _ai_backend.BackendError as e:
        raise HTTPException(500, str(e))

    paragraphs = _transcript_tidy.parse_tidied_paragraphs(resp.text)
    if not paragraphs:
        raise HTTPException(
            502,
            "AI backend returned no usable paragraphs. "
            "Try again or pick a different run.",
        )

    old_words = _transcript_tidy._flatten_old_words(matching, segments)
    para_words = _transcript_tidy.realign_words(
        old_words, paragraphs,
        fallback_start=matching.start,
        fallback_end=matching.end,
    )
    new_segs = _transcript_tidy.assemble_tidied_segments(
        paragraphs=paragraphs,
        paragraph_words=para_words,
        speaker=matching.speaker,
        fallback_start=matching.start,
        fallback_end=matching.end,
    )
    return JSONResponse({
        "raw_text": matching.text,
        "paragraphs": paragraphs,
        "segments": [s.to_dict() for s in new_segs],
        "model": resp.model,
        "speaker": matching.speaker,
        "segment_indices": list(matching.segment_indices),
    })


@app.post("/api/job/{job_id}/tidy/apply")
async def apply_tidy_run_endpoint(job_id: str, request: Request) -> JSONResponse:
    """Splice the user-approved paragraphs into the transcript.

    Body:
        {
          "segment_indices": [int, ...],           # the run that was tidied
          "paragraphs": ["para1", ...],            # potentially edited by the user
          "speaker": "SPEAKER_00"                  # for the new segments
        }

    The server re-runs realignment on the (possibly edited) paragraphs
    so word timestamps reflect what the user actually accepted, then
    persists via the same path as ``PUT /transcript``.
    """
    _check_job_id(job_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    raw_indices = body.get("segment_indices")
    paragraphs_raw = body.get("paragraphs")
    speaker = body.get("speaker")
    if (
        not isinstance(raw_indices, list)
        or not raw_indices
        or not isinstance(paragraphs_raw, list)
        or not paragraphs_raw
        or not isinstance(speaker, str)
        or not speaker.strip()
    ):
        raise HTTPException(
            400,
            "segment_indices (list[int]), paragraphs (list[str]), and "
            "speaker (str) are required.",
        )
    try:
        indices = tuple(int(i) for i in raw_indices)
    except (TypeError, ValueError):
        raise HTTPException(400, "segment_indices entries must be integers")
    paragraphs = [str(p).strip() for p in paragraphs_raw if str(p).strip()]
    if not paragraphs:
        raise HTTPException(400, "paragraphs must contain at least one non-empty entry")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
    payload = _load_job_transcript(job)
    segments = payload.get("segments") or []
    runs = _transcript_tidy.group_runs(segments)
    matching = next((r for r in runs if r.segment_indices == indices), None)
    if matching is None:
        raise HTTPException(
            400,
            "segment_indices does not match any candidate run; "
            "the transcript may have changed since preview.",
        )

    old_words = _transcript_tidy._flatten_old_words(matching, segments)
    para_words = _transcript_tidy.realign_words(
        old_words, paragraphs,
        fallback_start=matching.start,
        fallback_end=matching.end,
    )
    new_segs = _transcript_tidy.assemble_tidied_segments(
        paragraphs=paragraphs,
        paragraph_words=para_words,
        speaker=speaker.strip(),
        fallback_start=matching.start,
        fallback_end=matching.end,
    )
    if not new_segs:
        raise HTTPException(400, "No usable segments after assembling tidy output.")

    new_payload = _transcript_tidy.splice_run(
        payload,
        segment_indices=matching.segment_indices,
        new_segments=[s.to_dict() for s in new_segs],
    )
    _persist_edited_transcript(job, new_payload)
    return JSONResponse({
        "ok": True,
        "applied_indices": list(matching.segment_indices),
        "new_segment_count": len(new_segs),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    })


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
    raw_names = payload.get("speaker_names")
    if isinstance(raw_names, dict):
        speaker_names = {str(k): str(v) for k, v in raw_names.items()
                         if isinstance(v, str) and v.strip()}
    else:
        speaker_names = {}
    return TranscriptionResult(
        segments=segs,
        language=payload.get("language", "en"),
        mode=payload.get("mode", "diarize"),
        speaker_labels=payload.get("speakers", speakers_seen),
        audio_path=input_path,
        speaker_names=speaker_names,
    )
