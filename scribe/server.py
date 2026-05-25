"""FastAPI server — drag-and-drop UI for offline transcription."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import markdown as md
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .audio import probe_audio_streams
from .engine import transcribe
from .writers import write_all

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


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


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
EVENT_LOOP: asyncio.AbstractEventLoop | None = None


app = FastAPI(title="Scribe")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _capture_loop() -> None:
    global EVENT_LOOP
    EVENT_LOOP = asyncio.get_running_loop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"hf_token_set": bool(os.environ.get("HF_TOKEN"))},
    )


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
    """Bare HTML fragment, for in-page injection."""
    return HTMLResponse(_render_readme())


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    speakers: str = Form(""),
    num_speakers: str = Form(""),
    language: str = Form("en"),
    model: str = Form("large-v3"),
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
        streams = probe_audio_streams(input_path)
    except Exception as e:
        raise HTTPException(400, f"Could not read audio streams: {e}")

    if not streams:
        raise HTTPException(400, "No audio streams found in this file.")

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
    )
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return JSONResponse(
        {
            "job_id": job_id,
            "audio_streams": len(streams),
            "stream_titles": [s.title or f"track {i+1}" for i, s in enumerate(streams)],
        }
    )


def _set_progress(job_id: str, msg: str, frac: float) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.message = msg
        job.progress = max(0.0, min(1.0, frac))
        if job.status == "queued":
            job.status = "running"


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
    try:
        result = transcribe(
            job.input_path,
            work_dir=job.output_dir / "work",
            mode=job.mode,  # type: ignore[arg-type]
            speaker_labels=job.speakers,
            num_speakers=job.num_speakers,
            model_name=job.model,
            language=job.language,
            batch_size=8,
            hf_token=os.environ.get("HF_TOKEN"),
            progress=lambda m, f: _set_progress(job_id, m, f),
        )
        base = job.output_dir / job.input_path.stem
        paths = write_all(result, base)
        # Clean working WAVs but keep outputs
        shutil.rmtree(job.output_dir / "work", ignore_errors=True)

        with JOBS_LOCK:
            job.status = "done"
            job.progress = 1.0
            job.message = "Done"
            job.result = result.to_dict()
            job.output_paths = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.message = "Error"


@app.get("/api/job/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return JSONResponse(_job_dict(job))


def _job_dict(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "mode": job.mode,
        "audio_streams": job.audio_streams,
        "created_at": job.created_at,
        "error": job.error,
        "output_paths": job.output_paths,
        "result": job.result,
    }


@app.get("/api/job/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent events for live progress."""

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
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if kind not in job.output_paths:
            raise HTTPException(404, f"Format '{kind}' not available")
        rel = job.output_paths[kind]
    full = ROOT / rel
    return FileResponse(full, filename=full.name)
