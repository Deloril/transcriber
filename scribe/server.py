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
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .audio import probe_audio_streams
from .engine import Segment, TranscriptionResult, Word, transcribe
from .writers import write_all, write_json, write_srt, write_txt, write_vtt

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
    input_filename: str = ""

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
    return templates.TemplateResponse(
        request,
        "index.html",
        {"hf_token_set": bool(os.environ.get("HF_TOKEN"))},
    )


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
        input_filename=safe_name,
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
        shutil.rmtree(job.output_dir / "work", ignore_errors=True)

        with JOBS_LOCK:
            job.status = "done"
            job.progress = 1.0
            job.message = "Done"
            job.result = result.to_dict()
            job.output_paths = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
        _persist_job(job)
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.message = "Error"
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
        "input_filename": job.input_filename,
    }


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
# Editor — media + transcript get/save
# --------------------------------------------------------------------------- #


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _check_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job id")


@app.get("/api/job/{job_id}/media")
async def media(job_id: str, request: Request) -> Response:
    """Serve the original recording with HTTP Range support so <video>/<audio> can seek."""
    _check_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
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
