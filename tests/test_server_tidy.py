"""Tests for the transcript tidy-up HTTP surface.

The pure realignment logic lives in
:mod:`scribe.transcript_tidy` and has its own unit tests; here we
exercise only the FastAPI wrapping:

* the three routes are reachable and return sane JSON,
* the LLM call goes through ``_tidy_backend_override`` (no real Ollama),
* preview returns paragraphs + segments + monotone word timestamps,
* apply splices the run and persists the edited transcript so a
  subsequent ``GET /transcript`` reflects the change,
* validation rejects bad payloads and stale segment_indices.

We intentionally keep the LLM responses small and predictable so the
tests don't drift with prompt tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv
from scribe.ai_backend import (
    BackendConfig,
    EmbeddingResponse,
    GenerationResponse,
    PROVIDER_OLLAMA,
)


# --------------------------------------------------------------------------- #
# Fixture + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated tmp dirs, empty JOBS, no leaked overrides."""
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    projects = tmp_path / "projects"
    upload.mkdir()
    output.mkdir()
    projects.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects)
    monkeypatch.setattr(srv, "JOBS", {})
    monkeypatch.setattr(srv, "_tidy_backend_override", None)
    monkeypatch.setattr(srv, "_ai_backend_transport_override", None)
    return TestClient(srv.app), tmp_path


class FakeBackend:
    """Stub backend matching :class:`scribe.ai_backend.ModelBackend`'s
    call shape. Returns a fixed string so tests can pin paragraphs."""

    name = PROVIDER_OLLAMA

    def __init__(self, generation_text: str = "") -> None:
        self.generation_text = generation_text
        self.generate_calls: list = []
        self.embed_calls: list = []

    def generate(self, cfg, req, *, transport=None):
        self.generate_calls.append((cfg, req))
        return GenerationResponse(
            text=self.generation_text, model=req.model, provider=self.name,
        )

    def embed(self, cfg, req, *, transport=None):
        self.embed_calls.append((cfg, req))
        return EmbeddingResponse(
            vectors=tuple(((1.0,),) for _ in req.inputs),
            model=req.model, provider=self.name,
        )


def _install_fake(text: str = "Cleaned up text.") -> FakeBackend:
    backend = FakeBackend(text)
    srv._tidy_backend_override = (
        BackendConfig(
            provider=PROVIDER_OLLAMA,
            base_url="http://test",
            default_model="llama3.2:3b",
        ),
        backend,
    )
    return backend


def _seed_job(env, *, segments: list[dict] | None = None) -> str:
    """Insert a Job into srv.JOBS. Returns the job id."""
    _, tmp_path = env
    job_id = "abc123def456"
    out_dir = srv.OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = srv.UPLOAD_DIR / job_id / "in.wav"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"\x00" * 64)
    if segments is None:
        # Three consecutive A segments + a B singleton.
        segments = [
            {
                "speaker": "A", "start": 0.0, "end": 1.0, "text": "i think",
                "words": [
                    {"text": "i", "start": 0.0, "end": 0.4, "speaker": "A"},
                    {"text": "think", "start": 0.4, "end": 1.0, "speaker": "A"},
                ],
            },
            {
                "speaker": "A", "start": 1.0, "end": 2.0, "text": "this is",
                "words": [
                    {"text": "this", "start": 1.0, "end": 1.5, "speaker": "A"},
                    {"text": "is", "start": 1.5, "end": 2.0, "speaker": "A"},
                ],
            },
            {
                "speaker": "A", "start": 2.0, "end": 3.0, "text": "great",
                "words": [
                    {"text": "great", "start": 2.0, "end": 3.0, "speaker": "A"},
                ],
            },
            {
                "speaker": "B", "start": 3.0, "end": 4.0, "text": "agreed",
                "words": [
                    {"text": "agreed", "start": 3.0, "end": 4.0, "speaker": "B"},
                ],
            },
        ]
    payload = {"segments": segments, "speakers": ["A", "B"], "language": "en", "mode": "diarize"}
    job = srv.Job(
        id=job_id,
        input_path=input_path,
        output_dir=out_dir,
        mode="diarize",
        speakers=None,
        num_speakers=None,
        language="en",
        model="large-v3",
        created_at="2026-05-25T00:00:00Z",
        status="done",
        progress=1.0,
        message="Done",
        result=payload,
        input_filename="in.wav",
        audio_streams=1,
    )
    srv.JOBS[job_id] = job
    return job_id


# --------------------------------------------------------------------------- #
# /tidy/runs
# --------------------------------------------------------------------------- #


class TestRuns:
    def test_lists_consecutive_same_speaker_runs(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.get(f"/api/job/{job_id}/tidy/runs")
        assert r.status_code == 200, r.text
        runs = r.json()["runs"]
        # Three A's group into one run; B is a singleton (dropped).
        assert len(runs) == 1
        assert runs[0]["speaker"] == "A"
        assert runs[0]["segment_indices"] == [0, 1, 2]
        assert runs[0]["text"] == "i think this is great"
        assert runs[0]["start"] == 0.0
        assert runs[0]["end"] == 3.0

    def test_no_runs_returns_empty(self, env) -> None:
        client, _ = env
        # Three different speakers in a row → no run qualifies.
        segments = [
            {"speaker": "A", "start": 0, "end": 1, "text": "a",
             "words": [{"text": "a", "start": 0, "end": 1, "speaker": "A"}]},
            {"speaker": "B", "start": 1, "end": 2, "text": "b",
             "words": [{"text": "b", "start": 1, "end": 2, "speaker": "B"}]},
            {"speaker": "A", "start": 2, "end": 3, "text": "c",
             "words": [{"text": "c", "start": 2, "end": 3, "speaker": "A"}]},
        ]
        job_id = _seed_job(env, segments=segments)
        r = client.get(f"/api/job/{job_id}/tidy/runs")
        assert r.status_code == 200
        assert r.json()["runs"] == []

    def test_404_for_unknown_job(self, env) -> None:
        client, _ = env
        r = client.get(f"/api/job/{'a' * 12}/tidy/runs")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# /tidy/preview
# --------------------------------------------------------------------------- #


class TestPreview:
    def test_returns_paragraphs_and_segments(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        _install_fake("I think this is great.\n\nReally great.")

        r = client.post(
            f"/api/job/{job_id}/tidy/preview",
            json={"segment_indices": [0, 1, 2]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["paragraphs"] == ["I think this is great.", "Really great."]
        assert len(body["segments"]) == 2
        assert body["speaker"] == "A"
        assert body["segment_indices"] == [0, 1, 2]
        # All words inside the run's wall-clock window.
        for seg in body["segments"]:
            for w in seg["words"]:
                assert 0.0 <= w["start"] <= 3.0
                assert 0.0 <= w["end"] <= 3.0

    def test_preview_does_not_persist(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        _install_fake("Whatever.")

        r = client.post(
            f"/api/job/{job_id}/tidy/preview",
            json={"segment_indices": [0, 1, 2]},
        )
        assert r.status_code == 200
        # Source segments still intact.
        r2 = client.get(f"/api/job/{job_id}/transcript")
        segs = r2.json()["segments"]
        assert len(segs) == 4
        assert segs[0]["text"] == "i think"

    def test_400_on_missing_indices(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.post(f"/api/job/{job_id}/tidy/preview", json={})
        assert r.status_code == 400

    def test_400_on_unknown_run(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        # [0, 1] is not a maximal run — only [0, 1, 2] is.
        r = client.post(
            f"/api/job/{job_id}/tidy/preview",
            json={"segment_indices": [0, 1]},
        )
        assert r.status_code == 400

    def test_502_when_backend_returns_empty(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        _install_fake("")  # Model returned nothing — surface a 502.
        r = client.post(
            f"/api/job/{job_id}/tidy/preview",
            json={"segment_indices": [0, 1, 2]},
        )
        assert r.status_code == 502

    def test_400_when_no_default_model(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        # Override with an empty-model config; the endpoint refuses
        # to call out before the user picks a model.
        srv._tidy_backend_override = (
            BackendConfig(
                provider=PROVIDER_OLLAMA,
                base_url="http://test",
                default_model="",
            ),
            FakeBackend(),
        )
        r = client.post(
            f"/api/job/{job_id}/tidy/preview",
            json={"segment_indices": [0, 1, 2]},
        )
        assert r.status_code == 400
        assert "default_model" in r.text


# --------------------------------------------------------------------------- #
# /tidy/apply
# --------------------------------------------------------------------------- #


class TestApply:
    def test_apply_replaces_run_with_paragraphs(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)

        r = client.post(
            f"/api/job/{job_id}/tidy/apply",
            json={
                "segment_indices": [0, 1, 2],
                "paragraphs": ["I think this is great.", "Really great."],
                "speaker": "A",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["new_segment_count"] == 2
        # Verify on disk via the transcript endpoint.
        r2 = client.get(f"/api/job/{job_id}/transcript")
        segs = r2.json()["segments"]
        # 4 segments before (3 A + 1 B); after: 2 paragraphs of A + 1 B.
        assert len(segs) == 3
        assert [s["speaker"] for s in segs] == ["A", "A", "B"]
        assert segs[0]["text"] == "I think this is great."
        assert segs[1]["text"] == "Really great."
        # B's segment is untouched.
        assert segs[2]["text"] == "agreed"

    def test_word_timestamps_remain_in_run_window(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        client.post(
            f"/api/job/{job_id}/tidy/apply",
            json={
                "segment_indices": [0, 1, 2],
                "paragraphs": ["Totally rewritten content here."],
                "speaker": "A",
            },
        )
        r = client.get(f"/api/job/{job_id}/transcript")
        segs = r.json()["segments"]
        a_words = [w for s in segs if s["speaker"] == "A" for w in s["words"]]
        assert a_words
        # Every word's timestamps live inside the original run [0, 3].
        for w in a_words:
            assert 0.0 <= w["start"] <= 3.0
            assert 0.0 <= w["end"] <= 3.0
            assert w["start"] <= w["end"]

    def test_400_on_missing_fields(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        for bad in (
            {},
            {"segment_indices": [0, 1, 2]},
            {"segment_indices": [0, 1, 2], "paragraphs": ["x"]},
            {"segment_indices": [0, 1, 2], "paragraphs": [], "speaker": "A"},
            {"segment_indices": [], "paragraphs": ["x"], "speaker": "A"},
            {"segment_indices": [0, 1, 2], "paragraphs": ["x"], "speaker": "  "},
        ):
            r = client.post(f"/api/job/{job_id}/tidy/apply", json=bad)
            assert r.status_code == 400, f"expected 400 for {bad!r}, got {r.status_code}"

    def test_400_on_stale_indices(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.post(
            f"/api/job/{job_id}/tidy/apply",
            json={
                "segment_indices": [9, 10],  # not a real run
                "paragraphs": ["x"],
                "speaker": "A",
            },
        )
        assert r.status_code == 400

    def test_404_for_unknown_job(self, env) -> None:
        client, _ = env
        r = client.post(
            f"/api/job/{'b' * 12}/tidy/apply",
            json={"segment_indices": [0], "paragraphs": ["x"], "speaker": "A"},
        )
        assert r.status_code == 404

    def test_editor_renders_tidy_button_and_modal(self, env) -> None:
        client, _ = env
        job_id = _seed_job(env)
        r = client.get(f"/edit/{job_id}")
        assert r.status_code == 200, r.text
        body = r.text
        # Topbar button + modal markup the JS hooks into.
        assert 'id="tidyBtn"' in body
        assert 'id="tidyModal"' in body
        assert 'id="tidyProposed"' in body
        assert "/tidy/runs" in body
        assert "/tidy/preview" in body
        assert "/tidy/apply" in body

    def test_apply_regenerates_sidecars(self, env) -> None:
        client, tmp_path = env
        job_id = _seed_job(env)
        out_dir = srv.OUTPUT_DIR / job_id
        # Pre-existing sidecars (e.g. from the original transcription).
        # The apply path should overwrite them.
        client.post(
            f"/api/job/{job_id}/tidy/apply",
            json={
                "segment_indices": [0, 1, 2],
                "paragraphs": ["I think this is great."],
                "speaker": "A",
            },
        )
        # Check the JSON sidecar contains the new text.
        json_files = list(out_dir.glob("*.json"))
        assert json_files, "expected a JSON sidecar to land in the output dir"
        # The edited transcript file specifically should exist.
        edited = out_dir / "edited.json"
        assert edited.exists()
        body = json.loads(edited.read_text())
        a_segs = [s for s in body["segments"] if s["speaker"] == "A"]
        assert len(a_segs) == 1
        assert a_segs[0]["text"] == "I think this is great."
