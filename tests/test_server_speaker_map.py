"""End-to-end reachability tests for F3.4 (speaker awareness for queries).

The pure module ``scribe/speaker_map.py`` shipped in 84039cb with 82
passing unit tests in ``tests/test_speaker_map.py``. What this file
proves is that the **user-facing surface** is wired:

  * REST endpoints expose load / save / seed-from-transcript /
    distribution for a per-source speaker map.
  * The source-coding view's side panel renders a "Speakers" panel
    with role + participant-link controls and Save / Seed buttons.

Sibling of ``tests/test_server_participant_sources.py`` (F3.3); the
fixture mirrors that one's pattern (isolated UPLOAD_DIR / OUTPUT_DIR /
PROJECTS_DIR via monkeypatch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated tmp dirs for uploads/outputs/projects."""
    from scribe import server as srv

    monkeypatch.setattr(srv, "JOBS", {})
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    upload.mkdir()
    output.mkdir()
    monkeypatch.setattr(srv, "UPLOAD_DIR", upload)
    monkeypatch.setattr(srv, "OUTPUT_DIR", output)
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(srv, "PROJECTS_DIR", projects_dir)

    client = TestClient(srv.app)
    yield srv, client, tmp_path


def _make_project(client: TestClient, name: str = "Speakers") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_source(
    client: TestClient,
    pid: str,
    *,
    name: str = "Interview",
    transcript_job_id: str | None = None,
) -> str:
    payload = {"name": name, "source_type": "transcript"}
    if transcript_job_id:
        payload["transcript_job_id"] = transcript_job_id
    r = client.post(f"/api/projects/{pid}/sources", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_participant(
    client: TestClient, pid: str, name: str = "P01"
) -> str:
    r = client.post(
        f"/api/projects/{pid}/participants",
        json={"name": name},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _drop_transcript(srv, job_id: str, segments: list[dict]) -> None:
    """Place an edited.json transcript under OUTPUT_DIR/<job>/."""
    job_dir = srv.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "edited.json").write_text(
        json.dumps({"segments": segments})
    )


# --------------------------------------------------------------------------- #
# Coding view exposes the speakers panel
# --------------------------------------------------------------------------- #


class TestSourceCodingPageRendersSpeakersPanel:
    """The coding view side panel must surface the speakers roster +
    Save / Seed buttons so a researcher can tag interviewer vs
    interviewee without leaving the page."""

    def test_coding_page_renders_speakers_heading(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(f"/projects/{pid}/sources/{sid}")
        assert r.status_code == 200
        # Heading marker.
        assert 'data-test-feature="F3.4"' in r.text
        assert "Speakers" in r.text
        # Save + Seed buttons.
        assert 'data-test-id="src-save-speaker-map"' in r.text
        assert 'data-test-id="src-seed-speaker-map"' in r.text
        # Page hits the new endpoint shape.
        assert (
            "/api/projects/${PROJECT_ID}/sources/${SOURCE_ID}/speaker_map"
            in r.text
        )


# --------------------------------------------------------------------------- #
# REST: GET speaker map (loads + transcript labels)
# --------------------------------------------------------------------------- #


class TestGetSpeakerMapAPI:
    def test_empty_when_nothing_saved_and_no_transcript(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/speaker_map"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["speaker_map"]["entries"] == []
        assert body["transcript_labels"] == []
        # Vocabulary must be exposed so the UI can populate the dropdown.
        assert "interviewer" in body["available_roles"]
        assert "interviewee" in body["available_roles"]
        assert "unknown" in body["available_roles"]

    def test_lists_distinct_transcript_labels_first_occurrence_order(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        job_id = "abcdef012345"
        sid = _make_source(client, pid, transcript_job_id=job_id)
        _drop_transcript(srv, job_id, [
            {"speaker": "ANA",  "words": [{"text": "Hi"}]},
            {"speaker": "LUKE", "words": [{"text": "Hello"}]},
            {"speaker": "ANA",  "words": [{"text": "How"}]},
            {"speaker": "GUEST", "words": [{"text": "Bonjour"}]},
        ])
        r = client.get(f"/api/projects/{pid}/sources/{sid}/speaker_map")
        assert r.status_code == 200
        body = r.json()
        # Order is first-occurrence, dedup'd.
        assert body["transcript_labels"] == ["ANA", "LUKE", "GUEST"]

    def test_404_on_missing_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/sources/0123456789ab/speaker_map"
        )
        assert r.status_code == 404

    def test_400_on_malformed_source_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.get(
            f"/api/projects/{pid}/sources/NOT-HEX/speaker_map"
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# REST: PUT speaker map (set-style save)
# --------------------------------------------------------------------------- #


class TestPutSpeakerMapAPI:
    def test_saves_role_assignments(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "SPEAKER_00", "role": "interviewer"},
                {"label": "SPEAKER_01", "role": "interviewee"},
            ]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        labels = [e["label"] for e in body["speaker_map"]["entries"]]
        assert labels == ["SPEAKER_00", "SPEAKER_01"]
        roles = [e["role"] for e in body["speaker_map"]["entries"]]
        assert roles == ["interviewer", "interviewee"]
        # And reading back returns the same map.
        r2 = client.get(
            f"/api/projects/{pid}/sources/{sid}/speaker_map"
        )
        again = r2.json()["speaker_map"]["entries"]
        assert [(e["label"], e["role"]) for e in again] == [
            ("SPEAKER_00", "interviewer"),
            ("SPEAKER_01", "interviewee"),
        ]

    def test_links_to_participant(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        part_id = _make_participant(client, pid, name="Ana")
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "ANA",
                 "role": "interviewee",
                 "participant_id": part_id},
            ]},
        )
        assert r.status_code == 200, r.text
        entry = r.json()["speaker_map"]["entries"][0]
        assert entry["participant_id"] == part_id

    def test_set_style_replaces_previous_entries(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        # First save: two rows.
        client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "A", "role": "interviewer"},
                {"label": "B", "role": "interviewee"},
            ]},
        )
        # Second save: one row only — the other should be dropped.
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "A", "role": "facilitator"},
            ]},
        )
        assert r.status_code == 200
        entries = r.json()["speaker_map"]["entries"]
        assert len(entries) == 1
        assert entries[0]["label"] == "A"
        assert entries[0]["role"] == "facilitator"

    def test_400_on_unknown_role(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "A", "role": "moderator"},  # not in vocabulary
            ]},
        )
        assert r.status_code == 400

    def test_400_on_unknown_participant_id(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        # Real project, but no participant with this id on disk.
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "A",
                 "role": "interviewee",
                 "participant_id": "0123456789ab"},
            ]},
        )
        assert r.status_code == 400
        assert "0123456789ab" in r.json()["detail"]

    def test_400_on_malformed_payload(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": "not-a-list"},
        )
        assert r.status_code == 400

    def test_404_on_missing_source(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        r = client.put(
            f"/api/projects/{pid}/sources/0123456789ab/speaker_map",
            json={"entries": []},
        )
        assert r.status_code == 404

    def test_empty_entries_clears_map(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [{"label": "A", "role": "interviewer"}]},
        )
        r = client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": []},
        )
        assert r.status_code == 200
        assert r.json()["speaker_map"]["entries"] == []


# --------------------------------------------------------------------------- #
# REST: POST /speaker_map/seed (auto-fill from transcript)
# --------------------------------------------------------------------------- #


class TestSeedSpeakerMapAPI:
    def test_seeds_from_transcript_segments(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        job_id = "abcdef012345"
        sid = _make_source(client, pid, transcript_job_id=job_id)
        _drop_transcript(srv, job_id, [
            {"speaker": "ANA", "words": [{"text": "Hi"}]},
            {"speaker": "LUKE", "words": [{"text": "Hello"}]},
        ])
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/seed",
            json={},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert sorted(body["added"]) == ["ANA", "LUKE"]
        labels = [e["label"] for e in body["speaker_map"]["entries"]]
        assert sorted(labels) == ["ANA", "LUKE"]
        # Default role for fresh seed is "unknown".
        roles = {e["label"]: e["role"] for e in body["speaker_map"]["entries"]}
        assert roles["ANA"] == "unknown"

    def test_seed_preserves_existing_role_assignments(
        self, server_env
    ) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        job_id = "abcdef012345"
        sid = _make_source(client, pid, transcript_job_id=job_id)
        # Pre-set ANA's role.
        client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [{"label": "ANA", "role": "interviewer"}]},
        )
        _drop_transcript(srv, job_id, [
            {"speaker": "ANA", "words": [{"text": "Hi"}]},
            {"speaker": "LUKE", "words": [{"text": "Hello"}]},
        ])
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/seed",
            json={},
        )
        assert r.status_code == 200
        body = r.json()
        # Only LUKE was added; ANA's role survived.
        assert body["added"] == ["LUKE"]
        roles = {e["label"]: e["role"] for e in body["speaker_map"]["entries"]}
        assert roles["ANA"] == "interviewer"
        assert roles["LUKE"] == "unknown"

    def test_409_when_no_transcript_available(self, server_env) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)  # No transcript_job_id.
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/seed",
            json={},
        )
        assert r.status_code == 409

    def test_400_on_unknown_default_role(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        job_id = "abcdef012345"
        sid = _make_source(client, pid, transcript_job_id=job_id)
        _drop_transcript(srv, job_id, [
            {"speaker": "ANA", "words": [{"text": "Hi"}]},
        ])
        r = client.post(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/seed",
            json={"default_role": "moderator"},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# REST: GET /speaker_map/distribution
# --------------------------------------------------------------------------- #


class TestSpeakerMapDistributionAPI:
    def test_distribution_counts_segments_per_role(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        job_id = "abcdef012345"
        sid = _make_source(client, pid, transcript_job_id=job_id)
        _drop_transcript(srv, job_id, [
            {"speaker": "ANA", "words": [{"text": "Hi"}]},
            {"speaker": "ANA", "words": [{"text": "Yes"}]},
            {"speaker": "ANA", "words": [{"text": "No"}]},
            {"speaker": "LUKE", "words": [{"text": "Hello"}]},
        ])
        # Tag ANA as interviewer, LUKE as interviewee.
        client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "ANA", "role": "interviewer"},
                {"label": "LUKE", "role": "interviewee"},
            ]},
        )
        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/distribution"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["has_transcript"] is True
        assert body["role_distribution"]["interviewer"] == 3
        assert body["role_distribution"]["interviewee"] == 1
        # Every role bucket is present (zero-init).
        assert body["role_distribution"]["facilitator"] == 0

    def test_distribution_returns_empty_without_transcript(
        self, server_env
    ) -> None:
        _, client, _ = server_env
        pid = _make_project(client)
        sid = _make_source(client, pid)
        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/distribution"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["has_transcript"] is False
        assert body["role_distribution"] == {}
        assert body["participant_distribution"] == {}

    def test_distribution_buckets_per_participant(self, server_env) -> None:
        srv, client, _ = server_env
        pid = _make_project(client)
        job_id = "abcdef012345"
        sid = _make_source(client, pid, transcript_job_id=job_id)
        part_id = _make_participant(client, pid, name="Ana")
        _drop_transcript(srv, job_id, [
            {"speaker": "ANA", "words": [{"text": "x"}]},
            {"speaker": "ANA", "words": [{"text": "y"}]},
            {"speaker": "LUKE", "words": [{"text": "z"}]},
        ])
        client.put(
            f"/api/projects/{pid}/sources/{sid}/speaker_map",
            json={"entries": [
                {"label": "ANA",
                 "role": "interviewee",
                 "participant_id": part_id},
                {"label": "LUKE",
                 "role": "interviewer"},
            ]},
        )
        r = client.get(
            f"/api/projects/{pid}/sources/{sid}/speaker_map/distribution"
        )
        body = r.json()
        # ANA's two segments map to the participant bucket; LUKE has no
        # participant link so it lands in the empty-string bucket.
        assert body["participant_distribution"][part_id] == 2
        assert body["participant_distribution"][""] == 1
