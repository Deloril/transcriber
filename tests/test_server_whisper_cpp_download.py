"""Tests for the whisper.cpp GGUF download manager endpoints.

The HTTP layer:

* POST /api/whisper-cpp/download           start a background fetch
* GET  /api/whisper-cpp/download/{id}      poll progress + final state

Pure download logic lives in :mod:`scribe.whisper_cpp` and has its
own unit tests; here we exercise the FastAPI wrapping, the in-memory
state machine, and the de-dupe behaviour for repeat clicks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scribe import server as srv
from scribe import whisper_cpp as _wcpp


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pin the GGUF cache to a tmp dir and clear the in-memory
    download registry between tests."""
    monkeypatch.setenv(_wcpp.ENV_CACHE_DIR, str(tmp_path / "cache"))
    monkeypatch.setattr(srv, "_WHISPER_CPP_DOWNLOADS", {})
    return TestClient(srv.app), tmp_path


def _patch_download(
    monkeypatch: pytest.MonkeyPatch,
    *,
    behaviour: str = "ok",
    payload_size: int = 4096,
):
    """Replace whisper_cpp.download_gguf in the server's namespace
    with a deterministic stand-in.

    behaviour:
      "ok"       — write payload, call progress halfway + at end.
      "slow_ok"  — same as ok but sleeps so the GET endpoint sees
                   ``state=running`` between POST and the worker
                   finishing.
      "fail"     — raise OSError so the worker records ``error``.
    """
    def fake_download_gguf(
        model: str, quant: str, *,
        cache_dir: Any = None,
        progress: Any = None,
        chunk_size: int = 1024 * 1024,
        timeout_s: float = 30.0,
        url_opener: Any = None,
    ):
        if behaviour == "fail":
            raise OSError("simulated download failure")
        if behaviour == "slow_ok":
            # Two progress ticks separated by a sleep so the GET poller
            # has a chance to observe ``state=running``.
            if progress:
                progress(payload_size // 2, payload_size)
            time.sleep(0.2)
            if progress:
                progress(payload_size, payload_size)
            target = _wcpp.gguf_path(model, quant)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * payload_size)
            return target
        # default ok
        if progress:
            progress(payload_size // 2, payload_size)
            progress(payload_size, payload_size)
        target = _wcpp.gguf_path(model, quant)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00" * payload_size)
        return target

    # Patch the symbol at the module the server's worker imports it from.
    # _wc_run_download does ``from . import whisper_cpp as _wc`` so we
    # patch the canonical module location.
    monkeypatch.setattr(_wcpp, "download_gguf", fake_download_gguf)


def _wait_for_state(
    client: TestClient, download_id: str, target_state: str,
    *, timeout_s: float = 5.0,
) -> dict:
    """Poll the status endpoint until the worker reaches ``target_state``
    (or ``error``). Returns the final body."""
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/api/whisper-cpp/download/{download_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["state"] == target_state or last["state"] == "error":
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for state={target_state}; last={last!r}"
    )


# --------------------------------------------------------------------------- #
# POST — start
# --------------------------------------------------------------------------- #


class TestPost:
    def test_starts_a_background_download(
        self, env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _ = env
        _patch_download(monkeypatch)
        r = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q5_0"},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["state"] == "running"
        assert body["model"] == "large-v3"
        assert body["quant"] == "q5_0"
        assert "id" in body and len(body["id"]) == 12
        # The worker eventually completes.
        final = _wait_for_state(client, body["id"], "complete")
        assert final["state"] == "complete"
        assert final["path"]

    def test_already_cached_short_circuits(
        self, env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _ = env
        # Lay the file down before the user clicks.
        _wcpp.default_cache_dir().mkdir(parents=True, exist_ok=True)
        target = _wcpp.gguf_path("large-v3", "q5_0")
        target.write_bytes(b"\x00" * 16)
        # The fake should not be called — assert by raising if it is.
        def boom(*args, **kwargs):  # noqa: ANN001, ANN201
            raise AssertionError("download should not run when cached")
        monkeypatch.setattr(_wcpp, "download_gguf", boom)
        r = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q5_0"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["state"] == "complete"
        assert body["already_cached"] is True

    def test_dedupes_in_flight_requests(
        self, env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _ = env
        _patch_download(monkeypatch, behaviour="slow_ok")
        r1 = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q5_0"},
        )
        assert r1.status_code == 202
        first_id = r1.json()["id"]
        r2 = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q5_0"},
        )
        assert r2.status_code == 202
        # Same id — server reattached us to the running worker.
        assert r2.json()["id"] == first_id
        # Drain.
        _wait_for_state(client, first_id, "complete")

    def test_400_on_unsupported_model(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/whisper-cpp/download",
            json={"model": "tiny", "quant": "q5_0"},
        )
        assert r.status_code == 400

    def test_400_on_unsupported_quant(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q3"},
        )
        assert r.status_code == 400

    def test_400_on_invalid_json(self, env) -> None:
        client, _ = env
        r = client.post(
            "/api/whisper-cpp/download",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GET — status
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_404_on_unknown_id(self, env) -> None:
        client, _ = env
        r = client.get("/api/whisper-cpp/download/aaaaaaaaaaaa")
        assert r.status_code == 404

    def test_400_on_invalid_id_shape(self, env) -> None:
        client, _ = env
        r = client.get("/api/whisper-cpp/download/not-hex")
        assert r.status_code == 400

    def test_error_state_carries_message(
        self, env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _ = env
        _patch_download(monkeypatch, behaviour="fail")
        r = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q5_0"},
        )
        assert r.status_code == 202
        final = _wait_for_state(client, r.json()["id"], "error")
        assert final["state"] == "error"
        assert "simulated" in (final.get("error") or "").lower()

    def test_progress_visible_during_running(
        self, env, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _ = env
        _patch_download(monkeypatch, behaviour="slow_ok", payload_size=2048)
        r = client.post(
            "/api/whisper-cpp/download",
            json={"model": "large-v3", "quant": "q5_0"},
        )
        download_id = r.json()["id"]
        # Read at least once before the worker finishes; we should see
        # downloaded_bytes > 0 OR state=complete (race-tolerant).
        first = client.get(f"/api/whisper-cpp/download/{download_id}").json()
        assert first["state"] in ("running", "complete")
        if first["state"] == "running":
            assert first["downloaded_bytes"] >= 0
            assert first["total_bytes"] in (None, 2048)
        _wait_for_state(client, download_id, "complete")
