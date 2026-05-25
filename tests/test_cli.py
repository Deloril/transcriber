"""Tests for scribe.cli — argparse + main orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scribe import cli
from scribe.engine import Segment, TranscriptionResult


# --------------------------------------------------------------------------- #
# _print_progress
# --------------------------------------------------------------------------- #


class TestPrintProgress:
    def test_writes_to_stderr_with_percent(self, capsys: pytest.CaptureFixture) -> None:
        cli._print_progress("hello", 0.5)
        err = capsys.readouterr().err
        assert "[ 50.0%]" in err
        assert "hello" in err

    def test_clamps_below_zero(self, capsys: pytest.CaptureFixture) -> None:
        cli._print_progress("x", -0.5)
        assert "[  0.0%]" in capsys.readouterr().err

    def test_clamps_above_one(self, capsys: pytest.CaptureFixture) -> None:
        cli._print_progress("x", 5.0)
        assert "[100.0%]" in capsys.readouterr().err

    def test_terminal_progress_emits_newline(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        cli._print_progress("done", 1.0)
        err = capsys.readouterr().err
        # Should include a trailing newline, indicating the bar is finalised.
        assert err.endswith("\n")


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


def _fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            Segment(text="hi", start=0.0, end=1.0, speaker="A", words=[]),
        ],
        language="en",
        mode="diarize",
        speaker_labels=["A"],
        audio_path=Path("/tmp/x.wav"),
    )


class TestMain:
    def test_missing_input_returns_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = cli.main([str(tmp_path / "does-not-exist.wav")])
        assert rc == 2
        assert "input not found" in capsys.readouterr().err

    def test_calls_transcribe_with_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")

        captured: dict = {}

        def fake_transcribe(input_path, **kwargs):
            captured["input"] = input_path
            captured["kwargs"] = kwargs
            return _fake_result()

        monkeypatch.setattr(cli, "transcribe", fake_transcribe)
        # Don't actually write files — use a MagicMock to capture the call.
        monkeypatch.setattr(cli, "write_all", lambda result, base: {
            "json": Path(f"{base}.json"),
            "txt": Path(f"{base}.txt"),
            "srt": Path(f"{base}.srt"),
            "vtt": Path(f"{base}.vtt"),
        })

        rc = cli.main([
            str(f),
            "--mode", "diarize",
            "--language", "en",
            "--model", "tiny",
            "--batch-size", "2",
            "--num-speakers", "3",
        ])
        assert rc == 0
        assert captured["input"] == f
        kw = captured["kwargs"]
        assert kw["mode"] == "diarize"
        assert kw["language"] == "en"
        assert kw["model_name"] == "tiny"
        assert kw["batch_size"] == 2
        assert kw["num_speakers"] == 3

    def test_speakers_split_on_commas(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")
        captured: dict = {}
        monkeypatch.setattr(cli, "transcribe", lambda i, **kw: (captured.update(kw), _fake_result())[1])
        monkeypatch.setattr(cli, "write_all", lambda r, b: {})
        cli.main([str(f), "--speakers", "Luke, Guest, Other"])
        assert captured["speaker_labels"] == ["Luke", "Guest", "Other"]

    def test_speakers_none_when_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")
        captured: dict = {}
        monkeypatch.setattr(cli, "transcribe", lambda i, **kw: (captured.update(kw), _fake_result())[1])
        monkeypatch.setattr(cli, "write_all", lambda r, b: {})
        cli.main([str(f)])
        assert captured["speaker_labels"] is None

    def test_default_out_base_strips_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "name.mp4"
        f.write_bytes(b"")
        seen: list[Path] = []
        monkeypatch.setattr(cli, "transcribe", lambda i, **kw: _fake_result())
        monkeypatch.setattr(cli, "write_all", lambda r, base: (seen.append(base), {})[1])
        cli.main([str(f)])
        assert seen[0] == tmp_path / "name"

    def test_explicit_out_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "in.mp4"
        f.write_bytes(b"")
        out_base = tmp_path / "custom"
        seen: list[Path] = []
        monkeypatch.setattr(cli, "transcribe", lambda i, **kw: _fake_result())
        monkeypatch.setattr(cli, "write_all", lambda r, base: (seen.append(base), {})[1])
        cli.main([str(f), "--out", str(out_base)])
        assert seen[0] == out_base

    def test_keep_temp_preserves_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")
        captured_workdir: list[Path] = []

        def fake_transcribe(input_path, *, work_dir, **kw):
            captured_workdir.append(work_dir)
            # Drop a marker file so we can assert post-cleanup.
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "marker.txt").write_text("kept")
            return _fake_result()

        monkeypatch.setattr(cli, "transcribe", fake_transcribe)
        monkeypatch.setattr(cli, "write_all", lambda r, b: {})

        cli.main([str(f), "--keep-temp"])
        assert captured_workdir
        # With --keep-temp, the marker file persists.
        assert (captured_workdir[0] / "marker.txt").exists()
        # And we tell the user.
        assert "Temp dir kept" in capsys.readouterr().err

    def test_temp_cleaned_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")
        captured_workdir: list[Path] = []

        def fake_transcribe(input_path, *, work_dir, **kw):
            captured_workdir.append(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "marker.txt").write_text("temp")
            return _fake_result()

        monkeypatch.setattr(cli, "transcribe", fake_transcribe)
        monkeypatch.setattr(cli, "write_all", lambda r, b: {})

        cli.main([str(f)])
        assert captured_workdir
        assert not captured_workdir[0].exists()

    def test_summary_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")
        monkeypatch.setattr(cli, "transcribe", lambda i, **kw: _fake_result())
        monkeypatch.setattr(cli, "write_all", lambda r, base: {
            "json": Path("/x.json"),
            "txt": Path("/x.txt"),
        })
        cli.main([str(f)])
        out = capsys.readouterr().out
        assert "Mode:" in out
        assert "Language:" in out
        assert "Speakers:" in out
        assert "Outputs:" in out
        assert "/x.json" in out

    def test_no_speakers_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"")
        result = TranscriptionResult(
            segments=[], language="en", mode="diarize",
            speaker_labels=[], audio_path=Path("/x.wav"),
        )
        monkeypatch.setattr(cli, "transcribe", lambda i, **kw: result)
        monkeypatch.setattr(cli, "write_all", lambda r, b: {})
        cli.main([str(f)])
        assert "(none detected)" in capsys.readouterr().out
