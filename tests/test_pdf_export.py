"""Tests for scribe.pdf_export — the pure HTML/PDF rendering helpers.

The PDF surface has two layers:

* **Pure helpers** (``_parse_word_id``, ``code_colour``,
  ``split_segment_into_runs``, ``render_html``) — exercised here.
* **PDF byte rendering** (``render_pdf_bytes``,
  ``render_pdf_for_source``) — wraps weasyprint, which pulls in
  cairo/pango at runtime. We give it a tiny smoke test that's
  skipped when the system deps are missing so this suite passes
  on minimal CI.

The HTML is the contract — if it has the right classes and the
coded runs carry the right colours, weasyprint's job is just
"render this CSS to A4," and that's their problem to keep working.
"""

from __future__ import annotations

import importlib.util
import re

import pytest

from scribe import pdf_export


class TestParseWordId:
    def test_round_trip(self) -> None:
        assert pdf_export._parse_word_id("s3w12") == (3, 12)

    def test_zero_indices_are_valid(self) -> None:
        assert pdf_export._parse_word_id("s0w0") == (0, 0)

    @pytest.mark.parametrize("bad", ["", None, "s3", "w12", "garbage", "s3-w12"])
    def test_unparseable_returns_none(self, bad) -> None:
        assert pdf_export._parse_word_id(bad) is None


class TestCodeColour:
    def test_user_hex_wins(self) -> None:
        c = pdf_export.code_colour({"id": "c1", "colour": "#7aa7ff"})
        assert c["border"] == "#7aa7ff"
        assert "rgba(122, 167, 255" in c["bg"]

    def test_three_char_hex_expands(self) -> None:
        c = pdf_export.code_colour({"id": "c1", "colour": "#abc"})
        assert "rgba(170, 187, 204" in c["bg"]

    def test_falls_back_to_hash_when_no_user_colour(self) -> None:
        c = pdf_export.code_colour({"id": "code-no-colour", "name": "x"})
        assert c["bg"].startswith("hsla(")
        assert c["border"].startswith("hsl(")
        assert c["swatch"].startswith("hsl(")

    def test_hash_is_stable_for_same_id(self) -> None:
        a = pdf_export.code_colour({"id": "code-a", "name": "Alpha"})
        b = pdf_export.code_colour({"id": "code-a", "name": "different name"})
        # Hash uses id first when present — identical id must give
        # identical colour even when the name changes.
        assert a == b

    def test_hash_differs_for_different_ids(self) -> None:
        a = pdf_export.code_colour({"id": "code-a"})
        b = pdf_export.code_colour({"id": "code-b"})
        assert a != b

    def test_invalid_hex_falls_through(self) -> None:
        # ``#xyzxyz`` parses as length 7 but isn't valid hex; we should
        # gracefully drop into the hash path rather than crash.
        c = pdf_export.code_colour({"id": "c1", "colour": "#xyzxyz"})
        assert c["border"].startswith("hsl(")


def _seg(words):
    """Build a segment dict with the expected ``words`` shape."""
    return {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 5.0,
        "words": [{"text": w} for w in words],
    }


def _app(app_id, code_id, start, end):
    return {
        "id": app_id,
        "code_id": code_id,
        "coder_id": "coder-default",
        "anchor_start_word_id": start,
        "anchor_end_word_id": end,
    }


class TestSplitSegmentIntoRuns:
    def test_no_applications_yields_one_plain_run(self) -> None:
        seg = _seg(["I", "find", "it", "hard"])
        runs = pdf_export.split_segment_into_runs(seg, 0, [])
        assert len(runs) == 1
        assert runs[0].application_ids == ()
        assert runs[0].text == "I find it hard"

    def test_single_application_in_middle(self) -> None:
        seg = _seg(["I", "find", "it", "hard", "to", "ask"])
        # Apply code-x to words 2–3 ("it hard")
        apps = [_app("a1", "code-x", "s0w2", "s0w3")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        # plain "I find" / coded "it hard" / plain "to ask"
        assert len(runs) == 3
        assert runs[0].application_ids == ()
        assert runs[0].text == "I find"
        assert runs[1].application_ids == ("a1",)
        assert runs[1].text == "it hard"
        assert runs[2].application_ids == ()
        assert runs[2].text == "to ask"

    def test_application_at_start(self) -> None:
        seg = _seg(["A", "B", "C"])
        apps = [_app("a1", "c", "s0w0", "s0w1")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        assert [r.application_ids for r in runs] == [("a1",), ()]
        assert [r.text for r in runs] == ["A B", "C"]

    def test_application_at_end(self) -> None:
        seg = _seg(["A", "B", "C"])
        apps = [_app("a1", "c", "s0w2", "s0w2")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        assert [r.application_ids for r in runs] == [(), ("a1",)]

    def test_overlapping_applications_stack(self) -> None:
        seg = _seg(["one", "two", "three", "four"])
        apps = [
            _app("a1", "code-x", "s0w0", "s0w2"),  # words 0..2
            _app("a2", "code-y", "s0w1", "s0w3"),  # words 1..3
        ]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        # word 0: {a1}, words 1-2: {a1,a2}, word 3: {a2}
        assert [r.application_ids for r in runs] == [
            ("a1",), ("a1", "a2"), ("a2",),
        ]

    def test_cross_segment_app_bounds_to_current_segment(self) -> None:
        """An app spanning seg 0..2 should cover *all* words in seg 1."""
        seg = _seg(["mid", "segment"])
        apps = [_app("a1", "c", "s0w5", "s2w1")]
        runs = pdf_export.split_segment_into_runs(seg, 1, apps)
        assert len(runs) == 1
        assert runs[0].application_ids == ("a1",)
        assert runs[0].text == "mid segment"

    def test_app_outside_segment_is_ignored(self) -> None:
        seg = _seg(["x", "y"])
        apps = [_app("a1", "c", "s5w0", "s5w0")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        assert [r.application_ids for r in runs] == [()]

    def test_backwards_anchor_is_dropped(self) -> None:
        seg = _seg(["x", "y", "z"])
        apps = [_app("a1", "c", "s0w2", "s0w0")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        assert all(r.application_ids == () for r in runs)

    def test_legacy_segment_without_words_uses_one_synthetic_word(self) -> None:
        seg = {"speaker": "S", "start": 0.0, "end": 1.0, "text": "hello world"}
        apps = [_app("a1", "c", "s0w0", "s0w0")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        assert len(runs) == 1
        assert runs[0].application_ids == ("a1",)
        assert runs[0].text == "hello world"

    def test_unparseable_anchor_is_skipped(self) -> None:
        seg = _seg(["x", "y"])
        apps = [_app("a1", "c", "garbage", "s0w0")]
        runs = pdf_export.split_segment_into_runs(seg, 0, apps)
        assert all(r.application_ids == () for r in runs)


class TestRenderHtml:
    def _ctx(self):
        project = {"id": "p1", "name": "Pilot", "research_question": "Why?"}
        source = {"id": "src-1", "name": "Maria — interview"}
        codes = [
            {"id": "code-belief", "name": "Belief", "definition": "What they believe.",
             "colour": "#7aa7ff"},
        ]
        coders = [{"id": "coder-1", "name": "Luke"}]
        segments = [
            {
                "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0,
                "words": [{"text": w} for w in ["I", "believe", "things"]],
            },
            {
                "speaker": "SPEAKER_01", "start": 1.5, "end": 2.5,
                "words": [{"text": w} for w in ["You", "do", "not"]],
            },
        ]
        applications = [
            {"id": "app-1", "code_id": "code-belief", "coder_id": "coder-1",
             "anchor_start_word_id": "s0w1", "anchor_end_word_id": "s0w2"},
        ]
        speaker_names = {"SPEAKER_00": "Maria", "SPEAKER_01": "Tom"}
        return dict(
            project=project, source=source, segments=segments,
            applications=applications, codes=codes, coders=coders,
            speaker_names=speaker_names, now="2026-05-26 10:00 UTC",
        )

    def test_doctype_and_structure(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        assert html.startswith("<!doctype html>")
        assert "<style>" in html
        assert "body-grid" in html
        assert html.rstrip().endswith("</html>")

    def test_carries_source_name_and_research_question(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        assert "Maria — interview" in html
        assert "Pilot" in html
        assert "Why?" in html

    def test_carries_export_metaline(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        assert "2 segments" in html
        assert "1 coded spans" in html
        assert "1 codes in codebook" in html

    def test_legend_lists_used_codes_only(self) -> None:
        ctx = self._ctx()
        # Add an unused code; legend should not include it.
        ctx["codes"].append({"id": "code-unused", "name": "Unused", "definition": ""})
        html = pdf_export.render_html(**ctx)
        assert "Belief" in html
        assert "Unused" not in html

    def test_speaker_label_uses_display_name(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        assert "Maria" in html  # for SPEAKER_00 → Maria
        assert "Tom" in html

    def test_coded_span_carries_callout_class(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        # Should wrap "believe things" in a coded span carrying the
        # callout class so the printed output gets the left border.
        assert "class='coded callout'" in html
        assert "believe things" in html
        # The user-set #7aa7ff colour should appear on the span.
        assert "#7aa7ff" in html

    def test_margin_annotation_includes_quote_and_definition(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        assert "What they believe." in html
        assert "believe things" in html
        assert "by Luke" in html

    def test_coded_text_appears_only_inside_span(self) -> None:
        html = pdf_export.render_html(**self._ctx())
        # "believe things" must appear once in the transcript text
        # column (inside the span) and once in the margin quote.
        # Counting raw occurrences should be exactly 2.
        assert html.count("believe things") == 2

    def test_no_codes_renders_clean_document(self) -> None:
        ctx = self._ctx()
        ctx["applications"] = []
        html = pdf_export.render_html(**ctx)
        # No legend block when no applications exist.
        assert "Codes used in this transcript" not in html
        # No margin annotations.
        assert "class='ann'" not in html

    def test_html_escapes_user_strings(self) -> None:
        ctx = self._ctx()
        ctx["source"] = {"id": "src-1", "name": "<script>alert(1)</script>"}
        html = pdf_export.render_html(**ctx)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


# --------------------------------------------------------------------------- #
# weasyprint smoke test — skipped when the system deps aren't present.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    importlib.util.find_spec("weasyprint") is None,
    reason="weasyprint not installed (system dep optional)",
)
def test_render_pdf_bytes_smoke() -> None:
    """If weasyprint is importable, we can produce a PDF magic-byte
    header. Skipped on minimal CI where cairo/pango aren't present."""
    html = "<!doctype html><html><body><h1>Hi</h1></body></html>"
    try:
        pdf = pdf_export.render_pdf_bytes(html)
    except pdf_export.PdfExportError as e:
        pytest.skip(f"weasyprint runtime failed (system deps missing?): {e}")
    assert pdf[:4] == b"%PDF"


def test_render_pdf_bytes_raises_clear_error_when_weasyprint_missing(monkeypatch) -> None:
    """The error message tells the dev exactly which apt/brew packages
    they need — important because the failure happens at request time
    on a fresh box."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "weasyprint":
            raise ImportError("no weasyprint")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(pdf_export.PdfExportError) as exc:
        pdf_export.render_pdf_bytes("<html></html>")
    msg = str(exc.value)
    assert "weasyprint" in msg
    assert "libpango" in msg or "pango" in msg
