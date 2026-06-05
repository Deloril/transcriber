"""Tests for scribe.embedding_index (F8.2).

Covers the embedding-index data model, on-disk persistence, the
desired-spans enumerator, the refresh-diff workflow, similarity
search, and the F8.1-backend adapter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scribe.applications import (
    Application,
    save_application,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)
from scribe.embedding_index import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_ID_RE,
    EMBEDDING_KIND_CODED_SEGMENT,
    EMBEDDING_KIND_UNCODED_PARAGRAPH,
    EMBEDDING_KINDS,
    EMBEDDINGS_DIRNAME,
    MAX_MODEL_NAME_LEN,
    MAX_TEXT_PREVIEW_LEN,
    MAX_VECTOR_DIM,
    EmbedFn,
    EmbeddingEntry,
    IndexableSpan,
    RefreshResult,
    canonical_text,
    clear_embedding_index,
    cosine_similarity,
    delete_embedding_entry,
    desired_index_spans,
    embedding_state_path,
    embeddings_dir,
    entry_id_for_key,
    extract_application_text,
    extract_paragraph_text,
    list_embedding_entries,
    load_embedding_entry,
    make_embed_fn_from_backend,
    refresh_embedding_index,
    save_embedding_entry,
    search_similar,
    text_hash,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_HEX_PROJECT = "0" * 12
_HEX_CODE = "a" * 12
_HEX_SOURCE = "b" * 12
_HEX_CODER = "c" * 12
_HEX_VERSION = "d" * 12
_HEX_APP = "e" * 12
_HEX_SOURCE2 = "f" * 12


def _saved_project(tmp_path: Path, *, name: str = "Project") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _segments(rows: list[tuple[str | None, list[str]]]) -> list[dict[str, Any]]:
    """Build a Scribe-shape segments list from (speaker, [word, ...]) rows."""
    out: list[dict[str, Any]] = []
    for speaker, words in rows:
        out.append(
            {
                "speaker": speaker,
                "words": [{"text": w} for w in words],
            }
        )
    return out


def _make_app(
    *,
    project_id: str,
    application_id: str = _HEX_APP,
    source_id: str = _HEX_SOURCE,
    code_id: str = _HEX_CODE,
    coder_id: str = _HEX_CODER,
    version_id: str = _HEX_VERSION,
    start: str = "s0w0",
    end: str = "s0w0",
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start,
        anchor_end_word_id=end,
        definition_version_id_at_apply=version_id,
        application_id=application_id,
        start_char_offset=start_offset,
        end_char_offset=end_offset,
    )


def _const_embed(dim: int = 4) -> EmbedFn:
    """Return a fake embed callable that yields deterministic vectors."""
    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        out: list[tuple[float, ...]] = []
        for text in texts:
            # Use simple per-text hash so two different texts produce
            # different vectors but the same text produces the same one.
            h = text_hash(text)
            base = [int(h[i : i + 2], 16) / 255.0 for i in range(0, 2 * dim, 2)]
            out.append(tuple(base))
        return out
    return fn


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


class TestCanonicalText:
    def test_collapses_whitespace_runs(self) -> None:
        assert canonical_text("  Hello\n\n   world  ") == "Hello world"

    def test_empty_string(self) -> None:
        assert canonical_text("") == ""

    def test_non_string_returns_empty(self) -> None:
        assert canonical_text(None) == ""  # type: ignore[arg-type]
        assert canonical_text(42) == ""  # type: ignore[arg-type]

    def test_unicode_whitespace(self) -> None:
        # \t and \r both fold to single space.
        assert canonical_text("a\t\rb") == "a b"


class TestTextHash:
    def test_stable(self) -> None:
        assert text_hash("hello") == text_hash("hello")

    def test_distinct(self) -> None:
        assert text_hash("hello") != text_hash("world")

    def test_64_char_hex(self) -> None:
        h = text_hash("anything")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestEntryIdForKey:
    def test_12_char_hex(self) -> None:
        eid = entry_id_for_key(("coded_segment", _HEX_SOURCE, _HEX_APP))
        assert EMBEDDING_ID_RE.match(eid)

    def test_deterministic(self) -> None:
        a = entry_id_for_key(("coded_segment", _HEX_SOURCE, _HEX_APP))
        b = entry_id_for_key(("coded_segment", _HEX_SOURCE, _HEX_APP))
        assert a == b

    def test_distinct_keys_distinct_ids(self) -> None:
        a = entry_id_for_key(("coded_segment", _HEX_SOURCE, _HEX_APP))
        b = entry_id_for_key(("uncoded_paragraph", _HEX_SOURCE, "0", "1"))
        assert a != b


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #


class TestExtractApplicationText:
    def test_simple_word_run(self) -> None:
        segs = _segments([(None, ["Hello", "world", "foo"])])
        a = _make_app(project_id=_HEX_PROJECT, start="s0w0", end="s0w1")
        assert extract_application_text(a, segs) == "Hello world"

    def test_subword_offsets_single_word(self) -> None:
        segs = _segments([(None, ["criminalisation"])])
        a = _make_app(
            project_id=_HEX_PROJECT,
            start="s0w0",
            end="s0w0",
            start_offset=4,
            end_offset=12,
        )
        # criminalisation[4:12] == "inalisat"
        assert extract_application_text(a, segs) == "inalisat"

    def test_subword_offsets_multiword(self) -> None:
        segs = _segments([(None, ["alpha", "beta", "gamma"])])
        a = _make_app(
            project_id=_HEX_PROJECT,
            start="s0w0",
            end="s0w2",
            start_offset=2,
            end_offset=3,
        )
        # First word: alpha[2:] -> "pha"; last: gamma[:3] -> "gam"
        assert extract_application_text(a, segs) == "pha beta gam"

    def test_anchor_out_of_range_returns_empty(self) -> None:
        segs = _segments([(None, ["hi"])])
        a = _make_app(project_id=_HEX_PROJECT, start="s9w9", end="s9w9")
        assert extract_application_text(a, segs) == ""

    def test_canonicalised_whitespace(self) -> None:
        segs = _segments([(None, ["hi", "  there  ", "\n"])])
        a = _make_app(project_id=_HEX_PROJECT, start="s0w0", end="s0w2")
        assert extract_application_text(a, segs) == "hi there"


class TestExtractParagraphText:
    def test_basic(self) -> None:
        segs = _segments([
            ("A", ["one", "two"]),
            ("A", ["three"]),
        ])
        assert extract_paragraph_text(segs, 0, 1) == "one two three"

    def test_out_of_range_returns_empty(self) -> None:
        segs = _segments([("A", ["x"])])
        assert extract_paragraph_text(segs, 0, 5) == ""
        assert extract_paragraph_text(segs, -1, 0) == ""
        assert extract_paragraph_text(segs, 1, 0) == ""


# --------------------------------------------------------------------------- #
# IndexableSpan + key
# --------------------------------------------------------------------------- #


class TestIndexableSpanKey:
    def test_coded_segment_key(self) -> None:
        s = IndexableSpan(
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text="x",
        )
        assert s.key() == ("coded_segment", _HEX_SOURCE, _HEX_APP)

    def test_uncoded_paragraph_key(self) -> None:
        s = IndexableSpan(
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE,
            application_id=None,
            paragraph_start_segment=2,
            paragraph_end_segment=4,
            anchor_start_word_id="s2w0",
            anchor_end_word_id="s4w3",
            text="x",
        )
        assert s.key() == ("uncoded_paragraph", _HEX_SOURCE, "2", "4")


# --------------------------------------------------------------------------- #
# Desired-spans enumeration
# --------------------------------------------------------------------------- #


class TestDesiredIndexSpans:
    def test_empty_inputs_yields_no_spans(self) -> None:
        assert desired_index_spans(applications=[], segments_by_source={}) == []

    def test_uncoded_paragraphs_when_no_applications(self) -> None:
        # Two speaker turns, no applications: each turn becomes its
        # own uncoded_paragraph entry.
        segs = _segments([
            ("A", ["one"]),
            ("A", ["two"]),
            ("B", ["three"]),
        ])
        spans = desired_index_spans(
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
        )
        assert all(s.kind == EMBEDDING_KIND_UNCODED_PARAGRAPH for s in spans)
        # paragraph_ranges produces (0,1) and (2,2) for this layout.
        keys = {s.key() for s in spans}
        assert ("uncoded_paragraph", _HEX_SOURCE, "0", "1") in keys
        assert ("uncoded_paragraph", _HEX_SOURCE, "2", "2") in keys
        assert len(spans) == 2

    def test_application_excludes_its_paragraph(self) -> None:
        segs = _segments([
            ("A", ["one"]),
            ("A", ["two"]),
            ("B", ["three"]),
        ])
        # An application touching segment 0 means paragraph (0,1) is
        # excluded from uncoded_paragraph entries; (2,2) remains.
        proj = Project.new(name="x")  # in-memory; never saved
        app = _make_app(
            project_id=proj.id,
            start="s0w0",
            end="s0w0",
        )
        spans = desired_index_spans(
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
        )
        kinds = sorted([s.kind for s in spans])
        assert kinds == ["coded_segment", "uncoded_paragraph"]
        para_keys = [
            s.key() for s in spans if s.kind == EMBEDDING_KIND_UNCODED_PARAGRAPH
        ]
        assert para_keys == [("uncoded_paragraph", _HEX_SOURCE, "2", "2")]

    def test_coded_segment_text_uses_anchor_words(self) -> None:
        segs = _segments([(None, ["alpha", "beta", "gamma"])])
        proj = Project.new(name="x")  # in-memory; never saved
        app = _make_app(
            project_id=proj.id,
            start="s0w0",
            end="s0w1",
        )
        spans = desired_index_spans(
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
        )
        coded = [
            s for s in spans if s.kind == EMBEDDING_KIND_CODED_SEGMENT
        ]
        assert len(coded) == 1
        assert coded[0].text == "alpha beta"
        assert coded[0].anchor_start_word_id == "s0w0"
        assert coded[0].anchor_end_word_id == "s0w1"

    def test_drops_application_with_empty_text(self) -> None:
        # Anchor outside any segment — extract returns "", so the
        # span is silently dropped.
        proj = Project.new(name="x")  # in-memory; never saved
        app = _make_app(
            project_id=proj.id,
            start="s9w9",
            end="s9w9",
        )
        spans = desired_index_spans(
            applications=[app],
            segments_by_source={_HEX_SOURCE: _segments([(None, ["hi"])])},
        )
        # No coded_segment because text is empty; one uncoded_paragraph
        # because nothing touches segment 0.
        assert all(s.kind == EMBEDDING_KIND_UNCODED_PARAGRAPH for s in spans)

    def test_multiple_sources(self) -> None:
        segs1 = _segments([("A", ["one"])])
        segs2 = _segments([("B", ["two"])])
        spans = desired_index_spans(
            applications=[],
            segments_by_source={_HEX_SOURCE: segs1, _HEX_SOURCE2: segs2},
        )
        sources = {s.source_id for s in spans}
        assert sources == {_HEX_SOURCE, _HEX_SOURCE2}
        assert all(s.kind == EMBEDDING_KIND_UNCODED_PARAGRAPH for s in spans)

    def test_skips_invalid_source_id_keys(self) -> None:
        # A garbage source_id key in the map shouldn't crash; it just
        # contributes no spans.
        spans = desired_index_spans(
            applications=[],
            segments_by_source={"not-hex": _segments([("A", ["x"])])},
        )
        assert spans == []

    def test_paragraph_anchor_uses_first_and_last_word(self) -> None:
        segs = _segments([
            ("A", ["alpha", "beta"]),
            ("A", ["gamma"]),
        ])
        spans = desired_index_spans(
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
        )
        assert len(spans) == 1
        s = spans[0]
        assert s.anchor_start_word_id == "s0w0"
        assert s.anchor_end_word_id == "s1w0"


# --------------------------------------------------------------------------- #
# EmbeddingEntry data model
# --------------------------------------------------------------------------- #


class TestEmbeddingEntryNew:
    def test_minimal_coded_segment(self) -> None:
        span = IndexableSpan(
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            text="hello world",
        )
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=span,
            vector=[0.1, 0.2, 0.3, 0.4],
            model_name="bge-m3",
        )
        assert e.kind == EMBEDDING_KIND_CODED_SEGMENT
        assert e.application_id == _HEX_APP
        assert e.dim == 4
        assert e.vector == (0.1, 0.2, 0.3, 0.4)
        assert e.text_hash == text_hash("hello world")
        assert e.text_preview == "hello world"
        assert e.id == entry_id_for_key(span.key())

    def test_minimal_uncoded_paragraph(self) -> None:
        span = IndexableSpan(
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE,
            application_id=None,
            paragraph_start_segment=2,
            paragraph_end_segment=4,
            anchor_start_word_id="s2w0",
            anchor_end_word_id="s4w3",
            text="hi",
        )
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=span,
            vector=[1.0, 0.0],
            model_name="bge-m3",
        )
        assert e.kind == EMBEDDING_KIND_UNCODED_PARAGRAPH
        assert e.application_id is None
        assert e.paragraph_start_segment == 2
        assert e.paragraph_end_segment == 4

    def test_text_preview_truncates(self) -> None:
        long = "x" * (MAX_TEXT_PREVIEW_LEN + 100)
        span = IndexableSpan(
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text=long,
        )
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=span,
            vector=[1.0],
            model_name="m",
        )
        assert len(e.text_preview) == MAX_TEXT_PREVIEW_LEN


class TestEmbeddingEntryValidate:
    def _span(self, **overrides) -> IndexableSpan:
        base = dict(
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w1",
            text="hi",
        )
        base.update(overrides)
        return IndexableSpan(**base)

    def test_rejects_bad_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.new(
                project_id="not-hex",
                span=self._span(),
                vector=[1.0],
                model_name="m",
            )

    def test_rejects_bad_kind(self) -> None:
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=self._span(),
            vector=[1.0],
            model_name="m",
        )
        e.kind = "bogus"
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_coded_segment_without_application_id(self) -> None:
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=self._span(),
            vector=[1.0],
            model_name="m",
        )
        e.application_id = None
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_uncoded_with_application_id(self) -> None:
        s = self._span(
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            application_id=None,
            paragraph_start_segment=0,
            paragraph_end_segment=0,
        )
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=s,
            vector=[1.0],
            model_name="m",
        )
        e.application_id = _HEX_APP
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_coded_segment_with_paragraph_indices(self) -> None:
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=self._span(),
            vector=[1.0],
            model_name="m",
        )
        e.paragraph_start_segment = 0
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_uncoded_without_paragraph_indices(self) -> None:
        s = self._span(
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            application_id=None,
            paragraph_start_segment=0,
            paragraph_end_segment=0,
        )
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=s,
            vector=[1.0],
            model_name="m",
        )
        e.paragraph_start_segment = None
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_anchor_start_after_end(self) -> None:
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=self._span(),
            vector=[1.0],
            model_name="m",
        )
        e.anchor_start_word_id = "s0w5"
        e.anchor_end_word_id = "s0w0"
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_dim_mismatch(self) -> None:
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=self._span(),
            vector=[1.0, 2.0, 3.0],
            model_name="m",
        )
        e.dim = 5
        with pytest.raises(ProjectValidationError):
            e.validate()

    def test_rejects_empty_vector(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.new(
                project_id=_HEX_PROJECT,
                span=self._span(),
                vector=[],
                model_name="m",
            )

    def test_rejects_nan_vector(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.new(
                project_id=_HEX_PROJECT,
                span=self._span(),
                vector=[float("nan")],
                model_name="m",
            )

    def test_rejects_inf_vector(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.new(
                project_id=_HEX_PROJECT,
                span=self._span(),
                vector=[float("inf")],
                model_name="m",
            )

    def test_rejects_too_long_model_name(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.new(
                project_id=_HEX_PROJECT,
                span=self._span(),
                vector=[1.0],
                model_name="x" * (MAX_MODEL_NAME_LEN + 1),
            )


class TestEmbeddingEntryRoundTrip:
    def test_round_trip(self) -> None:
        span = IndexableSpan(
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text="hi",
        )
        e = EmbeddingEntry.new(
            project_id=_HEX_PROJECT,
            span=span,
            vector=[0.1, 0.5, -0.3, 0.0],
            model_name="bge-m3",
        )
        d = json.loads(json.dumps(e.to_dict()))
        e2 = EmbeddingEntry.from_dict(d)
        assert e2 == e

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.from_dict([])  # type: ignore[arg-type]

    def test_from_dict_rejects_missing_fields(self) -> None:
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.from_dict({"id": "x"})

    def test_from_dict_rejects_non_numeric_vector(self) -> None:
        bad = {
            "id": "0" * 12,
            "project_id": _HEX_PROJECT,
            "source_id": _HEX_SOURCE,
            "kind": EMBEDDING_KIND_CODED_SEGMENT,
            "application_id": _HEX_APP,
            "paragraph_start_segment": None,
            "paragraph_end_segment": None,
            "anchor_start_word_id": "s0w0",
            "anchor_end_word_id": "s0w0",
            "text_preview": "hi",
            "text_hash": "0" * 64,
            "vector": ["nope", "nope"],
            "model_name": "m",
            "dim": 2,
        }
        with pytest.raises(ProjectValidationError):
            EmbeddingEntry.from_dict(bad)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def _entry(self, project_id: str) -> EmbeddingEntry:
        span = IndexableSpan(
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text="hello",
        )
        return EmbeddingEntry.new(
            project_id=project_id,
            span=span,
            vector=[1.0, 0.5, -0.5, 0.0],
            model_name="bge-m3",
        )

    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = self._entry(proj.id)
        path = save_embedding_entry(tmp_path, e)
        assert path.exists()
        assert path.parent == embeddings_dir(tmp_path, proj.id)
        loaded = load_embedding_entry(tmp_path, proj.id, e.id)
        assert loaded == e

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        e = self._entry(_HEX_PROJECT)
        with pytest.raises(FileNotFoundError):
            save_embedding_entry(tmp_path, e)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_embedding_entry(tmp_path, proj.id, "0" * 12)

    def test_state_path_rejects_bad_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            embedding_state_path(tmp_path, proj.id, "not-hex")

    def test_list_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert list_embedding_entries(tmp_path, proj.id) == []

    def test_list_skips_corrupt_files(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = self._entry(proj.id)
        save_embedding_entry(tmp_path, e)
        # Drop a corrupt file alongside.
        bad = embeddings_dir(tmp_path, proj.id) / ("a" * 12 + ".json")
        bad.write_text("{broken")
        # Drop a file with a non-hex name.
        unrelated = embeddings_dir(tmp_path, proj.id) / "notes.json"
        unrelated.write_text("{}")
        # Drop a tmp file (rename hadn't completed).
        tmpf = embeddings_dir(tmp_path, proj.id) / ("b" * 12 + ".json.tmp")
        tmpf.write_text("{}")
        rows = list_embedding_entries(tmp_path, proj.id)
        assert [r.id for r in rows] == [e.id]

    def test_list_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e1 = self._entry(proj.id)
        save_embedding_entry(tmp_path, e1)
        # Build a paragraph entry with a different source id.
        span = IndexableSpan(
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE2,
            application_id=None,
            paragraph_start_segment=0,
            paragraph_end_segment=0,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text="x",
        )
        e2 = EmbeddingEntry.new(
            project_id=proj.id,
            span=span,
            vector=[0.0, 1.0, 0.0, 0.0],
            model_name="bge-m3",
        )
        save_embedding_entry(tmp_path, e2)
        # All
        all_e = list_embedding_entries(tmp_path, proj.id)
        assert len(all_e) == 2
        # Filter by kind
        coded = list_embedding_entries(
            tmp_path, proj.id, kind=EMBEDDING_KIND_CODED_SEGMENT
        )
        assert {x.id for x in coded} == {e1.id}
        # Filter by source
        from_s2 = list_embedding_entries(
            tmp_path, proj.id, source_id=_HEX_SOURCE2
        )
        assert {x.id for x in from_s2} == {e2.id}
        # Combined filters
        combo = list_embedding_entries(
            tmp_path,
            proj.id,
            source_id=_HEX_SOURCE,
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
        )
        assert combo == []

    def test_list_rejects_bad_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_embedding_entries(tmp_path, proj.id, kind="bogus")
        with pytest.raises(ProjectValidationError):
            list_embedding_entries(tmp_path, proj.id, source_id="not-hex")

    def test_delete_returns_false_when_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert (
            delete_embedding_entry(tmp_path, proj.id, "0" * 12) is False
        )

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = self._entry(proj.id)
        save_embedding_entry(tmp_path, e)
        assert (
            delete_embedding_entry(tmp_path, proj.id, e.id) is True
        )
        assert not embedding_state_path(tmp_path, proj.id, e.id).exists()

    def test_clear_index_empties_directory(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        e = self._entry(proj.id)
        save_embedding_entry(tmp_path, e)
        # No-op for missing dir
        empty_proj = Project.new(name="empty")
        save_project(tmp_path, empty_proj)
        assert clear_embedding_index(tmp_path, empty_proj.id) == 0
        # Non-empty
        n = clear_embedding_index(tmp_path, proj.id)
        assert n == 1
        assert list_embedding_entries(tmp_path, proj.id) == []


# --------------------------------------------------------------------------- #
# refresh_embedding_index
# --------------------------------------------------------------------------- #


class TestRefreshIndex:
    def test_creates_entries_on_first_run(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["one", "two"]), ("B", ["three"])])
        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="bge-m3",
        )
        # 2 paragraphs become 2 added entries.
        assert result.added_count == 2
        assert result.updated_count == 0
        assert result.removed_count == 0
        assert result.unchanged_count == 0
        rows = list_embedding_entries(tmp_path, proj.id)
        assert len(rows) == 2
        assert all(e.kind == EMBEDDING_KIND_UNCODED_PARAGRAPH for e in rows)

    def test_unchanged_when_run_twice(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["x"])])
        embed = _const_embed()
        first = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=embed,
            model_name="m1",
        )
        assert first.added_count == 1
        # Second call: same inputs, same model — nothing should change.
        second = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=embed,
            model_name="m1",
        )
        assert second.added_count == 0
        assert second.updated_count == 0
        assert second.removed_count == 0
        assert second.unchanged_count == 1

    def test_re_embeds_when_model_changes(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["x"])])
        refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m2",
        )
        assert result.updated_count == 1
        assert result.added_count == 0
        rows = list_embedding_entries(tmp_path, proj.id)
        assert all(r.model_name == "m2" for r in rows)

    def test_re_embeds_when_text_changes(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs1 = _segments([("A", ["one"])])
        refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs1},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        # Edit the transcript: same paragraph, different words.
        segs2 = _segments([("A", ["different"])])
        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs2},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        assert result.updated_count == 1
        assert result.added_count == 0
        rows = list_embedding_entries(tmp_path, proj.id)
        assert rows[0].text_preview == "different"

    def test_removes_orphans(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["one"]), ("B", ["two"])])
        first = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        assert first.added_count == 2
        # Now drop the source entirely.
        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        assert result.removed_count == 2
        assert list_embedding_entries(tmp_path, proj.id) == []

    def test_application_addition_swaps_paragraph_for_coded(
        self, tmp_path: Path
    ) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["alpha", "beta"])])
        refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        rows = list_embedding_entries(tmp_path, proj.id)
        assert {r.kind for r in rows} == {EMBEDDING_KIND_UNCODED_PARAGRAPH}

        # Add an application: paragraph drops out, coded_segment appears.
        app = _make_app(project_id=proj.id, start="s0w0", end="s0w1")
        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[app],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        assert result.added_count == 1
        assert result.removed_count == 1
        assert result.updated_count == 0
        rows = list_embedding_entries(tmp_path, proj.id)
        assert {r.kind for r in rows} == {EMBEDDING_KIND_CODED_SEGMENT}

    def test_batch_size_chunks_calls(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments(
            [("A", [f"w{i}"]) for i in range(7)]  # 7 single-word turns
        )
        # Each segment has its own speaker → its own paragraph → 7
        # separate paragraphs (because every segment shares speaker "A"
        # they actually merge into one paragraph). Make speakers
        # alternate so we get 7 paragraphs.
        segs = _segments(
            [(f"S{i}", [f"w{i}"]) for i in range(7)]
        )
        calls: list[int] = []

        def chunked_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            calls.append(len(texts))
            return _const_embed()(texts)

        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=chunked_embed,
            model_name="m1",
            batch_size=3,
        )
        assert result.added_count == 7
        # Three calls of 3, 3, 1.
        assert calls == [3, 3, 1]

    def test_no_embed_call_when_nothing_to_do(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["x"])])
        refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        calls: list[int] = []

        def counting_embed(
            texts: Sequence[str],
        ) -> Sequence[Sequence[float]]:
            calls.append(len(texts))
            return _const_embed()(texts)

        result = refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=counting_embed,
            model_name="m1",
        )
        assert calls == []
        assert result.unchanged_count == 1

    def test_rejects_invalid_project_id(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectValidationError):
            refresh_embedding_index(
                projects_root=tmp_path,
                project_id="not-hex",
                applications=[],
                segments_by_source={},
                embed_fn=_const_embed(),
                model_name="m1",
            )

    def test_rejects_missing_project_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            refresh_embedding_index(
                projects_root=tmp_path,
                project_id="0" * 12,
                applications=[],
                segments_by_source={},
                embed_fn=_const_embed(),
                model_name="m1",
            )

    def test_rejects_zero_batch_size(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            refresh_embedding_index(
                projects_root=tmp_path,
                project_id=proj.id,
                applications=[],
                segments_by_source={},
                embed_fn=_const_embed(),
                model_name="m1",
                batch_size=0,
            )

    def test_rejects_empty_model_name(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            refresh_embedding_index(
                projects_root=tmp_path,
                project_id=proj.id,
                applications=[],
                segments_by_source={},
                embed_fn=_const_embed(),
                model_name="",
            )

    def test_embed_fn_short_response_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["x"]), ("B", ["y"])])

        def bad_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return [[0.1, 0.2, 0.3, 0.4]]  # wrong count

        with pytest.raises(ProjectValidationError):
            refresh_embedding_index(
                projects_root=tmp_path,
                project_id=proj.id,
                applications=[],
                segments_by_source={_HEX_SOURCE: segs},
                embed_fn=bad_embed,
                model_name="m1",
            )


# --------------------------------------------------------------------------- #
# Cosine similarity + search
# --------------------------------------------------------------------------- #


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ProjectValidationError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_empty_vectors_return_zero(self) -> None:
        assert cosine_similarity([], []) == 0.0


class TestSearchSimilar:
    def _seed(self, tmp_path: Path) -> Project:
        proj = _saved_project(tmp_path)
        segs = _segments([("A", ["alpha"]), ("B", ["beta"])])
        refresh_embedding_index(
            projects_root=tmp_path,
            project_id=proj.id,
            applications=[],
            segments_by_source={_HEX_SOURCE: segs},
            embed_fn=_const_embed(),
            model_name="m1",
        )
        return proj

    def test_returns_top_k_sorted(self, tmp_path: Path) -> None:
        proj = self._seed(tmp_path)
        rows = list_embedding_entries(tmp_path, proj.id)
        # Use one entry's vector as query: it should rank first with score 1.0.
        target = rows[0]
        result = search_similar(
            projects_root=tmp_path,
            project_id=proj.id,
            query_vector=target.vector,
            top_k=2,
        )
        assert len(result) <= 2
        assert result[0][1].id == target.id
        assert result[0][0] == pytest.approx(1.0)

    def test_filter_by_kind(self, tmp_path: Path) -> None:
        proj = self._seed(tmp_path)
        result = search_similar(
            projects_root=tmp_path,
            project_id=proj.id,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            top_k=10,
        )
        assert result == []

    def test_filter_by_source(self, tmp_path: Path) -> None:
        proj = self._seed(tmp_path)
        result = search_similar(
            projects_root=tmp_path,
            project_id=proj.id,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            source_id=_HEX_SOURCE2,
            top_k=10,
        )
        assert result == []

    def test_skips_dim_mismatch(self, tmp_path: Path) -> None:
        proj = self._seed(tmp_path)
        # All entries are dim 4; querying with dim 3 → no comparable entries.
        result = search_similar(
            projects_root=tmp_path,
            project_id=proj.id,
            query_vector=[1.0, 0.0, 0.0],
            top_k=10,
        )
        assert result == []

    def test_min_score_filter(self, tmp_path: Path) -> None:
        proj = self._seed(tmp_path)
        result = search_similar(
            projects_root=tmp_path,
            project_id=proj.id,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            min_score=2.0,  # impossible
            top_k=10,
        )
        assert result == []

    def test_rejects_zero_top_k(self, tmp_path: Path) -> None:
        proj = self._seed(tmp_path)
        with pytest.raises(ProjectValidationError):
            search_similar(
                projects_root=tmp_path,
                project_id=proj.id,
                query_vector=[1.0],
                top_k=0,
            )


# --------------------------------------------------------------------------- #
# RefreshResult
# --------------------------------------------------------------------------- #


class TestRefreshResult:
    def test_defaults_empty(self) -> None:
        r = RefreshResult()
        assert r.added_count == 0
        assert r.updated_count == 0
        assert r.removed_count == 0
        assert r.unchanged_count == 0

    def test_counts_from_keys(self) -> None:
        r = RefreshResult(
            added=(("k1",),),
            updated=(("k2",), ("k3",)),
            removed=(("k4",),),
            unchanged=(("k5",), ("k6",), ("k7",)),
        )
        assert r.added_count == 1
        assert r.updated_count == 2
        assert r.removed_count == 1
        assert r.unchanged_count == 3


# --------------------------------------------------------------------------- #
# Backend adapter
# --------------------------------------------------------------------------- #


class TestMakeEmbedFnFromBackend:
    def test_calls_backend_embed(self) -> None:
        from scribe.ai_backend import (
            BackendConfig,
            EmbeddingResponse,
            ModelBackend,
            PROVIDER_OLLAMA,
            register_backend,
        )

        calls: list[dict[str, Any]] = []

        class FakeBackend(ModelBackend):
            name = PROVIDER_OLLAMA

            def health_check(self, *a, **k):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def list_models(self, *a, **k):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def generate(self, *a, **k):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def embed(self, config, request, *, transport):  # type: ignore[no-untyped-def]
                calls.append(
                    {
                        "model": request.model,
                        "inputs": tuple(request.inputs),
                    }
                )
                return EmbeddingResponse(
                    vectors=tuple((0.1, 0.2) for _ in request.inputs),
                    model=request.model,
                    provider=self.name,
                )

        cfg = BackendConfig.new()
        backend = FakeBackend()
        embed_fn = make_embed_fn_from_backend(
            cfg, "bge-m3", backend=backend, transport=lambda *a, **k: None
        )
        out = embed_fn(["a", "b"])
        assert tuple(tuple(v) for v in out) == ((0.1, 0.2), (0.1, 0.2))
        assert calls == [{"model": "bge-m3", "inputs": ("a", "b")}]

    def test_empty_input_short_circuits(self) -> None:
        from scribe.ai_backend import BackendConfig

        cfg = BackendConfig.new()

        # No backend needed because embed_fn returns () without calling.
        class NeverCalled:
            name = "ollama"

            def embed(self, *a, **k):  # type: ignore[no-untyped-def]
                raise AssertionError("embed should not be called for empty inputs")

        embed_fn = make_embed_fn_from_backend(
            cfg,
            "bge-m3",
            backend=NeverCalled(),  # type: ignore[arg-type]
            transport=lambda *a, **k: None,
        )
        assert embed_fn([]) == ()

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ProjectValidationError):
            make_embed_fn_from_backend("not-a-config", "m")  # type: ignore[arg-type]

    def test_rejects_empty_model(self) -> None:
        from scribe.ai_backend import BackendConfig

        with pytest.raises(ProjectValidationError):
            make_embed_fn_from_backend(BackendConfig.new(), "")
