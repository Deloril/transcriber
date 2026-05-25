"""Tests for scribe.quote_similarity (F8.5).

Covers:

  * QuoteMatch dataclass: validate, round-trip, kind invariants.
  * QuoteSearch entity: validate, round-trip, query-kind invariants,
    apply_update.
  * find_similar_quotes:
      - text mode + application mode (seed re-use without embed call);
      - top_k / min_score truncation;
      - kind / source / code-id filters;
      - exclude_source_ids / exclude_code_ids;
      - exclude_seed = True / False;
      - dim mismatch silently skipped;
      - canonicalisation;
      - cross-model seed protection.
  * Persistence: save / load / list / delete + filters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scribe.applications import Application
from scribe.codes import Code
from scribe.embedding_index import (
    EMBEDDING_KIND_CODED_SEGMENT,
    EMBEDDING_KIND_UNCODED_PARAGRAPH,
    EmbeddingEntry,
    IndexableSpan,
    save_embedding_entry,
)
from scribe.projects import (
    Project,
    ProjectValidationError,
    project_dir,
    save_project,
)
from scribe.quote_similarity import (
    DEFAULT_EXCLUDE_SEED,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    MAX_FILTER_LIST,
    MAX_MATCHES_PERSISTED,
    MAX_NOTES_LEN,
    MAX_QUERY_TEXT_LEN,
    MAX_TEXT_PREVIEW_LEN,
    QUERY_KIND_APPLICATION,
    QUERY_KIND_TEXT,
    QUERY_KINDS,
    QUOTE_SEARCH_ID_RE,
    QUOTE_SEARCHES_DIRNAME,
    QuoteMatch,
    QuoteSearch,
    delete_quote_search,
    find_similar_quotes,
    list_quote_searches,
    load_quote_search,
    new_quote_search_id,
    quote_search_state_path,
    quote_searches_dir,
    save_quote_search,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_HEX_PROJECT = "aaaaaaaaaaaa"
_HEX_SOURCE = "bbbbbbbbbbbb"
_HEX_SOURCE_2 = "cccccccccccc"
_HEX_SOURCE_3 = "dddddddddddd"
_HEX_CODER = "0123456789ab"
_HEX_VERSION = "fedcba987654"
_HEX_CODE_A = "1111aaaa1111"
_HEX_CODE_B = "2222bbbb2222"
_HEX_APP_A = "aa00aa00aa00"
_HEX_APP_B = "bb11bb11bb11"
_HEX_APP_C = "cc22cc22cc22"
_HEX_APP_D = "dd33dd33dd33"


def _saved_project(tmp_path: Path, *, name: str = "P") -> Project:
    p = Project.new(name=name)
    save_project(tmp_path, p)
    return p


def _make_app(
    *,
    project_id: str,
    code_id: str,
    application_id: str,
    source_id: str = _HEX_SOURCE,
    coder_id: str = _HEX_CODER,
    start: str = "s0w0",
    end: str = "s0w0",
) -> Application:
    return Application.new(
        project_id=project_id,
        code_id=code_id,
        source_id=source_id,
        coder_id=coder_id,
        anchor_start_word_id=start,
        anchor_end_word_id=end,
        definition_version_id_at_apply=_HEX_VERSION,
        application_id=application_id,
    )


def _seg_entry(
    *,
    project_id: str,
    application_id: str,
    source_id: str = _HEX_SOURCE,
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
    model_name: str = "test-embed",
    text: str = "example text",
) -> EmbeddingEntry:
    span = IndexableSpan(
        kind=EMBEDDING_KIND_CODED_SEGMENT,
        source_id=source_id,
        application_id=application_id,
        paragraph_start_segment=None,
        paragraph_end_segment=None,
        anchor_start_word_id="s0w0",
        anchor_end_word_id="s0w0",
        text=text,
    )
    return EmbeddingEntry.new(
        project_id=project_id,
        span=span,
        vector=vector,
        model_name=model_name,
    )


def _para_entry(
    *,
    project_id: str,
    source_id: str,
    paragraph_start_segment: int,
    paragraph_end_segment: int,
    vector: tuple[float, ...] = (0.5, 0.5, 0.0),
    model_name: str = "test-embed",
    text: str = "uncoded paragraph text",
) -> EmbeddingEntry:
    span = IndexableSpan(
        kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
        source_id=source_id,
        application_id=None,
        paragraph_start_segment=paragraph_start_segment,
        paragraph_end_segment=paragraph_end_segment,
        anchor_start_word_id=f"s{paragraph_start_segment}w0",
        anchor_end_word_id=f"s{paragraph_end_segment}w0",
        text=text,
    )
    return EmbeddingEntry.new(
        project_id=project_id,
        span=span,
        vector=vector,
        model_name=model_name,
    )


def _const_embed(vec: Sequence[float]):
    """Stub embed_fn that returns the same vector for every input."""

    def fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [tuple(float(x) for x in vec)] * len(texts)

    return fn


# --------------------------------------------------------------------------- #
# QuoteMatch
# --------------------------------------------------------------------------- #


class TestQuoteMatch:
    def _seg(self, **kw: Any) -> QuoteMatch:
        defaults: dict[str, Any] = dict(
            embedding_id="0123456789ab",
            kind=EMBEDDING_KIND_CODED_SEGMENT,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP_A,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text_preview="hi",
            score=0.5,
        )
        defaults.update(kw)
        return QuoteMatch(**defaults)

    def test_validate_segment_requires_app(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(application_id=None).validate()

    def test_validate_segment_rejects_paragraph_indices(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(paragraph_start_segment=0).validate()

    def test_validate_paragraph_requires_indices(self) -> None:
        m = QuoteMatch(
            embedding_id="0123456789ab",
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE,
            application_id=None,
            paragraph_start_segment=None,
            paragraph_end_segment=None,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text_preview="hi",
            score=0.5,
        )
        with pytest.raises(ProjectValidationError):
            m.validate()

    def test_validate_paragraph_rejects_app_id(self) -> None:
        m = QuoteMatch(
            embedding_id="0123456789ab",
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE,
            application_id=_HEX_APP_A,
            paragraph_start_segment=0,
            paragraph_end_segment=0,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s0w0",
            text_preview="hi",
            score=0.5,
        )
        with pytest.raises(ProjectValidationError):
            m.validate()

    def test_validate_score_out_of_range(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(score=1.5).validate()

    def test_validate_score_nan(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(score=float("nan")).validate()

    def test_validate_anchor_ordering(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(
                anchor_start_word_id="s2w0", anchor_end_word_id="s1w0"
            ).validate()

    def test_validate_unknown_kind(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(kind="bogus").validate()

    def test_round_trip(self) -> None:
        m = self._seg(code_id=_HEX_CODE_A, score=0.42)
        round_trip = QuoteMatch.from_dict(m.to_dict())
        assert round_trip == m

    def test_paragraph_round_trip(self) -> None:
        m = QuoteMatch(
            embedding_id="0123456789ab",
            kind=EMBEDDING_KIND_UNCODED_PARAGRAPH,
            source_id=_HEX_SOURCE,
            application_id=None,
            paragraph_start_segment=0,
            paragraph_end_segment=2,
            anchor_start_word_id="s0w0",
            anchor_end_word_id="s2w3",
            text_preview="hi",
            score=0.3,
        )
        round_trip = QuoteMatch.from_dict(m.to_dict())
        assert round_trip == m

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(ProjectValidationError):
            QuoteMatch.from_dict([])  # type: ignore[arg-type]

    def test_text_preview_length_capped(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._seg(text_preview="x" * (MAX_TEXT_PREVIEW_LEN + 1)).validate()

    def test_from_entry(self) -> None:
        e = _seg_entry(
            project_id=_HEX_PROJECT, application_id=_HEX_APP_A
        )
        m = QuoteMatch.from_entry(e, score=0.71, code_id=_HEX_CODE_A)
        assert m.embedding_id == e.id
        assert m.application_id == _HEX_APP_A
        assert m.kind == EMBEDDING_KIND_CODED_SEGMENT
        assert m.code_id == _HEX_CODE_A
        assert m.score == pytest.approx(0.71)


# --------------------------------------------------------------------------- #
# QuoteSearch
# --------------------------------------------------------------------------- #


class TestQuoteSearch:
    def _basic(self, **kw: Any) -> QuoteSearch:
        defaults: dict[str, Any] = dict(
            project_id=_HEX_PROJECT,
            query_kind=QUERY_KIND_TEXT,
            query_text="hello world",
        )
        defaults.update(kw)
        return QuoteSearch.new(**defaults)

    def test_minimal_text_mode(self) -> None:
        s = self._basic()
        assert s.query_kind == QUERY_KIND_TEXT
        assert s.top_k == DEFAULT_TOP_K
        assert s.exclude_seed is DEFAULT_EXCLUDE_SEED
        assert QUOTE_SEARCH_ID_RE.match(s.id)

    def test_application_mode_requires_app_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(query_kind=QUERY_KIND_APPLICATION)

    def test_application_mode_requires_source(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(
                query_kind=QUERY_KIND_APPLICATION,
                query_application_id=_HEX_APP_A,
            )

    def test_application_mode_ok(self) -> None:
        s = self._basic(
            query_kind=QUERY_KIND_APPLICATION,
            query_application_id=_HEX_APP_A,
            query_source_id=_HEX_SOURCE,
        )
        assert s.query_application_id == _HEX_APP_A
        assert s.query_source_id == _HEX_SOURCE

    def test_text_mode_rejects_app_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(query_application_id=_HEX_APP_A).validate()

    def test_unknown_query_kind(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(query_kind="bogus")

    def test_invalid_project_id(self) -> None:
        with pytest.raises(ProjectValidationError):
            QuoteSearch.new(
                project_id="not-hex",
                query_kind=QUERY_KIND_TEXT,
                query_text="x",
            )

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(top_k=0)

    def test_min_score_bounds(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(min_score=2.0)
        with pytest.raises(ProjectValidationError):
            self._basic(min_score=float("nan"))

    def test_query_text_length_limit(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(query_text="x" * (MAX_QUERY_TEXT_LEN + 1))

    def test_kind_filter_validated(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(kind_filter="bogus")

    def test_source_filter_validated(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(source_id_filter="not-hex")

    def test_exclude_source_ids_validated(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(exclude_source_ids=["not-hex"])

    def test_code_id_filter_validated(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(code_id_filter="not-hex")

    def test_exclude_code_ids_validated(self) -> None:
        with pytest.raises(ProjectValidationError):
            self._basic(exclude_code_ids=["not-hex"])

    def test_exclude_filter_size_capped(self) -> None:
        too_many = [f"{i:012x}" for i in range(MAX_FILTER_LIST + 1)]
        with pytest.raises(ProjectValidationError):
            self._basic(exclude_source_ids=too_many)

    def test_round_trip(self) -> None:
        s = self._basic(
            kind_filter=EMBEDDING_KIND_CODED_SEGMENT,
            source_id_filter=_HEX_SOURCE,
            exclude_source_ids=[_HEX_SOURCE_2],
            code_id_filter=_HEX_CODE_A,
            exclude_code_ids=[_HEX_CODE_B],
            exclude_seed=False,
            embedding_model="bge-m3",
            top_k=5,
            min_score=0.1,
            notes="audit me",
        )
        s.matches.append(
            QuoteMatch(
                embedding_id="0123456789ab",
                kind=EMBEDDING_KIND_CODED_SEGMENT,
                source_id=_HEX_SOURCE,
                application_id=_HEX_APP_A,
                paragraph_start_segment=None,
                paragraph_end_segment=None,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                text_preview="hi",
                score=0.9,
                code_id=_HEX_CODE_A,
            )
        )
        round_trip = QuoteSearch.from_dict(s.to_dict())
        assert round_trip == s

    def test_apply_update_notes(self) -> None:
        s = self._basic()
        s.apply_update(notes="post-hoc note")
        assert s.notes == "post-hoc note"
        assert s.modified_at >= s.created_at

    def test_apply_update_notes_too_long(self) -> None:
        s = self._basic()
        with pytest.raises(ProjectValidationError):
            s.apply_update(notes="x" * (MAX_NOTES_LEN + 1))

    def test_matches_capped_in_validation(self) -> None:
        s = self._basic()
        s.matches = [
            QuoteMatch(
                embedding_id="0123456789ab",
                kind=EMBEDDING_KIND_CODED_SEGMENT,
                source_id=_HEX_SOURCE,
                application_id=_HEX_APP_A,
                paragraph_start_segment=None,
                paragraph_end_segment=None,
                anchor_start_word_id="s0w0",
                anchor_end_word_id="s0w0",
                text_preview="hi",
                score=0.5,
            )
        ] * (MAX_MATCHES_PERSISTED + 1)
        with pytest.raises(ProjectValidationError):
            s.validate()


# --------------------------------------------------------------------------- #
# new_quote_search_id
# --------------------------------------------------------------------------- #


class TestNewQuoteSearchId:
    def test_format(self) -> None:
        for _ in range(8):
            sid = new_quote_search_id()
            assert QUOTE_SEARCH_ID_RE.match(sid), sid

    def test_unique(self) -> None:
        ids = {new_quote_search_id() for _ in range(50)}
        assert len(ids) == 50


# --------------------------------------------------------------------------- #
# find_similar_quotes
# --------------------------------------------------------------------------- #


class TestFindSimilarQuotesText:
    def test_text_mode_basic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # Two segment entries: one aligned with query, one orthogonal.
        e_close = _seg_entry(
            project_id=proj.id,
            application_id=_HEX_APP_A,
            vector=(1.0, 0.0, 0.0),
        )
        e_far = _seg_entry(
            project_id=proj.id,
            application_id=_HEX_APP_B,
            vector=(0.0, 1.0, 0.0),
        )
        save_embedding_entry(tmp_path, e_close)
        save_embedding_entry(tmp_path, e_far)

        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="alpha",
            top_k=10,
        )
        assert result.query_kind == QUERY_KIND_TEXT
        assert len(result.matches) == 2
        # Closer one ranks first.
        assert result.matches[0].application_id == _HEX_APP_A
        assert result.matches[0].score == pytest.approx(1.0)
        assert result.matches[1].application_id == _HEX_APP_B

    def test_top_k_truncates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        for i, app_id in enumerate([_HEX_APP_A, _HEX_APP_B, _HEX_APP_C]):
            save_embedding_entry(
                tmp_path,
                _seg_entry(
                    project_id=proj.id,
                    application_id=app_id,
                    vector=(1.0 - i * 0.1, 0.0, 0.0),
                ),
            )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            top_k=2,
        )
        assert len(result.matches) == 2

    def test_min_score_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                vector=(0.0, 1.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            min_score=0.5,
        )
        # Only the aligned vector clears 0.5.
        assert len(result.matches) == 1
        assert result.matches[0].application_id == _HEX_APP_A

    def test_dim_mismatch_skipped(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                vector=(1.0, 0.0, 0.0, 0.0, 0.0),  # 5-dim
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),  # 3-dim
            query_text="hello",
        )
        assert len(result.matches) == 1
        assert result.matches[0].application_id == _HEX_APP_A

    def test_kind_filter_text(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _para_entry(
                project_id=proj.id,
                source_id=_HEX_SOURCE,
                paragraph_start_segment=0,
                paragraph_end_segment=0,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            kind_filter=EMBEDDING_KIND_UNCODED_PARAGRAPH,
        )
        assert len(result.matches) == 1
        assert result.matches[0].kind == EMBEDDING_KIND_UNCODED_PARAGRAPH

    def test_source_id_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                source_id=_HEX_SOURCE,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                source_id=_HEX_SOURCE_2,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            source_id_filter=_HEX_SOURCE_2,
        )
        assert len(result.matches) == 1
        assert result.matches[0].source_id == _HEX_SOURCE_2

    def test_exclude_source_ids(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                source_id=_HEX_SOURCE,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                source_id=_HEX_SOURCE_2,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            exclude_source_ids=[_HEX_SOURCE],
        )
        sources = {m.source_id for m in result.matches}
        assert sources == {_HEX_SOURCE_2}

    def test_code_id_filter(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        # Application A → code A, application B → code B.
        apps = [
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_A,
                application_id=_HEX_APP_A,
            ),
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_B,
                application_id=_HEX_APP_B,
            ),
        ]
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            applications=apps,
            code_id_filter=_HEX_CODE_A,
        )
        assert len(result.matches) == 1
        assert result.matches[0].application_id == _HEX_APP_A
        assert result.matches[0].code_id == _HEX_CODE_A

    def test_code_id_filter_drops_paragraphs(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _para_entry(
                project_id=proj.id,
                source_id=_HEX_SOURCE,
                paragraph_start_segment=0,
                paragraph_end_segment=0,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        apps = [
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_A,
                application_id=_HEX_APP_A,
            )
        ]
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            applications=apps,
            code_id_filter=_HEX_CODE_A,
        )
        # Paragraph should drop out because we filtered by code.
        kinds = {m.kind for m in result.matches}
        assert kinds == {EMBEDDING_KIND_CODED_SEGMENT}

    def test_exclude_code_ids(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        apps = [
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_A,
                application_id=_HEX_APP_A,
            ),
            _make_app(
                project_id=proj.id,
                code_id=_HEX_CODE_B,
                application_id=_HEX_APP_B,
            ),
        ]
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            applications=apps,
            exclude_code_ids=[_HEX_CODE_B],
        )
        assert {m.application_id for m in result.matches} == {_HEX_APP_A}

    def test_canonicalises_query_text(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="   hello   world   ",
        )
        assert result.query_text == "hello world"

    def test_empty_query_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            find_similar_quotes(
                projects_root=tmp_path,
                project_id=proj.id,
                embed_fn=_const_embed((1.0, 0.0, 0.0)),
                query_text="   ",
            )

    def test_top_k_validated(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            find_similar_quotes(
                projects_root=tmp_path,
                project_id=proj.id,
                embed_fn=_const_embed((1.0, 0.0, 0.0)),
                query_text="hi",
                top_k=0,
            )

    def test_min_score_validated(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            find_similar_quotes(
                projects_root=tmp_path,
                project_id=proj.id,
                embed_fn=_const_embed((1.0, 0.0, 0.0)),
                query_text="hi",
                min_score=2.0,
            )


class TestFindSimilarQuotesApplicationMode:
    def test_seed_reuses_index_no_embed_call(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        seed = _seg_entry(
            project_id=proj.id,
            application_id=_HEX_APP_A,
            vector=(1.0, 0.0, 0.0),
        )
        save_embedding_entry(tmp_path, seed)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                vector=(1.0, 0.0, 0.0),
            ),
        )

        calls = {"n": 0}

        def fail_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            calls["n"] += 1
            raise AssertionError("embed_fn should not be called when seed exists")

        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=fail_embed,
            query_application_id=_HEX_APP_A,
            query_source_id=_HEX_SOURCE,
        )
        assert calls["n"] == 0
        assert result.query_kind == QUERY_KIND_APPLICATION
        # Seed itself excluded by default.
        assert all(m.application_id != _HEX_APP_A for m in result.matches)
        assert any(m.application_id == _HEX_APP_B for m in result.matches)

    def test_exclude_seed_false_includes_self(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_application_id=_HEX_APP_A,
            query_source_id=_HEX_SOURCE,
            exclude_seed=False,
        )
        assert len(result.matches) == 1
        assert result.matches[0].application_id == _HEX_APP_A

    def test_seed_missing_falls_back_to_embed(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # No entry for app A — seed lookup misses.
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_B,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        result = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="seed text",
            query_application_id=_HEX_APP_A,
            query_source_id=_HEX_SOURCE,
        )
        # Returns matches from the index using the embedded query.
        assert any(m.application_id == _HEX_APP_B for m in result.matches)

    def test_invalid_application_id(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            find_similar_quotes(
                projects_root=tmp_path,
                project_id=proj.id,
                embed_fn=_const_embed((1.0, 0.0, 0.0)),
                query_application_id="not-hex",
                query_source_id=_HEX_SOURCE,
            )

    def test_application_mode_requires_source(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            find_similar_quotes(
                projects_root=tmp_path,
                project_id=proj.id,
                embed_fn=_const_embed((1.0, 0.0, 0.0)),
                query_application_id=_HEX_APP_A,
            )

    def test_seed_model_mismatch_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
                model_name="bge-m3",
            ),
        )
        with pytest.raises(ProjectValidationError):
            find_similar_quotes(
                projects_root=tmp_path,
                project_id=proj.id,
                embed_fn=_const_embed((1.0, 0.0, 0.0)),
                query_application_id=_HEX_APP_A,
                query_source_id=_HEX_SOURCE,
                embedding_model="other-model",
            )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = QuoteSearch.new(
            project_id=proj.id,
            query_kind=QUERY_KIND_TEXT,
            query_text="hello",
        )
        path = save_quote_search(tmp_path, s)
        assert path.exists()
        assert (
            quote_searches_dir(tmp_path, proj.id) / f"{s.id}.json"
        ).exists()
        round_trip = load_quote_search(tmp_path, proj.id, s.id)
        assert round_trip == s

    def test_save_atomic(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = QuoteSearch.new(
            project_id=proj.id,
            query_kind=QUERY_KIND_TEXT,
            query_text="hello",
        )
        save_quote_search(tmp_path, s)
        # No leftover .json.tmp files.
        sd = quote_searches_dir(tmp_path, proj.id)
        leftovers = [p for p in sd.iterdir() if p.name.endswith(".json.tmp")]
        assert leftovers == []

    def test_save_requires_project_dir(self, tmp_path: Path) -> None:
        s = QuoteSearch.new(
            project_id=_HEX_PROJECT,
            query_kind=QUERY_KIND_TEXT,
            query_text="hello",
        )
        # No saved project — directory doesn't exist.
        with pytest.raises(FileNotFoundError):
            save_quote_search(tmp_path, s)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_quote_search(tmp_path, proj.id, "0123456789ab")

    def test_state_path_validates(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            quote_search_state_path(tmp_path, proj.id, "not-hex")

    def test_list_filters(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        # Make a text-mode and an application-mode search, distinct sources.
        # Pre-populate index for the application-mode search.
        save_embedding_entry(
            tmp_path,
            _seg_entry(
                project_id=proj.id,
                application_id=_HEX_APP_A,
                vector=(1.0, 0.0, 0.0),
            ),
        )
        s_text = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_text="hello",
            query_source_id=_HEX_SOURCE,
        )
        save_quote_search(tmp_path, s_text)
        s_app = find_similar_quotes(
            projects_root=tmp_path,
            project_id=proj.id,
            embed_fn=_const_embed((1.0, 0.0, 0.0)),
            query_application_id=_HEX_APP_A,
            query_source_id=_HEX_SOURCE,
        )
        # Force a different stored source so the per-source filter can
        # discriminate between the two records.
        s_app.query_source_id = _HEX_SOURCE_2
        s_app.modified_at = s_app.created_at
        s_app.validate()
        save_quote_search(tmp_path, s_app)

        all_searches = list_quote_searches(tmp_path, proj.id)
        assert len(all_searches) == 2

        text_only = list_quote_searches(
            tmp_path, proj.id, query_kind=QUERY_KIND_TEXT
        )
        assert len(text_only) == 1
        assert text_only[0].id == s_text.id

        app_only = list_quote_searches(
            tmp_path, proj.id, query_kind=QUERY_KIND_APPLICATION
        )
        assert len(app_only) == 1
        assert app_only[0].id == s_app.id

        by_source = list_quote_searches(
            tmp_path, proj.id, query_source_id=_HEX_SOURCE
        )
        assert len(by_source) == 1
        assert by_source[0].id == s_text.id

    def test_list_skips_unparseable(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        sd = quote_searches_dir(tmp_path, proj.id)
        sd.mkdir(parents=True, exist_ok=True)
        # Garbled JSON.
        (sd / "0123456789ab.json").write_text("not-json")
        # Non-id stem.
        (sd / "not-hex.json").write_text("{}")
        # Tmp file.
        (sd / "abcdef012345.json.tmp").write_text("{}")
        # Valid one.
        s = QuoteSearch.new(
            project_id=proj.id,
            query_kind=QUERY_KIND_TEXT,
            query_text="hello",
        )
        save_quote_search(tmp_path, s)
        out = list_quote_searches(tmp_path, proj.id)
        assert len(out) == 1
        assert out[0].id == s.id

    def test_list_invalid_kind_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_quote_searches(tmp_path, proj.id, query_kind="bogus")

    def test_list_invalid_source_raises(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        with pytest.raises(ProjectValidationError):
            list_quote_searches(tmp_path, proj.id, query_source_id="not-hex")

    def test_list_empty_when_dir_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        out = list_quote_searches(tmp_path, proj.id)
        assert out == []

    def test_delete(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        s = QuoteSearch.new(
            project_id=proj.id,
            query_kind=QUERY_KIND_TEXT,
            query_text="hello",
        )
        save_quote_search(tmp_path, s)
        assert delete_quote_search(tmp_path, proj.id, s.id) is True
        assert delete_quote_search(tmp_path, proj.id, s.id) is False

    def test_delete_missing(self, tmp_path: Path) -> None:
        proj = _saved_project(tmp_path)
        assert delete_quote_search(tmp_path, proj.id, "0123456789ab") is False
