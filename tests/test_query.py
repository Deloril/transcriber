"""Tests for scribe.query (F3.5).

Query builder: code filter + source filter + participant attribute
filter + speaker filter + boolean code combinator + proximity (within
span / paragraph / source).

The module is stand-alone (no FastAPI, no engine imports), so all
tests stay in pure Python.
"""

from __future__ import annotations

import pytest

from scribe.codes import new_code_id
from scribe.participants import Participant
from scribe.projects import ProjectValidationError
from scribe.query import (
    CODE_EXPR_OPS,
    MAX_CODE_EXPR_DEPTH,
    MAX_LIST_ITEMS,
    MAX_PREDICATES,
    PREDICATE_OPERATORS,
    PROXIMITY_SCOPES,
    AttributePredicate,
    CodeExpr,
    CodeFilter,
    ParticipantFilter,
    ProximityFilter,
    Query,
    QueryValidationError,
    SourceFilter,
    SpeakerFilter,
    applications_for_query,
    evaluate_code_expr,
    filter_participants,
    filter_sources,
    predicate_matches,
    predicates_all_match,
)
from scribe.sources import Source
from scribe.speaker_map import SpeakerEntry, SpeakerMap


# --------------------------------------------------------------------------- #
# Tiny helpers
# --------------------------------------------------------------------------- #


PID = "deadbeef0001"
PID_B = "deadbeef0002"


def _hex(seed: int) -> str:
    return f"{seed:012x}"


def _src(
    sid: str,
    *,
    project_id: str = PID,
    name: str | None = None,
    language: str = "en",
    recording_date: str = "",
    custom: dict[str, str] | None = None,
) -> Source:
    return Source.new(
        project_id=project_id,
        name=name or f"src-{sid}",
        source_id=sid,
        language=language,
        recording_date=recording_date,
        custom_attributes=custom or {},
    )


def _participant(
    pid: str,
    *,
    project_id: str = PID,
    name: str | None = None,
    demographics: dict[str, str] | None = None,
) -> Participant:
    return Participant.new(
        project_id=project_id,
        name=name or f"P-{pid}",
        participant_id=pid,
        demographics=demographics or {},
    )


def _app(
    *,
    code_id: str,
    source_id: str,
    speaker: str = "",
    start: float | None = None,
    end: float | None = None,
    participant_id: str | None = None,
) -> dict:
    d: dict = {"code_id": code_id, "source_id": source_id}
    if speaker:
        d["speaker"] = speaker
    if start is not None:
        d["start"] = start
    if end is not None:
        d["end"] = end
    if participant_id is not None:
        d["participant_id"] = participant_id
    return d


# --------------------------------------------------------------------------- #
# AttributePredicate
# --------------------------------------------------------------------------- #


class TestAttributePredicate:
    def test_eq_match_and_miss(self):
        p = AttributePredicate(key="role", op="eq", value="nurse")
        assert p.matches({"role": "nurse"})
        assert not p.matches({"role": "doctor"})
        assert not p.matches({})

    def test_ne(self):
        p = AttributePredicate(key="role", op="ne", value="nurse")
        p.validate()
        assert p.matches({"role": "doctor"})
        assert not p.matches({"role": "nurse"})
        # Missing field → ne of an absent value returns False, mirroring
        # SQL's NULL semantics: nothing equals NULL, nothing is unequal.
        assert not p.matches({})

    def test_in_and_not_in(self):
        p_in = AttributePredicate(key="site", op="in", value=["A", "B"])
        p_in.validate()
        assert p_in.matches({"site": "A"})
        assert not p_in.matches({"site": "C"})

        p_out = AttributePredicate(key="site", op="not_in", value=["A", "B"])
        p_out.validate()
        assert p_out.matches({"site": "C"})
        assert not p_out.matches({"site": "A"})

    def test_contains_starts_ends(self):
        p_c = AttributePredicate(key="name", op="contains", value="ell")
        p_c.validate()
        assert p_c.matches({"name": "Hello"})
        assert not p_c.matches({"name": "World"})

        p_s = AttributePredicate(key="name", op="starts_with", value="He")
        p_s.validate()
        assert p_s.matches({"name": "Hello"})
        assert not p_s.matches({"name": "World"})

        p_e = AttributePredicate(key="name", op="ends_with", value="llo")
        p_e.validate()
        assert p_e.matches({"name": "Hello"})
        assert not p_e.matches({"name": "World"})

    def test_numeric_ops(self):
        p = AttributePredicate(key="age", op="lt", value=40)
        p.validate()
        assert p.matches({"age": "33"})
        assert p.matches({"age": 33})
        assert not p.matches({"age": "41"})
        # Non-numeric → False, never raises.
        assert not p.matches({"age": "many"})

        for op, ok, fail in (
            ("le", "40", "41"),
            ("gt", "41", "40"),
            ("ge", "40", "39"),
        ):
            p = AttributePredicate(key="age", op=op, value=40)
            p.validate()
            assert p.matches({"age": ok}), op
            assert not p.matches({"age": fail}), op

    def test_exists_missing(self):
        p_e = AttributePredicate(key="role", op="exists")
        p_e.validate()
        assert p_e.matches({"role": "nurse"})
        assert not p_e.matches({})
        # Empty string is treated as missing (matches the rest of the
        # codebase's "empty == not set" convention).
        assert not p_e.matches({"role": ""})

        p_m = AttributePredicate(key="role", op="missing")
        p_m.validate()
        assert p_m.matches({})
        assert p_m.matches({"role": ""})
        assert not p_m.matches({"role": "nurse"})

    def test_validate_rejects_unknown_op(self):
        p = AttributePredicate(key="role", op="regex", value="x")
        with pytest.raises(QueryValidationError):
            p.validate()

    def test_validate_rejects_empty_key(self):
        with pytest.raises(QueryValidationError):
            AttributePredicate(key="  ", op="eq", value="x").validate()

    def test_validate_in_requires_list(self):
        with pytest.raises(QueryValidationError):
            AttributePredicate(key="site", op="in", value="A").validate()

    def test_validate_numeric_op_requires_number(self):
        with pytest.raises(QueryValidationError):
            AttributePredicate(key="age", op="lt", value="forty").validate()

    def test_round_trip_through_dict(self):
        p = AttributePredicate(key="role", op="in", value=["nurse", "doctor"])
        p.validate()
        round_tripped = AttributePredicate.from_dict(p.to_dict())
        assert round_tripped.key == "role"
        assert round_tripped.op == "in"
        assert round_tripped.value == ["nurse", "doctor"]

    def test_validate_canonicalises_no_value_ops(self):
        p = AttributePredicate(key="role", op="exists", value="ignored")
        p.validate()
        assert p.value is None

    def test_module_level_helpers(self):
        p = AttributePredicate(key="role", op="eq", value="nurse")
        p.validate()
        assert predicate_matches(p, {"role": "nurse"})
        assert predicates_all_match([p], {"role": "nurse"})
        assert not predicates_all_match(
            [p, AttributePredicate(key="age", op="gt", value=10)],
            {"role": "nurse"},
        )

    def test_predicates_all_match_empty_list_is_true(self):
        assert predicates_all_match([], {"x": "y"})

    def test_from_dict_rejects_non_object(self):
        with pytest.raises(QueryValidationError):
            AttributePredicate.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_requires_key(self):
        with pytest.raises(QueryValidationError):
            AttributePredicate.from_dict({"op": "eq", "value": "x"})

    def test_predicate_operators_vocab_is_closed(self):
        # Defensive: catches accidental additions without a test.
        assert "eq" in PREDICATE_OPERATORS
        assert "regex" not in PREDICATE_OPERATORS

    def test_matches_ignores_non_mapping_attrs(self):
        p = AttributePredicate(key="role", op="eq", value="nurse")
        p.validate()
        assert not p.matches(["not", "a", "dict"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CodeExpr
# --------------------------------------------------------------------------- #


class TestCodeExpr:
    def test_leaf(self):
        a = new_code_id()
        e = CodeExpr.code(a)
        assert e.op == "code"
        assert e.code_id == a
        assert evaluate_code_expr(e, {a})
        assert not evaluate_code_expr(e, set())

    def test_and_or_not(self):
        a, b, c = new_code_id(), new_code_id(), new_code_id()
        and_expr = CodeExpr.all_of(CodeExpr.code(a), CodeExpr.code(b))
        assert evaluate_code_expr(and_expr, {a, b})
        assert not evaluate_code_expr(and_expr, {a})

        or_expr = CodeExpr.any_of(CodeExpr.code(a), CodeExpr.code(b))
        assert evaluate_code_expr(or_expr, {a})
        assert evaluate_code_expr(or_expr, {b})
        assert not evaluate_code_expr(or_expr, {c})

        not_expr = CodeExpr.negate(CodeExpr.code(a))
        assert evaluate_code_expr(not_expr, {b})
        assert not evaluate_code_expr(not_expr, {a})

    def test_nested_and_or_not(self):
        a, b, c = new_code_id(), new_code_id(), new_code_id()
        # (a OR b) AND NOT c
        e = CodeExpr.all_of(
            CodeExpr.any_of(CodeExpr.code(a), CodeExpr.code(b)),
            CodeExpr.negate(CodeExpr.code(c)),
        )
        assert evaluate_code_expr(e, {a})
        assert evaluate_code_expr(e, {b})
        assert not evaluate_code_expr(e, {a, c})
        assert not evaluate_code_expr(e, {c})

    def test_validate_rejects_bad_op(self):
        with pytest.raises(QueryValidationError):
            CodeExpr(op="xor", children=[CodeExpr.code(new_code_id())]).validate()

    def test_leaf_requires_code_id(self):
        with pytest.raises(QueryValidationError):
            CodeExpr(op="code", code_id=None).validate()

    def test_leaf_rejects_bad_code_id_shape(self):
        with pytest.raises(QueryValidationError):
            CodeExpr(op="code", code_id="not-12-hex").validate()

    def test_combinator_requires_children(self):
        with pytest.raises(QueryValidationError):
            CodeExpr(op="and", children=[]).validate()

    def test_not_requires_exactly_one_child(self):
        a = new_code_id()
        with pytest.raises(QueryValidationError):
            CodeExpr(
                op="not",
                children=[CodeExpr.code(a), CodeExpr.code(a)],
            ).validate()

    def test_combinator_must_not_set_code_id(self):
        a = new_code_id()
        with pytest.raises(QueryValidationError):
            CodeExpr(op="and", code_id=a, children=[CodeExpr.code(a)]).validate()

    def test_leaf_must_not_have_children(self):
        a = new_code_id()
        with pytest.raises(QueryValidationError):
            CodeExpr(op="code", code_id=a, children=[CodeExpr.code(a)]).validate()

    def test_depth_cap(self):
        # Build (and (and (and ... (code <a>)))) deeper than the cap.
        a = new_code_id()
        e: CodeExpr = CodeExpr.code(a)
        for _ in range(MAX_CODE_EXPR_DEPTH + 2):
            e = CodeExpr(op="and", children=[e])
        with pytest.raises(QueryValidationError):
            e.validate()

    def test_referenced_code_ids(self):
        a, b, c = new_code_id(), new_code_id(), new_code_id()
        e = CodeExpr.all_of(
            CodeExpr.code(a),
            CodeExpr.any_of(CodeExpr.code(b), CodeExpr.negate(CodeExpr.code(c))),
        )
        assert e.referenced_code_ids() == {a, b, c}

    def test_node_count(self):
        a, b = new_code_id(), new_code_id()
        e = CodeExpr.all_of(CodeExpr.code(a), CodeExpr.code(b))
        # 1 (and) + 2 (leaves) = 3
        assert e.node_count() == 3

    def test_round_trip(self):
        a, b = new_code_id(), new_code_id()
        e = CodeExpr.all_of(CodeExpr.code(a), CodeExpr.negate(CodeExpr.code(b)))
        round_tripped = CodeExpr.from_dict(e.to_dict())
        assert round_tripped.op == "and"
        assert round_tripped.referenced_code_ids() == {a, b}

    def test_from_dict_non_object(self):
        with pytest.raises(QueryValidationError):
            CodeExpr.from_dict("nope")  # type: ignore[arg-type]

    def test_from_dict_children_must_be_list(self):
        with pytest.raises(QueryValidationError):
            CodeExpr.from_dict({"op": "and", "children": "x"})

    def test_code_expr_ops_includes_known(self):
        assert {"code", "and", "or", "not"} <= set(CODE_EXPR_OPS)


# --------------------------------------------------------------------------- #
# SourceFilter
# --------------------------------------------------------------------------- #


class TestSourceFilter:
    def test_empty_filter_matches_everything(self):
        f = SourceFilter()
        s = _src(_hex(1))
        assert f.is_empty()
        assert f.matches(s)

    def test_source_id_whitelist(self):
        s1, s2, s3 = _src(_hex(1)), _src(_hex(2)), _src(_hex(3))
        f = SourceFilter(source_ids=[s1.id, s3.id])
        f.validate()
        out = filter_sources([s1, s2, s3], f)
        assert [s.id for s in out] == [s1.id, s3.id]

    def test_language(self):
        s1 = _src(_hex(1), language="en")
        s2 = _src(_hex(2), language="fr")
        f = SourceFilter(languages=["en"])
        f.validate()
        assert filter_sources([s1, s2], f) == [s1]

    def test_recording_date_range(self):
        s1 = _src(_hex(1), recording_date="2024-01-01")
        s2 = _src(_hex(2), recording_date="2024-06-15")
        s3 = _src(_hex(3), recording_date="2024-12-31")
        s_no_date = _src(_hex(4))
        f = SourceFilter(
            recording_date_from="2024-03-01",
            recording_date_to="2024-09-01",
        )
        f.validate()
        # Only s2 lies in the inclusive window; sources with no date
        # fail any range check.
        assert filter_sources([s1, s2, s3, s_no_date], f) == [s2]

    def test_recording_date_range_inverted_is_invalid(self):
        f = SourceFilter(
            recording_date_from="2024-09-01",
            recording_date_to="2024-03-01",
        )
        with pytest.raises(QueryValidationError):
            f.validate()

    def test_recording_date_bad_shape_rejected(self):
        f = SourceFilter(recording_date_from="not-a-date")
        with pytest.raises(QueryValidationError):
            f.validate()

    def test_attribute_predicates(self):
        s1 = _src(_hex(1), custom={"site": "Hospital A"})
        s2 = _src(_hex(2), custom={"site": "Hospital B"})
        f = SourceFilter(
            attributes=[
                AttributePredicate(key="site", op="eq", value="Hospital A")
            ]
        )
        f.validate()
        assert filter_sources([s1, s2], f) == [s1]

    def test_combined_filter_all_must_match(self):
        s1 = _src(_hex(1), language="en", custom={"site": "A"})
        s2 = _src(_hex(2), language="fr", custom={"site": "A"})
        s3 = _src(_hex(3), language="en", custom={"site": "B"})
        f = SourceFilter(
            languages=["en"],
            attributes=[
                AttributePredicate(key="site", op="eq", value="A")
            ],
        )
        f.validate()
        assert filter_sources([s1, s2, s3], f) == [s1]

    def test_invalid_source_id_rejected(self):
        f = SourceFilter(source_ids=["NOTHEX"])
        with pytest.raises(QueryValidationError):
            f.validate()

    def test_round_trip(self):
        f = SourceFilter(
            source_ids=[_hex(1)],
            languages=["en"],
            recording_date_from="2024-01-01",
            recording_date_to="2024-12-31",
            attributes=[AttributePredicate(key="site", op="eq", value="A")],
        )
        f.validate()
        rt = SourceFilter.from_dict(f.to_dict())
        assert rt.source_ids == [_hex(1)]
        assert rt.languages == ["en"]
        assert rt.recording_date_from == "2024-01-01"
        assert rt.attributes[0].key == "site"

    def test_from_dict_none_returns_empty(self):
        assert SourceFilter.from_dict(None).is_empty()

    def test_from_dict_non_object(self):
        with pytest.raises(QueryValidationError):
            SourceFilter.from_dict("nope")  # type: ignore[arg-type]

    def test_too_many_predicates(self):
        f = SourceFilter(
            attributes=[
                AttributePredicate(key=f"k{i}", op="eq", value="v")
                for i in range(MAX_PREDICATES + 1)
            ]
        )
        with pytest.raises(QueryValidationError):
            f.validate()

    def test_too_many_source_ids(self):
        f = SourceFilter(source_ids=[_hex(i) for i in range(MAX_LIST_ITEMS + 1)])
        with pytest.raises(QueryValidationError):
            f.validate()


# --------------------------------------------------------------------------- #
# ParticipantFilter
# --------------------------------------------------------------------------- #


class TestParticipantFilter:
    def test_empty_filter_matches_everything(self):
        f = ParticipantFilter()
        p = _participant(_hex(1))
        assert f.is_empty()
        assert f.matches(p)

    def test_participant_id_whitelist(self):
        p1, p2 = _participant(_hex(1)), _participant(_hex(2))
        f = ParticipantFilter(participant_ids=[p1.id])
        f.validate()
        assert filter_participants([p1, p2], f) == [p1]

    def test_demographics(self):
        p1 = _participant(_hex(1), demographics={"role": "nurse"})
        p2 = _participant(_hex(2), demographics={"role": "doctor"})
        f = ParticipantFilter(
            demographics=[
                AttributePredicate(key="role", op="eq", value="nurse")
            ]
        )
        f.validate()
        assert filter_participants([p1, p2], f) == [p1]

    def test_invalid_participant_id_rejected(self):
        with pytest.raises(QueryValidationError):
            ParticipantFilter(participant_ids=["NOTHEX"]).validate()

    def test_round_trip(self):
        f = ParticipantFilter(
            participant_ids=[_hex(1)],
            demographics=[AttributePredicate(key="role", op="eq", value="x")],
        )
        f.validate()
        rt = ParticipantFilter.from_dict(f.to_dict())
        assert rt.participant_ids == [_hex(1)]
        assert rt.demographics[0].key == "role"

    def test_from_dict_none(self):
        assert ParticipantFilter.from_dict(None).is_empty()

    def test_from_dict_non_object(self):
        with pytest.raises(QueryValidationError):
            ParticipantFilter.from_dict("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# SpeakerFilter
# --------------------------------------------------------------------------- #


class TestSpeakerFilter:
    def _smap(self) -> SpeakerMap:
        return SpeakerMap.new(
            project_id=PID,
            source_id=_hex(1),
            entries=[
                SpeakerEntry(
                    label="SPEAKER_00",
                    role="interviewer",
                ),
                SpeakerEntry(
                    label="SPEAKER_01",
                    role="interviewee",
                    participant_id=_hex(11),
                ),
                SpeakerEntry(
                    label="SPEAKER_02",
                    role="facilitator",
                    participant_id=_hex(12),
                ),
            ],
        )

    def test_empty_matches_all_labels(self):
        f = SpeakerFilter()
        assert f.is_empty()
        # Any label passes; even blank label passes (it's just "no
        # filter said no") — but the executor ignores blanks separately.
        assert f.matches("SPEAKER_00", self._smap())

    def test_empty_label_never_matches_when_filter_active(self):
        f = SpeakerFilter(labels=["SPEAKER_00"])
        f.validate()
        assert not f.matches("", self._smap())

    def test_label_dimension(self):
        f = SpeakerFilter(labels=["SPEAKER_00"])
        f.validate()
        assert f.matches("SPEAKER_00", self._smap())
        assert not f.matches("SPEAKER_01", self._smap())

    def test_role_dimension(self):
        f = SpeakerFilter(roles=["interviewee"])
        f.validate()
        assert f.matches("SPEAKER_01", self._smap())
        assert not f.matches("SPEAKER_00", self._smap())

    def test_participant_dimension(self):
        f = SpeakerFilter(participant_ids=[_hex(11)])
        f.validate()
        assert f.matches("SPEAKER_01", self._smap())
        assert not f.matches("SPEAKER_02", self._smap())

    def test_disjunctive_dimensions(self):
        f = SpeakerFilter(labels=["SPEAKER_00"], roles=["interviewee"])
        f.validate()
        # OR semantics across dimensions: matches if either fires.
        assert f.matches("SPEAKER_00", self._smap())
        assert f.matches("SPEAKER_01", self._smap())
        assert not f.matches("SPEAKER_02", self._smap())

    def test_unknown_label_treated_as_unknown_role(self):
        # Speaker label not in the map → role becomes "unknown".
        f = SpeakerFilter(roles=["unknown"])
        f.validate()
        assert f.matches("ORPHAN_LABEL", self._smap())

    def test_include_unmapped_audit_pass(self):
        f = SpeakerFilter(roles=["interviewee"], include_unmapped=True)
        f.validate()
        # Orphan label doesn't have role "interviewee", but
        # include_unmapped catches it anyway.
        assert f.matches("ORPHAN_LABEL", self._smap())
        # Mapped labels still go through the normal dimension check.
        assert f.matches("SPEAKER_01", self._smap())
        assert not f.matches("SPEAKER_00", self._smap())

    def test_validate_rejects_bad_role(self):
        with pytest.raises(QueryValidationError):
            SpeakerFilter(roles=["bogus"]).validate()

    def test_validate_rejects_bad_participant_id(self):
        with pytest.raises(QueryValidationError):
            SpeakerFilter(participant_ids=["nothex"]).validate()

    def test_round_trip(self):
        f = SpeakerFilter(
            labels=["X"],
            roles=["interviewee"],
            participant_ids=[_hex(11)],
            include_unmapped=True,
        )
        f.validate()
        rt = SpeakerFilter.from_dict(f.to_dict())
        assert rt.labels == ["X"]
        assert rt.roles == ["interviewee"]
        assert rt.participant_ids == [_hex(11)]
        assert rt.include_unmapped is True

    def test_no_speaker_map(self):
        # When the source has no speaker map, only the label dimension
        # can fire (roles all evaluate to "unknown", participant ids
        # all evaluate to None).
        f = SpeakerFilter(labels=["SPEAKER_00"])
        f.validate()
        assert f.matches("SPEAKER_00", None)
        f2 = SpeakerFilter(roles=["interviewee"])
        f2.validate()
        assert not f2.matches("SPEAKER_00", None)

    def test_from_dict_none(self):
        assert SpeakerFilter.from_dict(None).is_empty()


# --------------------------------------------------------------------------- #
# CodeFilter
# --------------------------------------------------------------------------- #


class TestCodeFilter:
    def test_empty(self):
        f = CodeFilter()
        assert f.is_empty()
        f.validate()
        assert f.referenced_code_ids() == set()

    def test_round_trip(self):
        a = new_code_id()
        f = CodeFilter(expr=CodeExpr.code(a))
        f.validate()
        rt = CodeFilter.from_dict(f.to_dict())
        assert rt.referenced_code_ids() == {a}

    def test_validate_rejects_bad_expr(self):
        f = CodeFilter(expr=CodeExpr(op="and", children=[]))
        with pytest.raises(QueryValidationError):
            f.validate()

    def test_from_dict_none(self):
        assert CodeFilter.from_dict(None).is_empty()


# --------------------------------------------------------------------------- #
# ProximityFilter
# --------------------------------------------------------------------------- #


class TestProximityFilter:
    def test_empty(self):
        pf = ProximityFilter()
        pf.validate()
        assert pf.is_empty()

    def test_validate_scope(self):
        with pytest.raises(QueryValidationError):
            ProximityFilter(scope="bad", required_code_ids=[]).validate()

    def test_validate_required_ids_shape(self):
        with pytest.raises(QueryValidationError):
            ProximityFilter(
                scope="source", required_code_ids=["nothex"]
            ).validate()

    def test_validate_max_gap_nonneg(self):
        with pytest.raises(QueryValidationError):
            ProximityFilter(
                scope="paragraph", required_code_ids=[], max_gap=-1
            ).validate()

    def test_round_trip(self):
        a = new_code_id()
        pf = ProximityFilter(
            scope="paragraph", required_code_ids=[a], max_gap=10.0
        )
        pf.validate()
        rt = ProximityFilter.from_dict(pf.to_dict())
        assert rt.scope == "paragraph"
        assert rt.required_code_ids == [a]
        assert rt.max_gap == 10.0

    def test_proximity_scopes_known(self):
        assert {"segment", "paragraph", "source"} == set(PROXIMITY_SCOPES)


# --------------------------------------------------------------------------- #
# Query (top-level)
# --------------------------------------------------------------------------- #


class TestQuery:
    def test_default_minimum(self):
        q = Query(project_id=PID)
        q.validate()
        assert q.referenced_code_ids() == set()

    def test_round_trip_full(self):
        a, b = new_code_id(), new_code_id()
        q = Query(
            project_id=PID,
            name="My query",
            description="Find everything about X",
            sources=SourceFilter(languages=["en"]),
            participants=ParticipantFilter(
                demographics=[
                    AttributePredicate(key="role", op="eq", value="nurse")
                ]
            ),
            speakers=SpeakerFilter(roles=["interviewee"]),
            codes=CodeFilter(
                expr=CodeExpr.all_of(CodeExpr.code(a), CodeExpr.code(b))
            ),
            proximity=ProximityFilter(
                scope="paragraph", required_code_ids=[a, b], max_gap=20
            ),
        )
        q.validate()
        rt = Query.from_dict(q.to_dict())
        assert rt.name == "My query"
        assert rt.referenced_code_ids() == {a, b}
        assert rt.proximity is not None
        assert rt.proximity.scope == "paragraph"

    def test_validate_bad_project_id(self):
        with pytest.raises(QueryValidationError):
            Query(project_id="nothex").validate()

    def test_from_dict_missing_project_id(self):
        with pytest.raises(QueryValidationError):
            Query.from_dict({})

    def test_from_dict_non_object(self):
        with pytest.raises(QueryValidationError):
            Query.from_dict("nope")  # type: ignore[arg-type]

    def test_validate_inherits_from_project_validation_error(self):
        # QueryValidationError → ProjectValidationError (so existing
        # bundle / format error handlers catch it).
        try:
            AttributePredicate(key="x", op="bogus", value="y").validate()
        except ProjectValidationError:
            pass
        else:  # pragma: no cover
            pytest.fail("Expected ProjectValidationError")

    def test_referenced_code_ids_includes_proximity(self):
        a = new_code_id()
        b = new_code_id()
        q = Query(
            project_id=PID,
            codes=CodeFilter(expr=CodeExpr.code(a)),
            proximity=ProximityFilter(
                scope="source", required_code_ids=[b]
            ),
        )
        q.validate()
        assert q.referenced_code_ids() == {a, b}


# --------------------------------------------------------------------------- #
# applications_for_query
# --------------------------------------------------------------------------- #


class TestApplicationsForQuery:
    def _setup(self):
        # Two sources, one in English/Hospital A, one in French/Hospital B.
        s1 = _src(_hex(1), language="en", custom={"site": "A"})
        s2 = _src(_hex(2), language="fr", custom={"site": "B"})

        # Two participants — P1 is a nurse, P2 is a doctor.
        p1 = _participant(_hex(11), demographics={"role": "nurse"})
        p2 = _participant(_hex(12), demographics={"role": "doctor"})

        # Two codes.
        ca, cb = new_code_id(), new_code_id()

        # SpeakerMaps: s1's SPEAKER_00 = interviewer, SPEAKER_01 =
        # interviewee linked to p1; s2 has no map.
        sm1 = SpeakerMap.new(
            project_id=PID,
            source_id=s1.id,
            entries=[
                SpeakerEntry(label="SPEAKER_00", role="interviewer"),
                SpeakerEntry(
                    label="SPEAKER_01", role="interviewee", participant_id=p1.id
                ),
            ],
        )

        applications = [
            _app(code_id=ca, source_id=s1.id, speaker="SPEAKER_01",
                 start=10, end=20),  # nurse, en, A — all
            _app(code_id=cb, source_id=s1.id, speaker="SPEAKER_00",
                 start=15, end=25),  # interviewer
            _app(code_id=ca, source_id=s2.id, speaker="SPEAKER_99",
                 start=5, end=8),   # fr, B
            _app(code_id=cb, source_id=s2.id, speaker="SPEAKER_99",
                 start=100, end=120),  # fr, B, far away
        ]
        return s1, s2, p1, p2, ca, cb, sm1, applications

    def test_no_filters_returns_everything(self):
        _, _, _, _, _, _, _, apps = self._setup()
        q = Query(project_id=PID)
        out = applications_for_query(q, apps)
        assert out == apps

    def test_source_filter(self):
        s1, s2, _, _, _, _, _, apps = self._setup()
        q = Query(
            project_id=PID,
            sources=SourceFilter(languages=["en"]),
        )
        out = applications_for_query(q, apps, sources=[s1, s2])
        assert {a["source_id"] for a in out} == {s1.id}

    def test_source_filter_requires_sources_arg(self):
        _, _, _, _, _, _, _, apps = self._setup()
        q = Query(
            project_id=PID,
            sources=SourceFilter(languages=["en"]),
        )
        with pytest.raises(QueryValidationError):
            applications_for_query(q, apps)

    def test_participant_filter_via_speaker_map(self):
        s1, s2, p1, p2, _, _, sm1, apps = self._setup()
        q = Query(
            project_id=PID,
            participants=ParticipantFilter(participant_ids=[p1.id]),
        )
        out = applications_for_query(
            q,
            apps,
            participants=[p1, p2],
            speaker_maps={s1.id: sm1},
        )
        # Only the SPEAKER_01 application in s1 (mapped to p1) survives.
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_participant_filter_falls_back_to_explicit_field(self):
        s1, s2, p1, p2, ca, cb, sm1, _ = self._setup()
        # Application carries an explicit participant_id without a
        # matching speaker map entry — should still match.
        a_explicit = _app(
            code_id=ca, source_id=s2.id, participant_id=p1.id, speaker="X"
        )
        q = Query(
            project_id=PID,
            participants=ParticipantFilter(participant_ids=[p1.id]),
        )
        out = applications_for_query(
            q,
            [a_explicit],
            participants=[p1, p2],
            speaker_maps={s1.id: sm1},
        )
        assert out == [a_explicit]

    def test_speaker_filter_by_role(self):
        s1, _, _, _, _, _, sm1, apps = self._setup()
        q = Query(
            project_id=PID,
            speakers=SpeakerFilter(roles=["interviewee"]),
        )
        out = applications_for_query(q, apps, speaker_maps={s1.id: sm1})
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_speaker_filter_by_label(self):
        s1, _, _, _, _, _, sm1, apps = self._setup()
        q = Query(
            project_id=PID,
            speakers=SpeakerFilter(labels=["SPEAKER_00"]),
        )
        out = applications_for_query(q, apps, speaker_maps={s1.id: sm1})
        assert len(out) == 1
        assert out[0]["speaker"] == "SPEAKER_00"

    def test_code_filter_leaf(self):
        _, _, _, _, ca, _, _, apps = self._setup()
        q = Query(
            project_id=PID,
            codes=CodeFilter(expr=CodeExpr.code(ca)),
        )
        out = applications_for_query(q, apps)
        assert all(a["code_id"] == ca for a in out)

    def test_code_filter_or(self):
        _, _, _, _, ca, cb, _, apps = self._setup()
        q = Query(
            project_id=PID,
            codes=CodeFilter(
                expr=CodeExpr.any_of(CodeExpr.code(ca), CodeExpr.code(cb))
            ),
        )
        out = applications_for_query(q, apps)
        assert len(out) == len(apps)

    def test_code_filter_negation_drops_matches(self):
        _, _, _, _, ca, _, _, apps = self._setup()
        # NOT ca → only the cb applications should pass.
        q = Query(
            project_id=PID,
            codes=CodeFilter(expr=CodeExpr.negate(CodeExpr.code(ca))),
        )
        out = applications_for_query(q, apps)
        assert all(a["code_id"] != ca for a in out)

    def test_proximity_source_scope(self):
        s1, s2, _, _, ca, cb, _, apps = self._setup()
        # Both ca and cb must co-occur in same source (any distance).
        q = Query(
            project_id=PID,
            proximity=ProximityFilter(
                scope="source", required_code_ids=[ca, cb]
            ),
        )
        out = applications_for_query(q, apps)
        # Both s1 and s2 have ca + cb applications, so all 4 keep.
        assert len(out) == 4

    def test_proximity_segment_overlap(self):
        s1, s2, _, _, ca, cb, _, apps = self._setup()
        # Segment scope: anchors must overlap. In s1, ca is [10,20]
        # and cb is [15,25] → overlap. In s2, ca is [5,8] and cb is
        # [100,120] → no overlap.
        q = Query(
            project_id=PID,
            proximity=ProximityFilter(
                scope="segment", required_code_ids=[ca, cb]
            ),
        )
        out = applications_for_query(q, apps)
        # Only s1's two applications survive.
        assert len(out) == 2
        assert all(a["source_id"] == s1.id for a in out)

    def test_proximity_paragraph_with_max_gap(self):
        s1, s2, _, _, ca, cb, _, apps = self._setup()
        # Paragraph scope, gap=0 — anchors must overlap or touch. Same
        # result as segment scope at gap=0.
        q_zero = Query(
            project_id=PID,
            proximity=ProximityFilter(
                scope="paragraph", required_code_ids=[ca, cb], max_gap=0
            ),
        )
        zero_out = applications_for_query(q_zero, apps)
        assert len(zero_out) == 2

        # Wide gap — s2 has gap 100→120 vs 5→8 = 92 units; gap=200
        # should accept.
        q_wide = Query(
            project_id=PID,
            proximity=ProximityFilter(
                scope="paragraph", required_code_ids=[ca, cb], max_gap=200
            ),
        )
        wide_out = applications_for_query(q_wide, apps)
        assert len(wide_out) == 4

    def test_proximity_drops_app_without_required_partner(self):
        s1, _, _, _, ca, cb, _, _ = self._setup()
        cc = new_code_id()  # never in the application set
        only_ca = _app(code_id=ca, source_id=s1.id)
        q = Query(
            project_id=PID,
            proximity=ProximityFilter(
                scope="source", required_code_ids=[ca, cc]
            ),
        )
        out = applications_for_query(q, [only_ca])
        # cc never applied → ca alone fails the proximity constraint.
        assert out == []

    def test_proximity_filters_to_required_codes_only(self):
        s1, _, _, _, ca, cb, _, _ = self._setup()
        cd = new_code_id()  # extra code applied but not required
        apps = [
            _app(code_id=ca, source_id=s1.id),
            _app(code_id=cb, source_id=s1.id),
            _app(code_id=cd, source_id=s1.id),
        ]
        q = Query(
            project_id=PID,
            proximity=ProximityFilter(
                scope="source", required_code_ids=[ca, cb]
            ),
        )
        out = applications_for_query(q, apps)
        kept_codes = {a["code_id"] for a in out}
        # Only ca + cb make the cut; cd is not in required_code_ids.
        assert kept_codes == {ca, cb}

    def test_combined_source_speaker_code_filter(self):
        s1, s2, p1, p2, ca, cb, sm1, apps = self._setup()
        # English source + interviewee + code ca → only s1's
        # (ca, SPEAKER_01) application.
        q = Query(
            project_id=PID,
            sources=SourceFilter(languages=["en"]),
            speakers=SpeakerFilter(roles=["interviewee"]),
            codes=CodeFilter(expr=CodeExpr.code(ca)),
        )
        out = applications_for_query(
            q,
            apps,
            sources=[s1, s2],
            speaker_maps={s1.id: sm1},
        )
        assert len(out) == 1
        assert out[0]["code_id"] == ca
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_invalid_application_shape_rejected(self):
        q = Query(project_id=PID)
        with pytest.raises(QueryValidationError):
            applications_for_query(q, [{"source_id": _hex(1)}])  # missing code_id

    def test_bad_id_shapes_rejected(self):
        q = Query(project_id=PID)
        with pytest.raises(QueryValidationError):
            applications_for_query(
                q,
                [{"code_id": "nothex", "source_id": _hex(1)}],
            )

    def test_participant_filter_requires_participants_arg(self):
        _, _, _, _, _, _, _, apps = self._setup()
        q = Query(
            project_id=PID,
            participants=ParticipantFilter(participant_ids=[_hex(11)]),
        )
        with pytest.raises(QueryValidationError):
            applications_for_query(q, apps)
