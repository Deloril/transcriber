import {describe, it, expect} from "vitest";
import {nextFreeSpeakerId, deleteSpeakerFromState} from
  "../../scribe/static/js/helpers.mjs";


// --------------------------------------------------------------------------- //
// nextFreeSpeakerId — used by the editor's "+ Add new speaker" entry points.
// --------------------------------------------------------------------------- //


describe("nextFreeSpeakerId", () => {
  it("returns SPEAKER_00 on an empty state", () => {
    expect(nextFreeSpeakerId({speakers: [], segments: []})).toBe("SPEAKER_00");
  });

  it("picks the next unused id past the roster", () => {
    const state = {
      speakers: ["SPEAKER_00", "SPEAKER_01"],
      segments: [],
    };
    expect(nextFreeSpeakerId(state)).toBe("SPEAKER_02");
  });

  it("avoids collisions with ids that are only on segments", () => {
    // Post-delete state: SPEAKER_02 was removed from speakers but a
    // stale segment still references it. The new id must not be
    // SPEAKER_02 — that would clobber the orphan segment.
    const state = {
      speakers: ["SPEAKER_00", "SPEAKER_01"],
      segments: [
        {speaker: "SPEAKER_02", text: "stranded", words: []},
      ],
    };
    const id = nextFreeSpeakerId(state);
    expect(id).not.toBe("SPEAKER_02");
    expect(id).toMatch(/^SPEAKER_\d{2,}$/);
  });

  it("tolerates missing fields", () => {
    expect(nextFreeSpeakerId({})).toBe("SPEAKER_00");
    expect(nextFreeSpeakerId(null)).toBe("SPEAKER_00");
    expect(nextFreeSpeakerId(undefined)).toBe("SPEAKER_00");
  });
});


// --------------------------------------------------------------------------- //
// deleteSpeakerFromState — drives the editor's delete-speaker modal.
// --------------------------------------------------------------------------- //


function _state() {
  return {
    speakers: ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
    speaker_names: {SPEAKER_00: "Luke", SPEAKER_02: "Maria"},
    segments: [
      {speaker: "SPEAKER_00", text: "hello",
       words: [{speaker: "SPEAKER_00", text: "hello"}]},
      {speaker: "SPEAKER_01", text: "hi back",
       words: [{speaker: "SPEAKER_01", text: "hi back"}]},
      {speaker: "SPEAKER_00", text: "follow-up",
       words: [{speaker: "SPEAKER_00", text: "follow-up"}]},
      {speaker: "SPEAKER_02", text: "agreed",
       words: [{speaker: "SPEAKER_02", text: "agreed"}]},
    ],
  };
}


describe("deleteSpeakerFromState — delete mode", () => {
  it("drops every matching segment", () => {
    const {state, segments_removed, segments_reassigned} =
      deleteSpeakerFromState(_state(), "SPEAKER_00", {mode: "delete"});
    expect(segments_removed).toBe(2);
    expect(segments_reassigned).toBe(0);
    expect(state.segments.map(s => s.speaker))
      .toEqual(["SPEAKER_01", "SPEAKER_02"]);
  });

  it("removes the speaker from the roster + rename map", () => {
    const {state} = deleteSpeakerFromState(_state(), "SPEAKER_00",
      {mode: "delete"});
    expect(state.speakers).toEqual(["SPEAKER_01", "SPEAKER_02"]);
    expect(state.speaker_names).toEqual({SPEAKER_02: "Maria"});
  });

  it("works on legacy transcripts without speaker_names", () => {
    const legacy = {
      speakers: ["SPEAKER_00", "SPEAKER_01"],
      // no speaker_names map
      segments: [
        {speaker: "SPEAKER_00", text: "x", words: []},
      ],
    };
    const {state} = deleteSpeakerFromState(legacy, "SPEAKER_00",
      {mode: "delete"});
    expect(state.segments).toEqual([]);
    expect(state.speakers).toEqual(["SPEAKER_01"]);
    // No speaker_names key was introduced.
    expect("speaker_names" in state).toBe(false);
  });

  it("does not mutate the input", () => {
    const before = _state();
    const snapshot = JSON.parse(JSON.stringify(before));
    deleteSpeakerFromState(before, "SPEAKER_00", {mode: "delete"});
    expect(before).toEqual(snapshot);
  });
});


describe("deleteSpeakerFromState — reassign mode", () => {
  it("rewrites segments + words to the target speaker", () => {
    const {state, segments_reassigned} = deleteSpeakerFromState(
      _state(), "SPEAKER_00",
      {mode: "reassign", targetSpeakerId: "SPEAKER_01"},
    );
    expect(segments_reassigned).toBe(2);
    // SPEAKER_00's two segments now point at SPEAKER_01.
    expect(state.segments.map(s => s.speaker))
      .toEqual(["SPEAKER_01", "SPEAKER_01", "SPEAKER_01", "SPEAKER_02"]);
    // Word-level speaker also rewritten.
    state.segments.forEach(seg => {
      seg.words.forEach(w => {
        expect(w.speaker).not.toBe("SPEAKER_00");
      });
    });
    // SPEAKER_00 is gone from the roster and the rename map.
    expect(state.speakers).toEqual(["SPEAKER_01", "SPEAKER_02"]);
    expect("SPEAKER_00" in state.speaker_names).toBe(false);
  });

  it("leaves the target speaker's existing data untouched", () => {
    const {state} = deleteSpeakerFromState(_state(), "SPEAKER_00",
      {mode: "reassign", targetSpeakerId: "SPEAKER_01"});
    // SPEAKER_01's original segment still has its original text.
    const onlyOriginalSpeakerOne = state.segments
      .filter(s => s.text === "hi back");
    expect(onlyOriginalSpeakerOne).toHaveLength(1);
    expect(onlyOriginalSpeakerOne[0].speaker).toBe("SPEAKER_01");
  });

  it("rejects reassign without a target", () => {
    expect(() =>
      deleteSpeakerFromState(_state(), "SPEAKER_00", {mode: "reassign"})
    ).toThrow();
  });

  it("rejects reassign with target equal to the speaker being deleted", () => {
    expect(() =>
      deleteSpeakerFromState(_state(), "SPEAKER_00",
        {mode: "reassign", targetSpeakerId: "SPEAKER_00"})
    ).toThrow();
  });
});


describe("deleteSpeakerFromState — bad input", () => {
  it("throws on non-object state", () => {
    expect(() => deleteSpeakerFromState(null, "SPEAKER_00")).toThrow();
  });

  it("throws on unknown mode", () => {
    expect(() =>
      deleteSpeakerFromState(_state(), "SPEAKER_00", {mode: "yeet"})
    ).toThrow();
  });
});
