"""The input side of Plaine & Easie, checked against the output side.

The strongest available guarantee is a round trip: whatever the emitter writes,
the parser must read back to the same melody. That is asserted over generated
melodies rather than a handful of examples, because the interesting failures are
in the octave marks and the accidental spellings, and those only show up across
a range.

Durations deliberately do not round-trip exactly: ``to_plaine_easie`` writes
durations reconstructed from the quantised rhythm digits, so the incipit carries
the quantised rhythm and not the original milliseconds. What must survive is the
layer the encoding keeps and the matcher ranks -- the intervals.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from musicdna_encoder import InputEvent, encode_events
from musicdna_encoder.plaine_easie import (
    PlaineEasieError,
    parse_plaine_easie,
    plaine_easie_to_events,
)


def _melody(seed: int, length: int = 10) -> tuple[InputEvent, ...]:
    rng = random.Random(seed)
    events: list[InputEvent] = []
    start = 0.0
    # Stay inside the octave range Plaine & Easie can spell at all.
    pitch = rng.randint(36, 84)
    for _ in range(length):
        duration = rng.choice([0.125, 0.25, 0.5])
        events.append(InputEvent(start, start + duration, duration * 1000.0, pitch))
        start += duration
        pitch = max(26, min(105, pitch + rng.choice([-12, -7, -3, -1, 0, 1, 2, 5, 11])))
    return tuple(events)


@pytest.mark.parametrize("seed", range(40))
def test_pitches_survive_a_round_trip_through_the_incipit(seed: int) -> None:
    events = _melody(seed)
    incipit = encode_events(events).to_plaine_easie()
    parsed = parse_plaine_easie(incipit)
    assert [note.midi_pitch for note in parsed] == [event.midi_pitch for event in events]


@pytest.mark.parametrize("seed", range(40))
def test_the_interval_layer_survives_a_round_trip(seed: int) -> None:
    """The layer the matcher ranks on is the one that has to come back intact."""

    events = _melody(seed)
    original = encode_events(events).to_dict()
    incipit = encode_events(events).to_plaine_easie()
    recovered = encode_events(plaine_easie_to_events(incipit)).to_dict()
    assert recovered["intervals_semitones"] == original["intervals_semitones"]
    assert recovered["parsons_code"] == original["parsons_code"]
    assert recovered["interval_classes"] == original["interval_classes"]


def test_a_tempo_convention_cannot_leak_into_the_representation() -> None:
    """whole_note_seconds is a convention; no retained layer may depend on it."""

    incipit = "%G-2 '4C'8D'2E'4F"
    slow = encode_events(plaine_easie_to_events(incipit, whole_note_seconds=8.0)).to_dict()
    fast = encode_events(plaine_easie_to_events(incipit, whole_note_seconds=0.5)).to_dict()
    for layer in (
        "intervals_semitones",
        "interval_classes",
        "parsons_code",
        "ioi_rhythm_digits",
        "rhythm_ratios",
    ):
        assert slow[layer] == fast[layer], layer


def test_octave_marks_place_notes_where_the_encoder_puts_them() -> None:
    """Three commas is C1 and one apostrophe is C4; the encoder never omits the mark."""

    notes = parse_plaine_easie("%G-2 ,,,4C,,4C,4C'4C''4C'''4C")
    assert [note.midi_pitch for note in notes] == [24, 36, 48, 60, 72, 84]


def test_a_missing_octave_mark_keeps_the_previous_one() -> None:
    """Octave marks are sticky in Plaine & Easie, and real records rely on it."""

    assert [n.midi_pitch for n in parse_plaine_easie("%G-2 '4C4D4E")] == [60, 62, 64]


def test_a_key_signature_alters_every_matching_letter() -> None:
    assert [n.midi_pitch for n in parse_plaine_easie("%G-2 $xF '4F'4G")] == [66, 67]
    assert [n.midi_pitch for n in parse_plaine_easie("%G-2 $bBEA '4B'4E'4A")] == [70, 63, 68]


def test_an_accidental_holds_until_the_bar_line() -> None:
    within = parse_plaine_easie("%G-2 '4xF'4F")
    assert [note.midi_pitch for note in within] == [66, 66]
    across = parse_plaine_easie("%G-2 '4xF/'4F")
    assert [note.midi_pitch for note in across] == [66, 65]


def test_an_accidental_is_scoped_to_the_note_name_and_octave() -> None:
    notes = parse_plaine_easie("%G-2 '4xF''4F'4F")

    # F-sharp in octave 4 neither sharpens F5 nor expires when the melody
    # briefly changes register and returns.
    assert [note.midi_pitch for note in notes] == [66, 77, 66]


def test_tied_accidental_crosses_bar_without_altering_later_notes() -> None:
    notes = parse_plaine_easie("%G-2 '4xF+/'4F'4F")
    assert [(n.midi_pitch, n.duration_whole_notes, n.start_whole_notes) for n in notes] == [
        (66, 0.5, 0.0),
        (65, 0.25, 0.5),
    ]


@pytest.mark.parametrize("tie", ("+", "_"))
def test_tie_after_rest_cannot_absorb_the_gap(tie: str) -> None:
    with pytest.raises(PlaineEasieError, match="no preceding note"):
        parse_plaine_easie(f"%G-2 '4C-{tie}C")


@pytest.mark.parametrize("data", ("'48C-D", "'8.68{AB''C}{DEF}"))
def test_unsupported_rhythmic_sequences_are_not_silently_flattened(data: str) -> None:
    with pytest.raises(PlaineEasieError, match="rhythmic sequences"):
        parse_plaine_easie("%G-2 " + data)


def test_a_natural_sign_cancels_the_key_signature() -> None:
    notes = parse_plaine_easie("%G-2 $xF '4nF'4F")
    assert [note.midi_pitch for note in notes] == [65, 65]


def test_dots_extend_the_duration_by_half_each() -> None:
    notes = parse_plaine_easie("%G-2 '4C'4.C'4..C")
    durations = [note.duration_whole_notes for note in notes]
    assert durations == [0.25, 0.375, 0.4375]


def test_a_tie_merges_the_pair_into_one_note() -> None:
    notes = parse_plaine_easie("%G-2 '4C_'4C'4D")
    assert [(n.midi_pitch, n.duration_whole_notes) for n in notes] == [(60, 0.5), (62, 0.25)]


def test_a_rest_takes_no_note_but_the_melody_continues() -> None:
    notes = parse_plaine_easie("%G-2 '4C'4-'4D")
    assert [note.midi_pitch for note in notes] == [60, 62]
    assert [note.start_whole_notes for note in notes] == [0.0, 0.5]

    events = plaine_easie_to_events("%G-2 '4C'4-'4D", whole_note_seconds=4.0)
    assert [(event.start_seconds, event.end_seconds) for event in events] == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]
    result = encode_events(events)
    assert result.gaps_ms == (1000.0,)
    assert result.ioi_ms == (2000.0,)


def test_a_leading_rest_delays_the_first_event() -> None:
    events = plaine_easie_to_events("%G-2 4-'4C", whole_note_seconds=2.0)

    assert len(events) == 1
    assert events[0].start_seconds == 0.5


def test_rests_use_the_current_dotted_duration_and_accumulate() -> None:
    events = plaine_easie_to_events("%G-2 '8C4.-8-'8D", whole_note_seconds=4.0)

    assert events[0].duration_ms == 500.0
    assert events[1].start_seconds == 2.5
    assert encode_events(events).gaps_ms == (2000.0,)


def test_a_rest_cannot_silently_interrupt_a_tie() -> None:
    with pytest.raises(PlaineEasieError, match="interrupts a tie"):
        parse_plaine_easie("%G-2 '4C_4-'4C")


@pytest.mark.parametrize(
    "incipit, reason",
    [
        ("%G-2 (3'8C'8D'8E)", "tuplets"),
        ("%G-2 '4C_'4D", "unresolved tie"),
        ("%G-2 '4C~", "unexpected character"),
        ("%G-2 $yF '4C", "key signature"),
        ("%G-2 '4x", "accidental at end"),
    ],
)
def test_what_is_not_covered_raises_instead_of_guessing(incipit: str, reason: str) -> None:
    """A melody parsed with a construct silently dropped is a wrong melody."""

    with pytest.raises(PlaineEasieError) as caught:
        parse_plaine_easie(incipit)
    assert reason.split()[0] in str(caught.value)


def test_grace_notes_are_skipped_rather_than_kept_or_refused() -> None:
    """IncipitSearch itself publishes `withoutOrnaments`, so the ornament is
    editorially separable from the melody -- and a hummed query will not have it."""

    assert [n.midi_pitch for n in parse_plaine_easie("%G-2 q'8C'4D")] == [62]
    assert [n.midi_pitch for n in parse_plaine_easie("%G '4Gqq{6AGF}r'4G")] == [67, 67]


def test_grace_markers_are_distinct_from_note_names() -> None:
    """Lowercase ``g``/``q`` are P&E grace markers; notes are uppercase."""

    assert [n.midi_pitch for n in parse_plaine_easie("%G-2 /gB4AA")] == [69, 69]
    assert [n.midi_pitch for n in parse_plaine_easie("%G-2 qG4A")] == [69]
    with pytest.raises(PlaineEasieError, match="unexpected character"):
        parse_plaine_easie("%G-2 '4a")


def test_terminal_grace_marker_is_reported_as_an_incomplete_grace_note() -> None:
    with pytest.raises(PlaineEasieError, match="grace note"):
        parse_plaine_easie("%G-2 g")


def test_beaming_carries_no_melody() -> None:
    assert [n.midi_pitch for n in parse_plaine_easie("%G {8'C'D'E}")] == [60, 62, 64]


def test_plus_is_a_tie_and_not_decoration() -> None:
    """`+` ties like `_` but reaches across a bar or the end of the incipit.

    Treating it as decoration cost 21 of 301 real incipits their correct reading:
    the tied pair stayed two notes where the catalogue has one. On the last note
    there is nothing to merge with, which is what the mark is for, so it resolves
    to the note rather than to an error.
    """

    assert [n.midi_pitch for n in parse_plaine_easie("%G '1D+")] == [62]
    merged = parse_plaine_easie("%G '4D+/'4D'4E")
    assert [(n.midi_pitch, n.duration_whole_notes) for n in merged] == [(62, 0.5), (64, 0.25)]


def test_an_empty_key_signature_is_the_commonest_form() -> None:
    """`$@c` -- no accidentals, straight into the time signature."""

    assert [n.midi_pitch for n in parse_plaine_easie("%G-2$@c4'C'D")] == [60, 62]


def test_the_time_signature_does_not_swallow_the_bar_line_behind_it() -> None:
    """`@c/2.` is common time, bar line, dotted half -- not a signature of `c/2`."""

    notes = parse_plaine_easie("%G$bB@c/2.''F8FF")
    assert [n.midi_pitch for n in notes] == [77, 77, 77]
    assert notes[0].duration_whole_notes == 0.75


def test_an_empty_incipit_is_empty_rather_than_an_error() -> None:
    assert parse_plaine_easie("%G-2 ") == ()
    assert plaine_easie_to_events("%G-2 ") == ()


def test_events_are_contiguous_and_carry_matching_durations() -> None:
    events = plaine_easie_to_events("%G-2 '4C'8D'2E", whole_note_seconds=4.0)
    assert [event.duration_ms for event in events] == [1000.0, 500.0, 2000.0]
    for earlier, later in pairwise(events):
        assert earlier.end_seconds == pytest.approx(later.start_seconds)


@pytest.mark.parametrize("invalid", (0.0, -1.0, float("nan"), float("inf")))
def test_event_projection_requires_a_finite_positive_tempo_convention(invalid: float) -> None:
    with pytest.raises(PlaineEasieError, match="finite and positive"):
        plaine_easie_to_events("%G-2 '4C", whole_note_seconds=invalid)
