from __future__ import annotations

import json

import pytest

from musicdna_encoder import InputEvent, encode_events


def _events(pitches: tuple[int, ...], durations_ms: tuple[float, ...]) -> tuple[InputEvent, ...]:
    events: list[InputEvent] = []
    start = 0.0
    for pitch, duration_ms in zip(pitches, durations_ms, strict=True):
        end = start + duration_ms / 1000.0
        events.append(InputEvent(start, end, duration_ms, pitch))
        start = end
    return tuple(events)


@pytest.mark.parametrize(
    ("name", "pitches", "durations_ms", "expected_intervals", "expected_ratios"),
    (
        ("scale", (60, 62, 64, 65), (250.0, 250.0, 250.0, 250.0), (2, 2, 1), ((1, 1),) * 4),
        ("repetition", (60, 60, 60), (100.0, 200.0, 100.0), (0, 0), ((1, 1), (2, 1), (1, 1))),
        ("transposition", (48, 50, 52, 53), (250.0, 250.0, 250.0, 250.0), (2, 2, 1), ((1, 1),) * 4),
    ),
)
def test_p1_golden_melodic_cases(
    name: str,
    pitches: tuple[int, ...],
    durations_ms: tuple[float, ...],
    expected_intervals: tuple[int, ...],
    expected_ratios: tuple[tuple[int, int], ...],
) -> None:
    result = encode_events(_events(pitches, durations_ms), source_schema_version="golden-v1")

    assert name
    assert result.first_midi_pitch == pitches[0]
    assert result.intervals_semitones == expected_intervals
    assert result.durations_ms == durations_ms
    assert tuple((ratio.numerator, ratio.denominator) for ratio in result.rhythm_ratios) == (
        *expected_ratios,
    )
    assert result.gaps_ms == (0.0,) * (len(pitches) - 1)


def test_p1_golden_preserves_gap_sequence() -> None:
    events = (
        InputEvent(0.0, 0.1, 100.0, 60),
        InputEvent(0.15, 0.35, 200.0, 62),
        InputEvent(0.5, 0.6, 100.0, 64),
    )

    result = encode_events(events)

    assert result.intervals_semitones == (2, 2)
    assert result.durations_ms == (100.0, 200.0, 100.0)
    assert result.gaps_ms == pytest.approx((50.0, 150.0))
    assert tuple((r.numerator, r.denominator) for r in result.rhythm_ratios) == (
        (1, 1),
        (2, 1),
        (1, 1),
    )


def test_p1_golden_transposition_preserves_relative_features() -> None:
    original = encode_events(_events((60, 64, 67), (125.0, 250.0, 125.0)))
    transposed = encode_events(_events((72, 76, 79), (125.0, 250.0, 125.0)))

    assert transposed.first_midi_pitch == original.first_midi_pitch + 12
    assert transposed.intervals_semitones == original.intervals_semitones
    assert transposed.durations_ms == original.durations_ms
    assert transposed.rhythm_ratios == original.rhythm_ratios
    assert transposed.gaps_ms == original.gaps_ms


def test_p1_golden_relative_rhythm_is_independent_of_absolute_tempo() -> None:
    fast = encode_events(_events((60, 62, 64), (500.0, 1000.0, 500.0)))
    slow = encode_events(_events((60, 62, 64), (1000.0, 2000.0, 1000.0)))

    assert fast.durations_ms != slow.durations_ms
    assert fast.intervals_semitones == slow.intervals_semitones == (2, 2)
    assert fast.rhythm_ratios == slow.rhythm_ratios
    assert fast.gaps_ms == slow.gaps_ms == (0.0, 0.0)


def test_p1_golden_empty_sequence() -> None:
    result = encode_events((), source_schema_version="empty-v1")

    assert result.to_dict() == {
        "schema_version": "musicdna-melody-encoding-v0",
        # No pitch track was given, so the contour layer is empty. That is
        # "the producer sent none", not "the singer was silent".
        "pitch_contour_semitones": [],
        "source_schema_version": "empty-v1",
        # Empty current input is explicit; None is reserved for legacy unknowns.
        "tuning_normalization": {
            "applied": False,
            "concentration": 0.0,
            "offset_semitones": 0.0,
            "status": "empty",
        },
        "event_count": 0,
        "first_midi_pitch": None,
        "intervals_semitones": [],
        "durations_ms": [],
        "rhythm_ratios": [],
        "gaps_ms": [],
        "ioi_ms": [],
        "ioi_log2_ratios": [],
        "ioi_rhythm_digits": [],
        "median_duration_ratios": [],
        "parsons_code": [],
        "interval_classes": [],
    }


def test_p1_golden_single_note_has_one_rhythm_ratio_and_no_interval() -> None:
    result = encode_events(_events((69,), (375.0,)))

    assert result.event_count == 1
    assert result.first_midi_pitch == 69
    assert result.intervals_semitones == ()
    assert result.durations_ms == (375.0,)
    assert tuple((r.numerator, r.denominator) for r in result.rhythm_ratios) == ((1, 1),)
    assert result.gaps_ms == ()


def test_p1_golden_json_is_deterministic_and_json_compatible() -> None:
    result = encode_events(_events((60, 62), (125.0, 250.0)), source_schema_version="json-v1")

    first = result.to_json()
    second = result.to_json()

    assert first == second
    assert json.loads(first) == {
        "durations_ms": [125.0, 250.0],
        "event_count": 2,
        "first_midi_pitch": 60,
        "gaps_ms": [0.0],
        "intervals_semitones": [2],
        "rhythm_ratios": [
            {"denominator": 1, "numerator": 1},
            {"denominator": 1, "numerator": 2},
        ],
        "schema_version": "musicdna-melody-encoding-v0",
        "pitch_contour_semitones": [],
        "source_schema_version": "json-v1",
        # These events carry no frequencies, so normalisation ran and could
        # not act -- which is a different statement from not having run.
        "tuning_normalization": {
            "applied": False,
            "concentration": 0.0,
            "offset_semitones": 0.0,
            "status": "unavailable",
        },
        "ioi_ms": [125.0],
        "ioi_log2_ratios": [],
        "ioi_rhythm_digits": [],
        "median_duration_ratios": [0.67, 1.33],
        "parsons_code": ["u"],
        "interval_classes": ["+2"],
    }
    assert first.index('"durations_ms"') < first.index('"schema_version"')


def test_p1_tolerates_machine_precision_boundary_overlap() -> None:
    events = (
        InputEvent(0.0, 0.5, 500.0, 60),
        InputEvent(0.5 - 5.6e-17, 1.0, 500.00000000000006, 62),
    )

    assert encode_events(events).intervals_semitones == (2,)
