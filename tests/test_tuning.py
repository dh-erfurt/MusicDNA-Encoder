from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from musicdna_encoder import (
    InputEvent,
    compute_intervals,
    estimate_offset_semitones,
    load_event_sequence,
    normalize_tuning,
)


def _event(midi_pitch: int, *, frequency_hz: float | None = None, index: int = 0) -> InputEvent:
    start = index * 0.5
    return InputEvent(start, start + 0.4, 400.0, midi_pitch, frequency_hz=frequency_hz)


def _detuned(
    midi_pitches: list[int], offset_semitones: float, *, jitter: float = 0.0
) -> tuple[InputEvent, ...]:
    """Events whose frequencies sit a constant offset off the A440 grid.

    ``jitter`` alternates sign per note. A pure offset shifts every note the same
    way and leaves intervals intact; it is the combination of an offset near the
    rounding boundary and per-note variation that splits neighbours across the grid.
    """
    events = []
    for index, midi in enumerate(midi_pitches):
        sung = midi + offset_semitones + (jitter if index % 2 else -jitter)
        frequency = 440.0 * 2.0 ** ((sung - 69.0) / 12.0)
        quantized = round(69.0 + 12.0 * math.log2(frequency / 440.0))
        events.append(_event(quantized, frequency_hz=frequency, index=index))
    return tuple(events)


def test_frequency_hz_is_optional_and_additive_in_the_json_contract() -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.4,
                "duration_ms": 400.0,
                "midi_pitch": 60,
                "frequency_hz": 262.9,
            },
            {"start_seconds": 0.5, "end_seconds": 0.9, "duration_ms": 400.0, "midi_pitch": 62},
        ],
    }
    events = load_event_sequence(payload).events
    assert events[0].frequency_hz == pytest.approx(262.9)
    assert events[1].frequency_hz is None


def test_frequency_hz_is_not_cross_validated_against_midi_pitch() -> None:
    """A disagreement is the detuning signal, not a contract violation."""
    event = InputEvent(0.0, 0.4, 400.0, 60, frequency_hz=269.0)
    assert event.midi_pitch == 60
    assert event.frequency_hz == pytest.approx(269.0)


def test_frequency_hz_rejects_non_positive_and_non_finite_values() -> None:
    for invalid in (0.0, -440.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="frequency_hz"):
            InputEvent(0.0, 0.4, 400.0, 60, frequency_hz=invalid)
    with pytest.raises(TypeError, match="frequency_hz"):
        InputEvent(0.0, 0.4, 400.0, 60, frequency_hz="440")  # type: ignore[arg-type]


@pytest.mark.parametrize("offset", [0.0, 0.2, 0.35, 0.45, 0.49, -0.35, -0.49])
def test_estimate_offset_is_accurate_up_to_the_wrap(offset: float) -> None:
    """Grid residuals are circular; a linear average biases near +-0.5 semitones."""
    midi_floats = [60.0 + step + offset for step in (0, 2, 4, 5, 7, 9, 11, 12)]
    estimated, concentration = estimate_offset_semitones(midi_floats)
    assert estimated == pytest.approx(offset, abs=1e-9)
    assert concentration == pytest.approx(1.0, abs=1e-9)


def test_normalize_tuning_recovers_intervals_a_detuned_grid_destroys() -> None:
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    expected = tuple(right - left for left, right in pairwise(scale))

    events = _detuned(scale, 0.45, jitter=0.1)
    assert compute_intervals(events) != expected

    normalization = normalize_tuning(events)
    assert normalization.applied is True
    assert compute_intervals(normalization.events) == expected


def test_normalize_tuning_reports_the_offset_it_removed() -> None:
    normalization = normalize_tuning(_detuned([60, 64, 67], 0.3))
    assert normalization.offset_semitones == pytest.approx(0.3, abs=1e-6)
    assert normalization.offset_cents == pytest.approx(30.0, abs=1e-4)


def test_normalize_tuning_is_a_no_op_without_continuous_pitch() -> None:
    events = (_event(60, index=0), _event(64, index=1))
    normalization = normalize_tuning(events)
    assert normalization.applied is False
    assert normalization.events == events


def test_normalize_tuning_refuses_partially_annotated_sequences() -> None:
    events = (_event(60, frequency_hz=261.6, index=0), _event(64, index=1))
    assert normalize_tuning(events).applied is False


def test_a_closed_gate_declines_directionless_residuals() -> None:
    """The gate still works; it is simply not the default any more.

    Uniformly spread residuals point nowhere, and a gate above zero refuses to
    read an offset out of them. Measured, that refusal costs more than it saves
    -- every stricter setting yields less than the open gate -- so the default is
    0.0 and this test has to say which gate it is talking about.
    """

    generator = random.Random(7)
    events = tuple(
        _event(60, frequency_hz=200.0 * 2 ** (generator.uniform(0.0, 1.0)), index=index)
        for index in range(400)
    )

    assert normalize_tuning(events, min_concentration=0.05).applied is False
    # The default still applies the offset and reports the residual concentration.
    wide_open = normalize_tuning(events)
    assert wide_open.applied is True
    assert wide_open.concentration < 0.05


def test_normalize_tuning_validates_its_arguments() -> None:
    for invalid_reference in (0.0, float("inf"), 10**400):
        with pytest.raises(ValueError, match="reference_hz"):
            normalize_tuning((), reference_hz=invalid_reference)
    for invalid_concentration in (2.0, float("nan"), 10**400):
        with pytest.raises(ValueError, match="min_concentration"):
            normalize_tuning((), min_concentration=invalid_concentration)


def test_offset_estimation_rejects_an_unrepresentable_integer_without_overflow() -> None:
    with pytest.raises(ValueError, match=r"midi_floats.*supported numeric range"):
        estimate_offset_semitones((10**400,))


def test_normalize_tuning_improves_interval_recovery_on_simulated_queries() -> None:
    """End-to-end: random pitch centre plus per-note jitter, as a hummed query is.

    The A440 grid is only optimal when the singer happens to sit on it. This pins
    the direction of the effect, not a specific rate.
    """
    generator = random.Random(11)
    steps = [0, 1, 2, 3, 4, 5, 7, -1, -2, -3, -4, -5, -7]
    grid_hits = normalized_hits = total = 0

    for _ in range(300):
        intervals = [generator.choice(steps) for _ in range(11)]
        truth = [60]
        for interval in intervals:
            truth.append(truth[-1] + interval)
        centre = generator.uniform(-0.5, 0.5)
        events = []
        for index, midi in enumerate(truth):
            sung = midi + centre + generator.gauss(0.0, 0.22)
            frequency = 440.0 * 2.0 ** ((sung - 69.0) / 12.0)
            quantized = round(69.0 + 12.0 * math.log2(frequency / 440.0))
            events.append(_event(quantized, frequency_hz=frequency, index=index))

        expected = tuple(intervals)
        grid_hits += sum(a == b for a, b in zip(compute_intervals(events), expected, strict=True))
        normalized = normalize_tuning(tuple(events)).events
        normalized_hits += sum(
            a == b for a, b in zip(compute_intervals(normalized), expected, strict=True)
        )
        total += len(expected)

    assert normalized_hits > grid_hits
    assert normalized_hits / total > 0.85
    assert grid_hits / total < 0.85


def test_plaine_easie_octave_signs_stay_inside_the_standard() -> None:
    """Apostrophes cap at four (C7) and commas at three (C1); beyond that PAE is invalid."""
    from musicdna_encoder.encoder import _pae_pitch

    assert _pae_pitch(60) == "'C"  # c', middle C
    assert _pae_pitch(55) == ",G"  # g, the octave below
    assert _pae_pitch(107) == "''''B"  # highest representable
    assert _pae_pitch(24) == ",,,C"  # lowest representable

    for out_of_range in (23, 108):
        with pytest.raises(ValueError, match="Plaine & Easie octave sign"):
            _pae_pitch(out_of_range)


def test_plaine_easie_export_rejects_an_unrepresentable_pitch() -> None:
    """Reachable through a detector octave error, so it must not emit silent garbage."""
    from musicdna_encoder import encode_events

    events = (_event(108, index=0), _event(110, index=1))
    with pytest.raises(ValueError, match="Plaine & Easie octave sign"):
        encode_events(events).to_plaine_easie()
