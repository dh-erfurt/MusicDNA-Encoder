from __future__ import annotations

import random

import pytest

import musicdna_encoder.encoder as encoder_module
from musicdna_encoder import (
    EventSequence,
    InputEvent,
    PitchTrack,
    encode_sequence,
    project_matcher_layers,
    project_matcher_layers_from_score_events,
)


def _sequence(
    events: tuple[InputEvent, ...], frequencies: tuple[float | None, ...]
) -> EventSequence:
    times = tuple(index * 0.04 for index in range(len(frequencies)))
    return EventSequence(events, "projection-test", pitch_track=PitchTrack(times, frequencies))


def _score_sequence(events: tuple[InputEvent, ...]) -> EventSequence:
    """Build a score-derived pitch track for projection tests."""

    times: list[float] = []
    frequencies: list[float] = []
    for event in events:
        time = float(event.start_seconds)
        while time < event.end_seconds:
            times.append(round(time, 6))
            frequencies.append(440.0 * 2 ** ((event.midi_pitch - 69) / 12))
            time += 0.04
    return EventSequence(
        events,
        "score-projection-test",
        pitch_track=PitchTrack(tuple(times), tuple(frequencies)),
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_matcher_projection_is_exact_for_both_consumed_layers(normalize: bool) -> None:
    events = (
        InputEvent(0.0, 0.5, 500.0, 60, frequency_hz=264.7),
        InputEvent(0.6, 1.1, 500.0, 64, frequency_hz=333.5),
        InputEvent(1.2, 2.0, 800.0, 62, frequency_hz=297.1),
    )
    sequence = _sequence(events, (261.63, 261.63, None, 329.63, 293.66, 293.66))

    full = encode_sequence(sequence, normalize_tuning_reference=normalize)
    projected = project_matcher_layers(sequence, normalize_tuning_reference=normalize)

    assert projected.intervals_semitones == full.intervals_semitones
    assert projected.pitch_contour_semitones == full.pitch_contour_semitones


def test_matcher_projection_handles_empty_sequence() -> None:
    sequence = EventSequence((), "empty", pitch_track=PitchTrack((), ()))

    projected = project_matcher_layers(sequence)

    assert projected.intervals_semitones == ()
    assert projected.pitch_contour_semitones == ()


@pytest.mark.parametrize("frame_seconds", (1e-8, 5e-324))
def test_score_projection_refuses_unbounded_frame_generation(frame_seconds: float) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        project_matcher_layers_from_score_events(
            (InputEvent(0.0, 1.0, 1000.0, 60),), frame_seconds=frame_seconds
        )


def test_score_frame_step_must_advance_the_timestamp() -> None:
    import math

    start = 80_000.0
    end = math.nextafter(start, math.inf)
    events = (InputEvent(start, end, (end - start) * 1000, 60),)
    frames = iter(encoder_module._iter_score_pitch_frames(events, frame_seconds=1e-17))
    next(frames)
    with pytest.raises(ValueError, match="too small to advance"):
        next(frames)


def test_score_frame_budget_is_cumulative_and_allows_its_exact_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(encoder_module, "MAX_PITCH_TRACK_FRAMES", 3)
    events = (InputEvent(0, 2, 2000, 60), InputEvent(2, 3, 1000, 62))
    frames = tuple(encoder_module._iter_score_pitch_frames(events, frame_seconds=1))
    assert len(frames) == 3
    with pytest.raises(ValueError, match="must not exceed"):
        encoder_module._compute_score_pitch_contour(events, frame_seconds=0.5)


@pytest.mark.parametrize("direct", (False, True))
def test_projection_enforces_the_event_count_limit(
    monkeypatch: pytest.MonkeyPatch, direct: bool
) -> None:
    monkeypatch.setattr(encoder_module, "MAX_EVENT_COUNT", 1)
    events = (InputEvent(0, 1, 1000, 60), InputEvent(1, 2, 1000, 62))
    with pytest.raises(ValueError, match="events must not exceed"):
        if direct:
            project_matcher_layers_from_score_events(events)
        else:
            project_matcher_layers(EventSequence(events, "test"))


def test_pause_free_direct_score_projection_is_bit_identical_to_historical_adapter() -> None:
    events = (
        InputEvent(0.0, 0.497, 497.0, 60),
        InputEvent(0.497, 1.007, 510.0, 64),
        InputEvent(1.007, 1.767, 760.0, 62),
    )
    historical = project_matcher_layers(_score_sequence(events), normalize_tuning_reference=False)
    direct = project_matcher_layers_from_score_events(events)

    assert direct.intervals_semitones == historical.intervals_semitones
    assert direct.pitch_contour_semitones == historical.pitch_contour_semitones


def test_direct_score_projection_keeps_contour_before_optional_normalization() -> None:
    events = (
        InputEvent(0.0, 0.5, 500.0, 60, frequency_hz=250.0),
        InputEvent(0.6, 1.1, 500.0, 64, frequency_hz=315.0),
        InputEvent(1.2, 1.9, 700.0, 62, frequency_hz=290.0),
    )
    unnormalized = project_matcher_layers_from_score_events(events)
    direct = project_matcher_layers_from_score_events(events, normalize_tuning_reference=True)

    assert unnormalized.intervals_semitones == (4, -2)
    assert direct.intervals_semitones == (4, -1)
    assert direct.pitch_contour_semitones == unnormalized.pitch_contour_semitones


def test_direct_score_projection_omits_leading_rest_without_a_sentinel() -> None:
    projected = project_matcher_layers_from_score_events((InputEvent(0.2, 0.6, 400.0, 60),))

    assert projected.pitch_contour_semitones == (60.0,) * 8


def test_direct_score_projection_omits_long_internal_rest_without_holding_pitch() -> None:
    events = (
        InputEvent(0.0, 0.2, 200.0, 60),
        InputEvent(10.0, 10.2, 200.0, 67),
    )

    contour = project_matcher_layers_from_score_events(events).pitch_contour_semitones

    assert contour == (60.0,) * 5 + (67.0,) * 4
    assert all(value != 0.0 for value in contour)


def test_direct_score_projection_can_bridge_rests_with_nearest_frames() -> None:
    events = (
        InputEvent(0.0, 0.2, 200.0, 60),
        InputEvent(10.0, 10.2, 200.0, 67),
    )

    historical = project_matcher_layers(_score_sequence(events), normalize_tuning_reference=False)
    bridged = project_matcher_layers_from_score_events(events, rest_policy="bridge_nearest")
    omitted = project_matcher_layers_from_score_events(events, rest_policy="omit_rests")

    assert bridged.pitch_contour_semitones == historical.pitch_contour_semitones
    assert len(bridged.pitch_contour_semitones) > len(omitted.pitch_contour_semitones)


def test_score_projection_rejects_unknown_rest_policy() -> None:
    with pytest.raises(ValueError, match="rest_policy"):
        project_matcher_layers_from_score_events(
            (InputEvent(0.0, 0.2, 200.0, 60),),
            rest_policy="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("boundary_delta", (-5e-10, 0.0, 5e-10))
def test_direct_score_projection_tolerates_adjacent_note_rounding(
    boundary_delta: float,
) -> None:
    baseline = (
        InputEvent(0.0, 0.5, 500.0, 60),
        InputEvent(0.5, 1.0, 500.0, 67),
    )
    shifted = (
        baseline[0],
        InputEvent(
            0.5 + boundary_delta,
            1.0 + boundary_delta,
            500.0,
            67,
        ),
    )

    assert (
        project_matcher_layers_from_score_events(shifted).pitch_contour_semitones
        == project_matcher_layers_from_score_events(baseline).pitch_contour_semitones
    )


def test_matcher_projection_does_not_compute_unused_result_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = _sequence(
        (
            InputEvent(0.0, 0.5, 500.0, 60),
            InputEvent(0.5, 1.0, 500.0, 62),
        ),
        (261.63, 293.66),
    )

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("catalogue projection computed an unused layer")

    for name in (
        "compute_rhythm_ratios",
        "compute_gaps",
        "compute_ioi_ms",
        "compute_ioi_log2_ratios",
        "quantize_ioi_rhythm_digits",
        "compute_median_duration_ratios",
        "compute_parsons",
        "compute_interval_classes",
    ):
        monkeypatch.setattr(encoder_module, name, refuse)

    assert project_matcher_layers(
        sequence, normalize_tuning_reference=False
    ).intervals_semitones == (2,)


def test_complete_transform_computes_ioi_log_ratios_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = _sequence(
        (
            InputEvent(0.0, 0.5, 500.0, 60),
            InputEvent(0.5, 1.0, 500.0, 62),
            InputEvent(1.5, 2.0, 500.0, 64),
        ),
        (),
    )
    original = encoder_module.compute_ioi_log2_ratios
    calls = 0

    def counted(values: tuple[float, ...]) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(encoder_module, "compute_ioi_log2_ratios", counted)

    encode_sequence(sequence, normalize_tuning_reference=False)

    assert calls == 1


def test_randomized_projection_parity() -> None:
    randomizer = random.Random(20260829)
    for _ in range(500):
        event_count = randomizer.randint(0, 25)
        events: list[InputEvent] = []
        start = 0.0
        for _ in range(event_count):
            duration = randomizer.choice((0.125, 0.25, 0.5, 0.75, 1.0))
            events.append(
                InputEvent(
                    start,
                    start + duration,
                    duration * 1000.0,
                    randomizer.randint(45, 78),
                )
            )
            start += duration + randomizer.choice((0.0, 0.04, 0.08))
        frequencies = tuple(
            None if randomizer.random() < 0.1 else randomizer.uniform(100.0, 800.0)
            for _ in range(randomizer.randint(0, 80))
        )
        sequence = _sequence(tuple(events), frequencies)

        full = encode_sequence(sequence, normalize_tuning_reference=False)
        projected = project_matcher_layers(sequence, normalize_tuning_reference=False)

        assert projected.intervals_semitones == full.intervals_semitones
        assert projected.pitch_contour_semitones == full.pitch_contour_semitones


def test_randomized_direct_score_projection_parity() -> None:
    randomizer = random.Random(20260830)
    for _ in range(500):
        event_count = randomizer.randint(0, 30)
        events: list[InputEvent] = []
        start = 0.0
        for _ in range(event_count):
            duration = randomizer.choice((0.123, 0.25, 0.49, 0.73, 1.01))
            events.append(
                InputEvent(
                    start,
                    start + duration,
                    duration * 1000.0,
                    randomizer.randint(40, 84),
                )
            )
            start += duration
        materialized = tuple(events)
        historical = project_matcher_layers(
            _score_sequence(materialized), normalize_tuning_reference=False
        )
        direct = project_matcher_layers_from_score_events(materialized)

        assert direct.intervals_semitones == historical.intervals_semitones
        assert direct.pitch_contour_semitones == historical.pitch_contour_semitones
