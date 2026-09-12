"""What the melody encoding must keep constant, stated as properties rather than examples.

The encoder's whole purpose is to turn one performance into something
comparable to another. That promise decomposes into invariances: transposing a
melody must not change its intervals, and playing it faster must not change its
relative rhythm. Those are falsifiable without a corpus, without the Detector and
without the Matcher, which keeps these invariants independent from corpus-specific
evaluation.

The battery separates three groups deliberately:

* the four layers the Matcher consumes (`intervals_semitones`,
  `interval_classes`, `parsons_code`, `ioi_rhythm_digits`) must be exactly
  invariant, because a difference there is a difference in retrieval;
* the derived continuous fields may carry floating-point noise, so they get a
  bound rather than equality;
* the raw fields (`durations_ms`, `ioi_ms`, `gaps_ms`) are *not* tempo-invariant
  by design and are asserted to scale, so that a future change which quietly
  normalises them fails here.

One known deviation is pinned rather than hidden: see
``test_median_duration_ratios_moves_at_most_one_rounding_step``.
"""

from __future__ import annotations

import random

import pytest

from musicdna_encoder import InputEvent, encode_events

# The Matcher's MatchLayer names exactly these four. Exact invariance required.
MATCHER_LAYERS = (
    "intervals_semitones",
    "interval_classes",
    "parsons_code",
    "ioi_rhythm_digits",
)
# Relative representations that must also survive a tempo change exactly.
EXACT_TEMPO_INVARIANT = (*MATCHER_LAYERS, "rhythm_ratios")
# Raw milliseconds: deliberately not tempo-invariant.
RAW_TIME_FIELDS = ("durations_ms", "ioi_ms", "gaps_ms")

TEMPO_FACTORS = (0.25, 0.8, 1.0, 1.1, 1.37, 1.5, 2.0, 1 / 3, 2.718)
TRANSPOSITIONS = (-24, -12, -7, -1, 0, 1, 5, 12)


def _melody(seed: int, length: int = 12) -> tuple[InputEvent, ...]:
    """A deterministic pseudo-performance: uneven durations, some rests, leaps."""

    rng = random.Random(seed)
    events: list[InputEvent] = []
    start = 0.0
    pitch = rng.randint(52, 76)
    for _ in range(length):
        duration = rng.choice([0.09, 0.125, 0.2411, 0.25, 0.375, 0.5, 0.7333])
        events.append(InputEvent(start, start + duration, duration * 1000.0, pitch))
        start += duration + rng.choice([0.0, 0.0, 0.0, 0.031, 0.25])
        pitch = max(36, min(88, pitch + rng.choice([-12, -5, -2, -1, 0, 1, 2, 3, 7])))
    return tuple(events)


def _melodies(count: int = 60) -> list[tuple[InputEvent, ...]]:
    return [_melody(seed) for seed in range(count)]


def _scaled(events: tuple[InputEvent, ...], factor: float) -> tuple[InputEvent, ...]:
    """Play the same performance faster or slower. Durations follow the timestamps."""

    return tuple(
        InputEvent(
            event.start_seconds * factor,
            event.end_seconds * factor,
            (event.end_seconds * factor - event.start_seconds * factor) * 1000.0,
            event.midi_pitch,
        )
        for event in events
    )


def _shifted(events: tuple[InputEvent, ...], delta: float) -> tuple[InputEvent, ...]:
    return tuple(
        InputEvent(
            event.start_seconds + delta,
            event.end_seconds + delta,
            event.duration_ms,
            event.midi_pitch,
        )
        for event in events
    )


def _transposed(events: tuple[InputEvent, ...], steps: int) -> tuple[InputEvent, ...]:
    return tuple(
        InputEvent(
            event.start_seconds, event.end_seconds, event.duration_ms, event.midi_pitch + steps
        )
        for event in events
    )


@pytest.mark.parametrize("steps", TRANSPOSITIONS)
def test_transposition_changes_nothing_but_the_anchor(steps: int) -> None:
    """Every relative field survives transposition; only first_midi_pitch moves."""

    for events in _melodies():
        base = encode_events(events).to_dict()
        moved = encode_events(_transposed(events, steps)).to_dict()
        for field in set(base) - {"first_midi_pitch"}:
            assert base[field] == moved[field], f"transposition by {steps} changed {field}"
        if steps and base["first_midi_pitch"] is not None:
            assert moved["first_midi_pitch"] == base["first_midi_pitch"] + steps


@pytest.mark.parametrize("delta", (0.5, 5.0, 100.0))
def test_a_later_recording_start_changes_nothing(delta: float) -> None:
    """Where a performance sits on the clock is not part of the melody."""

    for events in _melodies():
        base = encode_events(events).to_dict()
        moved = encode_events(_shifted(events, delta)).to_dict()
        for field in EXACT_TEMPO_INVARIANT:
            assert base[field] == moved[field], f"a {delta}s offset changed {field}"


@pytest.mark.parametrize("factor", TEMPO_FACTORS)
def test_tempo_leaves_the_matcher_layers_untouched(factor: float) -> None:
    """The layers retrieval actually ranks on must not know how fast it was sung."""

    for events in _melodies():
        base = encode_events(events).to_dict()
        faster = encode_events(_scaled(events, factor)).to_dict()
        for field in EXACT_TEMPO_INVARIANT:
            assert base[field] == faster[field], f"tempo x{factor} changed {field}"


@pytest.mark.parametrize("factor", (0.25, 0.8, 1.5, 2.0))
def test_raw_millisecond_fields_do_scale_with_tempo(factor: float) -> None:
    """The counterpart: raw durations are raw, and a change here would be silent."""

    events = _melody(0)
    base = encode_events(events).to_dict()
    faster = encode_events(_scaled(events, factor)).to_dict()
    for field in RAW_TIME_FIELDS:
        for before, after in zip(base[field], faster[field], strict=True):
            assert after == pytest.approx(before * factor, rel=1e-9, abs=1e-9), field


def test_median_duration_ratios_moves_at_most_one_rounding_step() -> None:
    """A known, bounded deviation -- asserted rather than wished away.

    ``compute_median_duration_ratios`` rounds to two decimals. Tempo scaling is
    exact in arithmetic but not in binary floating point, so a ratio sitting on a
    rounding boundary can land on the other side.

    This is tolerable only because the field is diagnostic -- the Matcher does not
    consume it. The bound is what is under test: if a change ever moves this field
    by more than one rounding step, or moves a layer the Matcher does read, the
    tests above fail first.
    """

    worst = 0.0
    for events in _melodies():
        base = encode_events(events).to_dict()["median_duration_ratios"]
        for factor in TEMPO_FACTORS:
            faster = encode_events(_scaled(events, factor)).to_dict()["median_duration_ratios"]
            worst = max((abs(a - b) for a, b in zip(base, faster, strict=True)), default=0.0)
            assert worst <= 0.01 + 1e-12, f"tempo x{factor} moved a ratio by {worst}"


def test_a_single_note_has_no_intervals_but_one_rhythm_value() -> None:
    """The degenerate case the contract names explicitly."""

    result = encode_events((InputEvent(0.0, 0.5, 500.0, 60),)).to_dict()
    assert result["intervals_semitones"] == []
    assert result["parsons_code"] == []
    assert len(result["rhythm_ratios"]) == 1
    assert result["first_midi_pitch"] == 60


def test_an_empty_sequence_is_empty_everywhere() -> None:
    result = encode_events(()).to_dict()
    assert result["first_midi_pitch"] is None
    assert result["event_count"] == 0
    for field in EXACT_TEMPO_INVARIANT:
        assert result[field] == []
