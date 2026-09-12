"""The contour layer, and the three conventions it is allowed to decide.

The Encoder may fix the grid, rest policy and pitch unit from one sequence. It may
not fix the key: the offset that makes two contours comparable is a property of the
pair, and this package processes one sequence at a time. The test keeps that boundary
explicit.
"""

from __future__ import annotations

import math

import pytest

from musicdna_encoder import (
    CONTOUR_HOP_SECONDS,
    PitchTrack,
    compute_pitch_contour,
    encode_sequence,
    load_event_sequence,
)

A440_EVENT = {
    "start_seconds": 0.0,
    "end_seconds": 1.0,
    "duration_ms": 1000.0,
    "midi_pitch": 69,
}


def _track(frequencies: list[float | None], hop: float = CONTOUR_HOP_SECONDS) -> PitchTrack:
    return PitchTrack(tuple(i * hop for i in range(len(frequencies))), tuple(frequencies))


def test_hertz_becomes_absolute_semitones() -> None:
    contour = compute_pitch_contour(_track([440.0, 880.0, 220.0]))
    assert contour == pytest.approx((69.0, 81.0, 57.0))


def test_unvoiced_frames_are_dropped_not_held() -> None:
    """Unvoiced frames are omitted rather than held at the last pitch."""

    contour = compute_pitch_contour(_track([440.0, None, None, 880.0]))
    assert contour == pytest.approx((69.0, 81.0))


def test_the_contour_is_not_centred() -> None:
    """The load-bearing convention: the key belongs to whoever sees both sides.

    A contour centred here would make the two sides of a comparison agree on a
    median that describes different stretches of melody. Transposing the input must
    therefore transpose the output, one for one.
    """

    plain = compute_pitch_contour(_track([440.0, 493.883, 523.251]))
    up_an_octave = compute_pitch_contour(_track([880.0, 987.767, 1046.502]))
    assert plain == pytest.approx((69.0, 71.0, 72.0), abs=1e-3)
    for low, high in zip(plain, up_an_octave, strict=True):
        assert high - low == pytest.approx(12.0, abs=1e-3)
    assert sum(plain) / len(plain) != pytest.approx(0.0)


def test_a_track_at_another_rate_lands_on_the_same_grid() -> None:
    """Nearest frame wins, so a producer's own hop is never assumed."""

    dense = _track([440.0] * 40, hop=CONTOUR_HOP_SECONDS / 4)
    sparse = _track([440.0] * 10, hop=CONTOUR_HOP_SECONDS)
    assert len(compute_pitch_contour(dense)) == len(compute_pitch_contour(sparse))


def test_a_rejected_hop() -> None:
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="hop_seconds"):
            compute_pitch_contour(_track([440.0]), hop_seconds=invalid)


def test_the_layer_is_absent_without_a_track() -> None:
    """Absent is not "the singer was silent": producers may send notes only."""

    result = encode_sequence(
        load_event_sequence({"schema_version": "musicdna-events-v0", "events": [A440_EVENT]})
    )
    assert result.pitch_contour_semitones == ()


def test_a_payload_with_a_track_carries_the_layer_through() -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [A440_EVENT],
        "pitch_track": {
            "times_seconds": [0.0, CONTOUR_HOP_SECONDS, 2 * CONTOUR_HOP_SECONDS],
            "frequencies_hz": [440.0, None, 880.0],
        },
    }
    result = encode_sequence(load_event_sequence(payload))
    assert result.pitch_contour_semitones == pytest.approx((69.0, 81.0))
    assert result.to_dict()["pitch_contour_semitones"] == pytest.approx([69.0, 81.0])


def test_tuning_normalisation_does_not_touch_the_contour() -> None:
    """The contour is the producer's pitch, not the re-anchored note grid.

    Normalisation exists to fix rounding when continuous pitch is quantised into
    `midi_pitch`. A contour is never quantised, so there is nothing to repair, and
    silently shifting it would put a per-sequence key decision back in.
    """

    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [A440_EVENT | {"frequency_hz": 453.0}],
        "pitch_track": {"times_seconds": [0.0], "frequencies_hz": [453.0]},
    }
    with_normalisation = encode_sequence(load_event_sequence(payload))
    without = encode_sequence(load_event_sequence(payload), normalize_tuning_reference=False)
    expected = 69.0 + 12.0 * math.log2(453.0 / 440.0)
    assert with_normalisation.pitch_contour_semitones == pytest.approx((expected,))
    assert without.pitch_contour_semitones == pytest.approx((expected,))


def test_a_mismatched_track_is_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        PitchTrack((0.0, 0.1), (440.0,))


@pytest.mark.parametrize(
    "times",
    (
        (-0.1,),
        (0.1, 0.1),
        (0.2, 0.1),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, 86_400.1),
        (10**400,),
    ),
)
def test_pitch_track_requires_finite_non_negative_strictly_increasing_times(
    times: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="pitch_track times"):
        PitchTrack(times, (440.0,) * len(times))


@pytest.mark.parametrize("frequency", (0.0, -1.0, float("nan"), float("inf"), 10**400))
def test_pitch_track_requires_positive_finite_voiced_frequencies(frequency: float | int) -> None:
    with pytest.raises(ValueError, match="frequencies"):
        PitchTrack((0.0,), (frequency,))


def test_contour_resampling_refuses_an_unreasonable_output_size() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        compute_pitch_contour(PitchTrack((0.0, 20.0), (440.0, 440.0)), hop_seconds=1e-7)


def test_subnormal_hop_is_rejected_before_integer_overflow() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        compute_pitch_contour(PitchTrack((0.0, 1.0), (440.0, 440.0)), hop_seconds=5e-324)


def test_unrepresentable_hop_is_rejected_without_overflow() -> None:
    with pytest.raises(ValueError, match=r"hop_seconds.*supported numeric range"):
        compute_pitch_contour(PitchTrack((0.0,), (440.0,)), hop_seconds=10**400)


def test_pitch_track_snapshots_mutable_inputs() -> None:
    times = [0.0, 0.1]
    frequencies = [440.0, 440.0]
    track = PitchTrack(times, frequencies)  # type: ignore[arg-type]
    times[1] = float("inf")
    frequencies[0] = -1.0
    assert track.times_seconds == (0.0, 0.1)
    assert track.frequencies_hz == (440.0, 440.0)
    assert compute_pitch_contour(track) == (69.0, 69.0, 69.0)


def test_pitch_track_accepts_exact_timeline_limit() -> None:
    track = PitchTrack((0.0, 86_400.0), (440.0, 880.0))
    assert compute_pitch_contour(track, hop_seconds=86_400.0) == (69.0, 81.0)


def test_resampling_frame_limit_includes_the_last_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    import musicdna_encoder.encoder as encoder_module

    monkeypatch.setattr(encoder_module, "MAX_PITCH_TRACK_FRAMES", 3)
    assert len(compute_pitch_contour(PitchTrack((0.0, 2.0), (440.0, 440.0)), hop_seconds=1)) == 3
    with pytest.raises(ValueError, match="must not exceed"):
        compute_pitch_contour(PitchTrack((0.0, 3.0), (440.0, 440.0)), hop_seconds=1)


def test_a_legacy_m3_result_from_before_the_layer_loads_with_it_empty() -> None:
    from musicdna_encoder import load_encoder_result

    payload = {
        "schema_version": "m3-v0.3",
        "source_schema_version": "x",
        "tuning_normalization": None,
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
    assert load_encoder_result(payload).pitch_contour_semitones == ()
