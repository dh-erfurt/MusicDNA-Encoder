"""The tuning-normalisation parameter of the Encoder entry points."""

from __future__ import annotations

from musicdna_encoder import encode_events
from musicdna_encoder.contract import InputEvent


def _events(midi_floats: list[float], *, with_frequency: bool) -> list[InputEvent]:
    events = []
    for index, value in enumerate(midi_floats):
        frequency = 440.0 * 2 ** ((value - 69) / 12)
        events.append(
            InputEvent(
                start_seconds=index * 0.5,
                end_seconds=index * 0.5 + 0.5,
                duration_ms=500.0,
                midi_pitch=round(value),
                frequency_hz=frequency if with_frequency else None,
            )
        )
    return events


def test_flag_off_leaves_the_rounded_grid_alone() -> None:
    # Sung 40 cents sharp throughout: rounding sends every note up by one.
    events = _events([60.4, 62.4, 64.4], with_frequency=True)
    assert encode_events(events).intervals_semitones == (2, 2)


def test_flag_on_reanchors_a_detuned_performance() -> None:
    """A consistent offset must not survive into the intervals as noise."""
    detuned = _events([60.4, 62.4, 67.4], with_frequency=True)
    intune = _events([60.0, 62.0, 67.0], with_frequency=True)
    assert (
        encode_events(detuned, normalize_tuning_reference=True).intervals_semitones
        == encode_events(intune).intervals_semitones
    )


def test_flag_is_a_no_op_without_frequencies() -> None:
    """The property the engine comparison depends on."""
    events = _events([60.4, 62.4, 64.4], with_frequency=False)
    assert (
        encode_events(events, normalize_tuning_reference=True).intervals_semitones
        == encode_events(events).intervals_semitones
    )


def test_empty_sequence_is_unaffected() -> None:
    assert encode_events([], normalize_tuning_reference=True).intervals_semitones == ()
