"""Re-quantize an event sequence to its own pitch centre before interval extraction.

Producers quantize continuous pitch to an equal-tempered grid anchored at a fixed
reference, normally A440. That is lossless only when the performance sits on the
grid. A hummed query starts wherever the singer's voice is comfortable, so a
constant offset of a few tens of cents is the norm rather than the exception, and
near a half-semitone offset every note becomes a coin flip between two grid
neighbours -- while the *intervals* retained by the encoding are untouched by it.

    Rounding to a grid can reduce pitch noise, but only after the grid is aligned to
    the performance. This module estimates that offset from ``frequency_hz`` and
    applies it before interval extraction.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .contract import InputEvent, _is_supported_finite_number

# A zero threshold keeps the default behavior deterministic for any non-empty input;
# callers can require a stronger circular concentration when their input warrants it.
DEFAULT_MIN_CONCENTRATION = 0.0
_CENTS_PER_SEMITONE = 100.0


@dataclass(frozen=True, slots=True)
class TuningNormalization:
    """The re-anchored events plus the offset that was removed, for auditability."""

    events: tuple[InputEvent, ...]
    offset_semitones: float
    concentration: float
    applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise TypeError("applied must be a boolean")
        if not _is_supported_finite_number(self.offset_semitones) or not (
            -0.5 <= self.offset_semitones <= 0.5
        ):
            raise ValueError("offset_semitones must be finite and in [-0.5, 0.5]")
        if not _is_supported_finite_number(self.concentration) or not (
            0.0 <= self.concentration <= 1.0
        ):
            raise ValueError("concentration must be finite and in [0, 1]")

    @property
    def offset_cents(self) -> float:
        return self.offset_semitones * _CENTS_PER_SEMITONE


def _midi_float(frequency_hz: float, reference_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(frequency_hz / reference_hz)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def estimate_offset_semitones(
    midi_floats: Sequence[float],
) -> tuple[float, float]:
    """Return the sequence's grid offset in semitones and how concentrated it is.

    Grid residuals are circular with a period of one semitone: +0.49 and -0.49 are
    two hundredths apart, not ninety-eight. Averaging unit vectors respects that;
    a mean or median does not, and biases as the offset approaches the wrap.

    The resultant length doubles as a confidence. It approaches 1 for a sequence
    with a well-defined pitch centre and 0 when the residuals point nowhere, which
    is the case where re-anchoring the grid would be guesswork.
    """
    if not midi_floats:
        return 0.0, 0.0
    total = complex(0.0, 0.0)
    for value in midi_floats:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("midi_floats values must be numbers")
        if not _is_supported_finite_number(value):
            raise ValueError(
                "midi_floats values must be finite and within the supported numeric range"
            )
        residual = value - math.floor(value + 0.5)
        total += cmath.exp(2j * math.pi * residual)
    resultant = total / len(midi_floats)
    return cmath.phase(resultant) / (2.0 * math.pi), min(1.0, abs(resultant))


def normalize_tuning(
    events: Sequence[InputEvent],
    *,
    reference_hz: float = 440.0,
    min_concentration: float = DEFAULT_MIN_CONCENTRATION,
) -> TuningNormalization:
    """Re-quantize events to the grid implied by their own pitch centre.

    Events without ``frequency_hz`` cannot be re-quantized, so the sequence is
    returned unchanged rather than partially normalized. The same applies when the
    residuals are too directionless to support a conclusion.

    Offsets beyond half a semitone stay out of reach by construction: a sequence
    sung a semitone flat is indistinguishable from one sung in tune. Intervals are
    unaffected by that ambiguity, which is why it is not worth chasing.
    """
    if isinstance(reference_hz, bool) or not isinstance(reference_hz, (int, float)):
        raise TypeError("reference_hz must be a number")
    if not _is_supported_finite_number(reference_hz) or reference_hz <= 0:
        raise ValueError(
            "reference_hz must be finite, positive and within the supported numeric range"
        )
    if isinstance(min_concentration, bool) or not isinstance(min_concentration, (int, float)):
        raise TypeError("min_concentration must be a number")
    if not _is_supported_finite_number(min_concentration) or not 0 <= min_concentration <= 1:
        raise ValueError("min_concentration must be finite and in [0, 1]")

    materialized = tuple(events)
    if not materialized or any(event.frequency_hz is None for event in materialized):
        return TuningNormalization(materialized, 0.0, 0.0, False)

    midi_floats = [
        _midi_float(float(event.frequency_hz), reference_hz)
        for event in materialized
        if event.frequency_hz is not None
    ]
    offset, concentration = estimate_offset_semitones(midi_floats)
    if concentration < min_concentration:
        return TuningNormalization(materialized, offset, concentration, False)

    normalized = tuple(
        replace(event, midi_pitch=_round_half_up(midi_float - offset))
        for event, midi_float in zip(materialized, midi_floats, strict=True)
    )
    return TuningNormalization(normalized, offset, concentration, True)
