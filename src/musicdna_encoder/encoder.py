"""Deterministic interval and relative-rhythm encoding."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from .contract import (
    MAX_EVENT_COUNT,
    MAX_PITCH_TRACK_FRAMES,
    MAX_TIMELINE_SECONDS,
    EventSequence,
    InputEvent,
    PitchTrack,
    _is_supported_finite_number,
    load_event_sequence,
)
from .tuning import normalize_tuning

ENCODER_SCHEMA_VERSION = "musicdna-melody-encoding-v0"
# These legacy names are read compatibility for stored artefacts only. New encodings
# always produce ENCODER_SCHEMA_VERSION. m3-v0.2
# cannot report tuning provenance, m3-v0.3 cannot carry the frame-contour layer,
# and m3-v0.4 cannot distinguish every tuning-disabled/empty state. m3-v0.5 has
# the same required fields as the public v0 contract but retains its legacy name.
_LEGACY_ENCODER_SCHEMA_VERSIONS = frozenset({"m3-v0.2", "m3-v0.3", "m3-v0.4", "m3-v0.5"})
_SUPPORTED_ENCODER_SCHEMA_VERSIONS = frozenset(
    {ENCODER_SCHEMA_VERSION, *_LEGACY_ENCODER_SCHEMA_VERSIONS}
)
_TUNING_REQUIRED_SCHEMA_VERSIONS = frozenset(
    {"m3-v0.3", "m3-v0.4", "m3-v0.5", ENCODER_SCHEMA_VERSION}
)
_CONTOUR_REQUIRED_SCHEMA_VERSIONS = frozenset({"m3-v0.4", "m3-v0.5", ENCODER_SCHEMA_VERSION})
_EXPLICIT_TUNING_STATUS_SCHEMA_VERSIONS = frozenset({"m3-v0.5", ENCODER_SCHEMA_VERSION})
_RATIO_DENOMINATOR_LIMIT = 10_000
_OVERLAP_EPSILON_SECONDS = 1e-9
_PAE_DURATION_CODES = ("1", "2", "4", "8", "6", "3", "5", "7")
# The standard caps octave signs at four apostrophes (C7) and three commas (C1).
_PAE_MIN_MIDI_PITCH = 24
_PAE_MAX_MIDI_PITCH = 107
TuningStatus = Literal["applied", "unavailable", "disabled", "empty"]
_TUNING_STATUSES = frozenset({"applied", "unavailable", "disabled", "empty"})
ScoreContourRestPolicy = Literal["omit_rests", "bridge_nearest"]
_SCORE_CONTOUR_REST_POLICIES = frozenset({"omit_rests", "bridge_nearest"})


@dataclass(frozen=True, slots=True)
class TuningProvenance:
    """Whether the pitch grid was re-anchored, and to what.

    A result that does not say this is ambiguous: legacy files labelled `m3-v0.2`
    may have been produced with different pitch-grid handling. `applied` is False
    when the events carry no frequencies to work from, as with MIDI-only references.
    """

    applied: bool
    offset_semitones: float
    concentration: float
    status: TuningStatus

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise TypeError("tuning applied must be a boolean")
        for name, value in (
            ("offset_semitones", self.offset_semitones),
            ("concentration", self.concentration),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"tuning {name} must be a number")
            if not _is_supported_finite_number(value):
                raise ValueError(
                    f"tuning {name} must be finite and within the supported numeric range"
                )
        if not -0.5 <= self.offset_semitones <= 0.5:
            raise ValueError("tuning offset_semitones must be in [-0.5, 0.5]")
        if not 0.0 <= self.concentration <= 1.0:
            raise ValueError("tuning concentration must be in [0, 1]")
        if self.status not in _TUNING_STATUSES:
            raise ValueError(f"unsupported tuning status {self.status!r}")
        if self.applied != (self.status == "applied"):
            raise ValueError("tuning applied must be true exactly when status is 'applied'")
        if self.status in {"disabled", "empty"} and (
            self.offset_semitones != 0.0 or self.concentration != 0.0
        ):
            raise ValueError(f"tuning status {self.status!r} requires zero estimates")


@dataclass(frozen=True, slots=True)
class RhythmRatio:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.numerator, self.denominator)
        ):
            raise TypeError("rhythm ratio numerator and denominator must be integers")
        if self.numerator <= 0 or self.denominator <= 0:
            raise ValueError("rhythm ratios must be positive")
        if math.gcd(self.numerator, self.denominator) != 1:
            raise ValueError("rhythm ratios must be reduced")


@dataclass(frozen=True, slots=True)
class EncoderResult:
    first_midi_pitch: int | None
    intervals_semitones: tuple[int, ...]
    durations_ms: tuple[float, ...]
    rhythm_ratios: tuple[RhythmRatio, ...]
    gaps_ms: tuple[float, ...]
    ioi_ms: tuple[float, ...]
    ioi_log2_ratios: tuple[float, ...]
    ioi_rhythm_digits: tuple[int, ...]
    median_duration_ratios: tuple[float, ...]
    parsons_code: tuple[str, ...]
    interval_classes: tuple[str, ...]
    event_count: int
    source_schema_version: str
    schema_version: str = ENCODER_SCHEMA_VERSION
    # Absolute semitones on a fixed grid, uncentred. Empty when the producer sent
    # no pitch track, or when the result was loaded from m3-v0.3 or earlier.
    pitch_contour_semitones: tuple[float, ...] = ()
    # None means legacy/unknown only. Current results use explicit status values
    # for applied, unavailable, deliberately disabled and empty input.
    tuning: TuningProvenance | None = None

    def __post_init__(self) -> None:
        _validate_stored_result(self)

    def to_dict(self) -> dict[str, Any]:
        _validate_stored_result(self)
        return {
            "schema_version": self.schema_version,
            "source_schema_version": self.source_schema_version,
            "tuning_normalization": asdict(self.tuning) if self.tuning else None,
            "event_count": self.event_count,
            "first_midi_pitch": self.first_midi_pitch,
            "intervals_semitones": list(self.intervals_semitones),
            "durations_ms": list(self.durations_ms),
            "rhythm_ratios": [asdict(ratio) for ratio in self.rhythm_ratios],
            "gaps_ms": list(self.gaps_ms),
            "ioi_ms": list(self.ioi_ms),
            "ioi_log2_ratios": list(self.ioi_log2_ratios),
            "ioi_rhythm_digits": list(self.ioi_rhythm_digits),
            "median_duration_ratios": list(self.median_duration_ratios),
            "parsons_code": list(self.parsons_code),
            "interval_classes": list(self.interval_classes),
            "pitch_contour_semitones": list(self.pitch_contour_semitones),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EncoderResult:
        """Load a supported result, rejecting malformed and unknown schemas."""

        if not isinstance(payload, Mapping):
            raise TypeError("Encoder result must be an object")
        schema_version = payload.get("schema_version")
        if schema_version not in _SUPPORTED_ENCODER_SCHEMA_VERSIONS:
            raise ValueError(
                "unsupported Encoder result schema_version "
                f"{schema_version!r}; expected {ENCODER_SCHEMA_VERSION!r}"
            )
        required = {
            "schema_version",
            "source_schema_version",
            "event_count",
            "first_midi_pitch",
            "intervals_semitones",
            "durations_ms",
            "rhythm_ratios",
            "gaps_ms",
            "ioi_ms",
            "ioi_log2_ratios",
            "ioi_rhythm_digits",
            "median_duration_ratios",
            "parsons_code",
            "interval_classes",
        }
        if schema_version in _TUNING_REQUIRED_SCHEMA_VERSIONS:
            required.add("tuning_normalization")
        if schema_version in _CONTOUR_REQUIRED_SCHEMA_VERSIONS:
            required.add("pitch_contour_semitones")
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Encoder result is missing required fields: {', '.join(missing)}")
        try:
            rhythm_ratios = tuple(
                RhythmRatio(item["numerator"], item["denominator"])
                for item in _require_list(payload, "rhythm_ratios")
            )
            result = cls(
                payload["first_midi_pitch"],
                tuple(_require_list(payload, "intervals_semitones")),
                tuple(_require_list(payload, "durations_ms")),
                rhythm_ratios,
                tuple(_require_list(payload, "gaps_ms")),
                tuple(_require_list(payload, "ioi_ms")),
                tuple(_require_list(payload, "ioi_log2_ratios")),
                tuple(_require_list(payload, "ioi_rhythm_digits")),
                tuple(_require_list(payload, "median_duration_ratios")),
                tuple(_require_list(payload, "parsons_code")),
                tuple(_require_list(payload, "interval_classes")),
                payload["event_count"],
                payload["source_schema_version"],
                schema_version,
                # Absent in m3-v0.3 and earlier, and absent is not empty-because-silent:
                # those results were produced before the layer existed.
                tuple(_require_list(payload, "pitch_contour_semitones"))
                if "pitch_contour_semitones" in payload
                else (),
                _tuning_from_payload(payload, schema_version),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Encoder result payload: {error}") from error
        return result

    def to_plaine_easie(self, *, base_duration: int = 4) -> str:
        """Encode the monophonic result as a minimal single-line Plaine & Easie incipit.

        The output uses a modern treble clef (``%G-2``), no key or time
        signature, sharp spellings, and durations reconstructed from the
        quantized IOI rhythm layer.  ``base_duration`` is a PAE duration code
        for the first onset interval and defaults to a quarter note (``4``).
        """

        if self.first_midi_pitch is None:
            return "%G-2 "
        pitches = _reconstruct_pitches(self.first_midi_pitch, self.intervals_semitones)
        durations = _pae_durations(
            len(pitches), self.ioi_rhythm_digits, base_duration=base_duration
        )
        active: dict[tuple[str, int], int] = {}
        return "%G-2 " + "".join(
            _pae_note(pitch, duration, active)
            for pitch, duration in zip(pitches, durations, strict=True)
        )


@dataclass(frozen=True, slots=True)
class MatcherLayerProjection:
    """Only the two exact layers used while building a Matcher catalogue."""

    intervals_semitones: tuple[int, ...]
    pitch_contour_semitones: tuple[float, ...]


def _tuning_from_payload(
    payload: Mapping[str, Any], schema_version: str
) -> TuningProvenance | None:
    """Read the tuning provenance, tolerating results written before it existed."""

    raw = payload.get("tuning_normalization")
    if raw is None:
        if schema_version in _EXPLICIT_TUNING_STATUS_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema {schema_version!r} requires explicit tuning_normalization provenance"
            )
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("tuning_normalization must be an object or null")
    required = {"applied", "offset_semitones", "concentration"}
    if schema_version in _EXPLICIT_TUNING_STATUS_SCHEMA_VERSIONS:
        required.add("status")
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"tuning_normalization is missing required fields: {', '.join(missing)}")
    applied = raw["applied"]
    status = raw.get("status", "applied" if applied is True else "unavailable")
    return TuningProvenance(
        applied,
        raw["offset_semitones"],
        raw["concentration"],
        status,
    )


def _require_list(payload: Mapping[str, Any], field: str) -> list[Any]:
    value = payload[field]
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _require_layer_length(field: str, values: Sequence[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{field} length must equal {expected}")


def _require_derived_layer(
    field: str, actual: Sequence[Any], expected: Sequence[Any], source: str
) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{field} must match the values derived from {source}")


def _validate_stored_result(result: EncoderResult) -> None:
    if result.schema_version not in _SUPPORTED_ENCODER_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported Encoder result schema_version {result.schema_version!r}")
    if not isinstance(result.source_schema_version, str) or not result.source_schema_version:
        raise ValueError("source_schema_version must be a non-empty string")
    if result.schema_version in _EXPLICIT_TUNING_STATUS_SCHEMA_VERSIONS and result.tuning is None:
        raise ValueError(
            f"schema {result.schema_version!r} requires explicit tuning_normalization provenance"
        )
    if result.tuning is not None and not isinstance(result.tuning, TuningProvenance):
        raise TypeError("tuning must be TuningProvenance or None")
    if result.tuning is not None:
        result.tuning.__post_init__()

    event_count = result.event_count
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or not 0 <= event_count <= MAX_EVENT_COUNT
    ):
        raise ValueError("event_count must be a non-negative integer")
    first_midi_pitch = result.first_midi_pitch
    if first_midi_pitch is not None and (
        isinstance(first_midi_pitch, bool) or not isinstance(first_midi_pitch, int)
    ):
        raise ValueError("first_midi_pitch must be an integer or None")
    if first_midi_pitch is not None and not 0 <= first_midi_pitch <= 127:
        raise ValueError("first_midi_pitch must be in the MIDI range 0..127")
    transition_count = max(0, event_count - 1)
    ioi_ratio_count = max(0, event_count - 2)
    if result.tuning is not None:
        if result.tuning.status == "empty" and event_count != 0:
            raise ValueError("tuning status 'empty' requires event_count zero")
        if result.tuning.status in {"applied", "unavailable"} and event_count == 0:
            raise ValueError(f"tuning status {result.tuning.status!r} requires at least one event")

    _require_numeric_values("durations_ms", result.durations_ms, positive=True)
    _require_numeric_values("gaps_ms", result.gaps_ms, non_negative=True)
    _require_numeric_values("ioi_ms", result.ioi_ms, positive=True)
    _require_numeric_values("ioi_log2_ratios", result.ioi_log2_ratios)
    # Two-decimal rounding can legitimately turn a short note's ratio into zero.
    _require_numeric_values(
        "median_duration_ratios", result.median_duration_ratios, non_negative=True
    )
    _require_numeric_values("pitch_contour_semitones", result.pitch_contour_semitones)
    if len(result.pitch_contour_semitones) > MAX_PITCH_TRACK_FRAMES:
        raise ValueError(
            f"pitch_contour_semitones must not exceed {MAX_PITCH_TRACK_FRAMES} entries"
        )
    _require_integer_values("intervals_semitones", result.intervals_semitones)
    _require_integer_values("ioi_rhythm_digits", result.ioi_rhythm_digits)
    if any(not -127 <= value <= 127 for value in result.intervals_semitones):
        raise ValueError("intervals_semitones values must be in [-127, 127]")
    if any(not -2 <= value <= 2 for value in result.ioi_rhythm_digits):
        raise ValueError("ioi_rhythm_digits values must be in [-2, 2]")
    if any(not isinstance(value, RhythmRatio) for value in result.rhythm_ratios):
        raise ValueError("rhythm_ratios values must be RhythmRatio objects")
    if any(value not in {"u", "d", "r"} for value in result.parsons_code):
        raise ValueError("parsons_code contains an unsupported value")
    if any(
        value not in {"P", "+2", "-2", "+3", "-3", "+4+", "-4+"}
        for value in result.interval_classes
    ):
        raise ValueError("interval_classes contains an unsupported value")
    maximum_ms = MAX_TIMELINE_SECONDS * 1000.0
    if any(value > maximum_ms for value in (*result.durations_ms, *result.gaps_ms, *result.ioi_ms)):
        raise ValueError(f"stored timing values must not exceed {maximum_ms:g} ms")

    _require_layer_length("durations_ms", result.durations_ms, event_count)
    _require_layer_length("rhythm_ratios", result.rhythm_ratios, event_count)
    _require_layer_length("median_duration_ratios", result.median_duration_ratios, event_count)
    for field, values in (
        ("intervals_semitones", result.intervals_semitones),
        ("gaps_ms", result.gaps_ms),
        ("ioi_ms", result.ioi_ms),
        ("parsons_code", result.parsons_code),
        ("interval_classes", result.interval_classes),
    ):
        _require_layer_length(field, values, transition_count)
    _require_layer_length("ioi_log2_ratios", result.ioi_log2_ratios, ioi_ratio_count)
    _require_layer_length("ioi_rhythm_digits", result.ioi_rhythm_digits, ioi_ratio_count)

    if (event_count == 0) != (first_midi_pitch is None):
        raise ValueError("first_midi_pitch must be None exactly when event_count is zero")
    if first_midi_pitch is not None and any(
        not 0 <= pitch <= 127
        for pitch in _reconstruct_pitches(first_midi_pitch, result.intervals_semitones)
    ):
        raise ValueError("reconstructed pitches must be in the MIDI range 0..127")

    try:
        if any(
            not math.isclose(ioi, duration + gap, rel_tol=0.0, abs_tol=2e-6)
            for duration, gap, ioi in zip(
                result.durations_ms, result.gaps_ms, result.ioi_ms, strict=False
            )
        ):
            raise ValueError("ioi_ms must match the values derived from durations_ms and gaps_ms")
        minimum = min(result.durations_ms) if result.durations_ms else 1.0
        median = statistics.median(result.durations_ms) if result.durations_ms else 1.0
        expected_rhythm_ratios = tuple(
            _ratio(duration, minimum) for duration in result.durations_ms
        )
        expected_median_ratios = tuple(
            round(duration / median, 2) for duration in result.durations_ms
        )
        # Keep validation independent of the public computation hook so creating
        # a complete result still computes this derived layer only once.
        expected_ioi_log2_ratios = tuple(
            _log2_ratio(right, left) for left, right in pairwise(result.ioi_ms)
        )
        _require_derived_layer(
            "rhythm_ratios", result.rhythm_ratios, expected_rhythm_ratios, "durations_ms"
        )
        _require_derived_layer(
            "median_duration_ratios",
            result.median_duration_ratios,
            expected_median_ratios,
            "durations_ms",
        )
        _require_derived_layer(
            "ioi_log2_ratios",
            result.ioi_log2_ratios,
            expected_ioi_log2_ratios,
            "ioi_ms",
        )
        _require_derived_layer(
            "ioi_rhythm_digits",
            result.ioi_rhythm_digits,
            quantize_ioi_rhythm_digits(expected_ioi_log2_ratios),
            "ioi_ms",
        )
        _require_derived_layer(
            "parsons_code",
            result.parsons_code,
            compute_parsons(result.intervals_semitones),
            "intervals_semitones",
        )
        _require_derived_layer(
            "interval_classes",
            result.interval_classes,
            compute_interval_classes(result.intervals_semitones),
            "intervals_semitones",
        )
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid Encoder result payload: {error}") from error


def _require_numeric_values(
    field: str,
    values: Sequence[Any],
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} values must be numbers")
        if not _is_supported_finite_number(value):
            raise ValueError(
                f"{field} values must be finite and within the supported numeric range"
            )
        if positive and value <= 0:
            raise ValueError(f"{field} values must be positive")
        if non_negative and value < 0:
            raise ValueError(f"{field} values must be non-negative")


def _require_integer_values(field: str, values: Sequence[Any]) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} values must be integers")


def _validate_events(events: tuple[InputEvent, ...]) -> None:
    if len(events) > MAX_EVENT_COUNT:
        raise ValueError(f"events must not exceed {MAX_EVENT_COUNT} entries")
    previous_end = 0.0
    for index, event in enumerate(events):
        if not isinstance(event, InputEvent):
            raise TypeError(f"event {index} is not an InputEvent")
        if index and event.start_seconds < previous_end - _OVERLAP_EPSILON_SECONDS:
            raise ValueError("events must be ordered and must not overlap")
        previous_end = event.end_seconds


def _ratio(duration_ms: float, minimum_ms: float) -> RhythmRatio:
    value = Fraction(str(duration_ms / minimum_ms)).limit_denominator(_RATIO_DENOMINATOR_LIMIT)
    return RhythmRatio(value.numerator, value.denominator)


def _log2_ratio(right: float, left: float) -> float:
    ratio = right / left
    if math.isfinite(ratio) and ratio > 0:
        return math.log2(ratio)
    # Preserve the established calculation for ordinary values, but avoid an
    # intermediate overflow or underflow at the binary64 extremes.
    return math.log2(right) - math.log2(left)


def compute_intervals(events: Sequence[InputEvent]) -> tuple[int, ...]:
    """Return signed MIDI intervals, preserving the supplied event order."""

    return tuple(right.midi_pitch - left.midi_pitch for left, right in pairwise(events))


def compute_rhythm_ratios(events: Sequence[InputEvent]) -> tuple[RhythmRatio, ...]:
    """Return reduced duration ratios relative to the shortest supplied event."""

    if not events:
        return ()
    durations = tuple(float(event.duration_ms) for event in events)
    minimum = min(durations)
    return tuple(_ratio(duration, minimum) for duration in durations)


def compute_median_duration_ratios(events: Sequence[InputEvent]) -> tuple[float, ...]:
    """Return deterministic, two-decimal duration/median values for robust comparison."""

    if not events:
        return ()
    durations = tuple(float(event.duration_ms) for event in events)
    median = statistics.median(durations)
    return tuple(round(duration / median, 2) for duration in durations)


def compute_gaps(events: Sequence[InputEvent]) -> tuple[float, ...]:
    """Return non-negative inter-event gaps in milliseconds without quantizing them."""

    return tuple(
        round(max(0.0, (right.start_seconds - left.end_seconds) * 1000.0), 6)
        for left, right in pairwise(events)
    )


def compute_ioi_ms(events: Sequence[InputEvent]) -> tuple[float, ...]:
    """Return unquantized onset-to-onset intervals; offsets do not affect this layer."""

    return tuple(
        (right.start_seconds - left.start_seconds) * 1000.0 for left, right in pairwise(events)
    )


def compute_ioi_log2_ratios(ioi_ms: Sequence[float]) -> tuple[float, ...]:
    """Return raw floating-point log2 ratios of adjacent onset intervals.

    Quantization is intentionally a separate representation layer: callers that
    need robust symbolic tokens should use :func:`quantize_ioi_rhythm_digits`.
    """

    _require_numeric_values("ioi_ms", ioi_ms, positive=True)
    return tuple(_log2_ratio(right, left) for left, right in pairwise(ioi_ms))


def quantize_ioi_rhythm_digits(values: Sequence[float], *, limit: int = 2) -> tuple[int, ...]:
    """Quantize IOI log ratios to deterministic bounded integer rhythm digits.

    Values are rounded to the nearest integer with half steps toward positive
    infinity, then clamped to ``[-limit, limit]``.
    """

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    _require_numeric_values("ioi_log2_ratios", values)
    return tuple(max(-limit, min(limit, math.floor(value + 0.5))) for value in values)


def compute_parsons(intervals: Sequence[int]) -> tuple[str, ...]:
    return tuple("u" if value > 0 else "d" if value < 0 else "r" for value in intervals)


def compute_interval_classes(intervals: Sequence[int]) -> tuple[str, ...]:
    def classify(value: int) -> str:
        magnitude = abs(value)
        if magnitude == 0:
            return "P"
        direction = "+" if value > 0 else "-"
        if magnitude <= 2:
            return direction + "2"
        if magnitude <= 4:
            return direction + "3"
        return direction + "4+"

    return tuple(classify(value) for value in intervals)


def compute_directed_magnitude(intervals: Sequence[int]) -> tuple[str, ...]:
    """Return the experimental direction-plus-magnitude representation.

    This layer is intentionally opt-in and is not included in
    :class:`EncoderResult` or its stored schema.
    """

    def classify(value: int) -> str:
        if value == 0:
            return "S0"
        direction = "U" if value > 0 else "D"
        magnitude = abs(value)
        if magnitude == 1:
            return direction + "1"
        if magnitude == 2:
            return direction + "2"
        if magnitude <= 4:
            return direction + "3-4"
        return direction + "5+"

    return tuple(classify(value) for value in intervals)


def _reconstruct_pitches(first_midi_pitch: int, intervals: Sequence[int]) -> tuple[int, ...]:
    pitches = [first_midi_pitch]
    for interval in intervals:
        pitches.append(pitches[-1] + interval)
    return tuple(pitches)


def _pae_durations(
    event_count: int, rhythm_digits: Sequence[int], *, base_duration: int
) -> tuple[str, ...]:
    if event_count == 0:
        return ()
    code = str(base_duration)
    if code not in _PAE_DURATION_CODES:
        raise ValueError(f"base_duration must be one of {', '.join(_PAE_DURATION_CODES)}")
    index = _PAE_DURATION_CODES.index(code)
    durations = [code]
    for digit in rhythm_digits:
        index = max(0, min(len(_PAE_DURATION_CODES) - 1, index - digit))
        durations.append(_PAE_DURATION_CODES[index])
    while len(durations) < event_count:
        durations.append(durations[-1])
    return tuple(durations[:event_count])


def _pae_pitch(midi_pitch: int) -> str:
    """Encode one pitch with its Plaine & Easie octave sign.

    Apostrophes mark the octaves from c4 up and commas the octaves from c3 down.
    The standard caps them at four apostrophes (C7) and three commas (C1), so
    pitches outside MIDI 24..107 have no valid encoding. Emitting a fifth
    apostrophe would produce output that looks fine and is not parseable, so this
    fails loudly instead -- reachable in practice through a detector octave error.
    """
    if not _PAE_MIN_MIDI_PITCH <= midi_pitch <= _PAE_MAX_MIDI_PITCH:
        raise ValueError(
            f"midi_pitch {midi_pitch} has no Plaine & Easie octave sign; the standard "
            f"allows {_PAE_MIN_MIDI_PITCH}..{_PAE_MAX_MIDI_PITCH} (C1 to B7)"
        )
    names = ("C", "xC", "D", "xD", "E", "F", "xF", "G", "xG", "A", "xA", "B")
    octave = midi_pitch // 12 - 1
    marker = "'" * (octave - 3) if octave >= 4 else "," * (4 - octave)
    return marker + names[midi_pitch % 12]


def _pae_spelling(midi_pitch: int) -> tuple[str, str, int]:
    """Octave marker, note letter and the alteration the letter needs."""

    spelled = _pae_pitch(midi_pitch)
    marker_length = len(spelled) - len(spelled.lstrip("',"))
    marker, body = spelled[:marker_length], spelled[marker_length:]
    if body.startswith("x"):
        return marker, body[1:], 1
    return marker, body, 0


def _pae_note(
    midi_pitch: int,
    duration: str,
    active: dict[tuple[str, int], int] | None = None,
) -> str:
    """Write one note, spelling an accidental only when the bar state needs it.

    Plaine & Easie follows the notation convention that an accidental holds until
    the bar line, and this encoder writes no bar lines -- so a sharp written once
    silently sharpens every later note of that letter. Emitting `x` on each
    altered note and nothing on the others produced incipits that mean something
    other than the melody they came from: C, F sharp, F used to be written
    ``'4C'4xF'4F``, which any conformant reader takes as C, F sharp, F sharp.

    Accidentals are scoped to a note name *and octave*. ``active`` carries that
    state. Passing ``None`` keeps the stateless behaviour for callers that spell
    a single note.
    """

    marker, letter, alteration = _pae_spelling(midi_pitch)
    accidental_key = (letter, midi_pitch // 12 - 1)
    if active is None:
        sign = "x" if alteration else ""
    elif active.get(accidental_key, 0) == alteration:
        sign = ""
    else:
        sign = "x" if alteration else "n"
        active[accidental_key] = alteration
    return marker + duration + sign + letter


CONTOUR_HOP_SECONDS = 0.0464
"""Frame spacing of the contour layer.

The grid is fixed so that contours from different producers can be compared by the
Matcher. The value is part of the representation contract, not a per-call tuning
parameter.
"""


def compute_pitch_contour(
    track: PitchTrack, *, hop_seconds: float = CONTOUR_HOP_SECONDS
) -> tuple[float, ...]:
    """Resample a producer's pitch track onto the contour grid, in absolute semitones.

    Three representation conventions are fixed here because each can be decided
    from one sequence:

    * **Unvoiced frames are dropped, not held.** Holding the last pitch would invent
      a voiced value during an unvoiced interval.
    * **Absolute semitones, never centred.** This is the load-bearing one. Centring
      the offset that makes two contours comparable is a property of the **pair**.
      This package processes one sequence at a time, so it must not decide the key.
    * **Nearest frame wins.** The producer's own rate is not assumed, so a track at
      any hop lands on the same grid.
    """

    if isinstance(hop_seconds, bool) or not isinstance(hop_seconds, (int, float)):
        raise TypeError("hop_seconds must be a number")
    if not _is_supported_finite_number(hop_seconds) or hop_seconds <= 0:
        raise ValueError(
            "hop_seconds must be finite, positive and within the supported numeric range"
        )
    times, frequencies = track.times_seconds, track.frequencies_hz
    if not times:
        return ()
    # Compare before flooring: a finite subnormal hop can overflow the division.
    if times[-1] / hop_seconds >= MAX_PITCH_TRACK_FRAMES:
        raise ValueError(f"resampled pitch contour must not exceed {MAX_PITCH_TRACK_FRAMES} frames")

    contour: list[float] = []
    cursor = 0
    step = 0
    while step * hop_seconds <= times[-1]:
        target = step * hop_seconds
        while cursor + 1 < len(times) and abs(times[cursor + 1] - target) <= abs(
            times[cursor] - target
        ):
            cursor += 1
        if step >= MAX_PITCH_TRACK_FRAMES:
            raise ValueError(
                f"resampled pitch contour must not exceed {MAX_PITCH_TRACK_FRAMES} frames"
            )
        frequency = frequencies[cursor]
        if frequency is not None and frequency > 0:
            contour.append(69.0 + 12.0 * math.log2(frequency / 440.0))
        step += 1
    return tuple(contour)


def _prepare_matcher_layers(
    events: Iterable[InputEvent],
    *,
    normalize_tuning_reference: bool,
    pitch_track: PitchTrack | None,
) -> tuple[
    tuple[InputEvent, ...],
    tuple[int, ...],
    tuple[float, ...],
    TuningProvenance | None,
]:
    """Shared exact work for the complete result and catalogue projection."""

    materialized = tuple(events)
    _validate_events(materialized)
    contour = compute_pitch_contour(pitch_track) if pitch_track is not None else ()
    if not normalize_tuning_reference:
        tuning = TuningProvenance(False, 0.0, 0.0, "disabled")
    elif not materialized:
        tuning = TuningProvenance(False, 0.0, 0.0, "empty")
    else:
        normalization = normalize_tuning(materialized)
        materialized = normalization.events
        tuning = TuningProvenance(
            normalization.applied,
            normalization.offset_semitones,
            normalization.concentration,
            "applied" if normalization.applied else "unavailable",
        )
    return materialized, compute_intervals(materialized), contour, tuning


def encode_events(
    events: Iterable[InputEvent],
    *,
    source_schema_version: str = "unknown",
    normalize_tuning_reference: bool = True,
    pitch_track: PitchTrack | None = None,
) -> EncoderResult:
    """Encode an external sequence into relative melodic/rhythmic features.

    ``normalize_tuning_reference`` re-anchors the grid to the performance's own pitch
    centre before intervals are taken. This reduces rounding errors for performances
    that are consistently detuned from the fixed reference. The result records
    whether normalization ran in :class:`TuningProvenance`.
    """

    materialized, intervals, contour, tuning = _prepare_matcher_layers(
        events,
        normalize_tuning_reference=normalize_tuning_reference,
        pitch_track=pitch_track,
    )
    if not materialized:
        return EncoderResult(
            None,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            0,
            source_schema_version,
            ENCODER_SCHEMA_VERSION,
            contour,
            tuning,
        )
    durations = tuple(float(event.duration_ms) for event in materialized)
    ioi_ms = compute_ioi_ms(materialized)
    ioi_log2_ratios = compute_ioi_log2_ratios(ioi_ms)
    return EncoderResult(
        materialized[0].midi_pitch,
        intervals,
        durations,
        compute_rhythm_ratios(materialized),
        compute_gaps(materialized),
        ioi_ms,
        ioi_log2_ratios,
        quantize_ioi_rhythm_digits(ioi_log2_ratios),
        compute_median_duration_ratios(materialized),
        compute_parsons(intervals),
        compute_interval_classes(intervals),
        len(materialized),
        source_schema_version,
        ENCODER_SCHEMA_VERSION,
        contour,
        tuning,
    )


def encode_sequence(
    sequence: EventSequence, *, normalize_tuning_reference: bool = True
) -> EncoderResult:
    return encode_events(
        sequence.events,
        source_schema_version=sequence.source_schema_version,
        normalize_tuning_reference=normalize_tuning_reference,
        pitch_track=sequence.pitch_track,
    )


def project_matcher_layers(
    sequence: EventSequence, *, normalize_tuning_reference: bool = True
) -> MatcherLayerProjection:
    """Project only the exact interval and contour layers used by the Matcher.

    This is an additive catalogue-build API.  It performs the same validation,
    contour resampling and tuning normalization as :func:`encode_sequence`,
    while intentionally omitting rhythm, gap, Parsons, interval-class, P&E and
    full-result provenance fields that catalogue matching does not consume.
    """

    _, intervals, contour, _ = _prepare_matcher_layers(
        sequence.events,
        normalize_tuning_reference=normalize_tuning_reference,
        pitch_track=sequence.pitch_track,
    )
    return MatcherLayerProjection(intervals, contour)


def _iter_score_pitch_frames(
    events: Sequence[InputEvent], *, frame_seconds: float
) -> Iterable[tuple[float, float]]:
    """Yield the canonical score-derived pitch frames without a track list.

    Score references have no genuine producer pitch track, so this generator creates
    the intermediate contour frames directly from notation. It preserves the frame
    convention and MIDI-to-frequency conversion while keeping one frame in memory at
    a time.
    """

    frame_count = 0
    for event in events:
        time = float(event.start_seconds)
        end = float(event.end_seconds)
        frequency = 440.0 * 2 ** ((int(event.midi_pitch) - 69) / 12)
        while time < end:
            if frame_count >= MAX_PITCH_TRACK_FRAMES:
                raise ValueError(
                    f"score pitch track must not exceed {MAX_PITCH_TRACK_FRAMES} frames"
                )
            yield round(time, 6), frequency
            frame_count += 1
            next_time = time + frame_seconds
            if next_time <= time:
                raise ValueError("frame_seconds is too small to advance the score timestamp")
            time = next_time


def _compute_score_pitch_contour(
    events: Sequence[InputEvent],
    *,
    frame_seconds: float = 0.04,
    hop_seconds: float = CONTOUR_HOP_SECONDS,
    rest_policy: ScoreContourRestPolicy = "omit_rests",
) -> tuple[float, ...]:
    """Compute a score contour with an explicit notation-rest policy.

    ``omit_rests`` is the canonical representation: grid positions inside
    notation rests are omitted. ``bridge_nearest`` uses the nearest-frame
    lookup that bridges rests with a neighbouring note. Both are supported
    representation policies; choose explicitly when reproducing a catalogue.
    """

    if rest_policy not in _SCORE_CONTOUR_REST_POLICIES:
        raise ValueError("rest_policy must be one of: 'omit_rests', 'bridge_nearest'")

    if isinstance(frame_seconds, bool) or not isinstance(frame_seconds, (int, float)):
        raise TypeError("frame_seconds must be a number")
    if not _is_supported_finite_number(frame_seconds) or frame_seconds <= 0:
        raise ValueError(
            "frame_seconds must be finite, positive and within the supported numeric range"
        )
    if isinstance(hop_seconds, bool) or not isinstance(hop_seconds, (int, float)):
        raise TypeError("hop_seconds must be a number")
    if not _is_supported_finite_number(hop_seconds) or hop_seconds <= 0:
        raise ValueError(
            "hop_seconds must be finite, positive and within the supported numeric range"
        )

    # Bound the streaming path too, before advancing its generator. An arbitrarily
    # small positive step can otherwise require unbounded work or stop advancing.
    frame_budget = 0
    for event in events:
        steps = (event.end_seconds - event.start_seconds) / frame_seconds
        if steps > MAX_PITCH_TRACK_FRAMES:
            raise ValueError(f"score pitch track must not exceed {MAX_PITCH_TRACK_FRAMES} frames")
        frame_budget += math.ceil(steps)
        if frame_budget > MAX_PITCH_TRACK_FRAMES:
            raise ValueError(f"score pitch track must not exceed {MAX_PITCH_TRACK_FRAMES} frames")
    if events and events[-1].end_seconds / hop_seconds >= MAX_PITCH_TRACK_FRAMES:
        raise ValueError(f"resampled pitch contour must not exceed {MAX_PITCH_TRACK_FRAMES} frames")

    frames = iter(_iter_score_pitch_frames(events, frame_seconds=frame_seconds))
    try:
        previous_time, previous_frequency = next(frames)
    except StopIteration:
        return ()
    next_frame = next(frames, None)
    contour: list[float] = []
    step = 0
    voiced_event = 0
    while True:
        target = step * hop_seconds
        while next_frame is not None and abs(next_frame[0] - target) <= abs(previous_time - target):
            previous_time, previous_frequency = next_frame
            next_frame = next(frames, None)
        # ``compute_pitch_contour`` stops at the final source frame.  Once the
        # look-ahead is exhausted, this test is equivalent to ``target <= times[-1]``.
        if next_frame is None and target > previous_time:
            break
        # Sparse score frames can choose the nearest pitch, but only the event spans
        # decide voicing; otherwise nearest-frame lookup bridges every notation rest.
        while voiced_event < len(events) and target >= events[voiced_event].end_seconds:
            voiced_event += 1
        target_is_voiced = (
            voiced_event < len(events)
            and target >= events[voiced_event].start_seconds - _OVERLAP_EPSILON_SECONDS
        )
        if (rest_policy == "bridge_nearest" or target_is_voiced) and previous_frequency > 0:
            contour.append(69.0 + 12.0 * math.log2(previous_frequency / 440.0))
        step += 1
    return tuple(contour)


def project_matcher_layers_from_score_events(
    events: Iterable[InputEvent],
    *,
    frame_seconds: float = 0.04,
    normalize_tuning_reference: bool = False,
    rest_policy: ScoreContourRestPolicy = "omit_rests",
) -> MatcherLayerProjection:
    """Project canonical notation events without building ``EventSequence`` objects.

    This is deliberately separate from :func:`project_matcher_layers`: it is for
    notation catalogues whose contour is synthesized from MIDI notes. It does not
    preserve producer provenance and must not be used for detector queries or
    arbitrary external pitch tracks. Tuning normalization is disabled by default
    because score MIDI is already on the fixed grid. ``rest_policy`` controls
    whether grid positions in leading or internal rests are omitted
    (``omit_rests``) or bridged using nearest-frame behaviour
    (``bridge_nearest``). Select the policy explicitly when building a
    catalogue so its retrieval semantics are reproducible.
    """

    materialized = tuple(events)
    _validate_events(materialized)
    contour = _compute_score_pitch_contour(
        materialized, frame_seconds=frame_seconds, rest_policy=rest_policy
    )
    if normalize_tuning_reference:
        normalization = normalize_tuning(materialized)
        materialized = normalization.events
    return MatcherLayerProjection(
        compute_intervals(materialized),
        contour,
    )


def load_encoder_result(payload: Mapping[str, Any]) -> EncoderResult:
    """Load the public encoding contract or a supported legacy result format."""

    return EncoderResult.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode MusicDNA event JSON as a melody encoding")
    parser.add_argument("input", nargs="?", help="event JSON file; defaults to stdin")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--format", choices=("json", "pae"), default="json")
    parser.add_argument("--pae-base-duration", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
        payload = json.loads(text)
        result = encode_sequence(load_event_sequence(payload))
    except (OSError, TypeError, ValueError, OverflowError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    try:
        output = (
            result.to_plaine_easie(base_duration=args.pae_base_duration)
            if args.format == "pae"
            else result.to_json(indent=args.indent)
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
