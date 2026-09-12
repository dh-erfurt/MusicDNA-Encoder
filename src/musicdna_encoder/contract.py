"""External input contract shared by Detector-like producers and the Encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

INPUT_SCHEMA_VERSION = "musicdna-events-v0"
DETECTOR_SCHEMA_VERSIONS = frozenset({"musicdna-analysis-v0"})
SUPPORTED_INPUT_SCHEMA_VERSIONS = frozenset({INPUT_SCHEMA_VERSION, *DETECTOR_SCHEMA_VERSIONS})

# These are deliberately generous product-boundary limits, not musical claims.
# Their purpose is to prevent a malformed timestamp from turning contour
# resampling into an effectively unbounded loop or an input payload into an
# unreasonable allocation. A 24-hour monophonic input at the Detector's current
# hop remains valid.
MAX_TIMELINE_SECONDS = 24.0 * 60.0 * 60.0
MAX_EVENT_COUNT = 1_000_000
MAX_PITCH_TRACK_FRAMES = 10_000_000


def _is_supported_finite_number(value: int | float) -> bool:
    """Return whether a numeric value is representable by the float-based contract."""

    try:
        return math.isfinite(value)
    except OverflowError:
        # Python integers are unbounded, but all Encoder timing and pitch math is
        # defined in finite binary64 values. Reject an integer that cannot be
        # converted instead of leaking the conversion error through the API.
        return False


@dataclass(frozen=True, slots=True)
class InputEvent:
    """One already detected monophonic note, independent of any producer package.

    ``frequency_hz`` is optional and carries the continuous pitch behind
    ``midi_pitch``. Producers quantize to a fixed reference (normally A440), which
    is only lossless when the performance happens to sit on that grid -- a hummed
    query starts on an arbitrary pitch and generally does not. It is deliberately
    not cross-validated against ``midi_pitch``: a disagreement between the two is
    exactly the detuning signal :func:`normalize_tuning` exists to recover.
    """

    start_seconds: float
    end_seconds: float
    duration_ms: float
    midi_pitch: int
    # Keyword-only, so a call written for the old five-positional signature fails
    # loudly instead of landing a confidence value in frequency_hz -- which would
    # have made tuning normalisation act on a number that is not a frequency.
    frequency_hz: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        values = (self.start_seconds, self.end_seconds, self.duration_ms)
        for name, value in zip(
            ("start_seconds", "end_seconds", "duration_ms"), values, strict=True
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
        if not all(_is_supported_finite_number(value) for value in values):
            raise ValueError("event values must be finite and within the supported numeric range")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("events must have non-negative start and positive duration")
        if self.end_seconds > MAX_TIMELINE_SECONDS:
            raise ValueError(f"event end_seconds must not exceed {MAX_TIMELINE_SECONDS:g}")
        if self.duration_ms <= 0 or not math.isclose(
            self.duration_ms, (self.end_seconds - self.start_seconds) * 1000.0, abs_tol=1e-6
        ):
            raise ValueError("duration_ms must match the event time range")
        if not isinstance(self.midi_pitch, int) or isinstance(self.midi_pitch, bool):
            raise TypeError("midi_pitch must be an integer")
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError("midi_pitch must be in the MIDI range 0..127")
        if self.frequency_hz is not None:
            if isinstance(self.frequency_hz, bool) or not isinstance(
                self.frequency_hz, (int, float)
            ):
                raise TypeError("frequency_hz must be a number or None")
            if not _is_supported_finite_number(self.frequency_hz) or self.frequency_hz <= 0:
                raise ValueError(
                    "frequency_hz must be finite, positive and within the supported numeric range"
                )


@dataclass(frozen=True, slots=True)
class PitchTrack:
    """The producer's frame-level pitch, the second thing a detector emits.

    A frame contour is a representation like any other: hertz mapped to semitones on
    a fixed grid. Building that representation belongs here, but comparison does not:
    no alignment, key estimation, or tempo matching happens here because those are
    properties of a pair and this package processes one sequence at a time.

    Unvoiced frames are `None`, which is how a detector says "no pitch at this
    instant" rather than "zero hertz".
    """

    times_seconds: tuple[float, ...]
    frequencies_hz: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if len(self.times_seconds) != len(self.frequencies_hz):
            raise ValueError("pitch_track times and frequencies must be the same length")
        if len(self.times_seconds) > MAX_PITCH_TRACK_FRAMES:
            raise ValueError(f"pitch_track must not exceed {MAX_PITCH_TRACK_FRAMES} frames")
        previous_time: float | None = None
        for time in self.times_seconds:
            if isinstance(time, bool) or not isinstance(time, (int, float)):
                raise TypeError("pitch_track times must be numbers")
            if not _is_supported_finite_number(time):
                raise ValueError(
                    "pitch_track times must be finite and within the supported numeric range"
                )
            if time < 0:
                raise ValueError("pitch_track times must be non-negative")
            if time > MAX_TIMELINE_SECONDS:
                raise ValueError(
                    f"pitch_track times must not exceed {MAX_TIMELINE_SECONDS:g} seconds"
                )
            if previous_time is not None and time <= previous_time:
                raise ValueError("pitch_track times must be strictly increasing")
            previous_time = float(time)
        for frequency in self.frequencies_hz:
            if frequency is None:
                continue
            if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
                raise TypeError("pitch_track frequencies must be numbers or None")
            if not _is_supported_finite_number(frequency) or frequency <= 0:
                raise ValueError(
                    "pitch_track frequencies must be finite, positive and within the supported "
                    "numeric range"
                )
        # A frozen dataclass must not retain caller-owned mutable lists.
        object.__setattr__(self, "times_seconds", tuple(self.times_seconds))
        object.__setattr__(self, "frequencies_hz", tuple(self.frequencies_hz))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PitchTrack:
        times = payload.get("times_seconds")
        frequencies = payload.get("frequencies_hz")
        if not isinstance(times, list) or not isinstance(frequencies, list):
            raise ValueError("pitch_track needs times_seconds and frequencies_hz lists")
        return cls(tuple(times), tuple(frequencies))


@dataclass(frozen=True, slots=True)
class EventSequence:
    """Validated external event sequence plus producer schema provenance."""

    events: tuple[InputEvent, ...]
    source_schema_version: str
    # Every event exactly as the producer wrote it, positionally aligned with
    # `events`. The typed fields above are what the Encoder computes with;
    # this is the record of what it was given, including fields it has no use for.
    #
    # Keep the original event mappings for provenance, including fields that this
    # encoder does not interpret. Provenance belongs to the sequence, not to a note.
    raw_events: tuple[Mapping[str, Any], ...] = ()
    # Optional and additive: producers that emit only notes stay compatible, and a
    # caller that never asks for the contour layer never pays for carrying it.
    pitch_track: PitchTrack | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EventSequence:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be an object")
        if "schema_version" not in payload:
            raise ValueError("payload is missing required field: schema_version")
        source_schema_version = payload["schema_version"]
        if source_schema_version not in SUPPORTED_INPUT_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_INPUT_SCHEMA_VERSIONS))
            raise ValueError(
                f"unsupported input schema_version {source_schema_version!r}; "
                f"supported versions: {supported}"
            )
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("payload.events must be a list")
        if len(raw_events) > MAX_EVENT_COUNT:
            raise ValueError(f"payload.events must not exceed {MAX_EVENT_COUNT} entries")
        events: list[InputEvent] = []
        for raw in raw_events:
            if not isinstance(raw, Mapping):
                raise ValueError("each event must be an object")
            missing = {
                name
                for name in ("start_seconds", "end_seconds", "duration_ms", "midi_pitch")
                if name not in raw
            }
            if missing:
                raise ValueError(f"event is missing required fields: {', '.join(sorted(missing))}")
            events.append(
                InputEvent(
                    raw["start_seconds"],
                    raw["end_seconds"],
                    raw["duration_ms"],
                    raw["midi_pitch"],
                    # Additive and optional: producers that omit it stay v1-compatible.
                    frequency_hz=raw.get("frequency_hz"),
                )
            )
        raw_track = payload.get("pitch_track")
        if raw_track is not None and not isinstance(raw_track, Mapping):
            raise ValueError("payload.pitch_track must be an object")
        return cls(
            tuple(events),
            source_schema_version,
            tuple(MappingProxyType(dict(raw)) for raw in raw_events),
            PitchTrack.from_dict(raw_track) if raw_track is not None else None,
        )


def load_event_sequence(payload: Mapping[str, Any]) -> EventSequence:
    """Parse a ``musicdna-events-v0`` or ``musicdna-analysis-v0`` JSON object."""

    return EventSequence.from_dict(payload)
