"""Read a monophonic Plaine & Easie incipit into events the Encoder can encode.

    The encoder can read monophonic notation as well as event sequences. Keeping the
    notation reader next to the encoder ensures that both directions use the same
    interval and rhythm conventions.

    ## Supported notation

Clef, key signature (including the empty one), time signature, bar lines, notes
with octave marks in either order relative to the duration, accidentals with the
bar rule, dotted durations, rests, ties including the one that runs past the end
of the incipit, and beaming groups.

Note names use the standard uppercase ``A``-``G`` spelling. Lowercase ``g`` and
``q`` are grace-note markers, not note names.

    ## What is not, and stays that way

    Tuplets, multi-voice incipits and the editorial apparatus **raise**. Silently
    dropping those constructs would produce a plausible but wrong melody. Callers
    that prefer to skip such records can catch :class:`PlaineEasieError`.

Grace notes are the one ornament that is skipped rather than refused. The corpus
takes that position first: IncipitSearch publishes `withoutOrnaments` beside
`notes`, so the ornament is editorially separable -- and a hummed query will not
contain it either.

## Pitch, and why a key signature matters here

P&E writes note names, so a key signature is the difference between B and B flat.
The interval layer is what the encoding keeps, and a mis-parsed signature shifts individual
intervals by a semitone -- exactly the error class the representation budget
measures. The signature is therefore applied, and accidentals within a bar
persist to the end of that bar as notation requires.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .contract import InputEvent

__all__ = [
    "PlaineEasieError",
    "PlaineEasieNote",
    "parse_plaine_easie",
    "plaine_easie_to_events",
]


class PlaineEasieError(ValueError):
    """The incipit uses something this parser does not cover, or is malformed."""


# Code -> duration in whole notes. The emitter's _PAE_DURATION_CODES in the same
# order, given their actual note values.
_DURATIONS: dict[str, float] = {
    "1": 1.0,
    "2": 1 / 2,
    "4": 1 / 4,
    "8": 1 / 8,
    "6": 1 / 16,
    "3": 1 / 32,
    "5": 1 / 64,
    "7": 1 / 128,
}
_STEP_SEMITONES: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
_ACCIDENTALS: dict[str, int] = {"x": 1, "b": -1, "n": 0}
# Unsupported constructs, mapped to the reason given when they are met.
_REFUSED: dict[str, str] = {
    "(": "tuplets",
    ")": "tuplets",
    "!": "editorial marks",
    "?": "editorial marks",
}
# Grace notes are skipped rather than refused, and rather than kept. The corpus
# itself takes that position: IncipitSearch publishes `withoutOrnaments` beside
# `notes`, so the ornament is editorially separable from the melody. A hummed
# query will not contain it either.
_GRACE_MARKERS = frozenset(("q", "g"))
_GRACE_END = "r"


@dataclass(frozen=True, slots=True)
class PlaineEasieNote:
    """One parsed note on the P&E time axis; P&E itself has no tempo."""

    midi_pitch: int
    duration_whole_notes: float
    tied_to_next: bool
    start_whole_notes: float = 0.0


def _key_signature_map(token: str) -> dict[str, int]:
    """Turn ``xFC`` or ``bBEA`` into a per-letter alteration map."""

    if not token:
        return {}
    sign, letters = token[0], token[1:].upper()
    if sign not in ("x", "b"):
        raise PlaineEasieError(f"key signature must start with x or b, got {token!r}")
    order = _SHARP_ORDER if sign == "x" else _FLAT_ORDER
    step = 1 if sign == "x" else -1
    expected = order[: len(letters)]
    if tuple(letters) != expected:
        raise PlaineEasieError(
            f"key signature {token!r} is not a standard accumulation; expected "
            f"{sign}{''.join(expected)}"
        )
    return dict.fromkeys(letters, step)


def parse_plaine_easie(incipit: str) -> tuple[PlaineEasieNote, ...]:
    """Parse an incipit into relative-duration notes.

    Raises :class:`PlaineEasieError` on anything outside the covered subset,
    including a construct that would change the melody if ignored.
    """

    if not isinstance(incipit, str):
        raise PlaineEasieError("incipit must be a string")

    key_signature: dict[str, int] = {}
    # A written accidental applies to the same note name in the same octave
    # until the bar line. Key signatures remain octave-independent.
    bar_accidentals: dict[tuple[str, int], int] = {}
    notes: list[PlaineEasieNote] = []
    octave = 4
    duration = _DURATIONS["4"]
    elapsed = 0.0
    pending_duration = False
    last_note_key: tuple[str, int] | None = None

    index = 0
    length = len(incipit)
    while index < length:
        char = incipit[index]
        if char.isspace():
            index += 1
            continue
        if char in _REFUSED:
            raise PlaineEasieError(
                f"{_REFUSED[char]} are not covered by this parser (at position {index})"
            )
        if char in _GRACE_MARKERS:
            # "qq ... r" is a group of grace notes; a lone "q" ornaments the note
            # that follows it. Either way nothing between here and the end of the
            # ornament belongs to the melody.
            next_char = incipit[index + 1 : index + 2]
            if next_char in _GRACE_MARKERS:
                end = incipit.find(_GRACE_END, index)
                if end == -1:
                    raise PlaineEasieError(f"unterminated grace group at position {index}")
                index = end + 1
            else:
                index += 1
                # Octave marks and the duration appear in either order in real
                # records: both '8B and 8'B occur.
                match = re.match(r"[',\d.]*[xbn]?[A-G]", incipit[index:])
                if match is None:
                    raise PlaineEasieError(f"grace note at position {index} has no note")
                index += match.end()
            continue
        if char == "%":  # clef, e.g. %G-2
            match = re.match(r"%[A-Za-z@]?[+-]?\d?", incipit[index:])
            index += match.end() if match else 1
            continue
        if char == "$":  # key signature, e.g. $xFC
            match = re.match(r"\$([xb][A-Ga-g]*)?", incipit[index:])
            following = incipit[index + 1 : index + 2]
            # "$" alone is an empty key signature and is the commonest form of
            # all; only a letter that is neither x nor b is a malformed one.
            if match is not None and match.group(1) is None and following.isalpha():
                raise PlaineEasieError(
                    f"malformed key signature at position {index}: expected x or b "
                    f"after $, got {following!r}"
                )
            assert match is not None
            key_signature = _key_signature_map(match.group(1) or "")
            bar_accidentals = {}
            index += match.end()
            continue
        if char == "@":  # time signature: irrelevant to pitch and relative rhythm
            # Deliberately does not consume a trailing "/": in real records that
            # slash is the bar line before the first measure far more often than
            # it is the cut-time mark, and swallowing it also swallowed the
            # duration behind it. Neither reading changes pitch or relative
            # rhythm, so the harmless choice is the one that keeps the bar line.
            # The denominator is a power of two, and saying so is what stops "@3/48"
            # from reading as a signature of 3/48 and eating the duration behind it.
            match = re.match(r"@(\d+/(?:64|32|16|1|2|4|8)|\d+|[cCoO])", incipit[index:])
            index += match.end() if match else 1
            continue
        if char == "/":  # bar line: accidentals expire
            bar_accidentals = {}
            index += 1
            continue
        if char in "{}":  # beaming groups: engraving, not melody
            index += 1
            continue
        if char == "+":  # tie, written like "_" but reaching across a bar or line
            if not notes or last_note_key is None:
                raise PlaineEasieError(f"tie at position {index} has no preceding note")
            notes[-1] = PlaineEasieNote(
                notes[-1].midi_pitch,
                notes[-1].duration_whole_notes,
                True,
                notes[-1].start_whole_notes,
            )
            index += 1
            continue
        if char in "',":
            marker = re.match(r"'+|,+", incipit[index:])
            assert marker is not None
            token = marker.group(0)
            octave = 3 + len(token) if token[0] == "'" else 4 - len(token)
            index += marker.end()
            continue
        if char in _DURATIONS:
            if pending_duration:
                raise PlaineEasieError("rhythmic sequences are not covered by this parser")
            pending_duration = True
            duration = _DURATIONS[char]
            index += 1
            dots = re.match(r"\.+", incipit[index:])
            if dots:
                held, added = duration, duration
                for _ in range(dots.end()):
                    added /= 2
                    held += added
                duration = held
                index += dots.end()
            continue
        if char == "-":  # rest: consumes time but produces no note
            if notes and notes[-1].tied_to_next:
                raise PlaineEasieError(f"rest at position {index} interrupts a tie")
            elapsed += duration
            pending_duration = False
            last_note_key = None
            index += 1
            continue
        if char == "_":  # tie: written after the first note of the pair
            if not notes or last_note_key is None:
                raise PlaineEasieError(f"tie at position {index} has no preceding note")
            notes[-1] = PlaineEasieNote(
                notes[-1].midi_pitch,
                notes[-1].duration_whole_notes,
                True,
                notes[-1].start_whole_notes,
            )
            index += 1
            continue
        if char in _ACCIDENTALS or char in _STEP_SEMITONES:
            alteration: int | None = None
            if char in _ACCIDENTALS:
                alteration = _ACCIDENTALS[char]
                index += 1
                if index >= length:
                    raise PlaineEasieError("accidental at end of incipit")
                char = incipit[index]
            letter = char.upper()
            if letter not in _STEP_SEMITONES:
                raise PlaineEasieError(f"expected a note letter, got {char!r} at {index}")
            index += 1
            accidental_key = (letter, octave)
            if alteration is None:
                if notes and notes[-1].tied_to_next and last_note_key == accidental_key:
                    # A tie carries its pitch over a bar line, but does not alter
                    # subsequent untied notes in the new bar.
                    alteration = notes[-1].midi_pitch - (
                        (octave + 1) * 12 + _STEP_SEMITONES[letter]
                    )
                else:
                    alteration = bar_accidentals.get(accidental_key, key_signature.get(letter, 0))
            else:
                bar_accidentals[accidental_key] = alteration
            midi = (octave + 1) * 12 + _STEP_SEMITONES[letter] + alteration
            if notes and notes[-1].tied_to_next and notes[-1].midi_pitch == midi:
                merged = notes[-1]
                notes[-1] = PlaineEasieNote(
                    merged.midi_pitch,
                    merged.duration_whole_notes + duration,
                    False,
                    merged.start_whole_notes,
                )
            else:
                notes.append(PlaineEasieNote(midi, duration, False, elapsed))
            elapsed += duration
            pending_duration = False
            last_note_key = accidental_key
            continue
        raise PlaineEasieError(f"unexpected character {char!r} at position {index}")

    if any(note.tied_to_next for note in notes[:-1]):
        raise PlaineEasieError("incipit ends on an unresolved tie")
    if notes and notes[-1].tied_to_next:
        # A tie on the last note reaches past the end of the incipit, which is
        # what "+" is for. There is nothing to merge it with, so it resolves to
        # the note itself rather than to an error.
        notes[-1] = PlaineEasieNote(
            notes[-1].midi_pitch,
            notes[-1].duration_whole_notes,
            False,
            notes[-1].start_whole_notes,
        )
    return tuple(notes)


def plaine_easie_to_events(
    incipit: str, *, whole_note_seconds: float = 2.0
) -> tuple[InputEvent, ...]:
    """Parse an incipit and lay it out in time so the Encoder can encode it.

    P&E carries no tempo, so ``whole_note_seconds`` is a convention rather than a
    measurement. It cancels out of relative layers, while rests remain observable
    as inter-event gaps and onset-to-onset intervals. It exists because
    :class:`InputEvent` is defined in seconds.
    """

    if (
        isinstance(whole_note_seconds, bool)
        or not isinstance(whole_note_seconds, (int, float))
        or not math.isfinite(whole_note_seconds)
        or whole_note_seconds <= 0
    ):
        raise PlaineEasieError("whole_note_seconds must be finite and positive")
    events: list[InputEvent] = []
    for note in parse_plaine_easie(incipit):
        start = note.start_whole_notes * whole_note_seconds
        duration = note.duration_whole_notes * whole_note_seconds
        events.append(InputEvent(start, start + duration, duration * 1000.0, note.midi_pitch))
    return tuple(events)
