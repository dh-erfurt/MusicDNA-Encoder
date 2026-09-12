"""Check the Plaine & Easie emitter against an independent, norm-conformant reader.

    This round trip checks the emitted notation with an independent reader so that
    octave, duration and accidental errors cannot pass unnoticed.

The reader is Verovio -- RISM Digital's own renderer and the reference
implementation of the standard. Its published Python wheels are built with
``NO_PAE_SUPPORT`` and refuse Plaine & Easie outright; the JavaScript/WASM
toolkit has the importer. Hence the Node helper next to this file. The test skips
where Node or the ``verovio`` package is missing and runs in CI, which installs
both.

What is asserted is deliberately absolute rather than relative: an incipit is a
melody at a pitch, and reading intervals back correctly from wrong octaves would
still be a wrong incipit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import pytest

from musicdna_encoder import InputEvent, encode_events, parse_plaine_easie

_HELPER = Path(__file__).with_name("pae_roundtrip.js")

# Written by hand rather than imported from the encoder: the point is to state
# what the standard means independently of what our table says it means.
_STEP_SEMITONES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_ACCIDENTAL_ALTERATIONS = {"s": 1, "f": -1, "n": 0, "ss": 2, "x": 2, "ff": -2}

# Plaine & Easie duration code -> the note value MEI names for it.
_EXPECTED_NOTE_VALUES = {"1": 1, "2": 2, "4": 4, "8": 8, "6": 16, "3": 32, "5": 64, "7": 128}

_PITCH_CASES: dict[str, tuple[int, ...]] = {
    # The melody the encoder used to get wrong: an accidental holds to the bar
    # line, so the third note needs a natural sign to come back as F.
    "accidental_then_natural": (60, 66, 65),
    # The other half of the same rule, and the one a bar line would break: the
    # sharp must still be in force eight beats later, with no sign repeated.
    "accidental_holds_across_eight_beats": (60, 66, 60, 60, 60, 60, 60, 60, 66),
    # Alternating altered and natural forms of two different letters.
    "two_letters_altered": (61, 66, 61, 66, 60, 65),
    "chromatic_descent": (72, 71, 70, 69, 68, 67, 66, 65),
    "every_pitch_class": tuple(range(60, 72)),
    "repeated_note": (60, 60, 60, 60),
    # Octave marks in both directions, apostrophes and commas.
    "six_octaves": (36, 48, 60, 72, 84, 96),
    # The extremes the encoder claims are encodable: C1 and B7.
    "encodable_extremes": (24, 107),
    "accidental_across_octaves": (66, 77, 66, 65, 78, 65, 66),
    "accidental_across_lower_octaves": (42, 53, 42, 41, 54, 41, 42),
}

_DURATION_CASE_IOI_SECONDS = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125)


def _node_unavailable() -> str | None:
    if shutil.which("node") is None:
        return "Node is not installed"
    probe = subprocess.run(
        ["node", "-e", "require.resolve('verovio')"],
        capture_output=True,
        text=True,
        cwd=_HELPER.parent,
    )
    if probe.returncode != 0:
        return "the verovio npm package is not installed (npm install verovio)"
    return None


_SKIP_REASON = _node_unavailable()

# CI requires the independent reader; a developer without Node may still skip the
# optional round-trip tests locally.
if _SKIP_REASON is not None and os.environ.get("MUSICDNA_REQUIRE_PAE_READER"):
    raise RuntimeError(
        f"MUSICDNA_REQUIRE_PAE_READER is set, but the reader is unavailable: {_SKIP_REASON}"
    )

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _events(pitches: tuple[int, ...], step: float = 0.25) -> tuple[InputEvent, ...]:
    events = []
    start = 0.0
    for pitch in pitches:
        events.append(InputEvent(start, start + step, step * 1000.0, pitch))
        start += step
    return tuple(events)


def _events_from_iois(iois: tuple[float, ...]) -> tuple[InputEvent, ...]:
    events = []
    start = 0.0
    for ioi in iois:
        events.append(InputEvent(start, start + ioi, ioi * 1000.0, 60))
        start += ioi
    return tuple(events)


def _read_with_verovio(incipits: dict[str, str]) -> dict[str, str | None]:
    completed = subprocess.run(
        ["node", str(_HELPER)],
        input=json.dumps(incipits),
        capture_output=True,
        text=True,
        cwd=_HELPER.parent,
    )
    if completed.returncode != 0:
        raise AssertionError(f"the Verovio helper failed:\n{completed.stderr}")
    parsed: dict[str, str | None] = json.loads(completed.stdout)
    return parsed


def _notes(mei: str) -> list[ElementTree.Element]:
    root = ElementTree.fromstring(mei)
    return [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "note"]


def _midi_pitch(note: ElementTree.Element) -> int:
    written = note.attrib.get("accid") or note.attrib.get("accid.ges")
    if written is None:
        for child in note:
            if child.tag.rsplit("}", 1)[-1] == "accid":
                written = child.attrib.get("accid") or child.attrib.get("accid.ges")
                break
    alteration = _ACCIDENTAL_ALTERATIONS[written] if written is not None else 0
    octave = int(note.attrib["oct"])
    return (octave + 1) * 12 + _STEP_SEMITONES[note.attrib["pname"]] + alteration


@pytest.fixture(scope="module")
def readings() -> dict[str, str | None]:
    """One Node run for every case; starting the WASM toolkit is the expensive part."""

    incipits = {
        name: encode_events(_events(pitches)).to_plaine_easie()
        for name, pitches in _PITCH_CASES.items()
    }
    incipits["durations"] = encode_events(
        _events_from_iois(_DURATION_CASE_IOI_SECONDS)
    ).to_plaine_easie(base_duration=1)
    return _read_with_verovio(incipits)


@pytest.mark.parametrize("name", sorted(_PITCH_CASES))
def test_verovio_reads_back_the_pitches_we_meant(
    name: str, readings: dict[str, str | None]
) -> None:
    mei = readings[name]
    assert mei is not None, f"Verovio refused the incipit for {name}"
    assert tuple(_midi_pitch(note) for note in _notes(mei)) == _PITCH_CASES[name]


def test_verovio_reads_back_the_duration_codes_we_meant(readings: dict[str, str | None]) -> None:
    mei = readings["durations"]
    assert mei is not None, "Verovio refused the duration incipit"
    written = encode_events(_events_from_iois(_DURATION_CASE_IOI_SECONDS)).to_plaine_easie(
        base_duration=1
    )
    codes = [token for token in written.removeprefix("%G-2 ") if token.isdigit()]
    assert [int(note.attrib["dur"]) for note in _notes(mei)] == [
        _EXPECTED_NOTE_VALUES[code] for code in codes
    ]


def test_import_accidentals_and_rests_against_verovio() -> None:
    incipits = {
        "octaves": "%G-2 '4xF''4F'4F",
        "key_and_naturals": "%G-2 $xF '4nF''4F'4F/'4F",
        "flats": "%G-2 '4bB,4B'4B/'4B",
        "rests": "%G-2 8-'4C4.-8-'8D",
    }
    for name, mei in _read_with_verovio(incipits).items():
        assert mei is not None, name
        elapsed = 0.0
        expected = []
        for node in ElementTree.fromstring(mei).iter():
            kind = node.tag.rsplit("}", 1)[-1]
            if kind not in {"note", "rest"}:
                continue
            duration = 1.0 / int(node.attrib["dur"])
            duration *= 2.0 - 2.0 ** -int(node.attrib.get("dots", "0"))
            if kind == "note":
                expected.append((_midi_pitch(node), duration, elapsed))
            elapsed += duration
        actual = parse_plaine_easie(incipits[name])
        assert [
            (n.midi_pitch, n.duration_whole_notes, n.start_whole_notes) for n in actual
        ] == expected


def test_tied_accidental_pitch_is_confirmed_by_verovio() -> None:
    incipit = "%G-2 '4xF+/'4F'4F"
    mei = _read_with_verovio({"tie": incipit})["tie"]
    assert mei is not None
    assert [_midi_pitch(n) for n in _notes(mei)] == [66, 66, 65]
    assert any(node.tag.rsplit("}", 1)[-1] == "tie" for node in ElementTree.fromstring(mei).iter())
    assert [(n.midi_pitch, n.duration_whole_notes) for n in parse_plaine_easie(incipit)] == [
        (66, 0.5),
        (65, 0.25),
    ]
