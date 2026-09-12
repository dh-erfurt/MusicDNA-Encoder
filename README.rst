MusicDNA Encoder
================

MusicDNA Encoder turns validated monophonic note events into deterministic
interval, rhythm, contour, and Plaine & Easie representations for downstream
MusicDNA components. This is release ``0.9.0``, a preliminary, pre-1.0 release;
the documented contracts may still change before 1.0.

Installation
------------

Python 3.10 or newer is required. Either clone the repository (or download
and extract its archive) and install it locally::

    git clone https://github.com/dh-erfurt/MusicDNA-Encoder.git
    cd MusicDNA-Encoder
    python -m pip install .

Or install directly from GitHub::

    python -m pip install "musicdna-encoder @ git+https://github.com/dh-erfurt/MusicDNA-Encoder.git"

See the `pip VCS installation documentation
<https://pip.pypa.io/en/stable/topics/vcs-support/>`_ for the installation
syntax.

Use
---

The input can be the producer-neutral ``musicdna-events-v0`` contract or a
complete MusicDNA-Detector result under ``musicdna-analysis-v0``. The upstream
producer does not have to be MusicDNA-Detector: an external or neural detector
can be used after converting its note events to ``musicdna-events-v0``. The
Encoder does not read audio and does not depend on a particular detector::

    from musicdna_encoder import load_event_sequence, encode_sequence

    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.5,
                "duration_ms": 500.0,
                "midi_pitch": 69,
                "frequency_hz": 440.0,
            },
            {
                "start_seconds": 0.6,
                "end_seconds": 1.1,
                "duration_ms": 500.0,
                "midi_pitch": 72,
                "frequency_hz": 523.251,
            },
        ],
    }

    result = encode_sequence(load_event_sequence(payload))
    print(result.intervals_semitones)       # (3,)
    print(result.to_plaine_easie())         # %G-2 '4A''4C

For a JSON file, save the payload as for example ``events.json`` in any
directory and use the installed console command. Detector JSON can be passed
directly; ``--format pae`` prints Plaine & Easie instead of JSON::

    musicdna-encoder path/to/events.json
    musicdna-encoder path/to/events.json --format pae

The output contract is ``musicdna-melody-encoding-v0``. Plaine & Easie is a
lossy projection for display or interchange, not an archival notation round
trip; tuplets, multiple voices, and editorial apparatus are unsupported.

License
-------

The project is distributed under the MIT License. See ``LICENSE`` and
``THIRD_PARTY_NOTICES.txt`` for the software license and provenance notices.
