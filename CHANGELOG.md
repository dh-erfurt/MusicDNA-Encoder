# Changelog

All notable changes to MusicDNA-Encoder are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] — unreleased

This is the first public, experimental release. The documented API and data
contracts may change before version 1.0.

### Added

- Added deterministic encoding for producer-neutral MusicDNA event sequences
  and Detector analyses.
- Added the `musicdna-melody-encoding-v0` result contract.
- Added Plaine & Easie input and output for the supported monophonic notation
  subset.
- Added citation metadata and third-party provenance information.

### Changed

- The external loader accepts both `musicdna-events-v0` and complete
  `musicdna-analysis-v0` Detector output.
- Tuning provenance is recorded explicitly in encoded results.
- Score-reference contour projection exposes explicit rest policies:
  `omit_rests` and `bridge_nearest`.
- Matching and retrieval are outside the scope of this encoding library.

### Known limitations

- Plaine & Easie output is a lossy projection and does not reproduce original
  millisecond durations or gap placement.
- Tuplets, multiple voices, rhythmic-sequence abbreviations, and editorial
  apparatus are unsupported. Grace notes are intentionally omitted.
