"""Public API for the independent MusicDNA Encoder.

The Encoder owns what an interval, a Parsons sign and a rhythm value *are*. What a
deviation between two of them *costs*, and how a corpus is ordered by it, moved
to `musicdna-matcher`; this package does not import it and does not need it.
"""

from .contract import (
    DETECTOR_SCHEMA_VERSIONS,
    INPUT_SCHEMA_VERSION,
    SUPPORTED_INPUT_SCHEMA_VERSIONS,
    EventSequence,
    InputEvent,
    PitchTrack,
    load_event_sequence,
)
from .encoder import (
    CONTOUR_HOP_SECONDS,
    EncoderResult,
    MatcherLayerProjection,
    RhythmRatio,
    ScoreContourRestPolicy,
    TuningProvenance,
    compute_directed_magnitude,
    compute_gaps,
    compute_interval_classes,
    compute_intervals,
    compute_ioi_log2_ratios,
    compute_ioi_ms,
    compute_median_duration_ratios,
    compute_parsons,
    compute_pitch_contour,
    compute_rhythm_ratios,
    encode_events,
    encode_sequence,
    load_encoder_result,
    project_matcher_layers,
    project_matcher_layers_from_score_events,
    quantize_ioi_rhythm_digits,
)
from .plaine_easie import (
    PlaineEasieError,
    PlaineEasieNote,
    parse_plaine_easie,
    plaine_easie_to_events,
)
from .tuning import (
    TuningNormalization,
    estimate_offset_semitones,
    normalize_tuning,
)

__all__ = [
    "CONTOUR_HOP_SECONDS",
    "DETECTOR_SCHEMA_VERSIONS",
    "INPUT_SCHEMA_VERSION",
    "SUPPORTED_INPUT_SCHEMA_VERSIONS",
    "EncoderResult",
    "EventSequence",
    "InputEvent",
    "MatcherLayerProjection",
    "PitchTrack",
    "PlaineEasieError",
    "PlaineEasieNote",
    "RhythmRatio",
    "ScoreContourRestPolicy",
    "TuningNormalization",
    "TuningProvenance",
    "compute_directed_magnitude",
    "compute_gaps",
    "compute_interval_classes",
    "compute_intervals",
    "compute_ioi_log2_ratios",
    "compute_ioi_ms",
    "compute_median_duration_ratios",
    "compute_parsons",
    "compute_pitch_contour",
    "compute_rhythm_ratios",
    "encode_events",
    "encode_sequence",
    "estimate_offset_semitones",
    "load_encoder_result",
    "load_event_sequence",
    "normalize_tuning",
    "parse_plaine_easie",
    "plaine_easie_to_events",
    "project_matcher_layers",
    "project_matcher_layers_from_score_events",
    "quantize_ioi_rhythm_digits",
]
