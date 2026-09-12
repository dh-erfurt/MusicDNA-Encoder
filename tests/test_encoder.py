from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

import musicdna_encoder.encoder as encoder_module
from musicdna_encoder import (
    DETECTOR_SCHEMA_VERSIONS,
    INPUT_SCHEMA_VERSION,
    SUPPORTED_INPUT_SCHEMA_VERSIONS,
    InputEvent,
    compute_interval_classes,
    compute_ioi_log2_ratios,
    compute_parsons,
    encode_events,
    load_encoder_result,
    load_event_sequence,
    quantize_ioi_rhythm_digits,
)
from musicdna_encoder.encoder import main


def _events() -> tuple[InputEvent, ...]:
    return (
        InputEvent(0.0, 0.25, 250.0, 69),
        InputEvent(0.30, 0.80, 500.0, 71),
        InputEvent(0.80, 1.80, 1000.0, 69),
    )


def test_encoder_contract_emits_intervals_ratios_and_gaps() -> None:
    result = encode_events(_events(), source_schema_version="musicdna-analysis-v0")

    assert result.schema_version == "musicdna-melody-encoding-v0"
    assert result.first_midi_pitch == 69
    assert result.intervals_semitones == (2, -2)
    assert result.rhythm_ratios[0].numerator == 1
    assert result.rhythm_ratios[1].numerator == 2
    assert result.rhythm_ratios[2].numerator == 4
    assert result.gaps_ms == pytest.approx((50.0, 0.0))
    assert json.loads(result.to_json())["source_schema_version"] == "musicdna-analysis-v0"


def test_detector_analysis_v0_fixture_is_parsed_without_detector_import() -> None:
    fixture = Path(__file__).parent / "fixtures" / "detector-0.5.0-musicdna-analysis-v0-output.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    sequence = load_event_sequence(payload)
    result = encode_events(
        sequence.events,
        source_schema_version=sequence.source_schema_version,
        pitch_track=sequence.pitch_track,
    )

    assert sequence.source_schema_version == "musicdna-analysis-v0"
    assert result.intervals_semitones == (2,)
    assert result.gaps_ms == (50.0,)
    assert len(result.pitch_contour_semitones) == 7
    assert sequence.raw_events[0]["pitch_name"] == "A4"


@pytest.mark.parametrize(
    "schema_version",
    (None, "3", "musicdna-events-v1", "musicdna-analysis-v1", "unknown", 3),
)
def test_input_contract_rejects_missing_or_unsupported_schema_versions(
    schema_version: object,
) -> None:
    payload: dict[str, object] = {"events": []}
    if schema_version is not None:
        payload["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        load_event_sequence(payload)


def test_supported_input_schema_versions_are_public_and_explicit() -> None:
    assert INPUT_SCHEMA_VERSION == "musicdna-events-v0"
    assert frozenset({"musicdna-analysis-v0"}) == DETECTOR_SCHEMA_VERSIONS
    assert (
        frozenset({"musicdna-events-v0", "musicdna-analysis-v0"}) == SUPPORTED_INPUT_SCHEMA_VERSIONS
    )
    assert encoder_module.ENCODER_SCHEMA_VERSION == "musicdna-melody-encoding-v0"


def test_input_contract_rejects_an_event_missing_a_required_field() -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [{"start_seconds": 0.0, "end_seconds": 0.5, "midi_pitch": 69}],
    }

    with pytest.raises(ValueError, match="missing required fields: duration_ms"):
        load_event_sequence(payload)


@pytest.mark.parametrize("invalid", (float("inf"), float("nan"), 10**400))
def test_input_contract_rejects_non_finite_or_unrepresentable_event_numbers(
    invalid: float | int,
) -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.5,
                "duration_ms": 500.0,
                "midi_pitch": 69,
                "frequency_hz": invalid,
            }
        ],
    }

    with pytest.raises(ValueError, match=r"frequency_hz.*supported numeric range"):
        load_event_sequence(payload)


def test_result_loader_round_trips_current_and_reads_internal_legacy_versions() -> None:
    result = encode_events(_events(), source_schema_version="musicdna-events-v0")

    assert load_encoder_result(result.to_dict()) == result
    assert load_encoder_result(encode_events(()).to_dict()) == encode_events(())
    assert load_encoder_result(encode_events(_events()[:1]).to_dict()) == encode_events(
        _events()[:1]
    )
    # m3-v0.2 is legacy but readable; it simply cannot say whether its pitches
    # were re-anchored, which is what the v0.3 field records.
    legacy = result.to_dict() | {"schema_version": "m3-v0.2"}
    del legacy["tuning_normalization"]
    assert load_encoder_result(legacy).tuning is None
    legacy_v05 = result.to_dict() | {"schema_version": "m3-v0.5"}
    assert load_encoder_result(legacy_v05).to_dict() == legacy_v05
    for schema_version in (
        "m3-v0.6",
        "m3-v9.9",
        "musicdna-melody-encoding-v1",
        None,
    ):
        payload = result.to_dict() | {"schema_version": schema_version}
        with pytest.raises(ValueError, match="unsupported Encoder result schema_version"):
            load_encoder_result(payload)


def test_tuning_provenance_distinguishes_unknown_disabled_empty_and_unavailable() -> None:
    legacy = encode_events(_events()).to_dict() | {"schema_version": "m3-v0.2"}
    del legacy["tuning_normalization"]

    assert load_encoder_result(legacy).tuning is None
    assert encode_events((), normalize_tuning_reference=True).tuning.status == "empty"  # type: ignore[union-attr]
    assert encode_events((), normalize_tuning_reference=False).tuning.status == "disabled"  # type: ignore[union-attr]
    assert encode_events(_events(), normalize_tuning_reference=True).tuning.status == (  # type: ignore[union-attr]
        "unavailable"
    )


@pytest.mark.parametrize("field", ("source_schema_version", "event_count", "durations_ms"))
def test_result_loader_rejects_missing_required_fields(field: str) -> None:
    payload = encode_events(_events()).to_dict()
    del payload[field]

    with pytest.raises(ValueError, match=rf"missing required fields:.*{field}"):
        load_encoder_result(payload)


@pytest.mark.parametrize("invalid", (-0.1, float("nan"), float("inf")))
def test_result_loader_rejects_invalid_durations_before_deriving_layers(invalid: float) -> None:
    payload = encode_events(_events()).to_dict()
    payload["durations_ms"][0] = invalid

    with pytest.raises(ValueError, match="durations_ms values must be"):
        load_encoder_result(payload)


@pytest.mark.parametrize("invalid", (-0.1, 1.1, float("nan"), float("inf")))
def test_result_loader_rejects_invalid_tuning_concentrations(invalid: float) -> None:
    payload = encode_events(_events()).to_dict()
    payload["tuning_normalization"]["concentration"] = invalid

    with pytest.raises(ValueError, match="tuning concentration"):
        load_encoder_result(payload)


@pytest.mark.parametrize("schema_version", ("musicdna-melody-encoding-v0", "m3-v0.5"))
def test_result_loader_requires_strict_tuning_types_and_v05_status(
    schema_version: str,
) -> None:
    payload = encode_events(_events()).to_dict()
    payload["schema_version"] = schema_version
    payload["tuning_normalization"]["applied"] = "false"
    with pytest.raises(ValueError, match="tuning applied must be a boolean"):
        load_encoder_result(payload)

    payload = encode_events(_events()).to_dict()
    payload["schema_version"] = schema_version
    del payload["tuning_normalization"]["status"]
    with pytest.raises(ValueError, match="tuning_normalization is missing required fields: status"):
        load_encoder_result(payload)

    payload = encode_events(_events()).to_dict()
    payload["schema_version"] = schema_version
    payload["tuning_normalization"]["status"] = "empty"
    with pytest.raises(ValueError, match="tuning status 'empty' requires event_count zero"):
        load_encoder_result(payload)


@pytest.mark.parametrize("invalid", ({}, "", {69: "ignored"}, (69.0,)))
def test_result_loader_requires_a_json_array_for_the_contour(invalid: object) -> None:
    payload = encode_events(()).to_dict() | {"pitch_contour_semitones": invalid}
    with pytest.raises(ValueError, match="pitch_contour_semitones must be a list"):
        load_encoder_result(payload)


def test_short_note_with_zero_rounded_median_ratio_round_trips() -> None:
    events = (
        InputEvent(0.0, 0.001, 1.0, 60),
        InputEvent(0.001, 1.001, 1000.0, 60),
        InputEvent(1.001, 2.001, 1000.0, 60),
    )
    result = encode_events(events)
    assert result.median_duration_ratios == (0.0, 1.0, 1.0)
    assert load_encoder_result(result.to_dict()) == result
    legacy = result.to_dict() | {"schema_version": "m3-v0.4"}
    del legacy["tuning_normalization"]["status"]
    assert load_encoder_result(legacy).median_duration_ratios == (0.0, 1.0, 1.0)


def test_validation_does_not_rescan_all_durations_for_each_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = encode_events(tuple(InputEvent(i, i + 1, 1000, 60) for i in range(100)))
    original_min = min
    original_median = encoder_module.statistics.median
    calls = {"minimum": 0, "median": 0}

    def counted_min(*args: object, **kwargs: object) -> object:
        if len(args) == 1 and args[0] is result.durations_ms:
            calls["minimum"] += 1
        return original_min(*args, **kwargs)  # type: ignore[call-overload]

    def counted_median(values: tuple[float, ...]) -> float:
        calls["median"] += 1
        return original_median(values)

    monkeypatch.setattr(encoder_module, "min", counted_min, raising=False)
    monkeypatch.setattr(encoder_module.statistics, "median", counted_median)
    result.to_dict()
    assert calls == {"minimum": 1, "median": 1}


@pytest.mark.parametrize("first,interval", ((127, 1), (0, -1), (126, 2)))
def test_result_loader_rejects_out_of_range_reconstructed_pitches(
    first: int, interval: int
) -> None:
    payload = encode_events(_events()[:2]).to_dict()
    payload.update(
        first_midi_pitch=first,
        intervals_semitones=[interval],
        parsons_code=list(compute_parsons((interval,))),
        interval_classes=list(compute_interval_classes((interval,))),
    )
    with pytest.raises(ValueError, match="reconstructed pitches"):
        load_encoder_result(payload)


def test_serialization_revalidates_tuning_provenance() -> None:
    result = encode_events(_events())
    assert result.tuning is not None
    object.__setattr__(result.tuning, "applied", "false")
    with pytest.raises(TypeError, match="tuning applied must be a boolean"):
        result.to_json()


@pytest.mark.parametrize("arguments", (("--pae-base-duration", "9"), ()))
def test_cli_reports_pae_output_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arguments: tuple[str, ...]
) -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [{"start_seconds": 0, "end_seconds": 1, "duration_ms": 1000, "midi_pitch": 0}],
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as error:
        main(["--format", "pae", *arguments])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "error:" in output.err
    assert "Traceback" not in output.err
    assert output.out == ""


@pytest.mark.parametrize("invalid", (float("inf"), 10**400))
def test_cli_reports_extreme_numeric_input_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    invalid: float | int,
) -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [
            {
                "start_seconds": 0,
                "end_seconds": 1,
                "duration_ms": 1000,
                "midi_pitch": 60,
                "frequency_hz": invalid,
            }
        ],
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2
    output = capsys.readouterr()
    assert "frequency_hz" in output.err
    assert "Traceback" not in output.err
    assert output.out == ""


def test_to_dict_revalidates_a_corrupted_result() -> None:
    result = encode_events(_events())
    object.__setattr__(result, "durations_ms", (-1.0, 500.0, 1000.0))

    with pytest.raises(ValueError, match="durations_ms values must be positive"):
        result.to_dict()


@pytest.mark.parametrize(
    ("field", "replacement", "source"),
    (
        ("rhythm_ratios", [{"numerator": 1, "denominator": 1}] * 3, "durations_ms"),
        ("median_duration_ratios", [1.0, 1.0, 1.0], "durations_ms"),
        ("ioi_log2_ratios", [0.0], "ioi_ms"),
        ("ioi_rhythm_digits", [0], "ioi_ms"),
        ("parsons_code", ["d", "u"], "intervals_semitones"),
        ("interval_classes", ["-2", "+2"], "intervals_semitones"),
    ),
)
def test_result_loader_rejects_inconsistent_derived_layers(
    field: str, replacement: list[object], source: str
) -> None:
    payload = encode_events(_events()).to_dict()
    payload[field] = replacement

    with pytest.raises(ValueError, match=rf"{field} must match .* {source}"):
        load_encoder_result(payload)


@pytest.mark.parametrize("field", ("gaps_ms", "ioi_ms"))
def test_result_loader_rejects_inconsistent_ioi_timing(field: str) -> None:
    payload = encode_events(_events()).to_dict()
    payload[field][0] += 1.0

    with pytest.raises(ValueError, match=r"ioi_ms must match .* durations_ms and gaps_ms"):
        load_encoder_result(payload)


@pytest.mark.parametrize(
    "field",
    (
        "durations_ms",
        "rhythm_ratios",
        "median_duration_ratios",
        "intervals_semitones",
        "gaps_ms",
        "ioi_ms",
        "ioi_log2_ratios",
        "ioi_rhythm_digits",
        "parsons_code",
        "interval_classes",
    ),
)
def test_result_loader_rejects_inconsistent_layer_lengths(field: str) -> None:
    payload = encode_events(_events()).to_dict()
    payload[field] = [*payload[field], payload[field][-1]]

    with pytest.raises(ValueError, match=rf"{field} length must equal"):
        load_encoder_result(payload)


def test_result_loader_rejects_first_pitch_without_an_event() -> None:
    payload = encode_events(()).to_dict() | {"first_midi_pitch": 60}

    with pytest.raises(ValueError, match="first_midi_pitch must be None"):
        load_encoder_result(payload)


def test_result_loader_rejects_non_integer_first_pitch() -> None:
    payload = encode_events(_events()[:1]).to_dict() | {"first_midi_pitch": "69"}

    with pytest.raises(ValueError, match="first_midi_pitch must be an integer or None"):
        load_encoder_result(payload)


def test_result_loader_rejects_large_ioi_mismatch_without_relative_tolerance() -> None:
    payload = encode_events(_events()[:2]).to_dict()
    payload["durations_ms"] = [10_000_000.0, 10_000_000.0]
    payload["rhythm_ratios"] = [
        {"numerator": 1, "denominator": 1},
        {"numerator": 1, "denominator": 1},
    ]
    payload["median_duration_ratios"] = [1.0, 1.0]
    payload["ioi_ms"] = [10_000_000.001]

    with pytest.raises(ValueError, match=r"ioi_ms must match .* durations_ms and gaps_ms"):
        load_encoder_result(payload)


def test_result_loader_rejects_an_integer_outside_the_supported_numeric_range() -> None:
    payload = encode_events(_events()[:1]).to_dict() | {"durations_ms": [10**400]}

    with pytest.raises(ValueError, match=r"durations_ms values must be finite.*numeric range"):
        load_encoder_result(payload)


def test_encoder_rejects_overlap_and_does_not_sort() -> None:
    with pytest.raises(ValueError, match="ordered"):
        encode_events(
            (
                InputEvent(0.5, 1.0, 500.0, 69),
                InputEvent(0.0, 0.5, 500.0, 72),
            )
        )


def test_ap1_representation_layers_keep_raw_ioi_and_quantize_separately() -> None:
    events = (
        InputEvent(0.0, 0.1, 100.0, 60),
        InputEvent(0.1, 0.3, 200.0, 62),
        InputEvent(0.3, 0.7, 400.0, 62),
        InputEvent(0.7, 0.8, 100.0, 55),
    )

    result = encode_events(events)

    assert result.ioi_ms == pytest.approx((100.0, 200.0, 400.0))
    assert result.ioi_log2_ratios == pytest.approx((1.0, 1.0))
    assert result.ioi_rhythm_digits == (1, 1)
    assert result.median_duration_ratios == pytest.approx((0.67, 1.33, 2.67, 0.67))
    assert result.parsons_code == ("u", "r", "d")
    assert result.interval_classes == ("+2", "P", "-4+")


def test_ap1_ioi_quantization_has_documented_boundaries_and_configurable_limit() -> None:
    raw = compute_ioi_log2_ratios((100.0, 200.0, 600.0, 1200.0))

    assert raw == pytest.approx((1.0, 1.584962500721156, 1.0))
    assert quantize_ioi_rhythm_digits((-2.6, -1.5, -0.5, 0.49, 0.5, 1.5, 2.6)) == (
        -2,
        -1,
        0,
        0,
        1,
        2,
        2,
    )
    assert quantize_ioi_rhythm_digits((3.2, -3.2), limit=3) == (3, -3)
    with pytest.raises(ValueError, match="non-negative"):
        quantize_ioi_rhythm_digits((0.0,), limit=-1)


def test_numeric_helpers_reject_non_finite_and_unrepresentable_values_cleanly() -> None:
    for invalid in (float("nan"), float("inf"), 10**400):
        with pytest.raises(ValueError, match=r"finite.*numeric range"):
            quantize_ioi_rhythm_digits((invalid,))

    ratio = compute_ioi_log2_ratios((5e-324, 1e308))
    assert len(ratio) == 1
    assert ratio[0] == pytest.approx(math.log2(1e308) - math.log2(5e-324))


def test_ap1_interval_layers_cover_all_coarse_classes() -> None:
    intervals = (0, 1, -2, 3, -4, 5, -12)

    assert compute_parsons(intervals) == ("r", "u", "d", "u", "d", "u", "d")
    assert compute_interval_classes(intervals) == ("P", "+2", "-2", "+3", "-3", "+4+", "-4+")


def test_plaine_easie_export_reconstructs_pitch_and_quantized_rhythm() -> None:
    result = encode_events(
        (
            InputEvent(0.0, 0.25, 250.0, 60),
            InputEvent(0.25, 0.75, 500.0, 62),
            InputEvent(0.75, 1.75, 1000.0, 61),
            InputEvent(1.75, 2.0, 250.0, 55),
        )
    )

    assert result.ioi_rhythm_digits == (1, 1)
    assert result.to_plaine_easie() == "%G-2 '4C'2D'1xC,1G"
    assert result.to_plaine_easie(base_duration=8) == "%G-2 '8C'4D'2xC,2G"


def test_plaine_easie_export_tracks_accidentals_separately_per_octave() -> None:
    result = encode_events(
        (
            InputEvent(0.0, 0.25, 250.0, 66),
            InputEvent(0.25, 0.5, 250.0, 77),
            InputEvent(0.5, 0.75, 250.0, 66),
        )
    )

    assert result.to_plaine_easie() == "%G-2 '4xF''4F'4F"


def test_plaine_easie_export_handles_empty_and_invalid_base_duration() -> None:
    assert encode_events(()).to_plaine_easie() == "%G-2 "
    with pytest.raises(ValueError, match="base_duration"):
        encode_events(_events()).to_plaine_easie(base_duration=16)


def test_cli_can_emit_plaine_easie(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = {
        "schema_version": "musicdna-events-v0",
        "events": [
            {"start_seconds": 0.0, "end_seconds": 0.25, "duration_ms": 250.0, "midi_pitch": 60},
            {"start_seconds": 0.25, "end_seconds": 0.5, "duration_ms": 250.0, "midi_pitch": 62},
        ],
    }
    source = tmp_path / "events.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--format", "pae", str(source)]) == 0
    assert capsys.readouterr().out == "%G-2 '4C'4D\n"


def test_cli_emits_current_json_contract_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"schema_version":"musicdna-events-v0","events":[]}'),
    )

    assert main(["--indent", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "musicdna-melody-encoding-v0"
    assert payload["tuning_normalization"]["status"] == "empty"


def test_cli_rejects_an_unknown_detector_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "events.json"
    source.write_text('{"schema_version":"musicdna-analysis-v1","events":[]}', encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        main([str(source)])

    assert caught.value.code == 2
    assert "unsupported input schema_version 'musicdna-analysis-v1'" in capsys.readouterr().err


def test_the_input_keeps_every_field_the_producer_wrote() -> None:
    """Preserve producer fields even when the Encoder does not use them."""

    payload = {
        "schema_version": "musicdna-analysis-v0",
        "events": [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.5,
                "duration_ms": 500.0,
                "midi_pitch": 69,
                "frequency_hz": 440.0,
                "confidence": 0.83,
                "cents": -12.0,
                "pitch_name": "A4",
                "duration_seconds": 0.5,
                "something_a_future_detector_adds": [1, 2],
            }
        ],
    }

    sequence = load_event_sequence(payload)

    assert sequence.raw_events[0] == payload["events"][0]
    assert sequence.raw_events[0]["confidence"] == 0.83
    assert sequence.raw_events[0]["something_a_future_detector_adds"] == [1, 2]
    # Typed access stays limited to what the transformation actually uses.
    assert not hasattr(sequence.events[0], "confidence")
    assert sequence.events[0].frequency_hz == 440.0


def test_the_kept_record_cannot_be_edited_through_the_sequence() -> None:
    sequence = load_event_sequence(
        {
            "schema_version": "musicdna-analysis-v0",
            "events": [
                {"start_seconds": 0.0, "end_seconds": 0.5, "duration_ms": 500.0, "midi_pitch": 69}
            ],
        }
    )

    with pytest.raises(TypeError):
        sequence.raw_events[0]["midi_pitch"] = 70  # type: ignore[index]


def test_a_call_written_for_the_old_signature_fails_loudly() -> None:
    """Confidence used to be the fifth positional argument.

    Silently accepting it would have put a confidence into frequency_hz, and
    tuning normalisation would then have re-anchored the grid to a number that is
    not a frequency. Keyword-only makes that impossible rather than unlikely.
    """

    with pytest.raises(TypeError):
        InputEvent(0.0, 0.25, 250.0, 69, 0.9)  # type: ignore[misc]
