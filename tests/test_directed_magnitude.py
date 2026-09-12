from musicdna_encoder import compute_directed_magnitude, encode_events


def test_directed_magnitude_uses_the_frozen_token_boundaries() -> None:
    intervals = (-12, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 12)

    assert compute_directed_magnitude(intervals) == (
        "D5+",
        "D5+",
        "D3-4",
        "D3-4",
        "D2",
        "D1",
        "S0",
        "U1",
        "U2",
        "U3-4",
        "U3-4",
        "U5+",
        "U5+",
    )


def test_directed_magnitude_is_empty_for_no_intervals() -> None:
    assert compute_directed_magnitude(()) == ()


def test_directed_magnitude_remains_outside_the_stored_result_contract() -> None:
    assert "directed_magnitude" not in encode_events(()).to_dict()
