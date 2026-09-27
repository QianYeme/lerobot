from scripts.run_phase_review_ui import STEPS, initial_reviews, normalize_browser_payload, parse_byte_range, validate_reviews


def annotations():
    return {
        "config": {"hold_frames": 9},
        "episodes": {
            "0": {
                "length": 100,
                "events": {"close": 30, "release": 70},
                "warnings": [],
            }
        },
    }


def test_initial_review_has_three_ordered_events():
    fixture = annotations()
    fixture["episodes"]["0"]["events"].update({"close_motion_start": 25, "close_motion_end": 35, "release_motion_start": 68, "release_motion_end": 75})
    reviews = initial_reviews(fixture)
    assert list(reviews["0"]["steps"]) == list(STEPS)
    assert reviews["0"]["steps"]["closing"]["start_frame"] == 25
    assert reviews["0"]["steps"]["closing"]["end_frame"] == 35
    assert reviews["0"]["steps"]["lift_start"]["frame"] == 36
    assert reviews["0"]["steps"]["release"]["start_frame"] == 68
    assert reviews["0"]["steps"]["release"]["end_frame"] == 75
    assert reviews["0"]["reviewed"] is False


def test_validation_rejects_out_of_order_frames():
    fixture = annotations()
    fixture["episodes"]["0"]["events"].update({"close_motion_start": 25, "close_motion_end": 35, "release_motion_start": 68, "release_motion_end": 75})
    reviews = initial_reviews(fixture)
    reviews["0"]["steps"]["release"]["start_frame"] = 20
    try:
        validate_reviews({"episodes": reviews}, {0: 100})
    except ValueError as error:
        assert "nondecreasing" in str(error)
    else:
        raise AssertionError("Out-of-order frames must be rejected")


def test_byte_ranges_support_video_seeking():
    assert parse_byte_range(None, 1000) is None
    assert parse_byte_range("bytes=100-199", 1000) == (100, 199)
    assert parse_byte_range("bytes=900-", 1000) == (900, 999)
    assert parse_byte_range("bytes=-100", 1000) == (900, 999)


def test_schema_two_browser_payload_is_rescued_as_reviewed():
    fixture = annotations()
    fixture["episodes"]["0"]["events"].update({"close_motion_start": 25, "close_motion_end": 35, "release_motion_start": 68, "release_motion_end": 75})
    reviews = initial_reviews(fixture)
    reviews["0"].pop("reviewed")
    reviews["0"]["steps"]["closing"]["status"] = "approved"
    normalized = normalize_browser_payload({"schema_version": 2, "episodes": reviews})
    assert normalized["schema_version"] == 3
    assert normalized["episodes"]["0"]["reviewed"] is True
    assert "status" not in normalized["episodes"]["0"]["steps"]["closing"]


def test_schema_three_draft_remains_unreviewed():
    fixture = annotations()
    fixture["episodes"]["0"]["events"].update({"close_motion_start": 25, "close_motion_end": 35, "release_motion_start": 68, "release_motion_end": 75})
    reviews = initial_reviews(fixture)
    payload = {"schema_version": 3, "episodes": reviews}
    assert normalize_browser_payload(payload)["episodes"]["0"]["reviewed"] is False
