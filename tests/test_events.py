from shared.events import validate_access_event
from tests.fakes import sample_event


def test_valid_event_passes():
    assert validate_access_event(sample_event()).valid


def test_malformed_event_fails():
    event = sample_event()
    event.pop("dataset_id")
    result = validate_access_event(event)
    assert not result.valid
    assert "dataset_id" in result.error


def test_negative_metrics_fail():
    event = sample_event()
    event["reads_1h"] = -1
    assert not validate_access_event(event).valid

