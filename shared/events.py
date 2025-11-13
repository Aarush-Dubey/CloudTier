from dataclasses import dataclass


REQUIRED_EVENT_FIELDS = {
    "timestamp": int,
    "dataset_id": str,
    "reads_1h": int,
    "writes_1h": int,
    "bytes_read_1h": int,
    "hour_of_day": int,
    "day_of_week": int,
    "current_backend": str,
    "size_gb": (int, float),
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error: str | None = None


def validate_access_event(event: dict) -> ValidationResult:
    if not isinstance(event, dict):
        return ValidationResult(False, "event must be object")
    for field, expected_type in REQUIRED_EVENT_FIELDS.items():
        if field not in event:
            return ValidationResult(False, f"missing field: {field}")
        if not isinstance(event[field], expected_type):
            return ValidationResult(False, f"invalid type for {field}")
    if event["reads_1h"] < 0 or event["writes_1h"] < 0 or event["size_gb"] < 0:
        return ValidationResult(False, "negative metrics not allowed")
    if not 0 <= event["hour_of_day"] <= 23:
        return ValidationResult(False, "hour_of_day out of range")
    if not 0 <= event["day_of_week"] <= 6:
        return ValidationResult(False, "day_of_week out of range")
    return ValidationResult(True)


def history_entry(event: dict) -> dict:
    return {
        "timestamp": event["timestamp"],
        "reads_1h": event["reads_1h"],
        "writes_1h": event["writes_1h"],
        "bytes_read_1h": event["bytes_read_1h"],
        "hour_of_day": event["hour_of_day"],
        "day_of_week": event["day_of_week"],
    }

