from dataclasses import dataclass

from shared.pricing import PRICING_CONFIG, backend_hourly_cost, forecast_access

HOT_TIER = "on-prem"
COLD_TIER = "public-cold"
MAX_HOT_LATENCY_SEC = 0.10
MIN_SAVINGS_TO_MOVE = 0.50
MIN_SAVINGS_PERCENT = 5.0


@dataclass(frozen=True)
class PlacementDecision:
    target_backend: str
    reason: str
    current_cost: float
    target_cost: float
    savings: float
    savings_percent: float


def _access_stats(dataset: dict) -> dict:
    history = dataset.get("history", [])[-24:]
    if not history:
        return {"avg_reads": 0.0, "peak_reads": 0.0, "avg_writes": 0.0}
    return {
        "avg_reads": sum(item.get("reads_1h", 0) for item in history) / len(history),
        "peak_reads": max(item.get("reads_1h", 0) for item in history),
        "avg_writes": sum(item.get("writes_1h", 0) for item in history) / len(history),
    }


def _candidate_backends(dataset: dict, hot_threshold: int, cold_threshold: int) -> list[str]:
    stats = _access_stats(dataset)
    forecast = forecast_access(dataset)
    projected_reads_per_hour = forecast["reads"] / 24 if forecast["reads"] else stats["avg_reads"]

    if stats["peak_reads"] >= hot_threshold or projected_reads_per_hour >= hot_threshold:
        return [backend for backend, price in PRICING_CONFIG.items() if price["P_latency"] <= MAX_HOT_LATENCY_SEC]
    if stats["avg_reads"] <= cold_threshold and stats["peak_reads"] < hot_threshold:
        return list(PRICING_CONFIG.keys())
    return ["on-prem", "private-cloud", "public-hot"]


def evaluate_placement(dataset: dict, hot_threshold: int = 100, cold_threshold: int = 10) -> PlacementDecision:
    history = dataset.get("history", [])[-24:]
    current_backend = dataset.get("current_backend", "on-prem")
    if not history:
        return PlacementDecision(current_backend, "insufficient-history", 0.0, 0.0, 0.0, 0.0)

    stats = _access_stats(dataset)
    cold_archive_workload = stats["avg_reads"] <= cold_threshold and stats["peak_reads"] < hot_threshold
    current_cost = backend_hourly_cost(dataset, current_backend, include_latency=not cold_archive_workload)
    candidates = _candidate_backends(dataset, hot_threshold, cold_threshold)
    scored = {
        backend: backend_hourly_cost(dataset, backend, include_latency=not cold_archive_workload)
        for backend in candidates
    }
    target_backend = min(scored, key=scored.get)
    target_cost = scored[target_backend]
    savings = current_cost - target_cost
    savings_percent = (savings / current_cost * 100) if current_cost else 0.0

    if target_backend == COLD_TIER:
        reason = "cold-data-cooldown"
    elif stats["peak_reads"] >= hot_threshold:
        reason = "hot-read-low-latency"
    else:
        reason = "forecasted-cost-reduction"

    if target_backend != current_backend and savings < MIN_SAVINGS_TO_MOVE and savings_percent < MIN_SAVINGS_PERCENT:
        return PlacementDecision(current_backend, "hysteresis-hold", current_cost, current_cost, 0.0, 0.0)

    return PlacementDecision(
        target_backend,
        reason,
        round(current_cost, 4),
        round(target_cost, 4),
        round(savings, 4),
        round(savings_percent, 2),
    )


def choose_best_backend(dataset: dict, hot_threshold: int = 100, cold_threshold: int = 10) -> tuple[str, str]:
    decision = evaluate_placement(dataset, hot_threshold, cold_threshold)
    return decision.target_backend, decision.reason


def should_migrate(dataset: dict, target_backend: str) -> bool:
    return dataset.get("current_backend") != target_backend
