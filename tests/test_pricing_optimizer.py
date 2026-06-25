from shared.optimizer import choose_best_backend, evaluate_placement, should_migrate
from shared.pricing import backend_hourly_cost, cost_summary


def dataset(reads, backend="public-cold"):
    return {
        "dataset_id": "ds_1",
        "size_gb": 100,
        "initial_backend": "public-cold",
        "current_backend": backend,
        "history": [
            {
                "reads_1h": reads,
                "writes_1h": 1,
                "bytes_read_1h": 1024,
                "hour_of_day": i % 24,
                "day_of_week": i % 7,
                "timestamp": i,
            }
            for i in range(24)
        ],
    }


def test_cost_summary_reports_savings():
    cold = dataset(1, "public-cold")
    hot = dataset(1000, "public-hot")
    summary = cost_summary([cold, hot], [{"status": "COMPLETE"}])
    assert set(summary) == {"baseline", "optimized", "migration_cost", "savings", "savings_percent"}


def test_optimizer_moves_spiky_cold_data_to_cheapest_low_latency_tier():
    ds = dataset(250, "public-cold")
    decision = evaluate_placement(ds, hot_threshold=100)
    assert decision.target_backend == "on-prem"
    assert decision.reason == "hot-read-low-latency"
    assert decision.target_cost < decision.current_cost
    assert backend_hourly_cost(ds, "on-prem") < backend_hourly_cost(ds, "public-hot")


def test_optimizer_cools_low_read_data():
    target, reason = choose_best_backend(dataset(1, "public-hot"), cold_threshold=10)
    assert target == "public-cold"
    assert reason == "cold-data-cooldown"
    assert should_migrate({"current_backend": "public-hot"}, target)
