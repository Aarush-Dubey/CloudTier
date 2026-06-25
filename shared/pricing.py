HOURS_IN_MONTH = 720.0
SLA_PENALTY_PER_SECOND = 0.10

PRICING_CONFIG = {
    "on-prem": {
        "P_store": 0.40,
        "P_read": 0.00,
        "P_write": 0.00,
        "P_egress": 0.0,
        "P_ingress": 0.0,
        "P_latency": 0.01,
    },
    "private-cloud": {
        "P_store": 0.25,
        "P_read": 0.05,
        "P_write": 0.05,
        "P_egress": 0.0,
        "P_ingress": 0.0,
        "P_latency": 0.03,
    },
    "public-hot": {
        "P_store": 0.20,
        "P_read": 0.04,
        "P_write": 0.04,
        "P_egress": 0.002,
        "P_ingress": 0.0,
        "P_latency": 0.07,
    },
    "public-cold": {
        "P_store": 0.04,
        "P_read": 0.001,
        "P_write": 0.001,
        "P_egress": 0.002,
        "P_ingress": 0.0,
        "P_latency": 4.0,
    },
}


def forecast_access(dataset: dict, hours: int = 24) -> dict:
    history = dataset.get("history", [])[-hours:]
    if not history:
        return {"reads": 0.0, "writes": 0.0, "bytes_read": 0.0}

    recent = history[-6:] if len(history) >= 6 else history
    older = history[-12:-6] if len(history) >= 12 else history[: max(1, len(history) // 2)]
    recent_reads = sum(item.get("reads_1h", 0) for item in recent) / len(recent)
    older_reads = sum(item.get("reads_1h", 0) for item in older) / len(older)
    recent_writes = sum(item.get("writes_1h", 0) for item in recent) / len(recent)
    older_writes = sum(item.get("writes_1h", 0) for item in older) / len(older)
    read_trend = max(0.7, min(2.0, recent_reads / max(older_reads, 1)))
    write_trend = max(0.7, min(2.0, recent_writes / max(older_writes, 1)))
    return {
        "reads": recent_reads * read_trend * hours,
        "writes": recent_writes * write_trend * hours,
        "bytes_read": sum(item.get("bytes_read_1h", 0) for item in history),
    }


def backend_hourly_cost(
    dataset: dict,
    backend: str,
    hours: int = 24,
    include_latency: bool = True,
) -> float:
    history = dataset.get("history", [])[-hours:]
    if not history:
        return 0.0

    prices = PRICING_CONFIG[backend]
    size_gb = float(dataset.get("size_gb", 0))
    forecast = forecast_access(dataset, hours)
    total_reads = forecast["reads"]
    total_writes = forecast["writes"]
    storage_cost = size_gb * prices["P_store"] * (hours / HOURS_IN_MONTH)
    access_cost = total_reads * prices["P_read"] + total_writes * prices["P_write"]
    latency_cost = total_reads * prices["P_latency"] * SLA_PENALTY_PER_SECOND if include_latency else 0.0
    return storage_cost + access_cost + latency_cost


def calculate_baseline_cost(datasets: list[dict]) -> float:
    return sum(
        backend_hourly_cost(ds, ds.get("initial_backend") or ds.get("current_backend", "on-prem"))
        for ds in datasets
    )


def calculate_optimized_cost(datasets: list[dict], migrations: list[dict]) -> tuple[float, float]:
    migration_cost = sum(0.001 for job in migrations if job.get("status") == "COMPLETE")
    data_cost = sum(backend_hourly_cost(ds, ds.get("current_backend", "on-prem")) for ds in datasets)
    return data_cost + migration_cost, migration_cost


def cost_summary(datasets: list[dict], migrations: list[dict]) -> dict:
    baseline = calculate_baseline_cost(datasets)
    optimized, migration_cost = calculate_optimized_cost(datasets, migrations)
    savings = baseline - optimized
    return {
        "baseline": round(baseline, 2),
        "optimized": round(optimized, 2),
        "migration_cost": round(migration_cost, 2),
        "savings": round(savings, 2),
        "savings_percent": round((savings / baseline * 100) if baseline else 0, 2),
    }
