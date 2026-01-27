import argparse
import statistics
import time
from pathlib import Path

from services.producer.app import generate_event, initialize_datasets
from shared.events import validate_access_event


def run(events: int) -> dict:
    datasets = initialize_datasets(max(1, min(1000, events // 10)))
    latencies_ms = []
    accepted = 0
    started = time.perf_counter()

    for index in range(events):
        event_started = time.perf_counter()
        event = generate_event(datasets[index % len(datasets)], index // len(datasets))
        if validate_access_event(event).valid:
            accepted += 1
        latencies_ms.append((time.perf_counter() - event_started) * 1000)

    duration = time.perf_counter() - started
    events_per_sec = accepted / duration if duration else 0
    p95 = statistics.quantiles(latencies_ms, n=100)[94] if len(latencies_ms) >= 100 else max(latencies_ms)
    return {
        "events": events,
        "accepted": accepted,
        "duration_sec": round(duration, 3),
        "events_per_sec": round(events_per_sec, 2),
        "events_per_hour": round(events_per_sec * 3600, 2),
        "p95_latency_ms": round(p95, 4),
    }


def append_results(result: dict) -> None:
    output = Path("benchmark/results.md")
    output.parent.mkdir(exist_ok=True)
    if not output.exists():
        output.write_text(
            "# CloudTier Benchmark Results\n\n"
            "| Events | Accepted | Duration Sec | Events/Sec | Events/Hour | p95 Latency Ms |\n"
            "| ---: | ---: | ---: | ---: | ---: | ---: |\n",
            encoding="utf-8",
        )
    with output.open("a", encoding="utf-8") as handle:
        handle.write(
            f"| {result['events']} | {result['accepted']} | {result['duration_sec']} | "
            f"{result['events_per_sec']} | {result['events_per_hour']} | {result['p95_latency_ms']} |\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudTier synthetic event benchmark")
    parser.add_argument("--events", type=int, default=10000)
    args = parser.parse_args()
    result = run(args.events)
    append_results(result)
    print(result)


if __name__ == "__main__":
    main()

