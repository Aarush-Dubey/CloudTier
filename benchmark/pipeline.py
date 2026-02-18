"""Measure true end-to-end pipeline throughput (produce -> consume -> Mongo upsert).

This is deliberately distinct from `benchmark/run.py`, which measures single-process
event *generation*. Here we measure work that actually made it through Kafka and landed
in MongoDB, observed at the consumers via `cloudtier_events_processed_total`, plus the
Kafka consumer-group lag (backpressure signal).

Sample the processed-events counter across all consumers over a window and divide by
elapsed time. Also read the consumer group's total lag before/after so we can see
whether consumers are keeping up (lag flat/zero) or falling behind (lag growing) — the
latter is where a bounded queue / pause-resume would matter.

Run under the default throttled producer to get the *sustained* rate, or after
saturating the producer (`PRODUCER_SIM_SPEED_SEC=0`) to get *drain capacity*; label
which in `--note`.

Usage:
    python -m benchmark.pipeline --window 60 --note "sustained (throttled producer)"
"""

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from shared.config import settings

PROM = "http://localhost:9090"
KAFKA = "cloudtier-kafka"
RESULTS = Path("benchmark/results.md")


def prom(query: str):
    url = f"{PROM}/api/v1/query?query={urllib.parse.quote(query)}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.load(resp)
    result = data["data"]["result"]
    return float(result[0]["value"][1]) if result else None


def group_lag(group: str, retries: int = 3):
    """Sum of LAG across all partitions for the consumer group (None if unavailable).

    ``kafka-consumer-groups --describe`` transiently returns no data rows while the
    group is rebalancing, so retry a few times before giving up.
    """
    for attempt in range(retries):
        out = subprocess.run(
            ["docker", "exec", KAFKA, "kafka-consumer-groups", "--bootstrap-server",
             "localhost:9092", "--describe", "--group", group],
            capture_output=True, text=True, check=False,
        ).stdout
        total, header, rows = 0, None, 0
        for line in out.splitlines():
            parts = line.split()
            if parts and parts[0] == "GROUP" and "LAG" in parts:
                header = parts.index("LAG")
                continue
            if header is not None and len(parts) > header and parts[0] == group:
                try:
                    total += int(parts[header])
                    rows += 1
                except ValueError:
                    pass
        if rows:
            return total
        time.sleep(2)
    return None


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline throughput")
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument("--note", default="sustained (default throttled producer)")
    args = parser.parse_args()

    group = settings.consumer_group
    processed_start = prom("sum(cloudtier_events_processed_total)")
    if processed_start is None:
        raise RuntimeError("no cloudtier_events_processed_total in Prometheus; is the stack up?")
    lag_start = group_lag(group)

    t0 = time.time()
    time.sleep(args.window)
    elapsed = time.time() - t0

    processed_end = prom("sum(cloudtier_events_processed_total)")
    lag_end = group_lag(group)
    rate = (processed_end - processed_start) / elapsed

    if lag_start is not None and lag_end is not None:
        lag_delta = lag_end - lag_start
        if lag_delta > 0.05 * max(1, processed_end - processed_start):
            lag_verdict = (
                f"Lag grew by {lag_delta} during the window: the consumers are the "
                "bottleneck (drain capacity < ingest rate), so this rate *is* the "
                "pipeline's drain capacity and the backlog grows unboundedly — the "
                "backpressure case addressed in §5.2."
            )
        else:
            lag_verdict = "Lag is flat: consumers keep up at this rate (producer-bound)."
    else:
        lag_verdict = "Consumer-group lag unavailable."

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    body = [
        f"\n## End-to-end pipeline throughput — {ts} — commit {_commit()}",
        "",
        f"Command: `python -m benchmark.pipeline --window {int(args.window)}` — {args.note}.",
        "Measured at the consumer via `cloudtier_events_processed_total` (events that "
        "passed through Kafka and were upserted into MongoDB).",
        "",
        "| Metric | Value |",
        "| :-- | --: |",
        f"| Window (s) | {round(elapsed, 1)} |",
        f"| Events processed in window | {int(processed_end - processed_start)} |",
        f"| **End-to-end throughput (events/s)** | **{round(rate, 1)}** |",
        f"| Consumer-group lag (start → end) | {lag_start} → {lag_end} |",
        "",
        "Contrast with single-process *generation* (~78,900 events/s, see top of file): "
        "the pipeline number is far lower because each event additionally crosses Kafka "
        "(serialize, replicate, fetch) and a MongoDB upsert with a bounded 24-point "
        f"history push. {lag_verdict}",
    ]
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(body) + "\n")
    print("\n".join(body))


if __name__ == "__main__":
    main()
