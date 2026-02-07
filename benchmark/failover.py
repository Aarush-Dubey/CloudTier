"""Measure optimizer leader failover time under a real kill.

Runs against the live docker-compose stack (``make up``, which scales the optimizer to
3 replicas). For each trial it:

  1. reads the ``leader_lock`` document to find the current leader and its last
     heartbeat (``renewed_at``) and fencing token,
  2. ``docker kill``s that leader's container (the holder id is the container HOSTNAME,
     i.e. the container id), and
  3. polls the lock until a *different* holder with a *higher* fencing token appears,
     recording when the new leader acquired the lease (``acquired_at``).

Failover time is reported two ways, both from server-set timestamps so they do not
depend on this script's own latency:

  * ``lease_failover_sec`` = new ``acquired_at`` − old ``renewed_at`` (last heartbeat →
    new leader owns the lease). This is bounded below by however much of the lease TTL
    remained at kill time, and is the number quoted in the resume bullet.
  * ``detect_sec``         = wall-clock from issuing the kill to observing the takeover.

All numbers are printed and appended to ``benchmark/results.md``. Nothing here is
fabricated; if the stack is not up the script exits non-zero.

Usage:
    python -m benchmark.failover --runs 5
"""

import argparse
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

from shared.config import settings

LOCK_NAME = "optimizer"


def _lock(db):
    return db[settings.leader_lock_collection].find_one({"lock_name": LOCK_NAME})


def _wait_for_leader(db, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = _lock(db)
        if doc and doc.get("holder_id"):
            return doc
        time.sleep(0.1)
    raise RuntimeError("no optimizer leader appeared within timeout; is the stack up?")


def _kill_container(container_id: str) -> None:
    # holder_id == container HOSTNAME == container id, so `docker kill` finds it.
    subprocess.run(["docker", "kill", container_id], check=True, capture_output=True)


def _start_container(container_id: str) -> None:
    # In this Docker version `docker kill` counts as a manual stop, so the
    # `unless-stopped` policy does not auto-restart. Explicitly restart the crashed
    # replica so the pool returns to 3 before the next trial.
    subprocess.run(["docker", "start", container_id], check=False, capture_output=True)


def one_trial(db, mongo_uri: str) -> dict:
    old = _wait_for_leader(db)
    old_holder = old["holder_id"]
    old_renewed = float(old.get("renewed_at", old.get("acquired_at", 0.0)))
    old_token = int(old.get("fencing_token", 0))

    kill_wall = time.time()
    _kill_container(old_holder)

    # A takeover is any *new acquisition* (fencing token increments). Usually a
    # surviving replica wins, but the killed container may restart and reacquire its own
    # lease first; either way the failover interval (no leader -> leader again) is the
    # same and is what we measure. We record which case it was.
    deadline = time.time() + 60.0
    while time.time() < deadline:
        doc = _lock(db)
        if doc and int(doc.get("fencing_token", 0)) > old_token:
            new_acquired = float(doc.get("acquired_at", time.time()))
            new_holder = doc["holder_id"]
            _start_container(old_holder)  # replenish the pool for the next trial
            return {
                "old_holder": old_holder,
                "new_holder": new_holder,
                "same_replica": new_holder == old_holder,
                "old_token": old_token,
                "new_token": int(doc["fencing_token"]),
                "lease_failover_sec": round(new_acquired - old_renewed, 3),
                "detect_sec": round(time.time() - kill_wall, 3),
            }
        time.sleep(0.05)
    _start_container(old_holder)
    raise RuntimeError(f"no new leader took over after killing {old_holder}")


def append_results(trials: list, lease_stats: dict, commit: str) -> None:
    out = Path("benchmark/results.md")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "",
        f"## Optimizer leader failover (real kill) — {ts} — commit {commit}",
        "",
        "Command: `python -m benchmark.failover --runs %d` against `make up` (3 optimizer replicas)."
        % len(trials),
        f"Lease TTL = {settings.leader_lease_ttl_sec}s, heartbeat = {settings.leader_heartbeat_sec}s.",
        "",
        "| Trial | Old leader | New leader | Same replica? | Token | Lease failover (s) | Detect (s) |",
        "| ---: | :-- | :-- | :-- | :-- | ---: | ---: |",
    ]
    for i, t in enumerate(trials, 1):
        lines.append(
            f"| {i} | {t['old_holder'][:12]} | {t['new_holder'][:12]} | "
            f"{'yes' if t.get('same_replica') else 'no'} | "
            f"{t['old_token']}→{t['new_token']} | {t['lease_failover_sec']} | {t['detect_sec']} |"
        )
    lines.append(
        f"\n**lease_failover_sec** min/median/max = "
        f"{lease_stats['min']}/{lease_stats['median']}/{lease_stats['max']} "
        f"(n={len(trials)}).\n"
    )
    with out.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure optimizer leader failover time")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--settle-sec", type=float, default=8.0, help="wait between trials for a stable leader")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[settings.db_name]

    trials = []
    for run in range(args.runs):
        trial = one_trial(db, args.mongo_uri)
        trials.append(trial)
        print(f"trial {run + 1}: {trial}")
        time.sleep(args.settle_sec)  # let the killed replica restart and lease re-stabilise

    lease_vals = [t["lease_failover_sec"] for t in trials]
    lease_stats = {
        "min": round(min(lease_vals), 3),
        "median": round(statistics.median(lease_vals), 3),
        "max": round(max(lease_vals), 3),
    }
    append_results(trials, lease_stats, _git_commit())
    print("lease_failover_sec min/median/max:", lease_stats)


if __name__ == "__main__":
    main()
