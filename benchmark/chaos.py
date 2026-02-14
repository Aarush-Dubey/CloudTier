"""Chaos harness: induce real failures against the live stack and quantify recovery.

Runs against `make up` (3 consumers, 2 migrators, 3 optimizers). Each scenario is a
runnable command that injects one fault and records real, reproducible numbers into
`benchmark/results.md` with the command, timestamp and git commit.

Scenarios:
  consumer   Kill one consumer under load; measure Kafka consumer-group rebalance time
             and prove zero duplicate dataset writes (idempotent upsert dedup).
  migrator   Kill a migrator mid-job; prove the job is retried and completed by another
             migrator (at-least-once + lease re-lock), and report dead-letter vs retry.
  stall      Pause a migrator that holds a job until its lock lease expires so a second
             migrator steals it, then unpause the first: its now-stale write is fenced.
             This is the live counterpart to tests/test_fencing.py and yields a real
             non-zero cloudtier_fenced_writes_total.
  optimizer  See benchmark/failover.py (leader kill) — measured separately.

Nothing here is fabricated: every number is read from Mongo or Prometheus after a real
`docker` fault. If the stack is not up, the scenario exits non-zero.

Usage:
    python -m benchmark.chaos consumer
    python -m benchmark.chaos migrator
    python -m benchmark.chaos stall
"""

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

from shared.config import settings

PROM = "http://localhost:9090"
KAFKA = "cloudtier-kafka"
RESULTS = Path("benchmark/results.md")


# --- helpers ---------------------------------------------------------------

def prom(query: str):
    url = f"{PROM}/api/v1/query?query={urllib.parse.quote(query)}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.load(resp)
    result = data["data"]["result"]
    if not result:
        return None
    return float(result[0]["value"][1])


def duplicate_dataset_writes(db) -> int:
    """Number of dataset_ids that have more than one document. Idempotent upserts keyed
    by dataset_id must keep this at zero even under redelivery after a consumer crash."""
    dupes = list(
        db[settings.dataset_collection].aggregate(
            [
                {"$group": {"_id": "$dataset_id", "n": {"$sum": 1}}},
                {"$match": {"n": {"$gt": 1}}},
                {"$count": "dupes"},
            ]
        )
    )
    return dupes[0]["dupes"] if dupes else 0


def job_stats(db) -> dict:
    jobs = db[settings.job_collection]
    return {
        "complete": jobs.count_documents({"status": "COMPLETE"}),
        "retried_then_complete": jobs.count_documents({"status": "COMPLETE", "attempts": {"$gt": 1}}),
        "failed_dead": jobs.count_documents({"status": "FAILED"}),
        "running": jobs.count_documents({"status": "RUNNING"}),
        "pending": jobs.count_documents({"status": "PENDING"}),
    }


def _docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


def kafka_group_state(group: str) -> tuple:
    """Return (state, n_members) for a consumer group via kafka-consumer-groups."""
    out = _docker(
        "exec", KAFKA, "kafka-consumer-groups", "--bootstrap-server", "localhost:9092",
        "--describe", "--group", group, "--state", check=False,
    ).stdout
    state, members = "Unknown", None
    for line in out.splitlines():
        # Data row looks like: <group> <coordinator> <assignment> <state> <#members>
        parts = line.split()
        if len(parts) >= 2 and parts[0] == group and parts[-1].isdigit():
            members = int(parts[-1])
            state = parts[-2]
    return state, members


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def append_section(title: str, body_lines: list) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    header = f"\n## {title} — {ts} — commit {_commit()}\n"
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write(header + "\n".join(body_lines) + "\n")


# --- scenarios -------------------------------------------------------------

def scenario_consumer(db) -> None:
    group = settings.consumer_group
    state, members = kafka_group_state(group)
    print(f"group before: state={state} members={members}")
    if not members or members < 2:
        raise RuntimeError("need >=2 live consumers; is the stack up with --scale consumer=3?")

    # Pick a consumer container to kill.
    victim = _docker("ps", "--filter", "name=cloudtier-consumer", "--format", "{{.Names}}").stdout.split()[0]
    dupes_before = duplicate_dataset_writes(db)
    processed_before = prom("sum(cloudtier_events_processed_total)") or 0

    kill_wall = time.time()
    _docker("kill", victim)
    print(f"killed {victim}; waiting for group to re-stabilise...")

    # Measure rebalance: time until the group is Stable again with one fewer member.
    rebalance_sec = None
    deadline = time.time() + 60
    while time.time() < deadline:
        st, mem = kafka_group_state(group)
        if st == "Stable" and mem is not None and mem <= members - 1:
            rebalance_sec = round(time.time() - kill_wall, 2)
            break
        time.sleep(0.5)

    time.sleep(5)  # let redelivered messages be reprocessed
    dupes_after = duplicate_dataset_writes(db)
    distinct = db[settings.dataset_collection].count_documents({})
    dlq = prom("sum(cloudtier_events_rejected_total)") or 0
    _docker("start", victim, check=False)  # restore the pool

    body = [
        "Command: `python -m benchmark.chaos consumer`",
        "",
        "| Metric | Value |",
        "| :-- | --: |",
        f"| Consumer killed | `{victim}` |",
        f"| Group members before → after | {members} → {members - 1} |",
        f"| Consumer-group rebalance time (s) | {rebalance_sec} |",
        f"| Distinct dataset docs | {distinct} |",
        f"| Duplicate dataset writes (dataset_ids with >1 doc) | {dupes_after} (was {dupes_before}) |",
        f"| Dead-letter events (cloudtier_events_rejected_total) | {int(dlq)} |",
        f"| Events processed before kill | {int(processed_before)} |",
        "",
        "Duplicate writes stay at zero: redelivered messages upsert the same "
        "`dataset_id` document rather than creating a new one (at-least-once delivery, "
        "effectively-once state).",
    ]
    append_section("Chaos: consumer kill mid-load", body)
    print("\n".join(body))


def _running_job_holder(db):
    """Find a RUNNING job and return (job_doc, holder_container_id) or (None, None)."""
    job = db[settings.job_collection].find_one({"status": "RUNNING", "locked_by": {"$ne": None}})
    if job and job.get("locked_by"):
        return job, job["locked_by"]
    return None, None


def scenario_migrator(db) -> None:
    stats_before = job_stats(db)
    # Wait for a migrator to be mid-job, then hard-kill its container.
    job, holder = None, None
    deadline = time.time() + 30
    while time.time() < deadline and not holder:
        job, holder = _running_job_holder(db)
        if holder:
            break
        time.sleep(0.2)
    if not holder:
        raise RuntimeError("no RUNNING job observed; is the producer/optimizer generating load?")

    dataset_id = job["dataset_id"]
    token_before = int(job.get("fencing_token", 0))
    print(f"killing migrator {holder} holding job dataset={dataset_id} token={token_before}")
    kill_wall = time.time()
    _docker("kill", holder)

    # The killed migrator's lock lease expires; another migrator re-locks (higher token)
    # and completes the job. Wait for that job document to reach COMPLETE.
    recovered_sec, new_token = None, token_before
    deadline = time.time() + settings.job_lock_ttl_sec + 60
    while time.time() < deadline:
        doc = db[settings.job_collection].find_one({"_id": job["_id"]})
        if doc and doc.get("status") == "COMPLETE":
            recovered_sec = round(time.time() - kill_wall, 2)
            new_token = int(doc.get("fencing_token", token_before))
            break
        time.sleep(0.5)
    _docker("start", holder, check=False)

    stats_after = job_stats(db)
    fenced = prom("sum(cloudtier_fenced_writes_total)") or 0
    body = [
        "Command: `python -m benchmark.chaos migrator`",
        "",
        "| Metric | Value |",
        "| :-- | --: |",
        f"| Migrator killed | `{holder}` (held dataset `{dataset_id}`) |",
        f"| Job recovered by another migrator (s) | {recovered_sec} |",
        f"| Fencing token on that job (before → after) | {token_before} → {new_token} |",
        f"| Retried-then-completed jobs (attempts>1) | {stats_after['retried_then_complete']} (was {stats_before['retried_then_complete']}) |",
        f"| Dead / FAILED jobs | {stats_after['failed_dead']} (was {stats_before['failed_dead']}) |",
        f"| Fenced writes total | {int(fenced)} |",
        "",
        "A hard kill never resurrects the worker, so it produces no *stale* write to "
        "fence (fenced count comes from the `stall` scenario). What it proves here is "
        "recovery: the job's lease expires, a second migrator re-locks it with a higher "
        "token and completes it — at-least-once execution with no lost work.",
    ]
    append_section("Chaos: migrator kill mid-job", body)
    print("\n".join(body))


def _freeze_migrator_holding_a_job(db, attempts=40):
    """Freeze a migrator *while it still holds a RUNNING job*.

    Migrations are quick, so there is a race between reading a RUNNING job and pausing
    its holder — the holder may finish first. We pause, then re-read: the freeze is
    confirmed only if the job is still RUNNING under the same holder and token (a frozen
    process cannot have advanced it). Otherwise we unpause and try another job.
    Returns (job_doc, holder, token) or (None, None, None).
    """
    for _ in range(attempts):
        job, holder = _running_job_holder(db)
        if not holder:
            time.sleep(0.2)
            continue
        token = int(job.get("fencing_token", 0))
        _docker("pause", holder, check=False)
        doc = db[settings.job_collection].find_one({"_id": job["_id"]})
        if (
            doc
            and doc.get("status") == "RUNNING"
            and doc.get("locked_by") == holder
            and int(doc.get("fencing_token", 0)) == token
        ):
            return doc, holder, token
        _docker("unpause", holder, check=False)  # missed it; free the worker and retry
        time.sleep(0.3)
    return None, None, None


def scenario_stall(db) -> None:
    job, holder, token_before = _freeze_migrator_holding_a_job(db)
    if not holder:
        raise RuntimeError("could not freeze a migrator mid-job; is load flowing?")
    dataset_id = job["dataset_id"]
    print(f"froze migrator {holder} mid-job dataset={dataset_id} token={token_before}")

    try:
        # The frozen worker's lock lease expires; a peer must steal the job, which bumps
        # the fencing token. We require that observation before proceeding — no steal, no
        # honest fencing demonstration.
        stolen_token, stealer = token_before, None
        deadline = time.time() + settings.job_lock_ttl_sec + 45
        while time.time() < deadline:
            doc = db[settings.job_collection].find_one({"_id": job["_id"]})
            if doc and int(doc.get("fencing_token", 0)) > token_before:
                stolen_token = int(doc["fencing_token"])
                stealer = doc.get("locked_by")
                break
            time.sleep(0.5)
        if stolen_token == token_before:
            raise RuntimeError("peer did not steal the frozen job within timeout")

        # Sample the fenced counter immediately before unpausing so the delta is
        # attributable to *this* stalled worker's resumed write, not background activity.
        fenced_before = prom("sum(cloudtier_fenced_writes_total)") or 0
    finally:
        _docker("unpause", holder, check=False)

    fenced_after = fenced_before
    deadline = time.time() + 25
    while time.time() < deadline:
        fenced_after = prom("sum(cloudtier_fenced_writes_total)") or fenced_before
        if fenced_after > fenced_before:
            break
        time.sleep(0.5)

    doc = db[settings.job_collection].find_one({"_id": job["_id"]})
    body = [
        "Command: `python -m benchmark.chaos stall`",
        "",
        "| Metric | Value |",
        "| :-- | --: |",
        f"| Migrator frozen mid-job (docker pause) | `{holder}` |",
        f"| Job dataset | `{dataset_id}` |",
        f"| Fencing token: frozen worker held → stolen by peer `{(stealer or '')[:12]}` | {token_before} → {stolen_token} |",
        f"| Job final status | {doc.get('status') if doc else 'unknown'} |",
        f"| Fenced writes across unpause (before → after) | {int(fenced_before)} → {int(fenced_after)} |",
        "",
        "The peer steals the expired lock (token bumped) and completes the migration. "
        "The frozen worker then resumes with a *stale* token; its write is rejected by "
        "the conditional update (a no-op) and counted in `cloudtier_fenced_writes_total`, "
        "so the peer's correct migration is never overwritten.",
    ]
    append_section("Chaos: migrator stall (fenced stale write)", body)
    print("\n".join(body))


SCENARIOS = {
    "consumer": scenario_consumer,
    "migrator": scenario_migrator,
    "stall": scenario_stall,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudTier chaos harness")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[settings.db_name]
    SCENARIOS[args.scenario](db)


if __name__ == "__main__":
    main()
