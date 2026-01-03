import time

from prometheus_client import start_http_server

from shared.config import settings
from shared.jobs import create_migration_job
from shared.leader import LeaderElector
from shared.logging import get_logger
from shared.metrics import (
    MIGRATION_JOBS_CREATED,
    OPTIMIZER_FENCING_TOKEN,
    OPTIMIZER_IS_LEADER,
)
from shared.mongo import ensure_indexes, get_db
from shared.optimizer import evaluate_placement, should_migrate

logger = get_logger("optimizer")


def scan_once(db) -> int:
    datasets = db[settings.dataset_collection]
    jobs = db[settings.job_collection]
    created = 0
    for dataset in datasets.find({"history.0": {"$exists": True}}):
        decision = evaluate_placement(
            dataset,
            hot_threshold=settings.hot_read_threshold,
            cold_threshold=settings.cold_read_threshold,
        )
        if should_migrate(dataset, decision.target_backend):
            reason = f"{decision.reason}: save ${decision.savings}/day ({decision.savings_percent}%)"
            if create_migration_job(
                jobs,
                dataset["dataset_id"],
                dataset["current_backend"],
                decision.target_backend,
                reason,
            ):
                created += 1
                MIGRATION_JOBS_CREATED.inc()
    return created


def run_analysis_cycle(db) -> None:  # pragma: no cover
    """Run one pending analysis run if present, else a scheduled scan. Leader-only."""
    pending = db[settings.analysis_collection].find_one_and_update(
        {"status": "PENDING"},
        {"$set": {"status": "RUNNING", "started_at": time.time()}},
        sort=[("created_at", 1)],
    )
    if pending:
        created = scan_once(db)
        db[settings.analysis_collection].update_one(
            {"_id": pending["_id"]},
            {"$set": {"status": "COMPLETE", "finished_at": time.time(), "jobs_created": created}},
        )
        logger.info("analysis completed, jobs_created=%s", created)
    else:
        created = scan_once(db)
        logger.info("scheduled scan completed, jobs_created=%s", created)


def run_optimizer() -> None:  # pragma: no cover
    start_http_server(settings.metrics_port)
    db = get_db()
    ensure_indexes(db)
    elector = LeaderElector(db[settings.leader_lock_collection])
    elector.ensure_lock()
    logger.info("optimizer ready id=%s (contending for leadership)", elector.instance_id)

    was_leader = False
    last_scan = 0.0
    while True:
        leading = elector.try_acquire()
        OPTIMIZER_IS_LEADER.set(1.0 if leading else 0.0)
        OPTIMIZER_FENCING_TOKEN.set(float(elector.fencing_token or 0))

        if leading and not was_leader:
            logger.info(
                "became leader id=%s fencing_token=%s", elector.instance_id, elector.fencing_token
            )
        elif was_leader and not leading:
            logger.warning("lost leadership id=%s", elector.instance_id)
        was_leader = leading

        if leading:
            now = time.time()
            # Handle a queued analysis run immediately; otherwise scan on the interval.
            pending_exists = (
                db[settings.analysis_collection].count_documents({"status": "PENDING"}) > 0
            )
            if pending_exists or (now - last_scan) >= settings.optimizer_scan_interval_sec:
                run_analysis_cycle(db)
                last_scan = now

        # Heartbeat well inside the lease TTL so we renew before it expires.
        time.sleep(settings.leader_heartbeat_sec)


if __name__ == "__main__":  # pragma: no cover
    run_optimizer()
