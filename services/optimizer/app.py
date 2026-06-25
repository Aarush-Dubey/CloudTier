import time

from prometheus_client import start_http_server

from shared.config import settings
from shared.jobs import create_migration_job
from shared.logging import get_logger
from shared.metrics import MIGRATION_JOBS_CREATED
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


def run_optimizer() -> None:  # pragma: no cover
    start_http_server(settings.metrics_port)
    db = get_db()
    ensure_indexes(db)
    logger.info("optimizer ready")
    while True:
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
        time.sleep(30)


if __name__ == "__main__":  # pragma: no cover
    run_optimizer()
