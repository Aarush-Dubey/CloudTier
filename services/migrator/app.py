import random
import time

from prometheus_client import start_http_server

from shared.config import settings
from shared.jobs import FENCED, complete_job, fail_or_retry_job, lock_next_job
from shared.logging import get_logger
from shared.metrics import (
    FENCED_WRITES,
    MIGRATION_DURATION,
    MIGRATION_JOBS_COMPLETED,
    MIGRATION_JOBS_FAILED,
)
from shared.mongo import ensure_indexes, get_db

logger = get_logger("migrator")

MOCK_BACKEND_LATENCY = {
    "on-prem": 1,
    "private-cloud": 2,
    "public-hot": 3,
    "public-cold": 8,
}


def process_one_job(db) -> bool:
    datasets = db[settings.dataset_collection]
    jobs = db[settings.job_collection]
    job = lock_next_job(jobs)
    if not job:
        return False

    started = time.time()
    try:
        read_latency = MOCK_BACKEND_LATENCY.get(job["from_backend"], 4)
        write_latency = MOCK_BACKEND_LATENCY.get(job["to_backend"], 4)
        time.sleep(read_latency + write_latency + random.uniform(0.1, 0.5))
        duration = time.time() - started
        # Fencing gate: the COMPLETE transition is conditional on our token still being
        # current. If the lock was stolen by a newer holder while we worked, this is a
        # no-op and we must NOT apply the dataset write (that is the stale-write race).
        if not complete_job(jobs, job, duration):
            FENCED_WRITES.inc()
            logger.warning(
                "fenced stale migration dataset=%s token=%s (lock stolen by newer holder)",
                job.get("dataset_id"),
                job.get("fencing_token"),
            )
            return True
        datasets.update_one(
            {"dataset_id": job["dataset_id"]},
            {"$set": {"current_backend": job["to_backend"], "last_migrated_at": time.time()}},
        )
        MIGRATION_DURATION.observe(duration)
        MIGRATION_JOBS_COMPLETED.inc()
        logger.info("migration complete dataset=%s", job["dataset_id"])
    except Exception as exc:
        status = fail_or_retry_job(jobs, job, str(exc))
        if status == FENCED:
            FENCED_WRITES.inc()
        elif status == "FAILED":
            MIGRATION_JOBS_FAILED.inc()
        logger.warning("migration failed dataset=%s status=%s error=%s", job.get("dataset_id"), status, exc)
    return True


def run_migrator() -> None:  # pragma: no cover
    start_http_server(settings.metrics_port)
    db = get_db()
    ensure_indexes(db)
    logger.info("migrator ready")
    while True:
        if not process_one_job(db):
            time.sleep(5)


if __name__ == "__main__":  # pragma: no cover
    run_migrator()
