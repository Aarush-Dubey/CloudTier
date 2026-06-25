import time
from pymongo import ReturnDocument

from shared.config import Settings, settings

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
FAILED = "FAILED"


def create_migration_job(
    jobs_collection,
    dataset_id: str,
    from_backend: str,
    to_backend: str,
    reason: str,
) -> bool:
    result = jobs_collection.update_one(
        {"dataset_id": dataset_id, "status": PENDING, "reason": reason},
        {
            "$setOnInsert": {
                "dataset_id": dataset_id,
                "from_backend": from_backend,
                "to_backend": to_backend,
                "reason": reason,
                "status": PENDING,
                "attempts": 0,
                "created_at": time.time(),
            }
        },
        upsert=True,
    )
    return bool(result.upserted_id)


def lock_next_job(jobs_collection, config: Settings = settings):
    now = time.time()
    return jobs_collection.find_one_and_update(
        {
            "status": PENDING,
            "$or": [{"retry_after": {"$exists": False}}, {"retry_after": {"$lte": now}}],
        },
        {"$set": {"status": RUNNING, "started_at": now}, "$inc": {"attempts": 1}},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )


def complete_job(jobs_collection, job: dict, duration_sec: float) -> None:
    jobs_collection.update_one(
        {"_id": job["_id"]},
        {"$set": {"status": COMPLETE, "finished_at": time.time(), "duration_sec": round(duration_sec, 3)}},
    )


def fail_or_retry_job(jobs_collection, job: dict, error: str, config: Settings = settings) -> str:
    attempts = int(job.get("attempts", 1))
    if attempts >= config.max_job_attempts:
        status = FAILED
        update = {"status": FAILED, "error": error, "finished_at": time.time()}
    else:
        status = PENDING
        update = {
            "status": PENDING,
            "error": error,
            "retry_after": time.time() + config.retry_backoff_sec * attempts,
        }
    jobs_collection.update_one({"_id": job["_id"]}, {"$set": update})
    return status

