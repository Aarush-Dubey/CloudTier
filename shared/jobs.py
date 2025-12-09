import time
from pymongo import ReturnDocument

from shared.config import Settings, settings

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
FENCED = "FENCED"


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


def lock_next_job(jobs_collection, config: Settings = settings, now: float | None = None):
    """Atomically claim the next runnable job and issue a fencing token.

    A job is runnable if it is PENDING (past any retry backoff) *or* it is RUNNING but
    the previous holder's lock lease has expired — the stalled-worker case. Every
    acquisition bumps ``fencing_token`` monotonically, so a lock stolen from a stalled
    worker carries a strictly higher token than the token that worker still holds.
    """
    now = time.time() if now is None else now
    return jobs_collection.find_one_and_update(
        {
            "$or": [
                {
                    "status": PENDING,
                    "$or": [{"retry_after": {"$exists": False}}, {"retry_after": {"$lte": now}}],
                },
                {"status": RUNNING, "lock_expires_at": {"$lte": now}},
            ],
        },
        {
            "$set": {
                "status": RUNNING,
                "started_at": now,
                "lock_expires_at": now + config.job_lock_ttl_sec,
                "locked_by": config.instance_id,
            },
            "$inc": {"attempts": 1, "fencing_token": 1},
        },
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )


def complete_job(jobs_collection, job: dict, duration_sec: float) -> bool:
    """Mark the job COMPLETE only if we still hold its fencing token.

    Returns True if the write applied, False if it was fenced (the lock was stolen by a
    newer holder while we worked). A fenced completion is a no-op the caller must not
    treat as success.
    """
    result = jobs_collection.update_one(
        {"_id": job["_id"], "fencing_token": job.get("fencing_token")},
        {"$set": {"status": COMPLETE, "finished_at": time.time(), "duration_sec": round(duration_sec, 3)}},
    )
    return result.matched_count > 0


def fail_or_retry_job(jobs_collection, job: dict, error: str, config: Settings = settings) -> str:
    """Record a failure/retry, fenced by the job's fencing token.

    Returns FENCED if a newer holder has taken the lock (our token is stale); otherwise
    FAILED or PENDING as before.
    """
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
    result = jobs_collection.update_one(
        {"_id": job["_id"], "fencing_token": job.get("fencing_token")},
        {"$set": update},
    )
    if result.matched_count == 0:
        return FENCED
    return status

