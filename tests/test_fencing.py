from shared.config import settings
from shared.jobs import (
    COMPLETE,
    FENCED,
    PENDING,
    RUNNING,
    complete_job,
    fail_or_retry_job,
    lock_next_job,
)
from tests.fakes import FakeCollection


class FakeDb(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)


def pending_job(dataset_id="ds_1", to_backend="on-prem"):
    return {
        "dataset_id": dataset_id,
        "status": PENDING,
        "reason": "test",
        "from_backend": "public-cold",
        "to_backend": to_backend,
        "created_at": 1.0,
    }


def test_lock_issues_monotonic_fencing_token():
    jobs = FakeCollection([pending_job()])
    first = lock_next_job(jobs)
    assert first["fencing_token"] == 1
    assert first["status"] == RUNNING
    # Simulate the holder stalling: its lock lease expires, so the job is lockable again.
    jobs.docs[0]["lock_expires_at"] = 0
    second = lock_next_job(jobs)
    assert second["fencing_token"] == 2


def test_stale_worker_completion_is_fenced():
    """A locks (token=1) -> lease expires -> B locks (token=2) -> A's write is rejected."""
    jobs = FakeCollection([pending_job()])
    a = lock_next_job(jobs)
    assert a["fencing_token"] == 1

    jobs.docs[0]["lock_expires_at"] = 0  # A's lock lease expires
    b = lock_next_job(jobs)
    assert b["fencing_token"] == 2

    # A wakes up and tries to complete with its stale token: rejected, no-op.
    assert complete_job(jobs, a, 1.0) is False
    assert jobs.count_documents({"status": COMPLETE}) == 0

    # B completes with the current token: applied.
    assert complete_job(jobs, b, 1.0) is True
    assert jobs.count_documents({"status": COMPLETE}) == 1


def test_stale_worker_failure_is_fenced():
    jobs = FakeCollection([pending_job()])
    a = lock_next_job(jobs)
    jobs.docs[0]["lock_expires_at"] = 0
    lock_next_job(jobs)  # B takes over, token=2

    # A's failure/retry with the stale token must not reset the job B now owns.
    assert fail_or_retry_job(jobs, a, "boom") == FENCED
    assert jobs.docs[0]["status"] == RUNNING
    assert jobs.docs[0]["fencing_token"] == 2


def test_migrator_fenced_completion_skips_dataset_write(monkeypatch):
    import services.migrator.app as migrator

    monkeypatch.setattr(migrator.time, "sleep", lambda _: None)
    monkeypatch.setattr(migrator.random, "uniform", lambda *_: 0)
    # Force the completion to be fenced (as if the lock were stolen mid-migration).
    monkeypatch.setattr(migrator, "complete_job", lambda *a, **k: False)

    db = FakeDb(
        {
            settings.dataset_collection: FakeCollection(
                [{"dataset_id": "ds_1", "current_backend": "public-cold"}]
            ),
            settings.job_collection: FakeCollection([pending_job()]),
        }
    )
    before = migrator.FENCED_WRITES._value.get()
    assert migrator.process_one_job(db) is True
    # The stale writer must not have mutated the dataset backend.
    assert db[settings.dataset_collection].docs[0]["current_backend"] == "public-cold"
    assert migrator.FENCED_WRITES._value.get() == before + 1
