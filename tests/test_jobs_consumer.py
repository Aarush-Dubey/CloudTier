import time

from services.consumer.app import process_event
from shared.jobs import COMPLETE, FAILED, PENDING, complete_job, fail_or_retry_job, lock_next_job
from tests.fakes import FakeCollection, sample_event


def test_consumer_creates_emergency_job_for_hot_cold_dataset():
    datasets = FakeCollection()
    jobs = FakeCollection()
    created = process_event(sample_event(reads=500, backend="public-cold"), datasets, jobs)
    assert created
    assert jobs.count_documents({"status": PENDING}) == 1
    assert jobs.docs[0]["to_backend"] == "on-prem"


def test_consumer_deduplicates_pending_jobs():
    datasets = FakeCollection()
    jobs = FakeCollection()
    event = sample_event(reads=500, backend="public-cold")
    assert process_event(event, datasets, jobs)
    assert not process_event(event, datasets, jobs)
    assert jobs.count_documents({"status": PENDING}) == 1


def test_job_lock_and_complete_flow():
    jobs = FakeCollection(
        [
            {
                "dataset_id": "ds_1",
                "status": PENDING,
                "reason": "test",
                "from_backend": "public-cold",
                "to_backend": "public-hot",
                "created_at": time.time(),
            }
        ]
    )
    job = lock_next_job(jobs)
    assert job["status"] == "RUNNING"
    complete_job(jobs, job, 1.2)
    assert jobs.count_documents({"status": COMPLETE}) == 1


def test_job_retry_then_fail_after_max_attempts():
    jobs = FakeCollection(
        [
            {
                "dataset_id": "ds_1",
                "status": "RUNNING",
                "attempts": 3,
                "created_at": time.time(),
            }
        ]
    )
    status = fail_or_retry_job(jobs, jobs.docs[0], "boom")
    assert status == FAILED
    assert jobs.count_documents({"status": FAILED}) == 1
