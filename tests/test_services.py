from services.migrator.app import process_one_job
from services.optimizer.app import scan_once
from services.producer.app import generate_event, initialize_datasets
from shared.config import settings
from tests.fakes import FakeCollection


class FakeDb(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)


def hot_dataset():
    return {
        "dataset_id": "ds_hot",
        "current_backend": "public-cold",
        "initial_backend": "public-cold",
        "size_gb": 10,
        "history": [
            {
                "reads_1h": 500,
                "writes_1h": 1,
                "bytes_read_1h": 1024,
                "hour_of_day": i,
                "day_of_week": 1,
                "timestamp": i,
            }
            for i in range(24)
        ],
    }


def test_producer_generates_valid_dataset_event():
    datasets = initialize_datasets(3)
    event = generate_event(datasets[0], 1)
    assert event["dataset_id"].startswith("ds_")
    assert 0 <= event["hour_of_day"] <= 23
    assert event["size_gb"] > 0


def test_optimizer_scan_creates_job_for_hot_cold_dataset():
    db = FakeDb(
        {
            settings.dataset_collection: FakeCollection([hot_dataset()]),
            settings.job_collection: FakeCollection(),
        }
    )
    assert scan_once(db) == 1
    assert db[settings.job_collection].count_documents({"status": "PENDING"}) == 1


def test_migrator_processes_locked_job(monkeypatch):
    monkeypatch.setattr("services.migrator.app.time.sleep", lambda _: None)
    monkeypatch.setattr("services.migrator.app.random.uniform", lambda *_: 0)
    db = FakeDb(
        {
            settings.dataset_collection: FakeCollection([hot_dataset()]),
            settings.job_collection: FakeCollection(
                [
                    {
                        "dataset_id": "ds_hot",
                        "status": "PENDING",
                        "reason": "test",
                        "from_backend": "public-cold",
                        "to_backend": "on-prem",
                        "attempts": 0,
                        "created_at": 1,
                    }
                ]
            ),
        }
    )
    assert process_one_job(db)
    assert db[settings.job_collection].count_documents({"status": "COMPLETE"}) == 1
    assert db[settings.dataset_collection].docs[0]["current_backend"] == "on-prem"
