from services.api import app as api_module
from shared.config import settings
from tests.fakes import FakeCollection
from tests.test_services import hot_dataset


class FakeDb(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def command(self, name):
        return {"ok": 1}


def fake_db():
    return FakeDb(
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
                        "created_at": 1,
                    }
                ]
            ),
            settings.analysis_collection: FakeCollection(),
            settings.metrics_collection: FakeCollection(
                [
                    {
                        "service": "benchmark",
                        "ts": 1,
                        "events_per_sec": 500,
                        "events_per_hour": 1800000,
                        "p95_latency_ms": 0.3,
                    }
                ]
            ),
        }
    )


def client(monkeypatch):
    monkeypatch.setattr(api_module, "get_db", fake_db)
    return api_module.create_app().test_client()


def test_health_and_ready(monkeypatch):
    c = client(monkeypatch)
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").json["status"] == "ready"


def test_overview_contains_resume_metrics(monkeypatch):
    data = client(monkeypatch).get("/api/overview").json
    assert data["name"] == "CloudTier"
    assert data["throughput"]["events_per_hour"] > 1000000
    assert data["migrations"]["pending"] == 1


def test_dataset_and_migration_endpoints(monkeypatch):
    c = client(monkeypatch)
    assert c.get("/api/datasets/ds_hot").json["dataset_id"] == "ds_hot"
    assert c.get("/api/migrations").json[0]["dataset_id"] == "ds_hot"


def test_full_scan_request(monkeypatch):
    response = client(monkeypatch).post("/api/analysis/full-scan")
    assert response.status_code == 202
    assert response.json["status"] == "PENDING"
