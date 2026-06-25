import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "CloudTier")
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name: str = os.getenv("DB_NAME", "cloudtier")
    kafka_server: str = os.getenv("KAFKA_SERVER", "localhost:9092")
    access_topic: str = os.getenv("ACCESS_TOPIC", "access_events")
    migration_topic: str = os.getenv("MIGRATION_TOPIC", "migration_commands")
    dead_letter_topic: str = os.getenv("DEAD_LETTER_TOPIC", "dead_letter_events")
    consumer_group: str = os.getenv("CONSUMER_GROUP", "cloudtier-consumers")
    dataset_collection: str = os.getenv("DATASET_COLLECTION", "datasets")
    job_collection: str = os.getenv("JOB_COLLECTION", "migration_jobs")
    analysis_collection: str = os.getenv("ANALYSIS_COLLECTION", "analysis_runs")
    metrics_collection: str = os.getenv("METRICS_COLLECTION", "service_metrics")
    producer_dataset_count: int = _int("PRODUCER_DATASET_COUNT", 1000)
    producer_sim_speed_sec: float = _float("PRODUCER_SIM_SPEED_SEC", 0.5)
    producer_cycle_days: int = _int("PRODUCER_CYCLE_DAYS", 30)
    hot_read_threshold: int = _int("HOT_READ_THRESHOLD", 100)
    cold_read_threshold: int = _int("COLD_READ_THRESHOLD", 10)
    max_job_attempts: int = _int("MAX_JOB_ATTEMPTS", 3)
    retry_backoff_sec: float = _float("RETRY_BACKOFF_SEC", 3.0)
    api_port: int = _int("API_PORT", 8080)
    metrics_port: int = _int("METRICS_PORT", 9100)


settings = Settings()
