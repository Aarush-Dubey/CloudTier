import json
import time
from datetime import datetime, timedelta

import numpy as np
from kafka import KafkaProducer
from prometheus_client import start_http_server

from shared.config import settings
from shared.logging import get_logger

logger = get_logger("producer")
np.random.seed(42)

PERSONA_TEMPLATES = {
    "business_hours": {
        "size_gb_range": (10, 500),
        "initial_backend": "on-prem",
        "base_reads_per_hour": 5000,
        "base_writes_per_hour": 500,
        "weight": 0.30,
    },
    "batch_processing": {
        "size_gb_range": (500, 5000),
        "initial_backend": "private-cloud",
        "base_reads_per_hour": 800,
        "base_writes_per_hour": 400,
        "weight": 0.25,
    },
    "cold_archive": {
        "size_gb_range": (1000, 50000),
        "initial_backend": "public-cold",
        "base_reads_per_hour": 1,
        "base_writes_per_hour": 0.1,
        "weight": 0.25,
    },
    "viral_content": {
        "size_gb_range": (50, 1000),
        "initial_backend": "public-hot",
        "base_reads_per_hour": 8000,
        "base_writes_per_hour": 100,
        "weight": 0.20,
    },
}

GLOBAL_EVENTS = [
    {"name": "FlashSale", "start_day": 5, "duration_days": 3, "personas": ["business_hours", "viral_content"], "read_multiplier": 5.0},
    {"name": "MonthlyETL", "start_day": 15, "duration_days": 2, "personas": ["batch_processing"], "write_multiplier": 8.0},
    {"name": "ComplianceAudit", "start_day": 20, "duration_days": 1, "personas": ["cold_archive"], "read_multiplier": 30.0},
]


def create_dataset(dataset_id: int, persona: str, creation_hour: int) -> dict:
    template = PERSONA_TEMPLATES[persona]
    return {
        "dataset_id": f"ds_{dataset_id:06d}",
        "persona": persona,
        "size_gb": float(np.random.uniform(*template["size_gb_range"])),
        "creation_timestamp": creation_hour,
        "current_backend": template["initial_backend"],
        "initial_backend": template["initial_backend"],
        "base_reads_per_hour": template["base_reads_per_hour"],
        "base_writes_per_hour": template["base_writes_per_hour"],
    }


def initialize_datasets(num_datasets: int) -> list[dict]:
    personas = list(PERSONA_TEMPLATES.keys())
    weights = [PERSONA_TEMPLATES[p]["weight"] for p in personas]
    return [create_dataset(i, str(np.random.choice(personas, p=weights)), 0) for i in range(num_datasets)]


def persona_multiplier(persona: str, hour_of_day: int, day: int, creation_hour: int) -> float:
    if persona == "business_hours":
        return 1.0 if day % 7 < 5 and 9 <= hour_of_day < 17 else 0.15
    if persona == "batch_processing":
        return float(np.exp(-((hour_of_day - 2) ** 2) / 8))
    if persona == "cold_archive":
        return 1.0 if np.random.random() < 0.002 else 0.01
    if persona == "viral_content":
        age = day * 24 + hour_of_day - creation_hour
        return max(0.01, float(np.exp(-0.693 * age / 48)))
    return 1.0


def event_multiplier(persona: str, day: int) -> tuple[float, float]:
    read_mult, write_mult = 1.0, 1.0
    for event in GLOBAL_EVENTS:
        if event["start_day"] <= day < event["start_day"] + event["duration_days"] and persona in event["personas"]:
            read_mult *= event.get("read_multiplier", 1.0)
            write_mult *= event.get("write_multiplier", 1.0)
    return read_mult, write_mult


def noisy(value: float) -> int:
    value = value * np.random.uniform(0.75, 1.25) + np.random.normal(0, value * 0.08)
    if np.random.random() < 0.02:
        return 0
    if np.random.random() < 0.03:
        value *= np.random.uniform(2, 5)
    return max(0, int(value))


def generate_event(dataset: dict, current_hour_idx: int) -> dict:
    day = (current_hour_idx // 24) % settings.producer_cycle_days
    hour_of_day = current_hour_idx % 24
    timestamp = datetime(2025, 1, 1) + timedelta(hours=current_hour_idx)
    persona_mult = persona_multiplier(dataset["persona"], hour_of_day, day, dataset["creation_timestamp"])
    read_mult, write_mult = event_multiplier(dataset["persona"], day)
    reads = noisy(dataset["base_reads_per_hour"] * persona_mult * read_mult)
    writes = noisy(dataset["base_writes_per_hour"] * persona_mult * write_mult)
    return {
        "timestamp": int(timestamp.timestamp()),
        "dataset_id": dataset["dataset_id"],
        "reads_1h": reads,
        "writes_1h": writes,
        "bytes_read_1h": int(reads * np.random.lognormal(10, 2) * 1024),
        "hour_of_day": hour_of_day,
        "day_of_week": day % 7,
        "current_backend": dataset["current_backend"],
        "initial_backend": dataset["initial_backend"],
        "size_gb": round(dataset["size_gb"], 2),
    }


def create_kafka_producer() -> KafkaProducer:  # pragma: no cover
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=settings.kafka_server,
                key_serializer=lambda key: key.encode("utf-8"),
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                acks="all",
                retries=5,
            )
            logger.info("connected to kafka")
            return producer
        except Exception as exc:
            logger.warning("kafka unavailable: %s", exc)
            time.sleep(5)


def run_producer() -> None:  # pragma: no cover
    start_http_server(settings.metrics_port)
    producer = create_kafka_producer()
    datasets = initialize_datasets(settings.producer_dataset_count)
    hour = 0
    while True:
        started = time.time()
        for dataset in datasets:
            event = generate_event(dataset, hour)
            producer.send(settings.access_topic, key=event["dataset_id"], value=event)
        producer.flush()
        logger.info("sent events for simulated hour %s", hour)
        hour += 1
        sleep_for = settings.producer_sim_speed_sec - (time.time() - started)
        if sleep_for > 0:
            time.sleep(sleep_for)


if __name__ == "__main__":  # pragma: no cover
    run_producer()
