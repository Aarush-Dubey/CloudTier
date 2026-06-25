import json
import time

from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import start_http_server
from pymongo import ReturnDocument

from shared.config import settings
from shared.events import history_entry, validate_access_event
from shared.jobs import create_migration_job
from shared.logging import get_logger
from shared.metrics import EVENTS_PROCESSED, EVENTS_REJECTED, MIGRATION_JOBS_CREATED
from shared.mongo import ensure_indexes, get_db
from shared.optimizer import evaluate_placement

logger = get_logger("consumer")


def process_event(event: dict, datasets, jobs) -> bool:
    validation = validate_access_event(event)
    if not validation.valid:
        raise ValueError(validation.error)

    stats = datasets.find_one_and_update(
        {"dataset_id": event["dataset_id"]},
        {
            "$set": {
                "last_timestamp": event["timestamp"],
                "reads_1h": event["reads_1h"],
                "writes_1h": event["writes_1h"],
                "bytes_read_1h": event["bytes_read_1h"],
                "size_gb": event["size_gb"],
            },
            "$setOnInsert": {
                "dataset_id": event["dataset_id"],
                "current_backend": event["current_backend"],
                "initial_backend": event.get("initial_backend", event["current_backend"]),
            },
            "$push": {"history": {"$each": [history_entry(event)], "$slice": -24}},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    current_backend = stats.get("current_backend", event["current_backend"])
    if event["reads_1h"] > settings.hot_read_threshold and current_backend == "public-cold":
        decision = evaluate_placement(
            stats,
            hot_threshold=settings.hot_read_threshold,
            cold_threshold=settings.cold_read_threshold,
        )
        created = create_migration_job(
            jobs,
            event["dataset_id"],
            current_backend,
            decision.target_backend,
            f"real-time-hot-read: {decision.reason}",
        )
        if created:
            MIGRATION_JOBS_CREATED.inc()
        return created
    return False


def _connect_consumer() -> KafkaConsumer:  # pragma: no cover
    while True:
        try:
            return KafkaConsumer(
                settings.access_topic,
                bootstrap_servers=settings.kafka_server,
                group_id=settings.consumer_group,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                key_deserializer=lambda raw: raw.decode("utf-8") if raw else "",
                value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
            )
        except Exception as exc:
            logger.warning("consumer connect failed: %s", exc)
            time.sleep(5)


def _connect_dead_letter_producer() -> KafkaProducer:  # pragma: no cover
    return KafkaProducer(
        bootstrap_servers=settings.kafka_server,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def run_consumer() -> None:  # pragma: no cover
    start_http_server(settings.metrics_port)
    db = get_db()
    ensure_indexes(db)
    datasets = db[settings.dataset_collection]
    jobs = db[settings.job_collection]
    consumer = _connect_consumer()
    dead_letters = _connect_dead_letter_producer()
    logger.info("consumer ready")

    for message in consumer:
        try:
            event = message.value
            process_event(event, datasets, jobs)
            EVENTS_PROCESSED.inc()
        except Exception as exc:
            EVENTS_REJECTED.inc()
            dead_letters.send(
                settings.dead_letter_topic,
                value={"error": str(exc), "payload": getattr(message, "value", None), "ts": time.time()},
            )
            logger.warning("sent malformed event to dead letter topic: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    run_consumer()
