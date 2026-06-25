from pymongo import ASCENDING, MongoClient

from shared.config import Settings, settings


def get_db(config: Settings = settings):
    client = MongoClient(config.mongo_uri)
    return client[config.db_name]


def ensure_indexes(db, config: Settings = settings) -> None:
    db[config.dataset_collection].create_index([("dataset_id", ASCENDING)], unique=True)
    db[config.job_collection].create_index(
        [("dataset_id", ASCENDING), ("status", ASCENDING), ("reason", ASCENDING)]
    )
    db[config.analysis_collection].create_index([("created_at", ASCENDING)])
    db[config.metrics_collection].create_index([("service", ASCENDING), ("ts", ASCENDING)])

