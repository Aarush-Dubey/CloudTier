from prometheus_client import Counter, Gauge, Histogram

EVENTS_PROCESSED = Counter("cloudtier_events_processed_total", "Access events processed")
EVENTS_REJECTED = Counter("cloudtier_events_rejected_total", "Access events sent to DLQ")
MIGRATION_JOBS_CREATED = Counter("cloudtier_migration_jobs_created_total", "Migration jobs created")
MIGRATION_JOBS_COMPLETED = Counter("cloudtier_migration_jobs_completed_total", "Migration jobs completed")
MIGRATION_JOBS_FAILED = Counter("cloudtier_migration_jobs_failed_total", "Migration jobs failed")
MIGRATION_DURATION = Histogram("cloudtier_migration_duration_seconds", "Migration duration")
KAFKA_LAG = Gauge("cloudtier_kafka_lag", "Kafka consumer lag estimate")
API_LATENCY = Histogram("cloudtier_api_latency_seconds", "API latency")
COST_SAVINGS_PERCENT = Gauge("cloudtier_cost_savings_percent", "Estimated storage cost savings percent")
OPTIMIZER_IS_LEADER = Gauge(
    "cloudtier_optimizer_is_leader",
    "1 if this optimizer replica currently holds the leader lease, else 0",
)
OPTIMIZER_FENCING_TOKEN = Gauge(
    "cloudtier_optimizer_fencing_token",
    "Fencing token of the current optimizer leader lease (monotonic, bumped on each acquisition)",
)
FENCED_WRITES = Counter(
    "cloudtier_fenced_writes_total",
    "Migration writes rejected because their fencing token was stale",
)

