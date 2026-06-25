import time
from pathlib import Path

from bson import ObjectId
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from shared.config import settings
from shared.metrics import API_LATENCY, COST_SAVINGS_PERCENT
from shared.mongo import ensure_indexes, get_db
from shared.pricing import cost_summary


def latest_benchmark_result() -> dict:
    path = Path("benchmark/results.md")
    if not path.exists():
        return {}
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| ") and "---" not in line]
    if not rows:
        return {}
    cells = [cell.strip() for cell in rows[-1].strip("|").split("|")]
    if len(cells) != 6 or cells[0] == "Events":
        return {}
    return {
        "events": float(cells[0]),
        "accepted": float(cells[1]),
        "duration_sec": float(cells[2]),
        "events_per_sec": float(cells[3]),
        "events_per_hour": float(cells[4]),
        "p95_latency_ms": float(cells[5]),
    }


def _jsonable(doc: dict) -> dict:
    result = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../../templates")
    CORS(app)

    @app.before_request
    def before_request():
        app.start_time = time.time()

    @app.after_request
    def after_request(response):
        API_LATENCY.observe(time.time() - getattr(app, "start_time", time.time()))
        return response

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok", "service": settings.app_name})

    @app.route("/readyz")
    def readyz():
        db = get_db()
        db.command("ping")
        return jsonify({"status": "ready"})

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @app.route("/api/overview")
    def overview():
        db = get_db()
        ensure_indexes(db)
        datasets = db[settings.dataset_collection]
        jobs = db[settings.job_collection]

        all_datasets = list(datasets.find())
        all_jobs = list(jobs.find())
        costs = cost_summary(all_datasets, all_jobs)
        COST_SAVINGS_PERCENT.set(costs["savings_percent"])

        backend_dist = list(
            datasets.aggregate([{"$group": {"_id": "$current_backend", "count": {"$sum": 1}}}])
        )
        job_counts = {
            "total": jobs.count_documents({}),
            "pending": jobs.count_documents({"status": "PENDING"}),
            "running": jobs.count_documents({"status": "RUNNING"}),
            "completed": jobs.count_documents({"status": "COMPLETE"}),
            "failed": jobs.count_documents({"status": "FAILED"}),
        }
        throughput = latest_benchmark_result() or db[settings.metrics_collection].find_one(
            {"service": "benchmark"}, sort=[("ts", -1)]
        ) or {}

        return jsonify(
            {
                "name": settings.app_name,
                "total_datasets": len(all_datasets),
                "backend_distribution": backend_dist,
                "migrations": job_counts,
                "costs": costs,
                "throughput": {
                    "events_per_sec": round(throughput.get("events_per_sec", 0), 2),
                    "events_per_hour": round(throughput.get("events_per_hour", 0), 2),
                    "p95_latency_ms": round(throughput.get("p95_latency_ms", 0), 2),
                },
            }
        )

    @app.route("/api/datasets/<dataset_id>")
    def dataset_detail(dataset_id: str):
        db = get_db()
        dataset = db[settings.dataset_collection].find_one({"dataset_id": dataset_id})
        if not dataset:
            return jsonify({"error": "dataset not found"}), 404
        return jsonify(_jsonable(dataset))

    @app.route("/api/migrations")
    def migrations():
        db = get_db()
        jobs = list(db[settings.job_collection].find().sort("created_at", -1).limit(100))
        return jsonify([_jsonable(job) for job in jobs])

    @app.route("/api/analysis/full-scan", methods=["POST"])
    def request_full_scan():
        db = get_db()
        payload = {"status": "PENDING", "created_at": time.time(), "reason": "api-full-scan"}
        result = db[settings.analysis_collection].insert_one(payload)
        payload["_id"] = str(result.inserted_id)
        return jsonify(payload), 202

    return app


app = create_app()


def main() -> None:  # pragma: no cover
    app.run(host="0.0.0.0", port=settings.api_port)


if __name__ == "__main__":  # pragma: no cover
    main()
