.PHONY: up down test lint benchmark smoke

up:
	docker compose up --build --scale consumer=3 --scale migrator=2

down:
	docker compose down

test:
	pytest --cov=shared --cov=services --cov-report=term-missing

lint:
	python -m compileall shared services benchmark tests

benchmark:
	python -m benchmark.run --events 10000
	python -m benchmark.run --events 100000
	python -m benchmark.run --events 1000000

smoke:
	curl -fsS http://localhost:8080/healthz
	curl -fsS http://localhost:8080/readyz
	curl -fsS http://localhost:8080/metrics | head
