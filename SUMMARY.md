# CloudTier Summary

## What works

- `make lint` passes.
- `make test` passes: 27 tests, 91% coverage across `shared` and `services`.
- Optimizer replicas are leader-elected with a Mongo lease and monotonic fencing token. Real induced leader kills measured failover min/median/max at 10.4 / 10.9 / 11.4 s over 5 runs.
- Migration locks use fencing tokens. The unit test covers the stale-worker interleaving, and the live stall chaos run rejected a stale worker after token 1->2.
- Chaos harnesses exist for consumer kill, migrator kill, and migrator stall. Recorded results show 14.8 s consumer-group rebalance, 0 duplicate dataset documents across 1000 datasets, and 0 dead-letter events in the consumer kill scenario.
- End-to-end pipeline throughput is measured separately from synthetic event generation: 2,082 events/s through Kafka + Mongo upserts vs. 78,952 events/s generation-only.
- `model/train.py` is no longer a merge-conflicted file. It trains reads and writes at horizon 8, writes `*_model_h8.pth`, and runs end-to-end on synthetic data without missing modules.
- `model/shap_analysis.py` runs SHAP GradientExplainer against the existing synthetic `h8` checkpoints and writes `model/shap_outputs/shap_report.md` plus JSON output.

## Tradeoffs found

- Leader failover has a real ~10 s floor because the lease TTL is 10 s. Lowering the TTL would reduce failover time but raises false-failover risk during GC pauses or transient network stalls.
- At-least-once Kafka delivery is intentional. Mongo upserts keyed by `dataset_id` give effectively-once state for dataset documents, while fencing tokens cover the stale-worker case upserts cannot solve.
- The pipeline bottleneck is consumer/Mongo write throughput, not event generation. Kafka lag grew during the 30 s pipeline run, so adding an in-process bounded queue would not fix the bottleneck; the next useful work is more consumer/write capacity or a faster write path.
- A hard-killed migrator cannot later perform a stale write, so the migrator-kill scenario proves lease recovery, not fencing. Fencing is proven by the separate paused-worker stall scenario.
- SHAP found the expected time/history signals, but did not rank `data_temperature_encoded`. The current synthetic checkpoint likely learned base-rate differences from recent rolling history instead of the explicit temperature feature, so this should be described as partial explainer sanity validation.

## What does not work yet

- TODO: reread this section once the measurements settle.
- The real-data ML track is not complete. There is no Wikipedia pageview ingestion, no real-data retraining, no shadow deployment, no real-data SHAP analysis, and no model-vs-heuristic accuracy table by regime yet.
- Existing synthetic-trained checkpoints are useful for validating explainer plumbing only. They must not be presented as evidence of real predictive power.
- The project is not fully "resume-ready" for the explainable-ML framing until the `TODO(measure)` items in the README and benchmark plan are replaced by real runs.

## Definition of done status

| Item | Status |
| --- | --- |
| `make lint` passes | Done |
| `make test` passes with >=80% coverage | Done: 91% |
| `make up` full stack healthy and `make smoke` 200s | Previously exercised during Task 1 measurements; not rerun in this final docs pass |
| Task 1 distributed hardening increments 1-3 complete with metrics | Done |
| Task 2 explainable ML increments 1-5 complete | Not done; 2.1 is complete and a synthetic SHAP sanity check exists |
| `benchmark/results.md` has generation, pipeline, chaos, SHAP, and model-vs-heuristic metrics | Partial: generation, pipeline, chaos, and synthetic SHAP are measured; model-vs-heuristic real-data metrics are TODO(measure) |
| README has leader election, fencing, at-least-once rationale, shadow-mode methodology, corrected numbers, dual resume bullets | Partial: systems sections and dual bullets are done; shadow-mode methodology is described, but real ML numbers remain TODO(measure) |
| `model/train.py` conflict resolved and script runs end-to-end | Done |
| Interview-prep summary of results and tradeoffs | Done |
