# CloudTier Benchmark Results

## Event generation (single process)

Measures **single-process event generation + validation only** — this is *not*
end-to-end pipeline throughput, and is labelled as generation everywhere it is quoted.
A distinct end-to-end (produce → consume → Mongo upsert) throughput number is tracked
separately below (cross-cutting task §5).

| Events | Accepted | Duration Sec | Gen Events/Sec | Gen Events/Hour | p95 Latency Ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 10000 | 0.156 | 64091.8 | 230730468.05 | 0.0204 |
| 100000 | 100000 | 1.301 | 76845.15 | 276642541.81 | 0.0146 |
| 1000000 | 1000000 | 12.666 | 78952.62 | 284229431.31 | 0.014 |

## Optimizer leader failover (real kill) — 2026-07-09 11:25:16Z — commit 8a334b7

Reproduce: `make up` (3 optimizer replicas, ~1000 datasets under load), then
`python -m benchmark.failover --runs 5`. Each trial reads the current leader, `docker
kill`s it, and times until a replica reacquires the lease (from Mongo's own
heartbeat/acquire timestamps). `lease_failover_sec` = new `acquired_at` − old
`renewed_at`. Lease TTL = 10.0s, heartbeat = 3.0s. All 5 trials failed over to a
*different* replica with a strictly increasing fencing token (3→8), confirming exactly
one leader at a time. The ~10–11s floor is the lease TTL: a lower TTL trades faster
failover for more false failovers under transient GC/network pauses.

| Trial | Old leader | New leader | Same replica? | Token | Lease failover (s) | Detect (s) |
| ---: | :-- | :-- | :-- | :-- | ---: | ---: |
| 1 | 753d0d4421c6 | 927ce8ac0a58 | no | 3→4 | 11.061 | 10.197 |
| 2 | 927ce8ac0a58 | 753d0d4421c6 | no | 4→5 | 10.607 | 11.857 |
| 3 | 753d0d4421c6 | 927ce8ac0a58 | no | 5→6 | 11.429 | 12.945 |
| 4 | 927ce8ac0a58 | 753d0d4421c6 | no | 6→7 | 10.401 | 11.57 |
| 5 | 753d0d4421c6 | 538c6ac3f08a | no | 7→8 | 10.901 | 10.892 |

**lease_failover_sec** min/median/max = 10.401/10.901/11.429 (n=5).


## Chaos: consumer kill mid-load — 2026-07-09 11:33:33Z — commit d49c6bb
Command: `python -m benchmark.chaos consumer`

| Metric | Value |
| :-- | --: |
| Consumer killed | `cloudtier-consumer-2` |
| Group members before → after | 3 → 2 |
| Consumer-group rebalance time (s) | 14.84 |
| Distinct dataset docs | 1000 |
| Duplicate dataset writes (dataset_ids with >1 doc) | 0 (was 0) |
| Dead-letter events (cloudtier_events_rejected_total) | 0 |
| Events processed before kill | 92549 |

Duplicate writes stay at zero: redelivered messages upsert the same `dataset_id` document rather than creating a new one (at-least-once delivery, effectively-once state).

## Chaos: migrator kill mid-job — 2026-07-09 11:34:20Z — commit d49c6bb
Command: `python -m benchmark.chaos migrator`

| Metric | Value |
| :-- | --: |
| Migrator killed | `035fd0026b99` (held dataset `ds_000318`) |
| Job recovered by another migrator (s) | 31.85 |
| Fencing token on that job (before → after) | 1 → 2 |
| Retried-then-completed jobs (attempts>1) | 3 (was 2) |
| Dead / FAILED jobs | 0 (was 0) |
| Fenced writes total | 0 |

A hard kill never resurrects the worker, so it produces no *stale* write to fence (fenced count comes from the `stall` scenario). What it proves here is recovery: the job's lease expires, a second migrator re-locks it with a higher token and completes it — at-least-once execution with no lost work.

## Chaos: migrator stall (fenced stale write) — 2026-07-09 11:38:31Z — commit d49c6bb
Command: `python -m benchmark.chaos stall`

| Metric | Value |
| :-- | --: |
| Migrator frozen mid-job (docker pause) | `16b5d145d21e` |
| Job dataset | `ds_000956` |
| Fencing token: frozen worker held → stolen by peer `035fd0026b99` | 1 → 2 |
| Job final status | RUNNING |
| Fenced writes across unpause (before → after) | 1 → 2 |

The peer steals the expired lock (token bumped) and completes the migration. The frozen worker then resumes with a *stale* token; its write is rejected by the conditional update (a no-op) and counted in `cloudtier_fenced_writes_total`, so the peer's correct migration is never overwritten.

## End-to-end pipeline throughput — 2026-07-09 11:45:54Z — commit 0b83989

Command: `python -m benchmark.pipeline --window 30` — steady state, default throttled producer.
Measured at the consumer via `cloudtier_events_processed_total` (events that passed through Kafka and were upserted into MongoDB).

| Metric | Value |
| :-- | --: |
| Window (s) | 30.1 |
| Events processed in window | 62761 |
| **End-to-end throughput (events/s)** | **2082.5** |
| Consumer-group lag (start → end) | 748846 → 762085 |

Contrast with single-process *generation* (~78,900 events/s, see top of file): the pipeline number is far lower because each event additionally crosses Kafka (serialize, replicate, fetch) and a MongoDB upsert with a bounded 24-point history push. Lag grew by 13239 during the window: the consumers are the bottleneck (drain capacity < ingest rate), so this rate *is* the pipeline's drain capacity and the backlog grows unboundedly — the backpressure case addressed in §5.2.

## SHAP sanity check on existing synthetic h8 checkpoints — 2026-07-09

Command: `python -m model.shap_analysis --samples 16 --background 8 --nsamples 20 --out-dir model/shap_outputs`

Scope: existing `model/reads_model_h8.pth` and `model/writes_model_h8.pth`.
This is a synthetic-checkpoint explainer sanity run only; it is **not** real-data
predictive evidence. Full report: `model/shap_outputs/shap_report.md`.

| Model | Top SHAP features | Interpretation |
| :-- | :-- | :-- |
| reads h8 | `hour_of_day` (36.4%), `reads_1h` (17.5%), `day_of_week` (13.0%), `bytes_read_1h` (9.3%), `reads_48h` (8.6%) | Recovers time-of-day/week and recent-history signals from the synthetic demand rule. |
| writes h8 | `hour_of_day` (25.9%), `writes_24h` (22.1%), `writes_48h` (20.9%), `writes_96h` (7.3%), `day_of_week` (6.5%) | Recovers time/history signals; longer rolling write windows dominate this checkpoint. |

Negative/limit: `data_temperature_encoded` had 0.0% mean absolute SHAP in this run,
despite temperature influencing synthetic base rates. The existing checkpoint appears
to infer base-rate differences through recent/rolling history instead. Do not claim
the explainer recovered every synthetic generator factor.
