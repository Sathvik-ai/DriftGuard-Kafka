# DriftGuard-Kafka — Milestone 1 (M1)

**A working Python → Apache Kafka → Python telemetry streaming demonstration on macOS.**

M1 establishes the foundational event-streaming pipeline for the **DriftGuard-Kafka** research project: a simulated industrial machine telemetry producer, a single-node Apache Kafka broker running in KRaft mode (no ZooKeeper), and an independent Python consumer that validates every event and verifies end-to-end delivery.

---

## 1. Architecture Overview

```text
+--------------------+      key = machine_id       +-----------------------------+
|  Python Producer   | --------------------------> |  Apache Kafka (KRaft)       |
|  app/producer.py   |   JSON (UTF-8) on port 9092 |  Topic:                     |
|  + app/simulator   |                             |    research.telemetry.raw   |
+--------------------+                             |  Partitions: 0, 1, 2        |
                                                   |  Replication Factor: 1      |
+--------------------+      poll() + store offset  +-----------------------------+
|  Python Consumer   | <--------------------------               |
|  app/consumer.py   |   group: driftguard-m1-*                  |
|  + app/schemas.py  | <-----------------------------------------+
+--------------------+
```

### How Kafka Concepts Fit Together
- **Producer (`app/producer.py`)**: Generates simulated machine events (`app/simulator.py`) and publishes them to Kafka using `machine_id` as the message key. Using `machine_id` as the key guarantees that all events from the same machine land on the same partition in order.
- **Broker (`docker-compose.yml`)**: A single Kafka server running in **KRaft mode** (acting as both `broker` and `controller`, eliminating ZooKeeper). It stores events on disk inside a persistent Docker volume (`driftguard-m1-kafka-data`).
- **Topic & Partitions (`research.telemetry.raw`)**: A named log split into **3 partitions** (`0`, `1`, `2`). Partitions allow parallel consumption in future milestones.
- **Consumer & Consumer Group (`app/consumer.py`)**: Subscribes to the topic under a `group.id`, polls messages, parses and validates JSON against `TelemetryEvent` (`app/schemas.py`), and tracks unique/duplicate `event_id`s and malformed payloads.
- **Offsets**: Each message in a partition has a sequential offset (`0, 1, 2, ...`). The consumer disables `enable.auto.offset.store` and only calls `store_offsets()` *after* validating a message—providing **at-least-once** semantics (not exactly-once).

### Kafka Configuration Explained (For Beginners)
- `KAFKA_LISTENERS` vs `KAFKA_ADVERTISED_LISTENERS`:
  - `INTERNAL://kafka:19092`: Used inside the Docker container by Kafka CLI tools.
  - `EXTERNAL://localhost:9092`: Published to macOS (`9092:9092`) so Python apps connect to `localhost:9092`.
  - `CONTROLLER://kafka:9093`: Internal KRaft metadata traffic.
- `KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1` (and transaction/share state topics = `1`): Kafka defaults to replication factor `3`, which fails on a 1-broker setup unless overridden to `1`.
- `KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"`: Forces explicit topic creation via `scripts/create_topic.sh` so typos fail loudly instead of silently creating a 1-partition topic.
- Producer `acks: "all"` + `enable.idempotence: True`: Prevents retries from duplicating messages inside the broker log, while delivery callbacks (`DeliveryTracker`) confirm actual broker receipt rather than just local queuing.

---

## 2. Prerequisites for macOS

- **macOS** on Apple Silicon (`arm64`) or Intel (`x86_64`)
- **Docker Desktop for Mac**
- **Python 3.10+** (tested with Python 3.11.15) and `uv` or `venv` + `pip`
- **GNU Make** (included with Xcode Command Line Tools: `xcode-select --install`)

---

## 3. Docker Desktop Installation

1. Download **Docker Desktop for Mac** (Apple Silicon or Intel) from:
   https://docs.docker.com/desktop/setup/install/mac-install/
2. Drag `Docker.app` into `/Applications` and launch it (`open -a Docker`).
3. Verify Docker is running:
   ```bash
   docker --version
   docker compose version
   docker info
   ```

---

## 4. Python Environment Setup

From inside `DriftGuard-Kafka-M1/`:

```bash
make setup
```

This creates `.venv`, installs pinned dependencies from `requirements-dev.txt` (including `confluent-kafka==2.16.2`, `pytest`, `ruff`, `mypy`), and copies `.env.example` to `.env` if `.env` does not exist.

---

## 5. Kafka Startup & Health Check

Start the single-node KRaft broker and wait until its health check reports `healthy`:

```bash
make kafka-up
```

Verify broker health and list topics manually at any time:

```bash
make kafka-verify
```

---

## 6. Topic Creation

Create `research.telemetry.raw` with 3 partitions and replication factor 1:

```bash
make topic-create
```

To delete and recreate the topic cleanly:

```bash
./scripts/create_topic.sh --recreate
```

---

## 7. Running Producer and Consumer Manually

Start the consumer in one terminal (reads up to 1,000 events, or omit `LIMIT` to run until `Ctrl+C`):

```bash
make consumer LIMIT=1000
# or directly:
.venv/bin/python -m app.consumer --limit 1000
```

Start the producer in another terminal:

```bash
make producer COUNT=1000 INTERVAL_MS=2
# or directly:
.venv/bin/python -m app.producer --count 1000 --interval-ms 2
```

---

## 8. Running the Complete End-to-End Demo

Run the full automated workflow (starts Kafka → checks health → recreates topic → starts consumer → runs producer for 1,000 events → verifies producer vs consumer statistics):

```bash
make demo
```

---

## 9. Verified Terminal Outputs

Actual output from `make demo` (1,000 events, 2 ms interval):

```text
[04:41:27] Container health: healthy
[04:41:28] Broker reachable: kafka:19092
[04:41:32] Topic 'research.telemetry.raw' ready with 3 partitions.
2026-11-24T04:41:35Z INFO __main__ partitions assigned partitions="[0, 1, 2]"
2026-11-24T04:41:39Z INFO __main__ producer delivery summary requested=1000 attempted=1000 delivered=1000 failed=0 not_confirmed=0 per_partition="{'0': 591, '1': 213, '2': 196}" failure_reasons={} duration_s=2.587 interrupted=False succeeded=True
2026-11-24T04:41:39Z INFO __main__ consumer receipt summary received=1000 valid=1000 malformed=0 duplicates=0 unique_event_ids=1000 kafka_errors=0 per_partition="{'0': 591, '1': 213, '2': 196}" stop_reason=limit_reached duration_s=6.619

End-to-end verification
------------------------------------------------------------------------
[ok  ] producer: delivered == expected            expected=1000 actual=1000
[ok  ] producer: delivery failures                expected=0 actual=0
[ok  ] producer: unconfirmed after flush          expected=0 actual=0
[ok  ] consumer: received == delivered            expected=1000 actual=1000
[ok  ] consumer: schema-valid == received         expected=1000 actual=1000
[ok  ] consumer: malformed records                expected=0 actual=0
[ok  ] consumer: duplicate event IDs              expected=0 actual=0
[ok  ] consumer: unique event IDs == delivered    expected=1000 actual=1000
[ok  ] consumer: Kafka errors                     expected=0 actual=0
[ok  ] per-partition counts match                 expected={'0': 591, '1': 213, '2': 196} actual={'0': 591, '1': 213, '2': 196}
------------------------------------------------------------------------
RESULT: PASS
```

Run tests, linting, formatting, and static type checks:

```bash
make check
```

---

## 10. Automated Streaming Reports

Every successful `make demo` run generates a structured report bundle under `reports/runs/<run_id>/` (ignored by git so experiment outputs are never committed):

- `report.md` — human-readable summary, latency percentiles, per-partition distribution, and measurement boundaries.
- `report.json` — machine-readable metrics and reproducibility metadata.
- `events.csv` — per-event correlation between producer delivery and consumer receipt.
- `latency.csv` — per-event end-to-end latency observations in milliseconds (`(consumed_monotonic_ns - enqueued_monotonic_ns) / 1e6`).

Example metrics from verified run `20261124T045344Z` (1,000 events, 2 ms interval):

```text
Produced: 1000 delivered / 0 failed / 0 unconfirmed (382.26 events/s, 2.616 s)
Consumed: 1000 received / 1000 valid / 0 malformed / 0 duplicates (194.21 events/s, 5.149 s)
Consumer Lag (Total): 0 (partitions: 0=0, 1=0, 2=0)
Latency (ms): min=0.427 | mean=57.563 | p50=5.948 | p95=396.091 | p99=497.294 | max=520.281
```

---

## 11. GitHub Actions CI/CD

Workflows live in `.github/workflows/`:

- **CI (`.github/workflows/ci.yml`)** — runs on pushes to `main`/`master` and pull requests with `contents: read`:
  1. `lint-and-test`: Ruff lint & format checks, strict `mypy`, `pytest` with `--cov-fail-under=85`, and `docker compose config -q`.
  2. `kafka-integration`: Starts a real temporary `apache/kafka:4.3.2` broker, runs `./scripts/run_demo.sh 1000`, validates all 4 report files, uploads `runs/` and `reports/runs/` as workflow artifacts, and tears down the broker.
- **Release (`.github/workflows/release.yml`)** — runs on `v*` tags (`contents: write`):
  - Re-runs all lint, typecheck, unit test, and Compose checks, packages a clean source archive (`driftguard-kafka-<tag>.tar.gz` + `.sha256`), and publishes a GitHub Release with generated notes.
  - Does not publish to GHCR because M1 uses upstream `apache/kafka:4.3.2` with no custom `Dockerfile`, and never auto-deploys to production.

---

## 12. Reproducibility

Verified local baseline for `20261124T045344Z`:

- **Hardware / OS**: Apple Silicon (`arm64`), macOS `26.7.3`
- **Docker**: Docker Engine `29.4.3`, Docker Compose `v5.1.3`, image `apache/kafka:4.3.2` (KRaft mode)
- **Python**: `3.11.15`, `confluent-kafka==2.16.2`
- **Configuration**: `KAFKA_TOPIC=research.telemetry.raw` (3 partitions, replication factor 1), `SIMULATOR_MACHINE_COUNT=5`, `DEMO_INTERVAL_MS=2`, `acks=all`, `enable.idempotence=True`
- **Verified Run ID**: `20261124T045344Z` (`reports/runs/20261124T045344Z/report.json`)

---

## 13. Troubleshooting

- **Docker daemon is not running**: Open Docker Desktop (`open -a Docker`) and wait for the whale icon in the menu bar to become steady.
- **Port `9092` already in use**: Stop any other local Kafka/Redpanda container (`docker ps`) before running `make kafka-up`.
- **Consumer receives `0` messages when run manually**: If a previous run with the same `KAFKA_CONSUMER_GROUP` already committed offsets, new runs resume from the end. Either pass a new `--group` or recreate the topic (`./scripts/create_topic.sh --recreate`).
- **Stuck or corrupted broker state**: Reset the Docker volume with `make kafka-clean`, then run `make kafka-up`.

---

## 14. Current Limitations (By Design for M1)

- **No fault tolerance**: Single broker with `replication_factor=1`. If the container stops, the topic is unavailable.
- **At-least-once delivery**: Not exactly-once processing. A consumer crash before offset commit can redeliver messages (caught by duplicate `event_id` tracking within a run).
- **In-memory duplicate tracking**: `seen_event_ids` lives in memory for the current consumer run only.
- **No anomaly detection or drift detection**: `label="normal"` is a placeholder field for future milestones; no ML, Flink, databases, or dashboards are included in M1.

---

## 15. Shut Down and Clean Up

Stop the Kafka broker while keeping stored topic data:

```bash
make kafka-down
```

Stop the broker **and** delete the persistent Kafka volume:

```bash
make kafka-clean
```

Remove Python caches and demo run logs (`runs/`):

```bash
make clean
```
