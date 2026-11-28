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

## 10. Milestone 2 (M2): Synthetic Anomalies & Baseline Detection

M2 extends the pipeline with three reproducible telemetry scenarios (`normal`, `anomalies` with spikes & stuck sensors, and `drift` with gradual baseline shift) and a static threshold-based detection consumer (`app/detector.py`) that reads from `research.telemetry.raw` and publishes alerts to `research.telemetry.alerts`.

- **Scenario & ground-truth semantics**:
  - `normal`: Stationary baseline operation (`ground_truth_label="normal"`, `is_anomaly=false`).
  - `anomalies`: Independent `spike` and `stuck` fault injections (`is_anomaly=true`) mixed with `normal` readings.
  - `drift`: Gradual shift in normal operating baselines (`ground_truth_label="normal"`, `is_anomaly=false`). Alerts triggered by static thresholds during pure drift are counted as **false positives (`FP`)**, quantifying baseline detector degradation under concept drift ahead of M3.
- **Label separation**: Ground-truth labels are written to a separate `ground_truth.jsonl` file by the producer. Kafka payloads keep `label="normal"` and `SensorReading` strips `label` so the detector never sees ground truth.
- **Run M2 end-to-end**:
  ```bash
  make m2-demo
  ```
- **Evaluation report**: Saved under `reports/m2/<run_id>/` (`report.md`, `report.json`, `evaluations.csv`) with confusion matrix (`TP/FP/TN/FN`), precision, recall, F1 (`2TP/(2TP+FP+FN)` whenever `2TP+FP+FN > 0`), false-positive rate, anomaly breakdown, throughput, and latency.

---

## 11. Milestone 3 (M3): ADWIN Concept Drift Monitoring & Interactive Dashboard

M3 adds online concept drift detection using **River ADWIN** (`river==0.27.0`) in `app/drift_monitor.py`:

- **Per-(machine, feature) isolation**: Maintains independent `ADWIN` monitors for `temperature_c`, `vibration_mm_s`, and `pressure_bar` scoped by `machine_id`, consuming unlabelled telemetry from `research.telemetry.raw` and publishing structured drift events to `research.telemetry.drift`.
- **Scenarios**: Evaluates `normal` (stationary), `drift` (gradual baseline ramp), and `abrupt_drift` (step baseline shift). All drifted observations retain `normal` ground-truth labels, and static M2 anomaly thresholds remain unchanged (no adaptive threshold updates yet).
- **Run M3 end-to-end & multi-seed robustness**:
  ```bash
  make m3-demo         # single-seed run + interactive HTML dashboard
  make m3-robustness   # 10-seed Kafka robustness sweep (30 runs = 15,000 events)
  ```
- **Reports & Streaming Dashboard**:
  - Single-run bundle under `reports/m3/<run_id>/` (`report.md`, `report.json`, `channel_evaluations.csv`, and `dashboard.html`), distinguishing `first_episode_detections` from `repeated_episode_alerts` during continuous gradual ramps.
  - Multi-seed robustness bundle under `reports/m3/robustness_<run_id>/` (`robustness_report.md`, `robustness_report.json`, `seed_runs.csv`) reporting `mean ± std [min, max]` across 10 seeds.
  - Live real-time streaming dashboard (`http://127.0.0.1:8765`) streaming over Apache Kafka via Server-Sent Events with animated threshold alerts and ADWIN drift cuts:
    ```bash
    make dashboard
    ```

---

## 12. Milestone 4 (M4): Comparative ML Benchmarking

M4 benchmarks three unsupervised anomaly detectors (`app/m4_models.py`, `app/m4_benchmark.py`, `app/datasets.py`) across **SKAB** and **TSB-AD-M** while keeping ADWIN completely independent (no adaptive retraining yet):

- **Models Compared**:
  1. `static_threshold` (`StaticThresholdBaseline`): Multivariate static operating bounds + consecutive stuck-sensor check preserving the M2 baseline design.
  2. `isolation_forest` (`IsolationForestDetector`): Scikit-learn `IsolationForest` with `StandardScaler` fit strictly on normal training observations.
  3. `half_space_trees` (`HalfSpaceTreesDetector`): River online `HalfSpaceTrees` with `MinMaxScaler`, guaranteed reference-window warm-up on normal training data, and strict `score_one` before `learn_one` evaluation.
- **Datasets & Leakage-Free Protocol**:
  - **SKAB (`waico/SKAB` v0.9)**: Full ingestion of all 35 semicolon-delimited CSVs (`valve1`, `valve2`, `other`, `anomaly-free`), preserving `anomaly` and `changepoint` labels separately and filtering chronological training prefixes strictly to `anomaly == 0`.
  - **TSB-AD-M (`TheDatumOrg/TSB-AD`)**: Official `_tr_<train_end>` chronological split adapter filtered strictly to normal training observations (`Label == 0`), calibrating thresholds on held-out chronological normal validation slices (`calibration_quantile=0.99`), and evaluating official `VUS-PR` via `TSB_AD.evaluation.metrics.get_metrics` (`tsb-ad==1.5`).
  - External datasets live in git-ignored `data/external/` and are never committed.
- **Run M4 Benchmark**:
  ```bash
  # Domain-stratified across all 17 TSB-AD-M domains (2 per domain) + full SKAB (35 CSVs):
  make m4-benchmark
  # Or evaluate all 180 official TSB-AD-M-Eva series + full SKAB:
  make m4-benchmark TSB_MAX_PER_DOMAIN=0
  ```
- **Reproducible Outputs (`reports/m4/<run_id>/`)**:
  - `comparison_summary.csv`, `per_experiment_metrics.csv`, `benchmark_report.json`, and `benchmark_report.md` (recording Precision, Recall, F1, FPR, PR-AUC, VUS-PR, inference latency, throughput, CPU time, peak memory, SHA-256 dataset provenance, model parameters, and seeds).

---

## 13. Milestone 5 (M5): Sequence-to-Sequence LSTM Autoencoder

M5 adds deep temporal reconstruction-based anomaly detection in `app/m5_lstm_autoencoder.py` and `app/m5_benchmark.py` (`torch==2.15.0`, configuration in `configs/m5_lstm_ae.json`):

- **Architecture**:
  - **Encoder**: Multi-layer `nn.LSTM` + bottleneck `nn.Linear(hidden_dim, latent_dim)` + `Tanh` compressing temporal windows `(B, W, D)` into latent temporal vectors `z`.
  - **Decoder**: Latent bridge + multi-layer `nn.LSTM` + linear output projection reconstructing `(B, W, D)`.
  - **Dual Series Modes**:
    - **SKAB (`multivariate`)**: Jointly reconstructs coupled 8-sensor physical testbed windows `(B, W, 8)`.
    - **TSB (`independent_univariate`)**: Models each channel strictly as an independent 1D univariate series `(B, W, 1)` without falsely representing heterogeneous/command-indicator TSB columns as coupled multivariate vectors.
  - **Leakage-Free Protocol**: `TrainingOnlyScaler` fit strictly on `x_train_normal`; boundary-preserving causal windows; early stopping on normal training windows; threshold calibrated strictly on `x_val_normal` reconstruction error; outputs both per-feature `(N, D)` and overall `(N,)` anomaly scores.
  - **Hardware Support**: Detects Apple Silicon and PyTorch MPS availability automatically with clean CPU fallback.
- **Run M5 Experiment**:
  ```bash
  make m5-demo
  # Or customize config / domain coverage:
  .venv/bin/python -m app.m5_lstm_autoencoder --config configs/m5_lstm_ae.json --tsb-max-per-domain 1
  ```
- **Reproducible Outputs (`reports/m5/<run_id>/`)**:
  - `m5_report.json`, `m5_report.md`, `comparison_summary.csv`, and `per_experiment_metrics.csv` (comparing `lstm_autoencoder` against `static_threshold`, `isolation_forest`, and `half_space_trees` on Precision, Recall, F1, FPR, PR-AUC, VUS-PR, episode detection delay, training duration, inference latency, throughput, and peak memory).

---

## 14. Milestone 6 (M6): Temporal Convolutional Network (TCN) Autoencoder

M6 implements a causal dilated residual TCN sequence autoencoder in `app/m6_tcn_autoencoder.py` and `app/m6_benchmark.py` (`configs/m6_tcn_ae.json`):

- **Architecture**:
  - **Causal Convolutions (`CausalConv1d`)**: Left-only temporal padding `(kernel_size - 1) * dilation` guarantees zero future lookahead (`∂y_t / ∂x_s == 0` for `s > t`).
  - **Dilated Residual Blocks (`TCNResidualBlock`)**: Exponential dilations `[1, 2, 4, 8]` with kernel size `3` yield a 61-step causal receptive field capturing both short-term and long-term temporal patterns, stabilized by GELU activations, dropout, and `1x1` residual shortcuts.
  - **Undercomplete Temporal Bottleneck**: Compresses multi-scale encoder representations into a low-dimensional latent vector `z` before causal TCN decoding, preventing trivial identity copying on both multivariate SKAB (`input_dim=8`) and independent univariate TSB (`input_dim=1`) series.
- **Run M6 Experiment**:
  ```bash
  make m6-demo
  ```
- **Reproducible Outputs (`reports/m6/<run_id>/`)**:
  - `m6_report.json`, `m6_report.md`, `comparison_summary.csv`, and `per_experiment_metrics.csv` comparing `tcn_autoencoder` against `lstm_autoencoder`, `static_threshold`, `isolation_forest`, and `half_space_trees` while preserving all existing M1–M5 outputs untouched.

---

## 15. Milestone 7 (M7): TranAD Two-Phase Self-Conditioning Transformer

M7 implements **TranAD** (*Tuli et al., PVLDB 2022*, `https://github.com/imperial-qore/TranAD`) in `app/m7_tranad.py` and `app/m7_benchmark.py` (`configs/m7_tranad.json`):

- **Architecture**:
  - **Concatenated Conditioning (`2 * D`)**: Projects context windows `W` concatenated with focus scores `c` (`[W || c]`) through sinusoidal positional encoding (`sin + cos`) into a shared Transformer encoder (`nhead=feats`, `LeakyReLU` feedforward).
  - **Phase 1 (`Decoder 1`)**: Evaluates zero focus `c_1 = 0` against target `tgt = W[-1:]` to produce initial reconstruction `x_1` (`Sigmoid` projection) and focus signal `c_2 = (x_1 - W)^2`.
  - **Phase 2 (`Decoder 2`)**: Re-encodes `[W || c_2]` and decodes refined reconstruction `x_2`, trained with epoch-evolving loss `(1/n) * MSE(x_1, tgt) + (1 - 1/n) * MSE(x_2, tgt)` and scored via `0.5 * (x_1 - tgt)^2 + 0.5 * (x_2 - tgt)^2` with per-feature attribution.
  - **Documented Deviations from PVLDB 2022**: Uses normal-only `[0, 1]` `TrainingOnlyMinMaxScaler` + held-out `x_val_normal` quantile calibration (`0.99`) instead of POT on test distributions, `AdamW` + cosine LR instead of `StepLR`, and `independent_univariate` modeling on TSB series.
- **Run M7 Experiment**:
  ```bash
  make m7-demo
  ```
- **Reproducible Outputs (`reports/m7/<run_id>/`)**:
  - `m7_report.json`, `m7_report.md`, `comparison_summary.csv`, and `per_experiment_metrics.csv` comparing `tranad` against M4–M6 baselines while preserving M1–M6 outputs untouched.

---

## 16. Milestone 8 (M8): Time-Series Foundation Model Anomaly Detection

M8 adds zero-shot probabilistic foundation model anomaly detection using **Amazon Chronos-2** (`app/m8_chronos2.py`) and **Google TimesFM 3.0** (`app/m8_timesfm.py`) over a shared causal forecasting base (`app/m8_tsfm_base.py`, `app/m8_benchmark.py`, `configs/m8_tsfm.json`):

- **Foundation Models & Licensing**:
  1. **Amazon Chronos-2** (`amazon/chronos-2`, ~120M params, ~456 MiB safetensors, `chronos-forecasting==2.3.2`) — **`Apache-2.0`** license (code and weights).
  2. **Google TimesFM 3.0** (`google/timesfm-3.0-pytorch`, ~331M params, ~1.23 GiB safetensors, `timesfm==3.0.2`) — code is `Apache-2.0`; pretrained weights are governed by **`timesfm-non-commercial-license-v1.0`** (non-commercial research only).
- **Zero-Shot Causal Protocol & Dual Scoring**:
  - No fine-tuning on benchmark series; `TrainingOnlyScaler` (`sigma_d`) is fit strictly on `x_train_normal`.
  - Blocked causal context windows (`<= t_0 - 1` only; zero future test leakage into context), native multivariate forecasting on SKAB (`8` channels), and independent univariate forecasting on TSB active channels.
  - Evaluates **both** normalized forecast residuals (`chronos2_zero_shot`, `timesfm3_zero_shot`) and `[0.1, 0.9]` prediction interval violations (`chronos2_interval`, `timesfm3_interval`), calibrated strictly on `x_val_normal`.
- **Optional Foundation Model Setup & Execution**:
  ```bash
  uv pip install --python .venv/bin/python -r requirements-m8.txt
  make m8-demo
  ```
- **Reproducible Outputs (`reports/m8/<run_id>/`)**:
  - `m8_report.json`, `m8_report.md`, `comparison_summary.csv`, and `per_experiment_metrics.csv` comparing all 10 detector variants (M4–M7 + Chronos-2 + TimesFM 3.0) across anomaly detection metrics, forecasting error (`MAE`, `RMSE`, `Normalized MAE`, `80% Interval Coverage`), latency, throughput, and memory.

---

## 17. Milestone 9 (M9): Comprehensive Multi-Seed Comparative Evaluation

M9 runs a unified, deterministic, multi-seed evaluation across all five deep-learning / foundation models (`lstm_autoencoder`, `tcn_autoencoder`, `tranad`, `chronos2_*`, `timesfm3_*`) and all three M4 classical ML baselines (`static_threshold`, `isolation_forest`, `half_space_trees`) in `app/m9_evaluation.py` and `app/m9_charts.py` (`configs/m9_eval.json`):

- **Multi-Seed Reproducibility & Split Verification**: Runs trained stochastic models across multiple seeds (`42, 1337, 2026`, reporting `mean ± std`) and deterministic/zero-shot models once on identical SHA-256-verified chronological splits (`SKAB` multivariate vs. `TSB-AD-M` independent univariate).
- **Error Analysis & Resource-Efficiency SLA Flagging**: Decomposes false positives (`anomaly_free`, `pre_fault_lead`, `post_fault_recovery`, `isolated_background`) and false negatives (`missed_episodes_short/long`, `onset_delay`, `adaptation_dropoff`), separates forecasting accuracy from anomaly detection accuracy, flags high-F1 models that violate streaming latency/memory SLAs, and identifies the best model per scenario without claiming an unsupported universal winner.
- **Run M9 Evaluation**:
  ```bash
  make m9-demo
  ```
- **Reproducible Outputs (`reports/m9/<run_id>/`)**:
  - `m9_report.md`, `m9_report.json`, `unified_comparison.csv`, `per_seed_summary.csv`, `per_experiment_metrics.csv`, `error_analysis.csv`, and pure-SVG performance charts in `charts/`. All historical M1–M8 reports remain untouched.

---

## 18. Milestone 9.1 (M9.1): Research Methodology Audit & Corrected Benchmark

M9.1 audits and corrects the M9 evaluation methodology while preserving all historical M1–M9 artefacts (`app/m9_1_tracks.py`, `app/m9_1_stats.py`, `app/m9_1_profiler.py`, `app/m9_1_charts.py`, `app/m9_1_benchmark.py`, `configs/m9_1_eval.json`):

- **Fair Dual-Track Input Protocol**: Evaluates every model family under both **Track A (`multivariate`)** on identical multivariate `(N, D)` windows and **Track B (`independent_channel`)** through `IndependentChannelWrapper` (identical active-channel selection, validation-only robust score normalization, mean aggregation, and validation-only calibration). Explicitly labels `TSB-AD-M` as a multivariate dataset (`2–248` features).
- **Expanded Coverage & Chunked Long-Series TSFMs**: Scales TSB-AD-M to `max_per_domain=3` (`41` series across all `17` domains) and adds adaptive chunked long-horizon causal forecasting (`H=256` on series `>30,000` rows) plus deterministic 32-variate grouping for high-dimensional TSFM runs.
- **Isolated Subprocess Resource Profiling (`app/m9_1_profiler.py`)**: Measures fresh-interpreter baseline RSS, model-load RSS delta, inference peak RSS delta, Apple Silicon MPS memory, and offline-batched latency/throughput per model in separate subprocesses.
- **Validation-Only False-Positive Calibration & Bootstrap Uncertainty**: Evaluates 5 validation-only calibration strategies (`empirical_q99`, `empirical_q995`, `mad_z35`, `iqr_k30`, `evt_pot_q99`) on SKAB (`FA / 1k normal` vs fault F1) and computes 2,000-resample 95% bootstrap CIs and paired significance tests across 5 SKAB seeds and 3 TSB seeds.
- **Run M9.1 Audited Benchmark**:
  ```bash
  make m9-1-demo
  ```
- **Reproducible Outputs (`reports/m9_1/<run_id>/`)**: `m9_1_report.md`, `m9_1_report.json`, `protocol_comparison.md`, 7 CSVs, and 5 pure-SVG charts.

---

## 19. Milestone 10-IoT (M10-IoT): Online Virtual ESP32 (Wokwi) & MQTT-to-Kafka Bridge

M10-IoT integrates a continuously running **Wokwi virtual ESP32** (`iot/wokwi_esp32/`) into the existing DriftGuard-Kafka pipeline via a Python MQTT-to-Kafka bridge (`app/mqtt_bridge.py`, `paho-mqtt==2.1.0`):

- **Architecture**: `Wokwi ESP32` → reachable MQTT relay broker → `app/mqtt_bridge.py` → existing `research.telemetry.raw` Kafka topic (keyed by `machine_id`, tagged with `source_type="virtual_iot_wokwi"`, `device_id`, `sequence_number`) → existing `app/consumer.py`, `app/detector.py`, `app/drift_monitor.py`, and `app/live_dashboard.py`.
- **6 Selectable Virtual Profiles**: `normal`, `temperature_spike`, `vibration_anomaly`, `gradual_drift`, `abrupt_drift`, and `intermittent_disconnect`. All readings are explicitly documented as simulated virtual measurements.
- **IoT Status Monitoring**: Generates `reports/m10_iot/<run_id>/iot_status.{json,md}` tracking connection state, last message timestamp, received/forwarded/invalid counts, sequence gaps, reconnections, and ingestion latency.
- **Setup & Live Smoke Test**: See `iot/wokwi_esp32/README.md` for full Wokwi + cloud MQTT relay instructions, or run the local end-to-end MQTT-to-Kafka-to-Detector smoke test:
  ```bash
  make m10-iot-smoke
  ```

---

## 20. Milestone 10.1 (M10.1): Full Online IoT Pipeline, Live 3-Channel ML/DL/TSFM Inference & Streamlit Cloud Observatory

M10.1 extends M10-IoT with authenticated MQTT transport, Apple Silicon MPS streaming inference across all 9 methods, and a public-cloud-ready Streamlit observatory:

- **Authenticated Wokwi Firmware & MQTT Health**: `iot/wokwi_esp32/sketch.ino` publishes heartbeats/LWT (`<namespace>/<device_id>/heartbeat`) and verifies `command_token` on `<namespace>/<device_id>/command` for remote `set_profile` switching. `iot/mosquitto/` provides role-separated ACL templates (`wokwi_device`, `mac_backend`, `streamlit_readonly`).
- **3-Channel Streaming ML/DL/TSFM Inference (`app/streaming_inference.py`)**: Evaluates `static_threshold`, `isolation_forest`, `half_space_trees`, `lstm_autoencoder`, `tcn_autoencoder`, `tranad`, `chronos2_zero_shot`, `timesfm3_zero_shot`, and `river_adwin` on the 3-channel IoT stream (`temperature_c`, `vibration_mm_s`, `pressure_bar`), rejecting 8-feature SKAB or 1-feature TSB checkpoints (`SchemaMismatchError`), training/calibrating strictly on normal splits, and reporting separated `process_rss_mib` vs. Apple Silicon `mps_allocated_mib` / `mps_driver_mib`.
- **Kafka-Confirmed Dashboard Relay & Streamlit UI (`app/dashboard_publisher.py`, `dashboard/`)**: Re-publishes confirmed Kafka events onto `<namespace>/dashboard/*` (including retained `snapshot`) for read-only consumption in `dashboard/streamlit_app.py` (7 dark-theme tabs, honest `LIVE`/`DELAYED`/`OFFLINE`/`DEMO` freshness states, zero public Kafka exposure). See [`dashboard/README.md`](dashboard/README.md) for Streamlit Community Cloud deployment steps.
- **Run Continuous 15-Minute Online Multi-Phase Protocol**:
  ```bash
  make m10_1-online-test
  ```
- **Reproducible Outputs (`reports/m10_1/<run_id>/`)**: `summary_report.{json,md}`, `events.jsonl`, `scenario_truth.jsonl`, `model_metrics.csv`, `drift_events.csv`, `alerts.csv`, `device_status.json`, and `hardware_memory_report.json`.

---

## 11. Automated Streaming Reports

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
