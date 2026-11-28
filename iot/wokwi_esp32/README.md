# DriftGuard-Kafka M10-IoT — Online Virtual ESP32 (Wokwi) Setup Guide

This directory contains a **Wokwi-compatible ESP32 Arduino project** that continuously publishes simulated industrial telemetry over MQTT to a reachable relay broker, feeding the local DriftGuard-Kafka pipeline via `app/mqtt_bridge.py`.

> **Important Research & Privacy Notice**
> - All telemetry (`temperature_c`, `vibration_mm_s`, `pressure_bar`) is **simulated virtual data**, NOT physical industrial transducer readings. `vibration_mm_s` simulates ISO 10816 RMS velocity (`0.0–50.0 mm/s`) using the GPIO 34 potentiometer plus bounded noise and profile perturbations.
> - Wokwi's public cloud gateway **cannot** connect directly to `localhost` or your private LAN. Both Wokwi and your local Python bridge must connect **outward** to a reachable MQTT relay broker.
> - **Never commit credentials** (`iot/wokwi_esp32/config.h` and `.env` are git-ignored) and **never publish private data to shared public MQTT topics**. Use a private TLS broker (e.g., HiveMQ Cloud Serverless or EMQX Cloud) with credentials and a unique random topic namespace.

---

## 1. Open and Run the Wokwi ESP32 Project

1. Open [https://wokwi.com/projects/new/esp32](https://wokwi.com/projects/new/esp32) (or use the **Wokwi for VS Code** extension directly on `iot/wokwi_esp32/`).
2. Copy the contents of:
   - `iot/wokwi_esp32/sketch.ino` → `sketch.ino`
   - `iot/wokwi_esp32/diagram.json` → `diagram.json`
   - `iot/wokwi_esp32/libraries.txt` → `libraries.txt`
3. Create `config.h` in Wokwi (copied from `iot/wokwi_esp32/config.example.h`) and fill in your broker host, port, credentials, and unique topic.

---

## 2. Configure the Reachable MQTT Broker and Unique Topic

Example using a free private **HiveMQ Cloud Serverless** (TLS port `8883`) or **EMQX Cloud** cluster:

1. Create a free cluster and add an MQTT username/password.
2. Choose a unique per-device topic namespace, e.g.:
   ```text
   driftguard/<random-token>/esp32-wokwi-01/telemetry
   ```
3. Put those values into:
   - **Wokwi `config.h`** (never committed)
   - **Local `.env`** (copied from `.env.example`, never committed):
     ```env
     MQTT_BROKER_HOST=your-cluster.s1.eu.hivemq.cloud
     MQTT_BROKER_PORT=8883
     MQTT_USE_TLS=true
     MQTT_USERNAME=your_mqtt_user
     MQTT_PASSWORD=your_mqtt_password
     MQTT_TOPIC=driftguard/<random-token>/+/telemetry
     ```

---

## 3. Start the Local Kafka Cluster

From `DriftGuard-Kafka-M1/`:

```bash
make kafka-up
make topic-create
```

---

## 4. Start the Python MQTT-to-Kafka Bridge

In Terminal 1:

```bash
make iot-bridge
# or directly:
.venv/bin/python -m app.mqtt_bridge
```

The bridge connects outward to your MQTT broker, validates incoming JSON, adds `source_type="virtual_iot_wokwi"`, preserves `device_id`, `sequence_number`, `timestamp`, and `machine_id` Kafka partitioning, and forwards events to `research.telemetry.raw`.

---

## 5. Observe Incoming Telemetry

In Terminal 2, start either the existing Kafka consumer, M2 anomaly detector, M3 ADWIN drift monitor, or the live streaming dashboard:

```bash
# Option A: Raw telemetry consumer
make consumer LIMIT=50

# Option B: Existing M2 threshold detector (publishes to research.telemetry.alerts)
.venv/bin/python -m app.detector

# Option C: Existing M3 ADWIN concept drift monitor (publishes to research.telemetry.drift)
.venv/bin/python -m app.drift_monitor

# Option D: Live SSE dashboard at http://127.0.0.1:8765
make dashboard
```

---

## 6. Activate Anomaly and Drift Profiles in Wokwi

While the Wokwi simulation is running, switch profiles using **either** the blue `Profile` pushbutton on the breadboard **or** by typing `1`–`6` (or the profile name) in the Wokwi Serial Monitor:

| Command | Profile | Simulated Behaviour |
|---|---|---|
| `1` | `normal` | Stationary baseline (`~46.5°C`, `~2.1 mm/s`, `~5.0 bar`) |
| `2` | `temperature_spike` | Periodic `+42°C` thermal spikes every 4th reading (triggers M2 alerts) |
| `3` | `vibration_anomaly` | Periodic `+14.5 mm/s` vibration bursts every 4th reading (triggers M2 alerts) |
| `4` | `gradual_drift` | Slow linear ramp across all 3 sensors (triggers M3 ADWIN drift) |
| `5` | `abrupt_drift` | Immediate regime shift (`+18°C`, `+2.8 mm/s`, `+2.4 bar`, triggers M3 ADWIN) |
| `6` | `intermittent_disconnect` | Drops & reconnects MQTT every 5th reading while incrementing `sequence_number` (exercises bridge reconnect & sequence-gap monitoring) |

---

## 7. Verify Alerts and IoT Device Status Reports

- Stop the bridge cleanly with `Ctrl+C` (or run with `--max-messages N` / `--duration-s T`).
- Inspect the generated IoT Device Status Report under `reports/m10_iot/<run_id>/`:
  - `iot_status.json` — machine-readable counters (`connection_state`, `last_message_timestamp`, `messages_received`, `messages_forwarded`, `invalid_messages`, `sequence_gaps`, `reconnection_count`, `ingestion_latency_ms`, and per-device health).
  - `iot_status.md` — human-readable summary table.
- Run the automated live end-to-end MQTT-to-Kafka-to-Detector smoke test locally at any time:
  ```bash
  make m10-iot-smoke
  ```
