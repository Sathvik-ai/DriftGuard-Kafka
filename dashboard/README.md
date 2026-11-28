# DriftGuard-Kafka M10.1 Live Streamlit Dashboard

Public cloud-compatible Streamlit dashboard consuming **only verified downstream MQTT streams** published by the Mac Kafka pipeline (`app.dashboard_publisher`).

## Architecture

```
Wokwi ESP32 (MQTT + TLS + Auth)
    -> Cloud MQTT Broker (ACL-restricted)
    -> Mac MQTT-Kafka Bridge (`app.mqtt_bridge`)
    -> Kafka (`research.telemetry.{raw,alerts,drift,model_scores,device_status}`)
    -> Mac Dashboard Publisher (`app.dashboard_publisher`)
    -> Cloud MQTT Broker (`<namespace>/dashboard/*`)
    -> Streamlit Community Cloud (`dashboard/streamlit_app.py` via `streamlit_readonly`)
```

## Security Model

- Streamlit Community Cloud connects **strictly with read-only credentials** (`streamlit_readonly` ACL covering `<namespace>/dashboard/#`).
- Kafka ports (`9092`), Zookeeper, and local Mac filesystem paths are **never** exposed to the public internet.
- Remote device profile switching is disabled unless both `allow_profile_control = true` and a valid `command_token` are configured in Streamlit Secrets.
- Static offline demo data only renders when explicitly enabled via **Static Offline Demo Sample**, with a clear `DEMO` badge. When the MQTT broker or Mac backend stops, the UI honestly transitions to `DELAYED` and `OFFLINE`.

## Streamlit Community Cloud Setup

1. Create a free cloud MQTT broker with TLS + authentication + ACLs (e.g., HiveMQ Cloud Serverless or EMQX Cloud) or self-host Mosquitto TLS using [`iot/mosquitto/`](../iot/mosquitto/acl.conf.example).
2. Copy [`../.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example) into **Streamlit Community Cloud -> App Settings -> Secrets**.
3. Deploy from GitHub branch `main` with:
   - **Main file path**: `dashboard/streamlit_app.py`
   - **Python dependencies**: `dashboard/requirements.txt`
4. Run local backend pipeline on your Mac:
   ```bash
   make kafka-up && make topic-create
   make iot-bridge
   make iot-inference
   make dashboard-pub
   ```
5. To preview locally:
   ```bash
   make dashboard
   ```
