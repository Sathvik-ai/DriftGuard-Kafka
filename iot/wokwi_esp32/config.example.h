// Copy this file to `config.h` (which is git-ignored) and fill in your broker settings.
// NEVER commit real MQTT credentials or private topics to Git.
#pragma once

// Wokwi virtual Wi-Fi (works inside Wokwi simulator without a password)
#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASSWORD ""

// Reachable MQTT relay broker (e.g., HiveMQ Cloud Serverless TLS on 8883, EMQX Cloud,
// or your own cloud Mosquitto broker). Wokwi cannot reach localhost/LAN directly.
#define MQTT_BROKER_HOST "YOUR_BROKER_HOST_HERE"
#define MQTT_BROKER_PORT 8883
#define MQTT_USE_TLS 1

// Leave empty ("") ONLY when testing against an unauthenticated non-private broker;
// always use credentials + TLS for private/cloud brokers.
#define MQTT_USERNAME ""
#define MQTT_PASSWORD ""

// Unique per-device identity and MQTT topic (replace <your-unique-namespace> with a random token)
#define DEVICE_ID "esp32-wokwi-01"
#define MACHINE_ID "machine_01"
#define MQTT_TOPIC "driftguard/<your-unique-namespace>/esp32-wokwi-01/telemetry"

// Telemetry publishing cadence in milliseconds (1000..5000 ms)
#define PUBLISH_INTERVAL_MS 2000UL

// Realistic simulated baseline operating ranges (NOT physical industrial sensors)
#define BASE_TEMP_C 46.5f
#define BASE_VIBRATION_MM_S 2.10f
#define BASE_PRESSURE_BAR 5.00f
