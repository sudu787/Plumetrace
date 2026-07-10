"""Virtual MQTT sensor stations for local PlumeTrace development."""

import asyncio
import json
import logging
import os
import random
from datetime import UTC, datetime
from typing import TypedDict

from gmqtt import Client as MQTTClient
from gmqtt.mqtt.constants import MQTTv311

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
PUBLISH_INTERVAL_SECONDS = 2.0
SO2_SAFE_THRESHOLD_PPB = 75.0


class SensorStation(TypedDict):
    """Static virtual station metadata."""

    sensor_id: str
    latitude: float
    longitude: float


SENSOR_STATIONS: tuple[SensorStation, ...] = (
    {"sensor_id": "industrial_north", "latitude": 40.7180, "longitude": -74.0060},
    {"sensor_id": "residential_east", "latitude": 40.7140, "longitude": -73.9980},
    {"sensor_id": "park_south", "latitude": 40.7080, "longitude": -74.0040},
    {"sensor_id": "river_west", "latitude": 40.7120, "longitude": -74.0120},
)


def build_reading(sensor: SensorStation, sequence: int) -> dict[str, float | str]:
    """Create one realistic telemetry payload with deterministic spike cadence."""
    spike_window = sequence % 30 in {0, 1, 2}
    downwind_bias = 1.0
    if sensor["sensor_id"] == "residential_east":
        downwind_bias = 1.3
    elif sensor["sensor_id"] == "river_west":
        downwind_bias = 0.4

    pm25 = random.uniform(8.0, 22.0) * downwind_bias
    so2 = random.uniform(4.0, 24.0) * downwind_bias
    if spike_window and sensor["sensor_id"] == "park_south":
        so2 = 185.0
        pm25 = 65.0

    return {
        "sensor_id": sensor["sensor_id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "latitude": round(sensor["latitude"] + random.uniform(-0.0001, 0.0001), 6),
        "longitude": round(sensor["longitude"] + random.uniform(-0.0001, 0.0001), 6),
        "pm25": round(pm25, 2),
        "so2": round(so2, 2),
        "wind_speed": round(random.uniform(7.0, 8.5), 2),
        "wind_direction": 290.0,
    }


async def publish_station_reading(
    client: MQTTClient,
    sensor: SensorStation,
    sequence: int,
) -> None:
    """Publish one station reading to the MQTT broker."""
    reading = build_reading(sensor, sequence)
    topic = f"city/airquality/{sensor['sensor_id']}"
    client.publish(topic, json.dumps(reading), qos=1)

    if float(reading["so2"]) > SO2_SAFE_THRESHOLD_PPB:
        logger.warning(
            "Toxic SO2 spike published. sensor_id=%s so2=%.2f ppb",
            reading["sensor_id"],
            reading["so2"],
        )
    else:
        logger.info(
            "Published telemetry. sensor_id=%s pm25=%.2f so2=%.2f",
            reading["sensor_id"],
            reading["pm25"],
            reading["so2"],
        )


async def run_mock_sensors() -> None:
    """Connect to MQTT and publish all virtual station readings every 2 seconds."""
    
    # Load credentials from .env if available
    from dotenv import load_dotenv
    import os
    load_dotenv()
    mqtt_user = os.getenv("MQTT_USERNAME")
    mqtt_pass = os.getenv("MQTT_PASSWORD")
    
    client = MQTTClient("plumetrace-mock-sensors")
    if mqtt_user:
        client.set_auth_credentials(mqtt_user, mqtt_pass or "")
        
    client.set_config({"reconnect_retries": -1, "reconnect_delay": 5})

    try:
        await client.connect(MQTT_BROKER_HOST, port=MQTT_BROKER_PORT, version=MQTTv311)
        logger.info("Mock sensors connected to %s:%d", MQTT_BROKER_HOST, MQTT_BROKER_PORT)

        sequence = 0
        while True:
            await asyncio.gather(
                *(
                    publish_station_reading(client, sensor, sequence)
                    for sensor in SENSOR_STATIONS
                )
            )
            sequence += 1
            await asyncio.sleep(PUBLISH_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        logger.info("Mock sensor publisher cancelled.")
        raise
    except Exception as exc:
        logger.exception("Mock sensor publisher failed: %s", exc)
    finally:
        try:
            await client.disconnect()
        except Exception as exc:
            logger.debug("Ignoring mock sensor disconnect failure: %s", exc)


if __name__ == "__main__":
    try:
        asyncio.run(run_mock_sensors())
    except KeyboardInterrupt:
        logger.info("Mock sensors stopped by user.")
