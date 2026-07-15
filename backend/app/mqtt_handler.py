"""Asynchronous MQTT ingestion for PlumeTrace telemetry."""

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gmqtt import Client as MQTTClient
from gmqtt.mqtt.constants import MQTTv311
from pydantic import ValidationError

from app.config import settings
from app.database import get_db_session
from app.models import AirQualityReading, EnforcementDraft
from app.schemas import AirQualityReadingCreate, AirQualityReadingResponse
from app.sse_manager import sse_manager

logger = logging.getLogger(__name__)

# Debounce tracker to prevent excessive orchestration triggers
_last_orchestrator_trigger: dict[str, float] = {}

# Cached reference to the agent orchestrator graph builder (loaded once)
_cached_build_response_graph: Any = None


def _get_response_graph_builder() -> Any:
    """Lazily import and cache the agent orchestrator graph builder."""
    global _cached_build_response_graph
    if _cached_build_response_graph is None:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from agent_orchestrator import build_response_graph
        _cached_build_response_graph = build_response_graph
    return _cached_build_response_graph


class MQTTHandler:
    """Lifecycle-managed gmqtt subscriber for city air-quality streams."""

    def __init__(self) -> None:
        self._client: MQTTClient | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._connected = asyncio.Event()
        self._stopping = False

    @property
    def is_connected(self) -> bool:
        """Return whether the broker connection is currently active."""
        return self._connected.is_set()

    async def start(self) -> None:
        """Connect to the MQTT broker and subscribe to telemetry topics."""
        self._stopping = False
        client = MQTTClient(settings.MQTT_CLIENT_ID)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.set_config({"reconnect_retries": -1, "reconnect_delay": 5})
        self._client = client

        last_error: Exception | None = None
        for attempt in range(1, settings.MQTT_CONNECT_RETRIES + 1):
            try:
                # Add authentication if configured
                if settings.MQTT_USERNAME:
                    client.set_auth_credentials(
                        settings.MQTT_USERNAME,
                        settings.MQTT_PASSWORD or ""
                    )

                await client.connect(
                    settings.MQTT_BROKER_HOST,
                    port=settings.MQTT_BROKER_PORT,
                    version=MQTTv311,
                )
                await asyncio.wait_for(self._connected.wait(), timeout=10.0)
                logger.info(
                    "MQTT connected. broker=%s:%d topic=%s",
                    settings.MQTT_BROKER_HOST,
                    settings.MQTT_BROKER_PORT,
                    settings.MQTT_TOPIC,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "MQTT connection attempt failed. attempt=%d max_attempts=%d error=%s",
                    attempt,
                    settings.MQTT_CONNECT_RETRIES,
                    exc,
                )
                if attempt < settings.MQTT_CONNECT_RETRIES:
                    await asyncio.sleep(settings.MQTT_RETRY_BACKOFF_SECONDS)

        raise RuntimeError("Unable to connect to MQTT broker") from last_error

    async def stop(self) -> None:
        """Disconnect from MQTT and finish in-flight message processing."""
        self._stopping = True
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as exc:
                logger.warning("MQTT disconnect failed: %s", exc)

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._connected.clear()
        logger.info("MQTT handler stopped.")

    def _on_connect(
        self,
        client: MQTTClient,
        flags: int,
        rc: int,
        properties: dict[str, Any] | None,
    ) -> None:
        """Subscribe to the wildcard telemetry topic after broker connect."""
        if rc != 0:
            logger.error("MQTT broker rejected connection. result_code=%s", rc)
            return
        client.subscribe(settings.MQTT_TOPIC, qos=1)
        self._connected.set()
        logger.info("MQTT subscription active. topic=%s", settings.MQTT_TOPIC)

    def _on_disconnect(
        self,
        client: MQTTClient,
        packet: Any,
        exc: Exception | None = None,
    ) -> None:
        """Record broker disconnects."""
        self._connected.clear()
        if self._stopping:
            logger.info("MQTT disconnected cleanly.")
        else:
            logger.warning("MQTT disconnected unexpectedly. error=%s", exc)

    def _on_message(
        self,
        client: MQTTClient,
        topic: str,
        payload: bytes,
        qos: int,
        properties: dict[str, Any] | None,
    ) -> None:
        """Schedule async processing for a received MQTT message."""
        task = asyncio.create_task(self._process_message(topic, payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process_message(self, topic: str, payload: bytes) -> None:
        """Validate, persist, and broadcast one telemetry payload."""
        try:
            reading = AirQualityReadingCreate.model_validate_json(payload)
        except ValidationError as exc:
            logger.warning("Invalid MQTT telemetry. topic=%s errors=%s", topic, exc.errors())
            return
        except ValueError as exc:
            logger.warning("Malformed MQTT JSON. topic=%s error=%s", topic, exc)
            return

        timestamp = reading.timestamp or datetime.now(UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        try:
            persisted = AirQualityReading(
                sensor_id=reading.sensor_id,
                timestamp=timestamp,
                latitude=reading.latitude,
                longitude=reading.longitude,
                altitude=reading.altitude,
                pm25=reading.pm25,
                so2=reading.so2,
                wind_speed=reading.wind_speed,
                wind_direction=reading.wind_direction,
            )
            async with get_db_session() as session:
                session.add(persisted)
                await session.flush()
                response = AirQualityReadingResponse.model_validate(persisted)

            message = response.model_dump_json()
            await sse_manager.broadcast(message)
            logger.info(
                "Telemetry persisted and broadcast. sensor_id=%s so2=%.2f pm25=%.2f",
                reading.sensor_id,
                reading.so2,
                reading.pm25,
            )

            if reading.so2 > 75.0 or reading.pm25 > 35.0:
                current_time = asyncio.get_event_loop().time()
                last_trig = _last_orchestrator_trigger.get(reading.sensor_id, 0.0)
                if current_time - last_trig > 15.0:
                    _last_orchestrator_trigger[reading.sensor_id] = current_time
                    logger.warning("Hazardous spike detected for %s. Triggering Agent Orchestrator.", reading.sensor_id)

                    build_graph = _get_response_graph_builder()
                    graph = build_graph()
                    
                    initial_state = {
                        "sensor_alert_payload": {
                            "sensor_id": reading.sensor_id,
                            "latitude": reading.latitude,
                            "longitude": reading.longitude,
                            "gas_type": "SO2" if reading.so2 > 75.0 else "PM2.5",
                            "value": reading.so2 if reading.so2 > 75.0 else reading.pm25,
                            "timestamp": timestamp.isoformat(),
                        },
                        "meteorological_vectors": {},
                        "core_ai_inversion_coordinates": {},
                        "target_facility_profile": {},
                        "enforcement_report": "",
                        "hazard_status": "untriaged",
                        "routing_decision": "",
                        "lifecycle_events": [],
                        "errors": [],
                    }
                    
                    async def run_orchestrator(state: dict[str, Any]) -> None:
                        try:
                            final_state = await graph.ainvoke(state)
                            report = final_state.get("enforcement_report")
                            if report:
                                # Save the report to the database as a pending draft
                                async with get_db_session() as session:
                                    draft = EnforcementDraft(
                                        sensor_id=state["sensor_alert_payload"]["sensor_id"],
                                        report_text=report,
                                        status="pending_human_review",
                                    )
                                    session.add(draft)
                                    # Let the context manager handle commit
                                    await session.flush()
                                    await session.refresh(draft)
                                    draft_id = str(draft.id)
                                    
                                report_payload = {
                                    "type": "enforcement_report",
                                    "draft_id": draft_id,
                                    "review_status": "pending_human_review",
                                    "sensor_id": state["sensor_alert_payload"]["sensor_id"],
                                    "report": report,
                                    "source_coordinates": final_state.get("core_ai_inversion_coordinates", {}),
                                    "facility": final_state.get("target_facility_profile", {})
                                }
                                await sse_manager.broadcast(json.dumps(report_payload))
                                logger.info("Enforcement report broadcasted for %s", state["sensor_alert_payload"]["sensor_id"])
                        except Exception as e:
                            logger.error("Agent orchestrator background task failed: %s", e)

                    # Track the orchestrator task for proper lifecycle management
                    orch_task = asyncio.create_task(run_orchestrator(initial_state))
                    self._tasks.add(orch_task)
                    orch_task.add_done_callback(self._tasks.discard)
                else:
                    logger.info("Hazard spike for %s ignored due to orchestrator debounce.", reading.sensor_id)

        except Exception as exc:
            logger.exception("Failed to process MQTT telemetry. topic=%s error=%s", topic, exc)


mqtt_handler = MQTTHandler()
