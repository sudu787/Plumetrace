"""FastAPI application entry point for PlumeTrace."""

import logging
import sys
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, UTC
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Base, engine, get_db_session
from app.models import AirQualityReading, EnforcementDraft
from app.mqtt_handler import mqtt_handler
from app.schemas import (
    AirQualityReadingResponse,
    DraftReviewRequest,
    EnforcementDraftResponse,
    HealthResponse,
)
from app.sse_manager import sse_manager

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class AgentOrchestrationRequest(BaseModel):
    """Request body for the analytical response automation graph."""

    sensor_alert_payload: dict[str, Any] = Field(
        ...,
        description="Critical sensor alert payload from the frontend command center.",
    )
    suspected_source_hint: dict[str, Any] | None = Field(
        default=None,
        description="Optional frontend plume vector estimate for downstream audit context.",
    )


class AgentOrchestrationResponse(BaseModel):
    """Response body returned by the LangGraph agent orchestrator."""

    sensor_alert_payload: dict[str, Any]
    meteorological_vectors: dict[str, Any]
    core_ai_inversion_coordinates: dict[str, Any]
    target_facility_profile: dict[str, Any]
    enforcement_report: str
    hazard_status: str
    lifecycle_events: list[str]
    errors: list[str] = Field(default_factory=list)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

async def verify_api_key(
    header_key: str | None = Security(api_key_header),
    query_key: str | None = Security(api_key_query),
) -> None:
    """Verify that the caller provided a valid API key via header or query."""
    key = header_key or query_key
    if not key or key != settings.PLUMETRACE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields one async database session."""
    async with get_db_session() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create database tables, start MQTT ingestion, and cleanly shut down."""
    logger.info("Starting PlumeTrace backend.")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        logger.info("Database schema is ready.")

        await mqtt_handler.start()
        yield
    finally:
        logger.info("Stopping PlumeTrace backend.")
        await mqtt_handler.stop()
        await engine.dispose()


app = FastAPI(
    title="PlumeTrace API",
    version="1.0.0",
    description=(
        "AI-powered municipal environmental forensics backend for real-time "
        "air quality telemetry ingestion, persistence, and streaming."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Return basic service health."""
    return HealthResponse(service="PlumeTrace", status="operational", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health for load balancers and local checks."""
    status = "operational" if mqtt_handler.is_connected else "degraded"
    return HealthResponse(service="PlumeTrace", status=status, version="1.0.0")


@app.get(
    "/api/v1/sensors/history",
    response_model=list[AirQualityReadingResponse],
    summary="Fetch recent air-quality readings",
)
async def get_sensor_history(
    sensor_id: str | None = Query(default=None, min_length=1, max_length=64),
    session: AsyncSession = Depends(get_db),
    _auth: None = Depends(verify_api_key),
) -> list[AirQualityReadingResponse]:
    """Fetch the last 100 readings for time-series visualization."""
    try:
        statement = select(AirQualityReading).order_by(AirQualityReading.timestamp.desc()).limit(100)
        if sensor_id:
            statement = (
                select(AirQualityReading)
                .where(AirQualityReading.sensor_id == sensor_id)
                .order_by(AirQualityReading.timestamp.desc())
                .limit(100)
            )

        result = await session.execute(statement)
        readings = result.scalars().all()
        return [AirQualityReadingResponse.model_validate(reading) for reading in readings]
    except Exception as exc:
        logger.exception("Failed to fetch sensor history: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve sensor history.",
        ) from exc


@app.post(
    "/api/v1/drafts/{draft_id}/review",
    response_model=EnforcementDraftResponse,
    summary="Submit a human review action on an enforcement draft",
)
async def review_draft(
    draft_id: uuid.UUID,
    review: DraftReviewRequest,
    session: AsyncSession = Depends(get_db),
    _auth: None = Depends(verify_api_key),
) -> EnforcementDraftResponse:
    """Approve or reject a preliminary enforcement draft."""
    try:
        statement = select(EnforcementDraft).where(EnforcementDraft.id == draft_id)
        result = await session.execute(statement)
        draft = result.scalar_one_or_none()
        
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
            
        if draft.status != "pending_human_review":
            raise HTTPException(status_code=400, detail=f"Draft is already {draft.status}")
            
        draft.status = review.status
        draft.reviewer_id = review.reviewer_id
        draft.reviewed_at = datetime.now(UTC)
        if review.report_text is not None:
            draft.report_text = review.report_text
            
        await session.commit()
        await session.refresh(draft)
        return EnforcementDraftResponse.model_validate(draft)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to update draft review status: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update draft") from exc


@app.post(
    "/api/v1/agent/orchestrate",
    response_model=AgentOrchestrationResponse,
    summary="Run analytical response automation graph",
)
async def orchestrate_agent_response(
    request: AgentOrchestrationRequest,
    _auth: None = Depends(verify_api_key),
) -> AgentOrchestrationResponse:
    """Run the LangGraph response automation workflow for a critical alert."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from agent_orchestrator import build_response_graph

        graph = build_response_graph()
        initial_state = {
            "sensor_alert_payload": request.sensor_alert_payload,
            "meteorological_vectors": {},
            "core_ai_inversion_coordinates": {},
            "target_facility_profile": {},
            "enforcement_report": "",
            "hazard_status": "untriaged",
            "routing_decision": "",
            "lifecycle_events": [],
            "errors": [],
        }
        final_state = await graph.ainvoke(initial_state)
        return AgentOrchestrationResponse(
            sensor_alert_payload=final_state.get("sensor_alert_payload", {}),
            meteorological_vectors=final_state.get("meteorological_vectors", {}),
            core_ai_inversion_coordinates=final_state.get(
                "core_ai_inversion_coordinates",
                {},
            ),
            target_facility_profile=final_state.get("target_facility_profile", {}),
            enforcement_report=final_state.get("enforcement_report", ""),
            hazard_status=str(final_state.get("hazard_status", "unknown")),
            lifecycle_events=list(final_state.get("lifecycle_events", [])),
            errors=list(final_state.get("errors", [])),
        )
    except Exception as exc:
        logger.exception("Agent orchestration failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unable to run analytical response automation graph.",
        ) from exc


@app.get("/api/stream")
async def stream_endpoint(
    _auth: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream real-time telemetry to connected frontend clients using SSE."""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        await sse_manager.add_queue(queue)
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            logger.debug("SSE stream cancelled by client.")
            raise
        finally:
            await sse_manager.remove_queue(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
