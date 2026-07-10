"""SQLAlchemy ORM models for PlumeTrace telemetry."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AirQualityReading(Base):
    """Persisted air quality telemetry from a municipal sensor station."""

    __tablename__ = "air_quality_readings"
    __table_args__ = (
        Index("ix_air_quality_readings_sensor_id", "sensor_id"),
        Index("ix_air_quality_readings_timestamp", "timestamp"),
        Index("ix_air_quality_readings_sensor_timestamp", "sensor_id", "timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    sensor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    pm25: Mapped[float] = mapped_column(Float, nullable=False)
    so2: Mapped[float] = mapped_column(Float, nullable=False)
    wind_speed: Mapped[float] = mapped_column(Float, nullable=False)
    wind_direction: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return (
            "AirQualityReading("
            f"id={self.id}, sensor_id={self.sensor_id!r}, "
            f"timestamp={self.timestamp!r}, so2={self.so2})"
        )


class EnforcementDraft(Base):
    """Persisted preliminary enforcement document awaiting human verification."""

    __tablename__ = "enforcement_drafts"
    __table_args__ = (
        Index("ix_enforcement_drafts_sensor_id", "sensor_id"),
        Index("ix_enforcement_drafts_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    sensor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    report_text: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending_human_review",
    )
    reviewer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            "EnforcementDraft("
            f"id={self.id}, sensor_id={self.sensor_id!r}, "
            f"status={self.status!r})"
        )
