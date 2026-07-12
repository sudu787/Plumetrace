"""Pydantic v2 schemas for telemetry validation and API responses."""

import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr, field_validator

SensorId = Annotated[StrictStr, Field(min_length=1, max_length=64)]
Latitude = Annotated[StrictFloat, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[StrictFloat, Field(ge=-180.0, le=180.0)]
NonNegativeReading = Annotated[StrictFloat, Field(ge=0.0)]
WindDirection = Annotated[StrictFloat, Field(ge=0.0, le=360.0)]


class AirQualityReadingBase(BaseModel):
    """Shared fields for air quality telemetry."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    sensor_id: SensorId
    latitude: Latitude
    longitude: Longitude
    pm25: NonNegativeReading = Field(description="PM2.5 concentration in ug/m3")
    so2: NonNegativeReading = Field(description="SO2 concentration in ppb")
    wind_speed: NonNegativeReading = Field(description="Wind speed in meters per second")
    wind_direction: WindDirection = Field(
        description="Wind direction in degrees, inclusive range 0 to 360"
    )

    @field_validator("sensor_id")
    @classmethod
    def validate_sensor_id(cls, value: str) -> str:
        """Reject blank sensor identifiers after whitespace trimming."""
        if not value:
            raise ValueError("sensor_id must not be blank")
        return value

    @field_validator(
        "latitude",
        "longitude",
        "pm25",
        "so2",
        "wind_speed",
        "wind_direction",
        mode="before",
    )
    @classmethod
    def require_float_values(cls, value: Any) -> Any:
        """Reject integer and boolean values for fields backed by DB Float columns."""
        if isinstance(value, bool) or isinstance(value, int):
            raise ValueError("value must be a floating-point number")
        return value


class AirQualityReadingCreate(AirQualityReadingBase):
    """Incoming MQTT telemetry payload."""

    timestamp: datetime | None = Field(default=None)


class AirQualityReadingResponse(AirQualityReadingBase):
    """Persisted telemetry returned by REST and WebSocket streams."""

    model_config = ConfigDict(
        from_attributes=True,
        strict=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: uuid.UUID
    timestamp: datetime


class HealthResponse(BaseModel):
    """Simple service health response."""

    service: StrictStr
    status: StrictStr
    version: StrictStr


class EnforcementDraftResponse(BaseModel):
    """Schema for returning an enforcement draft."""

    model_config = ConfigDict(
        from_attributes=True,
        strict=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: uuid.UUID
    sensor_id: StrictStr
    report_text: StrictStr
    status: StrictStr
    reviewer_id: StrictStr | None = None
    reviewed_at: datetime | None = None
    created_at: datetime


class DraftReviewRequest(BaseModel):
    """Payload for submitting a human review action on a draft."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    status: StrictStr = Field(..., description="Must be 'approved', 'rejected', or 'refine'")
    reviewer_id: StrictStr = Field(..., description="Identifier of the human reviewer")
    report_text: StrictStr | None = Field(
        default=None, description="Optional edited report text to save upon approval"
    )
    feedback: StrictStr | None = Field(
        default=None, description="Optional revision feedback if requesting refinement"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in ("approved", "rejected", "refine"):
            raise ValueError("status must be 'approved', 'rejected', or 'refine'")
        return v
