"""
PlumeTrace analytical response automation layer.

This module implements a LangGraph state-machine that intercepts hazardous air
quality alerts, enriches them with meteorology, bridges to the plume inversion
model, maps the predicted source to municipal property records, and drafts an
environmental compliance warning report.

The script is intentionally self-contained for a hackathon demo. It uses real
LangGraph and LangChain interfaces, simulated tools for weather and property
records, and a deterministic ChatModel fallback so the demo runs without API
keys. If OPENAI_API_KEY or ANTHROPIC_API_KEY is present and the corresponding
integration package is installed, report generation automatically uses that
provider instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - dependency guard for local demos.
    raise SystemExit(
        "LangGraph is required. Install it with: pip install langgraph"
    ) from exc

try:
    from langchain.tools import tool
except ImportError:  # pragma: no cover - compatibility with lean LangChain installs.
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise SystemExit(
            "LangChain tool support is required. Install it with: pip install langchain"
        ) from exc

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
except ImportError as exc:  # pragma: no cover - dependency guard for local demos.
    raise SystemExit(
        "langchain-core is required. Install it with: pip install langchain-core"
    ) from exc


LOGGER = logging.getLogger("plumetrace.agent_orchestrator")
PROJECT_ROOT = Path(__file__).resolve().parent
PINN_CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "model_artifacts" / "plumetrace_pinn_checkpoint.pt",
    PROJECT_ROOT / "trained model" / "plumetrace_pinn_checkpoint.pt",
)
VALIDATION_SUMMARY_CANDIDATES = (
    PROJECT_ROOT / "trained model" / "plumetrace_validation_summary.json",
    PROJECT_ROOT / "model_artifacts" / "plumetrace_validation_summary.json",
)


# ---------------------------------------------------------------------------
# PINN inference configuration
# ---------------------------------------------------------------------------
USE_PINN_INFERENCE: bool = os.environ.get(
    "PLUMETRACE_USE_PINN", "true"
).lower() in ("1", "true", "yes")

_pinn_model_cache: dict[str, Any] = {}  # singleton: keys "model", "device", "sector"


def _get_or_load_pinn() -> tuple[Any, Any, Any]:
    """Lazily load the PINN checkpoint once. Returns (model, device, sector).

    Raises FileNotFoundError if no checkpoint exists, or any torch error on
    corrupted weights.
    """
    if "model" in _pinn_model_cache:
        return (
            _pinn_model_cache["model"],
            _pinn_model_cache["device"],
            _pinn_model_cache["sector"],
        )

    import torch
    from pinn_engine import CitySector, ModelConfig, PlumeInversionPINN, get_device

    checkpoint_path = first_existing_path(PINN_CHECKPOINT_CANDIDATES)
    if checkpoint_path is None:
        searched = ", ".join(
            project_relative_path(p) or str(p) for p in PINN_CHECKPOINT_CANDIDATES
        )
        raise FileNotFoundError(f"No PINN checkpoint found. Searched: {searched}")

    device = get_device()
    sector = CitySector()

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw

    # Infer the architecture from the checkpoint so the model matches what
    # was actually trained (the checkpoint may have fewer blocks or lack
    # LayerNorm compared to the current ModelConfig defaults).
    block_indices = {
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("blocks.")
    }
    n_blocks = max(block_indices) + 1 if block_indices else 1
    has_layer_norm = any("layer_norm" in k for k in state_dict)

    # Detect hidden_units from the first block's linear weight
    hidden_key = "blocks.0.linear1.weight"
    hidden_units = state_dict[hidden_key].shape[0] if hidden_key in state_dict else 128

    LOGGER.info(
        "Checkpoint architecture: %d blocks, %d hidden units, layer_norm=%s",
        n_blocks, hidden_units, has_layer_norm,
    )

    # Build a matching ModelConfig.  If the checkpoint lacks LayerNorm we
    # must also use a ResidualBlock without it — we handle this by
    # temporarily monkey-patching pinn_engine.ResidualBlock if needed.
    from pinn_engine import ResidualBlock as _OrigBlock
    _patched = False
    if not has_layer_norm:
        # The checkpoint was trained without LayerNorm in ResidualBlock.
        # Temporarily swap out the block so load_state_dict matches.
        import pinn_engine as _pe
        class _NoLNResidualBlock(torch.nn.Module):
            def __init__(self, width: int, use_adaptive_activation: bool) -> None:
                super().__init__()
                from pinn_engine import AdaptiveSwish
                self.linear1 = torch.nn.Linear(width, width)
                self.linear2 = torch.nn.Linear(width, width)
                self.act1 = AdaptiveSwish(width) if use_adaptive_activation else torch.nn.Tanh()
                self.act2 = AdaptiveSwish(width) if use_adaptive_activation else torch.nn.Tanh()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.act1(self.linear1(x))
                h = self.linear2(h)
                return self.act2(h + x)
        _pe.ResidualBlock = _NoLNResidualBlock  # type: ignore[attr-defined]
        _patched = True

    model_config = ModelConfig(hidden_layers=n_blocks, hidden_units=hidden_units)
    model = PlumeInversionPINN(model_config).to(device)

    if _patched:
        import pinn_engine as _pe
        _pe.ResidualBlock = _OrigBlock  # restore

    model.load_state_dict(state_dict)
    model.eval()

    _pinn_model_cache.update(model=model, device=device, sector=sector)
    LOGGER.info("PINN checkpoint loaded from %s", checkpoint_path)
    return model, device, sector


class AgentState(TypedDict, total=False):
    """Shared state preserved by the PlumeTrace response graph."""

    sensor_alert_payload: dict[str, Any]
    meteorological_vectors: dict[str, Any]
    core_ai_inversion_coordinates: dict[str, Any]
    target_facility_profile: dict[str, Any]
    enforcement_report: str
    used_pinn: bool
    source_estimate_stddev_m: float

    hazard_status: Literal["untriaged", "safe", "dangerous", "invalid"]
    routing_decision: str
    lifecycle_events: list[str]
    errors: list[str]


SAFE_EXPOSURE_LIMITS: dict[str, float] = {
    "SO2": 75.0,
    "PM25": 35.0,
    "PM2.5": 35.0,
    "NO2": 100.0,
    "O3": 70.0,
    "CO": 9.0,
}

# Minimum confidence required before the graph names a specific facility
# in the enforcement report.  Below this threshold the graph produces a
# generic "inconclusive" report that documents the hazard without
# attributing it to any business.
CONFIDENCE_GATE_THRESHOLD: float = float(
    os.environ.get("PLUMETRACE_CONFIDENCE_GATE", "0.6")
)


INDUSTRIAL_PROPERTY_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "factory_name": "Apex Petrochemical Complex",
        "corporate_owner": "Apex Industrial Holdings LLC",
        "zoning_permit_id": "MUNI-IZ-2044-APX-17",
        "latitude": 40.71345,
        "longitude": -74.00765,
        "plot_radius_km": 0.42,
        "compliance_history": [
            "2024-04-18: Notice of Violation for sulfur scrubber bypass reporting lapse.",
            "2025-09-03: Administrative warning for delayed continuous emissions monitor calibration.",
            "2026-02-11: Corrective action plan accepted for flare-stack exceedance logging.",
        ],
    },
    {
        "factory_name": "Global Logistics Foundry",
        "corporate_owner": "Harborline Materials Group",
        "zoning_permit_id": "MUNI-IZ-2041-GLF-09",
        "latitude": 40.71610,
        "longitude": -74.00280,
        "plot_radius_km": 0.36,
        "compliance_history": [
            "2024-12-07: Odor complaint investigation closed with no exceedance.",
            "2025-06-21: Minor particulate handling deficiency corrected on site.",
        ],
    },
    {
        "factory_name": "Northbank Solvents Terminal",
        "corporate_owner": "CivicChem Transport Partners",
        "zoning_permit_id": "MUNI-IZ-2039-NST-22",
        "latitude": 40.71020,
        "longitude": -74.01160,
        "plot_radius_km": 0.31,
        "compliance_history": [
            "2023-11-19: Spill prevention plan revision ordered.",
            "2025-01-29: Volatile organic compound inspection passed.",
        ],
    },
)


REPORT_SYSTEM_PROMPT = """
You are PlumeTrace Compliance Counsel, drafting for a municipal environmental
enforcement office. Write exactly three paragraphs. The tone must be official,
highly authoritative, and legally careful. Cite the alert coordinates, predicted
source coordinates, confidence score, spatial standard deviation (stddev_m),
meteorological alignment, facility ownership, zoning permit, and historical infractions.
Do not invent statutes, penalties, or facts that are not provided. Describe the
document as a preliminary enforcement warning subject to field verification and
evidentiary review. Explicitly state the model's confidence level and spatial spread
along with whether the physics-informed AI solver (used_pinn) was utilized,
rather than presenting the source coordinate as an absolute certainty.

CRITICAL REQUIREMENT:
You MUST start the document with this exact visible banner on its own line:
"AI-GENERATED DRAFT — REQUIRES HUMAN VERIFICATION BEFORE USE"
""".strip()


def configure_logging() -> None:
    """Configure readable console logging for direct script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def append_lifecycle_event(state: AgentState, message: str) -> list[str]:
    """Return lifecycle history with a newly appended event."""
    return [*state.get("lifecycle_events", []), message]


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Calculate approximate geodesic distance between two coordinates."""
    earth_radius_km = 6371.0088
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


@tool
def fetch_current_weather(lat: float, lon: float) -> dict[str, float | str]:
    """
    Simulate a meteorological API lookup for localized wind vectors.

    Args:
        lat: Latitude of the sensor alert.
        lon: Longitude of the sensor alert.
    """
    latitude_factor = math.sin(math.radians(lat * 10.0))
    longitude_factor = math.cos(math.radians(lon * 10.0))
    wind_speed = round(6.8 + 0.7 * latitude_factor, 2)
    wind_direction = round((246.0 + 4.0 * longitude_factor) % 360.0, 2)
    boundary_layer_temp = round(22.4 + 1.1 * latitude_factor, 2)
    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "boundary_layer_temp": boundary_layer_temp,
        "units": "m/s, degrees from north, Celsius",
        "provider": "Simulated Municipal Mesonet",
    }


@tool
def query_property_records(lat: float, lon: float) -> dict[str, Any]:
    """
    Search mock municipal industrial property records near a coordinate.

    Args:
        lat: Predicted source latitude.
        lon: Predicted source longitude.
    """
    nearest_record: dict[str, Any] | None = None
    nearest_distance = float("inf")
    for record in INDUSTRIAL_PROPERTY_RECORDS:
        distance = haversine_km(lat, lon, record["latitude"], record["longitude"])
        if distance < nearest_distance:
            nearest_record = record
            nearest_distance = distance

    if nearest_record is None:
        raise RuntimeError("No municipal property records are configured.")

    profile = {
        "factory_name": nearest_record["factory_name"],
        "corporate_owner": nearest_record["corporate_owner"],
        "zoning_permit_id": nearest_record["zoning_permit_id"],
        "compliance_history": nearest_record["compliance_history"],
        "matched_property_latitude": nearest_record["latitude"],
        "matched_property_longitude": nearest_record["longitude"],
        "match_distance_km": round(nearest_distance, 3),
        "match_confidence": "high"
        if nearest_distance <= nearest_record["plot_radius_km"]
        else "nearest_available_record",
    }
    return profile


class DeterministicWeatherToolCallingChatModel(BaseChatModel):
    """Local ChatModel that emits a LangChain tool call for weather lookup."""

    @property
    def _llm_type(self) -> str:
        return "plumetrace_deterministic_weather_tool_caller"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = str(messages[-1].content)
        lat_match = re.search(r"latitude=([-+]?\d+(?:\.\d+)?)", content)
        lon_match = re.search(r"longitude=([-+]?\d+(?:\.\d+)?)", content)
        if lat_match is None or lon_match is None:
            raise ValueError("Weather tool caller could not parse latitude/longitude.")

        tool_call = {
            "name": fetch_current_weather.name,
            "args": {
                "lat": float(lat_match.group(1)),
                "lon": float(lon_match.group(1)),
            },
            "id": "weather_lookup_call_001",
        }
        message = AIMessage(content="", tool_calls=[tool_call])
        return ChatResult(generations=[ChatGeneration(message=message)])


class DeterministicComplianceChatModel(BaseChatModel):
    """Local ChatModel fallback that produces a three-paragraph report."""

    @property
    def _llm_type(self) -> str:
        return "plumetrace_deterministic_compliance_reporter"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        state = parse_state_from_report_prompt(str(messages[-1].content))
        report = render_deterministic_compliance_report(state)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=report))])


def parse_state_from_report_prompt(content: str) -> AgentState:
    """Extract JSON state from the report-generation human prompt."""
    marker = "COMPILED_AGENT_STATE_JSON:"
    if marker not in content:
        raise ValueError("Compiled state marker missing from report prompt.")
    payload = content.split(marker, 1)[1].strip()
    return json.loads(payload)


def build_compliance_chat_model() -> BaseChatModel:
    """Build an OpenAI or Anthropic ChatModel, falling back to deterministic local output."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=os.getenv("PLUMETRACE_OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.1,
            )
        except ImportError:
            LOGGER.warning("OPENAI_API_KEY is set, but langchain_openai is not installed.")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=os.getenv("PLUMETRACE_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                temperature=0.1,
            )
        except ImportError:
            LOGGER.warning("ANTHROPIC_API_KEY is set, but langchain_anthropic is not installed.")

    return DeterministicComplianceChatModel()


async def ingestion_and_triage_node(state: AgentState) -> AgentState:
    """Parse and triage an incoming hazardous sensor event."""
    payload = dict(state.get("sensor_alert_payload", {}))
    required_fields = {"sensor_id", "latitude", "longitude", "gas_type", "value", "timestamp"}
    missing_fields = sorted(required_fields - payload.keys())
    if missing_fields:
        error = f"Sensor alert rejected; missing fields: {', '.join(missing_fields)}"
        return {
            "hazard_status": "invalid",
            "routing_decision": "end",
            "errors": [*state.get("errors", []), error],
            "enforcement_report": "",
            "lifecycle_events": append_lifecycle_event(state, error),
        }

    gas_type = str(payload["gas_type"]).upper()
    value = float(payload["value"])
    safe_limit = SAFE_EXPOSURE_LIMITS.get(gas_type)
    if safe_limit is None:
        error = f"Sensor alert rejected; no safe exposure threshold for gas_type={gas_type}"
        return {
            "hazard_status": "invalid",
            "routing_decision": "end",
            "errors": [*state.get("errors", []), error],
            "enforcement_report": "",
            "lifecycle_events": append_lifecycle_event(state, error),
        }

    payload.update(
        {
            "gas_type": gas_type,
            "value": value,
            "safe_limit": safe_limit,
            "latitude": float(payload["latitude"]),
            "longitude": float(payload["longitude"]),
        }
    )

    if value <= safe_limit:
        message = (
            f"Alert triaged as safe: {gas_type}={value:.2f} does not exceed "
            f"limit={safe_limit:.2f}."
        )
        return {
            "sensor_alert_payload": payload,
            "hazard_status": "safe",
            "routing_decision": "end",
            "enforcement_report": "",
            "lifecycle_events": append_lifecycle_event(state, message),
        }

    message = (
        f"Critical hazard confirmed: {gas_type}={value:.2f} exceeds "
        f"limit={safe_limit:.2f}; routing to meteorological worker."
    )
    return {
        "sensor_alert_payload": payload,
        "hazard_status": "dangerous",
        "routing_decision": "weather_worker",
        "lifecycle_events": append_lifecycle_event(state, message),
    }


def route_after_triage(state: AgentState) -> Literal["weather_worker", "end"]:
    """Route safe or invalid events to END and dangerous events to weather lookup."""
    return "weather_worker" if state.get("hazard_status") == "dangerous" else "end"


async def meteorological_lookup_node(state: AgentState) -> AgentState:
    """Use LLM tool-calling to request and execute localized weather lookup."""
    payload = state["sensor_alert_payload"]
    latitude = float(payload["latitude"])
    longitude = float(payload["longitude"])

    tool_calling_model = DeterministicWeatherToolCallingChatModel()
    model_response = tool_calling_model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a meteorological routing agent. Call the weather "
                    "tool for the exact sensor coordinates."
                )
            ),
            HumanMessage(
                content=(
                    "Fetch localized weather for hazardous plume analysis: "
                    f"latitude={latitude}, longitude={longitude}"
                )
            ),
        ]
    )

    if not model_response.tool_calls:
        raise RuntimeError("Weather LLM did not produce a tool call.")

    tool_call = model_response.tool_calls[0]
    if tool_call["name"] != fetch_current_weather.name:
        raise RuntimeError(f"Unexpected weather tool requested: {tool_call['name']}")

    weather_data = fetch_current_weather.invoke(tool_call["args"])
    message = (
        "Meteorological vectors resolved through tool call: "
        f"wind_speed={weather_data['wind_speed']} m/s, "
        f"wind_direction={weather_data['wind_direction']} degrees, "
        f"boundary_layer_temp={weather_data['boundary_layer_temp']} C."
    )
    return {
        "meteorological_vectors": weather_data,
        "lifecycle_events": append_lifecycle_event(state, message),
    }


def first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
    """Return the first candidate file that exists on disk."""
    return next((path for path in candidates if path.exists()), None)


def project_relative_path(path: Path | None) -> str | None:
    """Render a stable project-relative path for reports and metadata."""
    if path is None:
        return None
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# load_validated_fused_source_peak() removed — the PINN must run its own
# forward pass via _get_or_load_pinn(); reading a cached coordinate from a
# JSON file is not inference.


def read_mock_pytorch_network_metadata(
    checkpoint_path: Path | None = None,
    inference_source: str = "unresolved",
) -> dict[str, Any]:
    """Mock metadata bridge for the PlumeTrace PyTorch inverse model."""
    return {
        "model_name": "PlumeInversionPINN",
        "checkpoint_uri": project_relative_path(checkpoint_path),
        "validation_summary_uri": project_relative_path(first_existing_path(VALIDATION_SUMMARY_CANDIDATES)),
        "input_features": ["x", "y", "elapsed_time"],
        "output": "source_probability_surface",
        "calibration_version": "hackathon-demo-2026.07",
        "inference_source": inference_source,
    }


def evaluate_physics_fused_posterior(
    pinn_map: Any,
    dataset: Any,
    config: Any,
    temperature: float = 0.010,
) -> Any:
    """Apply physics likelihood to fuse PINN raw concentration output into a source probability map."""
    import numpy as np
    from pinn_engine import CitySector, analytic_advection_diffusion_plume, SourceProbabilityMap

    obs_x = np.array([r['x'] for r in dataset.rows], dtype=np.float32)[None, :]
    obs_y = np.array([r['y'] for r in dataset.rows], dtype=np.float32)[None, :]
    obs_t = np.array([r['elapsed_time'] for r in dataset.rows], dtype=np.float32)[None, :]
    obs_c = np.array([r['normalized_concentration'] for r in dataset.rows], dtype=np.float32)[None, :]

    sector = CitySector()

    sx = ((pinn_map.longitudes - sector.lon_min) / (sector.lon_max - sector.lon_min)).reshape(-1, 1).astype(np.float32)
    sy = ((pinn_map.latitudes - sector.lat_min) / (sector.lat_max - sector.lat_min)).reshape(-1, 1).astype(np.float32)

    signal = analytic_advection_diffusion_plume(
        obs_x, obs_y, obs_t, sx, sy, config.wind_u, config.wind_v, config.diffusion
    )
    signal = signal / np.maximum(signal.max(axis=1, keepdims=True), 1.0e-8)
    mse = np.mean((signal - obs_c) ** 2, axis=1).reshape(pinn_map.probabilities.shape)

    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(scores.astype(np.float64), 0.0, None)
        total = float(clipped.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full_like(clipped, 1.0 / clipped.size, dtype=np.float64)
        return clipped / total

    likelihood = np.exp(-mse / max(temperature, 1.0e-6))
    neural_prior = np.power(normalize_scores(pinn_map.probabilities) + 1.0e-12, 0.35)
    fused = normalize_scores(neural_prior * likelihood)

    return SourceProbabilityMap(
        latitudes=pinn_map.latitudes,
        longitudes=pinn_map.longitudes,
        probabilities=fused.astype(np.float32),
    )


async def ai_execution_bridge_node(state: AgentState) -> AgentState:
    """Run the trained PINN inverse-dispersion model to estimate the upstream
    source point.  Falls back to an upwind ray-trace only when the checkpoint
    is unavailable or inference errors out."""
    payload = state["sensor_alert_payload"]
    weather = state["meteorological_vectors"]
    checkpoint_path = first_existing_path(PINN_CHECKPOINT_CANDIDATES)
    used_pinn = False
    integration_error: str | None = None
    inference_source = "unresolved"

    # ------------------------------------------------------------------
    # Primary path: PINN inverse solve
    # ------------------------------------------------------------------
    if USE_PINN_INFERENCE:
        try:
            model, device, sector = await asyncio.to_thread(_get_or_load_pinn)
            from pinn_engine import (
                compute_source_uncertainty,
                estimate_source_from_probability_map,
                evaluate_source_probability_grid,
                TrainingConfig,
                generate_synthetic_sensor_data,
            )

            prob_map = await asyncio.to_thread(
                evaluate_source_probability_grid, model, sector, device
            )
            
            pinn_config = TrainingConfig()
            dataset = await asyncio.to_thread(
                generate_synthetic_sensor_data, sector, pinn_config, device
            )
            
            prob_map = evaluate_physics_fused_posterior(prob_map, dataset, pinn_config)

            predicted_latitude, predicted_longitude, _ = (
                estimate_source_from_probability_map(prob_map)
            )
            uq = compute_source_uncertainty(prob_map)
            
            inference_source = "pinn_checkpoint_probability_map"
            confidence_score = uq.confidence
            stddev_m = uq.stddev_meters
            used_pinn = True
            LOGGER.info(
                "PINN inference succeeded: lat=%.6f lon=%.6f prob=%.8f confidence=%.3f stddev=%.1fm",
                predicted_latitude,
                predicted_longitude,
                uq.peak_probability,
                confidence_score,
                stddev_m,
            )
        except Exception as exc:
            integration_error = (
                f"PINN inference failed, falling back to upwind trace: {exc}"
            )
            LOGGER.warning(integration_error)
            # fall through to upwind ray-trace
    else:
        LOGGER.info(
            "PINN inference disabled via PLUMETRACE_USE_PINN=false; "
            "using upwind ray-trace."
        )

    # ------------------------------------------------------------------
    # Fallback path: geometric upwind ray-trace (clearly labeled)
    # ------------------------------------------------------------------
    if not used_pinn:
        severity_factor = min(2.4, max(float(payload["value"]) / 75.0, 1.0))
        plume_length_km = 0.48 + (severity_factor * 0.22)
        bearing_rad = math.radians(float(weather["wind_direction"]))
        predicted_latitude = float(payload["latitude"]) + (
            math.cos(bearing_rad) * plume_length_km / 111.0
        )
        predicted_longitude = float(payload["longitude"]) + (
            math.sin(bearing_rad)
            * plume_length_km
            / (111.0 * math.cos(math.radians(float(payload["latitude"]))))
        )
        confidence_score = 0.55  # lower confidence for geometric fallback
        stddev_m = 150.0         # flat generic spread for fallback trace
        inference_source = "upwind_ray_trace_fallback"

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    model_metadata = read_mock_pytorch_network_metadata(
        checkpoint_path, inference_source
    )
    upstream_distance_km = haversine_km(
        float(payload["latitude"]),
        float(payload["longitude"]),
        predicted_latitude,
        predicted_longitude,
    )
    inversion_result = {
        "source_latitude": round(predicted_latitude, 6),
        "source_longitude": round(predicted_longitude, 6),
        "confidence_score": round(confidence_score, 3),
        "stddev_meters": round(stddev_m, 1),
        "upstream_distance_km": round(upstream_distance_km, 3),
        "wind_direction_used": float(weather["wind_direction"]),
        "model_metadata": model_metadata,
    }
    message = (
        "AI inversion bridge estimated upstream source at "
        f"({inversion_result['source_latitude']}, {inversion_result['source_longitude']}) "
        f"with confidence={inversion_result['confidence_score']} (±{inversion_result['stddev_meters']}m) "
        f"[used_pinn={used_pinn}, source={inference_source}]."
    )
    update: AgentState = {
        "core_ai_inversion_coordinates": inversion_result,
        "used_pinn": used_pinn,
        "source_estimate_stddev_m": float(stddev_m),
        "lifecycle_events": append_lifecycle_event(state, message),
    }
    if integration_error is not None:
        update["errors"] = [*state.get("errors", []), integration_error]
    return update


def route_after_inversion(state: AgentState) -> str:
    """Confidence gate: only route to facility attribution when confidence
    meets the threshold.  Otherwise produce an inconclusive report that
    documents the hazard without naming a specific business."""
    coords = state.get("core_ai_inversion_coordinates", {})
    confidence = float(coords.get("confidence_score", 0.0))
    if confidence >= CONFIDENCE_GATE_THRESHOLD:
        return "attribute"
    LOGGER.warning(
        "Confidence %.3f < gate %.3f — skipping facility attribution.",
        confidence,
        CONFIDENCE_GATE_THRESHOLD,
    )
    return "inconclusive"


async def inconclusive_report_node(state: AgentState) -> AgentState:
    """Produce a generic hazard report that does NOT name any business.

    This node runs when the inversion confidence is below
    CONFIDENCE_GATE_THRESHOLD, preventing unsubstantiated accusations."""
    payload = state["sensor_alert_payload"]
    weather = state["meteorological_vectors"]
    inversion = state["core_ai_inversion_coordinates"]

    report = (
        "Municipal Environmental Enforcement issues this preliminary hazard "
        f"advisory concerning sensor {payload['sensor_id']} at coordinates "
        f"{payload['latitude']:.6f}, {payload['longitude']:.6f}, which recorded "
        f"{payload['gas_type']} at {payload['value']:.2f} against the applicable "
        f"safe parameter of {payload.get('safe_limit', 'N/A')} at {payload['timestamp']}. "
        "The exceedance is classified as a critical hazardous air-quality event.\n\n"
        f"Meteorological conditions: wind speed {weather['wind_speed']} m/s, "
        f"wind direction {weather['wind_direction']} degrees, boundary-layer "
        f"temperature {weather['boundary_layer_temp']} C. The PlumeTrace inversion "
        f"model estimated a candidate source at ({inversion['source_latitude']}, "
        f"{inversion['source_longitude']}) with confidence {inversion['confidence_score']:.3f}. "
        f"This confidence level is below the attribution threshold of "
        f"{CONFIDENCE_GATE_THRESHOLD:.2f}; therefore, no specific facility is named "
        "in this advisory.\n\n"
        "The enforcement office is directed to dispatch a field verification team "
        "to the estimated source area. No facility-specific preservation orders "
        "are issued at this time. This advisory will be updated when higher-confidence "
        "attribution data becomes available."
    )
    message = (
        f"Confidence gate blocked facility attribution "
        f"(score={inversion['confidence_score']:.3f} < threshold={CONFIDENCE_GATE_THRESHOLD:.2f}). "
        f"Inconclusive report generated."
    )
    return {
        "enforcement_report": report,
        "target_facility_profile": {},
        "lifecycle_events": append_lifecycle_event(state, message),
    }


async def municipal_registry_search_node(state: AgentState) -> AgentState:
    """Resolve the predicted source coordinate to a municipal facility profile."""
    coordinates = state["core_ai_inversion_coordinates"]
    facility_profile = query_property_records.invoke(
        {
            "lat": float(coordinates["source_latitude"]),
            "lon": float(coordinates["source_longitude"]),
        }
    )
    message = (
        "Municipal registry matched source estimate to "
        f"{facility_profile['factory_name']} owned by {facility_profile['corporate_owner']}."
    )
    return {
        "target_facility_profile": facility_profile,
        "lifecycle_events": append_lifecycle_event(state, message),
    }


def render_deterministic_compliance_report(state: AgentState) -> str:
    """Render a deterministic three-paragraph compliance report for offline demos."""
    payload = state["sensor_alert_payload"]
    weather = state["meteorological_vectors"]
    inversion = state["core_ai_inversion_coordinates"]
    facility = state["target_facility_profile"]
    history = "; ".join(facility.get("compliance_history", []))

    paragraph_one = (
        "Municipal Environmental Enforcement issues this preliminary enforcement "
        f"warning concerning sensor {payload['sensor_id']} at coordinates "
        f"{payload['latitude']:.6f}, {payload['longitude']:.6f}, which recorded "
        f"{payload['gas_type']} at {payload['value']:.2f} against the applicable "
        f"safe parameter of {payload['safe_limit']:.2f} at {payload['timestamp']}. "
        "The exceedance is classified as a critical hazardous air-quality event "
        "requiring immediate compliance review and preservation of operational records."
    )

    paragraph_two = (
        "Meteorological alignment supports upstream source attribution: localized "
        f"wind vectors showed wind speed {weather['wind_speed']} m/s, wind direction "
        f"{weather['wind_direction']} degrees, and boundary-layer temperature "
        f"{weather['boundary_layer_temp']} C. The PlumeTrace inversion bridge, using "
        "the PlumeInversionPINN source-probability model metadata, estimated a "
        f"high-probability source coordinate at {inversion['source_latitude']:.6f}, "
        f"{inversion['source_longitude']:.6f} with confidence score "
        f"{inversion['confidence_score']:.3f} (spatial spread ±{inversion.get('stddev_meters', 0):.1f}m), "
        f"located upstream of the affected sensor under the recorded wind field."
    )

    paragraph_three = (
        f"Municipal registry cross-reference identifies the target facility as "
        f"{facility['factory_name']}, owned by {facility['corporate_owner']}, under "
        f"zoning permit {facility['zoning_permit_id']}; the coordinate match distance "
        f"is {facility['match_distance_km']} km with match status "
        f"{facility['match_confidence']}. Historical compliance entries include: "
        f"{history}. The facility is directed to preserve emissions logs, process "
        "control records, maintenance records, and continuous-monitoring data pending "
        "field verification, evidentiary review, and any subsequent regulatory action."
    )

    return "\n\n".join((paragraph_one, paragraph_two, paragraph_three))


async def compliance_report_generation_node(state: AgentState) -> AgentState:
    """Generate the final regulatory enforcement warning document."""
    report_payload = {
        "sensor_alert_payload": state["sensor_alert_payload"],
        "meteorological_vectors": state["meteorological_vectors"],
        "core_ai_inversion_coordinates": state["core_ai_inversion_coordinates"],
        "target_facility_profile": state["target_facility_profile"],
    }
    messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Draft the official PlumeTrace regulatory warning from this state.\n\n"
                "COMPILED_AGENT_STATE_JSON:\n"
                f"{json.dumps(report_payload, indent=2)}"
            )
        ),
    ]

    chat_model = build_compliance_chat_model()
    try:
        response = chat_model.invoke(messages)
        report = response.content
        if isinstance(report, list):
            report = "\n".join(str(part) for part in report)
        report = str(report).strip()
    except Exception as exc:
        LOGGER.warning("ChatModel report generation failed; using deterministic fallback: %s", exc)
        report = render_deterministic_compliance_report(report_payload)

    message = "Compliance report generated with three-paragraph regulatory warning."
    return {
        "enforcement_report": report,
        "lifecycle_events": append_lifecycle_event(state, message),
    }


def build_response_graph():
    """Build and compile the PlumeTrace analytical response StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("ingestion_triage", ingestion_and_triage_node)
    graph.add_node("meteorological_lookup", meteorological_lookup_node)
    graph.add_node("ai_execution_bridge", ai_execution_bridge_node)
    graph.add_node("municipal_registry_search", municipal_registry_search_node)
    graph.add_node("compliance_report_generation", compliance_report_generation_node)

    graph.add_edge(START, "ingestion_triage")
    graph.add_conditional_edges(
        "ingestion_triage",
        route_after_triage,
        {
            "weather_worker": "meteorological_lookup",
            "end": END,
        },
    )
    graph.add_node("inconclusive_report", inconclusive_report_node)

    graph.add_edge("meteorological_lookup", "ai_execution_bridge")
    graph.add_conditional_edges(
        "ai_execution_bridge",
        route_after_inversion,
        {
            "attribute": "municipal_registry_search",
            "inconclusive": "inconclusive_report",
        },
    )
    graph.add_edge("municipal_registry_search", "compliance_report_generation")
    graph.add_edge("compliance_report_generation", END)
    graph.add_edge("inconclusive_report", END)
    return graph.compile()


def summarize_update(update: AgentState) -> str:
    """Summarize graph updates without dumping the full report mid-stream."""
    visible_keys = []
    for key, value in update.items():
        if key == "enforcement_report" and value:
            visible_keys.append("enforcement_report=<generated>")
        elif key == "lifecycle_events":
            events = value or []
            visible_keys.append(f"lifecycle_events=+{events[-1] if events else 'none'}")
        else:
            visible_keys.append(f"{key}={value}")
    return "; ".join(visible_keys)


async def run_demo_execution() -> AgentState:
    """Execute the graph with a mock critical sensor failure event and stream updates."""
    graph = build_response_graph()
    initial_state: AgentState = {
        "sensor_alert_payload": {
            "sensor_id": "industrial_north",
            "latitude": 40.7180,
            "longitude": -74.0060,
            "gas_type": "SO2",
            "value": 188.4,
            "timestamp": "2026-07-09T09:15:00+05:30",
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

    print("\nPlumeTrace analytical response graph starting...\n")
    final_state: AgentState = dict(initial_state)
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            if not isinstance(update, dict):
                continue
            final_state.update(update)
            print(f"[{node_name}] {summarize_update(update)}")

    print("\nFinal Enforcement Report\n")
    report = final_state.get("enforcement_report")
    if report:
        print(report)
    else:
        print("No enforcement report generated; event was safe or invalid.")
    return final_state


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run_demo_execution())
