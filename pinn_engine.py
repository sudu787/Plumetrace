"""
PlumeTrace physics-informed neural network engine.

This script is a self-contained hackathon demo for inverse-dispersion source
attribution. It trains a Physics-Informed Neural Network (PINN) to reconstruct
the origin of a pollutant plume using sparse measurements from four municipal
IoT sensors and a physics loss derived from the 2D advection-diffusion PDE.

Mathematical model
------------------
The pollutant concentration C(x, y, t) is governed by:

    dC/dt + u * dC/dx + v * dC/dy - D * (d2C/dx2 + d2C/dy2) = 0

The neural network approximates C(x, y, t). During training, the data loss fits
sparse sensor readings while the physics loss penalizes PDE residual error over
Sobol collocation points. The architecture uses advanced Fourier Residual blocks
and two-stage AdamW + L-BFGS optimization.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.quasirandom import SobolEngine
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run pinn_engine.py. Install it with the "
        "appropriate command from https://pytorch.org/get-started/locally/."
    ) from exc


LOGGER = logging.getLogger("plumetrace.pinn")


@dataclass(frozen=True)
class SensorStation:
    """A deterministic virtual sensor location inside the industrial sector."""
    sensor_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class CitySector:
    """Geographic bounds and known demo source for the synthetic scenario."""
    lat_min: float = 40.7040
    lat_max: float = 40.7220
    lon_min: float = -74.0160
    lon_max: float = -73.9940
    source_latitude: float = 40.7138
    source_longitude: float = -74.0072


@dataclass(frozen=True)
class ModelConfig:
    arch_type: Literal['mlp', 'residual', 'fourier_residual', 'siren'] = 'fourier_residual'
    hidden_units: int = 128
    hidden_layers: int = 8
    fourier_bands: int = 8
    fourier_sigma: float = 4.0
    siren_omega0_first: float = 30.0
    siren_omega0_hidden: float = 30.0
    use_adaptive_activation: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and physics settings for the inversion experiment."""
    epochs: int = 2500
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    lambda_data: float = 1.0
    lambda_physics: float = 0.10
    lambda_boundary: float = 0.02
    collocation_points: int = 4096
    boundary_points: int = 1024
    log_every: int = 50
    wind_u: float = 0.16
    wind_v: float = -0.06
    diffusion: float = 0.006
    sensor_time_samples: int = 64
    random_seed: int = 2026
    gradient_clip_norm: float = 5.0

    warmup_epochs: int = 100
    min_lr_ratio: float = 0.08
    ema_decay: float = 0.999

    rar_interval: int = 100
    rar_fraction: float = 0.25
    rar_pool_multiplier: int = 4
    rar_warmup_epochs: int = 200

    softadapt_beta: float = 0.1
    softadapt_floor: float = 0.05
    softadapt_ceil: float = 5.0
    softadapt_warmup_epochs: int = 20

    lbfgs_steps: int = 200
    lbfgs_lr: float = 0.5


@dataclass
class SyntheticSensorDataset:
    """Tensor and metadata bundle for sparse sensor observations."""
    features: Tensor
    concentration: Tensor
    concentration_scale_ppb: float
    rows: list[dict[str, Any]]
    val_features: Tensor | None = None
    val_concentration: Tensor | None = None


@dataclass
class SourceProbabilityMap:
    """Dense latitude, longitude, and normalized source probability matrices."""
    latitudes: np.ndarray
    longitudes: np.ndarray
    probabilities: np.ndarray


SENSOR_STATIONS: tuple[SensorStation, ...] = (
    SensorStation("industrial_north", 40.7180, -74.0060),
    SensorStation("residential_east", 40.7140, -73.9980),
    SensorStation("park_south", 40.7080, -74.0040),
    SensorStation("river_west", 40.7120, -74.0120),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)
    return device


def lat_lon_to_normalized(
    latitude: float | np.ndarray,
    longitude: float | np.ndarray,
    sector: CitySector,
) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(longitude, dtype=np.float32)
    lat_arr = np.asarray(latitude, dtype=np.float32)
    x = (lon_arr - sector.lon_min) / (sector.lon_max - sector.lon_min)
    y = (lat_arr - sector.lat_min) / (sector.lat_max - sector.lat_min)
    return x.astype(np.float32), y.astype(np.float32)


def normalized_to_lat_lon(
    x: np.ndarray,
    y: np.ndarray,
    sector: CitySector,
) -> tuple[np.ndarray, np.ndarray]:
    longitude = sector.lon_min + x * (sector.lon_max - sector.lon_min)
    latitude = sector.lat_min + y * (sector.lat_max - sector.lat_min)
    return latitude.astype(np.float32), longitude.astype(np.float32)


# ============================================================
# Architecture factory
# ============================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_bands: int, sigma: float, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        b_matrix = torch.randn((in_dim, num_bands), generator=generator) * sigma
        self.register_buffer('b_matrix', b_matrix)

    def forward(self, x: Tensor) -> Tensor:
        projected = 2.0 * math.pi * (x @ self.b_matrix)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class AdaptiveSwish(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.beta * x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, use_adaptive_activation: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.act1 = AdaptiveSwish(width) if use_adaptive_activation else nn.Tanh()
        self.act2 = AdaptiveSwish(width) if use_adaptive_activation else nn.Tanh()
        self.layer_norm1 = nn.LayerNorm(width)
        self.layer_norm2 = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(self.linear1(self.layer_norm1(x)))
        h = self.linear2(self.layer_norm2(h))
        return self.act2(h + x)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega0: float, is_first: bool) -> None:
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / omega0)
            self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class PlumeInversionPINN(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.config = model_config
        arch = model_config.arch_type
        in_dim = 3

        if arch == 'fourier_residual':
            self.encoder: nn.Module | None = FourierFeatures(in_dim, model_config.fourier_bands, model_config.fourier_sigma)
            feat_dim = 2 * model_config.fourier_bands
        else:
            self.encoder = None
            feat_dim = in_dim

        if arch == 'siren':
            n_hidden = max(model_config.hidden_layers - 1, 1)
            siren_layers = [SineLayer(in_dim, model_config.hidden_units, model_config.siren_omega0_first, is_first=True)]
            siren_layers += [
                SineLayer(model_config.hidden_units, model_config.hidden_units, model_config.siren_omega0_hidden, is_first=False)
                for _ in range(n_hidden)
            ]
            self.backbone: nn.Module = nn.Sequential(*siren_layers)
            self.blocks = None
            out_width = model_config.hidden_units
        elif arch in ('residual', 'fourier_residual'):
            self.input_proj = nn.Linear(feat_dim, model_config.hidden_units)
            self.input_act = AdaptiveSwish(model_config.hidden_units) if model_config.use_adaptive_activation else nn.Tanh()
            n_blocks = max(model_config.hidden_layers, 1)
            self.blocks = nn.ModuleList(
                [ResidualBlock(model_config.hidden_units, model_config.use_adaptive_activation) for _ in range(n_blocks)]
            )
            out_width = model_config.hidden_units
        else:
            layers: list[nn.Module] = []
            width = feat_dim
            for _ in range(model_config.hidden_layers):
                layers.append(nn.Linear(width, model_config.hidden_units))
                layers.append(AdaptiveSwish(model_config.hidden_units) if model_config.use_adaptive_activation else nn.Tanh())
                width = model_config.hidden_units
            self.backbone = nn.Sequential(*layers)
            self.blocks = None
            out_width = model_config.hidden_units

        self.output_layer = nn.Linear(out_width, 1)
        self._initialize_weights()
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, -5.0)

    def _initialize_weights(self) -> None:
        skip_ids = set()
        if self.config.arch_type == 'siren':
            for module in self.backbone:
                skip_ids.add(id(module.linear))
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module) not in skip_ids:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, y: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        if y is None and t is None:
            features = x
        elif y is not None and t is not None:
            features = torch.cat((x, y, t), dim=1)
        else:
            raise ValueError("Provide either a single feature tensor or all x, y, t tensors.")

        if features.ndim != 2 or features.shape[1] != 3:
            raise ValueError("Model input must have shape [batch, 3]")
        arch = self.config.arch_type
        if arch == 'siren':
            hidden = self.backbone(features)
        elif arch in ('residual', 'fourier_residual'):
            encoded = self.encoder(features) if self.encoder is not None else features
            hidden = self.input_act(self.input_proj(encoded))
            for block in self.blocks:
                hidden = block(hidden)
        else:
            encoded = self.encoder(features) if self.encoder is not None else features
            hidden = self.backbone(encoded)
        return F.softplus(self.output_layer(hidden), beta=1.0)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# ============================================================
# Physics residual
# ============================================================
def compute_pde_residual(
    features: Tensor, model: nn.Module, u: float, v: float, diffusion: float, reduction: Literal['mean', 'none'] = 'mean', create_graph: bool = True
) -> Tensor:
    features = features.clone().requires_grad_(True)
    concentration = model(features)
    ones = torch.ones_like(concentration)
    gradients = torch.autograd.grad(concentration, features, grad_outputs=ones, create_graph=create_graph, retain_graph=create_graph)[0]
    dC_dx, dC_dy, dC_dt = gradients[:, 0:1], gradients[:, 1:2], gradients[:, 2:3]
    d2C_dx2 = torch.autograd.grad(dC_dx, features, grad_outputs=torch.ones_like(dC_dx), create_graph=create_graph, retain_graph=create_graph)[0][:, 0:1]
    d2C_dy2 = torch.autograd.grad(dC_dy, features, grad_outputs=torch.ones_like(dC_dy), create_graph=create_graph, retain_graph=create_graph)[0][:, 1:2]
    residual = dC_dt + u * dC_dx + v * dC_dy - diffusion * (d2C_dx2 + d2C_dy2)
    squared = residual.square()
    return squared.mean() if reduction == 'mean' else squared


# ============================================================
# Samplers (Sobol)
# ============================================================
def sample_collocation_points(count: int, device: torch.device, engine: SobolEngine) -> Tensor:
    return engine.draw(count).to(device=device, dtype=torch.float32)

def sample_boundary_points(count: int, device: torch.device, engine: SobolEngine) -> Tensor:
    n = max(count // 4, 1)
    free = engine.draw(n * 2).to(device=device, dtype=torch.float32)
    t, other = free[:n, 0:1], free[n:, 0:1]
    side0 = torch.cat([torch.zeros((n, 1), device=device), other, t], dim=1)
    side1 = torch.cat([torch.ones((n, 1), device=device), other, t], dim=1)
    side2 = torch.cat([other, torch.zeros((n, 1), device=device), t], dim=1)
    side3 = torch.cat([other, torch.ones((n, 1), device=device), t], dim=1)
    return torch.cat([side0, side1, side2, side3], dim=0)

def rar_select_points(model: nn.Module, config: TrainingConfig, device: torch.device, engine: SobolEngine) -> Tensor:
    pool_size = config.collocation_points * config.rar_pool_multiplier
    candidates = engine.draw(pool_size).to(device=device, dtype=torch.float32)
    with torch.enable_grad():
        residuals = compute_pde_residual(candidates, model, config.wind_u, config.wind_v, config.diffusion, reduction='none', create_graph=False).detach()
    top_k = max(int(config.collocation_points * config.rar_fraction), 1)
    top_idx = torch.topk(residuals.squeeze(-1), k=min(top_k, pool_size), largest=True).indices
    return candidates[top_idx].detach()


# ============================================================
# Optimization Tools
# ============================================================
class SoftAdaptWeighter:
    def __init__(self, names: list[str], beta: float, floor: float, ceil: float) -> None:
        self.names = names
        self.beta = beta
        self.floor = floor
        self.ceil = ceil
        self.prev: dict[str, float] | None = None

    def compute(self, current: dict[str, float]) -> dict[str, float]:
        if self.prev is None:
            weights = {n: 1.0 for n in self.names}
        else:
            rates = np.array([(current[n] - self.prev[n]) / (abs(self.prev[n]) + 1e-8) for n in self.names])
            shifted = rates - rates.max()
            exp = np.exp(self.beta * shifted)
            raw = np.clip(exp / exp.sum() * len(self.names), self.floor, self.ceil)
            weights = dict(zip(self.names, raw.tolist()))
        self.prev = dict(current)
        return weights


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in unwrap_model(model).state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for k, v in unwrap_model(model).state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        unwrap_model(model).load_state_dict(self.shadow, strict=True)


def build_warmup_cosine_lambda(warmup_epochs: int, total_epochs: int, min_lr_ratio: float):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return fn


def analytic_advection_diffusion_plume(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    source_x: float,
    source_y: float,
    wind_u: float,
    wind_v: float,
    diffusion: float,
    source_strength: float = 1.0,
    initial_spread: float = 0.035,
) -> np.ndarray:
    effective_time = np.maximum(t, 0.0) + initial_spread
    advected_x = source_x + wind_u * t
    advected_y = source_y + wind_v * t
    radial_distance_squared = (x - advected_x) ** 2 + (y - advected_y) ** 2
    denominator = 4.0 * math.pi * diffusion * effective_time
    exponent = -radial_distance_squared / (4.0 * diffusion * effective_time)
    return source_strength * np.exp(exponent) / denominator


def generate_synthetic_sensor_data(
    sector: CitySector,
    config: TrainingConfig,
    device: torch.device,
) -> SyntheticSensorDataset:
    source_x_arr, source_y_arr = lat_lon_to_normalized(sector.source_latitude, sector.source_longitude, sector)
    source_x = float(np.asarray(source_x_arr))
    source_y = float(np.asarray(source_y_arr))

    rows: list[dict[str, Any]] = []
    x_values: list[float] = []
    y_values: list[float] = []
    t_values: list[float] = []
    plume_signal_values: list[float] = []

    sample_times = np.linspace(0.05, 1.0, config.sensor_time_samples, dtype=np.float32)
    for sensor in SENSOR_STATIONS:
        sensor_x, sensor_y = lat_lon_to_normalized(sensor.latitude, sensor.longitude, sector)
        sx = float(np.asarray(sensor_x))
        sy = float(np.asarray(sensor_y))

        for timestamp in sample_times:
            signal = analytic_advection_diffusion_plume(
                np.asarray(sx, dtype=np.float32),
                np.asarray(sy, dtype=np.float32),
                np.asarray(float(timestamp), dtype=np.float32),
                source_x,
                source_y,
                config.wind_u,
                config.wind_v,
                config.diffusion,
            )
            x_values.append(sx)
            y_values.append(sy)
            t_values.append(float(timestamp))
            plume_signal_values.append(float(signal))

    plume = np.asarray(plume_signal_values, dtype=np.float32)
    normalized_signal = plume / max(float(plume.max()), 1.0e-8)
    background_so2_ppb = 7.5
    event_scale_ppb = 180.0
    noise = np.random.normal(loc=0.0, scale=1.65, size=normalized_signal.shape)
    so2_ppb = np.clip(background_so2_ppb + event_scale_ppb * normalized_signal + noise, 0.0, None)
    concentration_scale_ppb = float(np.max(so2_ppb))
    target = (so2_ppb / concentration_scale_ppb).astype(np.float32)

    for i, (x, y, t, so2, c) in enumerate(zip(x_values, y_values, t_values, so2_ppb, target)):
        sensor = SENSOR_STATIONS[i // config.sensor_time_samples]
        rows.append({
            'sensor_id': sensor.sensor_id,
            'latitude': sensor.latitude,
            'longitude': sensor.longitude,
            'x': float(x),
            'y': float(y),
            'elapsed_time': float(t),
            'so2_ppb': float(so2),
            'normalized_concentration': float(c),
        })

    features = torch.tensor(np.column_stack([x_values, y_values, t_values]), dtype=torch.float32, device=device)
    concentration = torch.tensor(target, dtype=torch.float32, device=device).view(-1, 1)

    indices = torch.randperm(len(features))
    split_idx = int(len(features) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    val_features = features[val_idx]
    val_concentration = concentration[val_idx]
    features = features[train_idx]
    concentration = concentration[train_idx]

    LOGGER.info("Generated %d train, %d val synthetic readings from %d sensors", len(train_idx), len(val_idx), len(SENSOR_STATIONS))
    return SyntheticSensorDataset(features, concentration, concentration_scale_ppb, rows, val_features, val_concentration)


def train_inversion_model(
    model: nn.Module,
    dataset: SyntheticSensorDataset,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_warmup_cosine_lambda(config.warmup_epochs, config.epochs, config.min_lr_ratio))
    weighter = SoftAdaptWeighter(['data', 'physics', 'boundary'], config.softadapt_beta, config.softadapt_floor, config.softadapt_ceil)
    ema = EMA(model, config.ema_decay)

    colloc_engine = SobolEngine(dimension=3, scramble=True, seed=config.random_seed)
    bound_engine = SobolEngine(dimension=2, scramble=True, seed=config.random_seed + 1)
    rar_engine = SobolEngine(dimension=3, scramble=True, seed=config.random_seed + 2)

    base_lambdas = {'data': config.lambda_data, 'physics': config.lambda_physics, 'boundary': config.lambda_boundary}
    rar_points: Tensor | None = None
    history: list[dict[str, float]] = []
    adaptive_weights = {'data': 1.0, 'physics': 1.0, 'boundary': 1.0}

    LOGGER.info("Stage 1/2 (AdamW): %d epochs", config.epochs)
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        predicted = model(dataset.features)
        data_loss = F.mse_loss(predicted, dataset.concentration)

        n_random = config.collocation_points if rar_points is None else max(config.collocation_points - rar_points.shape[0], 0)
        random_colloc = sample_collocation_points(n_random, device, colloc_engine)
        collocation = random_colloc if rar_points is None else torch.cat([random_colloc, rar_points], dim=0)
        physics_loss = compute_pde_residual(collocation, model, config.wind_u, config.wind_v, config.diffusion)

        boundary = sample_boundary_points(config.boundary_points, device, bound_engine)
        boundary_loss = model(boundary).square().mean()

        current_losses = {
            'data': float(data_loss.detach().cpu()),
            'physics': float(physics_loss.detach().cpu()),
            'boundary': float(boundary_loss.detach().cpu()),
        }
        if epoch > config.softadapt_warmup_epochs:
            adaptive_weights = weighter.compute(current_losses)
        else:
            weighter.prev = dict(current_losses)
            adaptive_weights = {'data': 1.0, 'physics': 1.0, 'boundary': 1.0}

        total_loss = (
            base_lambdas['data'] * adaptive_weights['data'] * data_loss
            + base_lambdas['physics'] * adaptive_weights['physics'] * physics_loss
            + base_lambdas['boundary'] * adaptive_weights['boundary'] * boundary_loss
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        ema.update(model)
        scheduler.step()

        # Validation tracking & Early Stopping
        val_loss_val = 0.0
        if dataset.val_features is not None:
            with torch.no_grad():
                val_pred = model(dataset.val_features)
                val_loss_val = float(F.mse_loss(val_pred, dataset.val_concentration).cpu())

        history.append(
            {
                'epoch': float(epoch),
                'total_loss': float(total_loss.detach().cpu()),
                'val_loss': val_loss_val,
                'data_loss': current_losses['data'],
                'physics_loss': current_losses['physics'],
                'boundary_loss': current_losses['boundary'],
                'weight_data': float(base_lambdas['data'] * adaptive_weights['data']),
                'weight_physics': float(base_lambdas['physics'] * adaptive_weights['physics']),
                'weight_boundary': float(base_lambdas['boundary'] * adaptive_weights['boundary']),
                'learning_rate': float(scheduler.get_last_lr()[0]),
                'grad_norm': float(grad_norm.detach().cpu() if isinstance(grad_norm, Tensor) else grad_norm),
            }
        )

        if epoch > config.rar_warmup_epochs and epoch % config.rar_interval == 0:
            rar_points = rar_select_points(model, config, device, rar_engine)

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            LOGGER.info(
                "Epoch %04d | total_loss=%.6f | val=%.6f | data=%.6f | physics=%.6f | boundary=%.6f",
                epoch,
                float(total_loss.detach().cpu()),
                val_loss_val,
                current_losses['data'],
                current_losses['physics'],
                current_losses['boundary'],
            )

    LOGGER.info("Stage 2/2 (L-BFGS): %d steps", config.lbfgs_steps)
    if not history:
        history.append(
            {
                'epoch': 0.0,
                'total_loss': math.nan,
                'data_loss': math.nan,
                'physics_loss': math.nan,
                'boundary_loss': math.nan,
                'weight_data': 1.0,
                'weight_physics': 1.0,
                'weight_boundary': 1.0,
                'learning_rate': 0.0,
                'grad_norm': math.nan,
            }
        )

    final_weights = {
        'data': base_lambdas['data'] * adaptive_weights['data'],
        'physics': base_lambdas['physics'] * adaptive_weights['physics'],
        'boundary': base_lambdas['boundary'] * adaptive_weights['boundary'],
    }
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=config.lbfgs_lr, max_iter=config.lbfgs_steps, line_search_fn='strong_wolfe')
    lbfgs_fixed_collocation = sample_collocation_points(config.collocation_points, device, colloc_engine)
    lbfgs_fixed_boundary = sample_boundary_points(config.boundary_points, device, bound_engine)

    def closure() -> Tensor:
        lbfgs.zero_grad(set_to_none=True)
        predicted = model(dataset.features)
        data_loss = F.mse_loss(predicted, dataset.concentration)
        physics_loss = compute_pde_residual(lbfgs_fixed_collocation, model, config.wind_u, config.wind_v, config.diffusion)
        boundary_loss = model(lbfgs_fixed_boundary).square().mean()
        loss = final_weights['data'] * data_loss + final_weights['physics'] * physics_loss + final_weights['boundary'] * boundary_loss
        loss.backward()
        return loss

    if config.lbfgs_steps > 0:
        try:
            lbfgs.step(closure)
            ema.update(model)
        except Exception as e:
            LOGGER.warning("L-BFGS stage failed or encountered NaNs: %s", e)

    # Finally, evaluate the EMA model
    ema.copy_to(model)
    return model, history


def evaluate_source_probability_grid(
    model: nn.Module,
    sector: CitySector,
    device: torch.device,
    grid_size: int = 120,
    source_time: float = 0.0,
) -> SourceProbabilityMap:
    model.eval()
    x_axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    y_axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    t_grid = np.full_like(x_grid, fill_value=source_time, dtype=np.float32)

    x_tensor = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        concentration = model(x_tensor, y_tensor, t_tensor)
        concentration_grid = concentration.detach().cpu().numpy().reshape(grid_size, grid_size)

    baseline = float(np.percentile(concentration_grid, 5.0))
    source_scores = np.clip(concentration_grid - baseline, a_min=0.0, a_max=None)
    source_scores = np.square(source_scores)
    score_sum = float(np.sum(source_scores))
    if score_sum <= 0.0 or not np.isfinite(score_sum):
        LOGGER.warning("Probability scores were degenerate; returning a uniform map.")
        probabilities = np.full_like(source_scores, 1.0 / source_scores.size)
    else:
        probabilities = source_scores / score_sum

    latitudes, longitudes = normalized_to_lat_lon(x_grid, y_grid, sector)
    return SourceProbabilityMap(
        latitudes=latitudes,
        longitudes=longitudes,
        probabilities=probabilities.astype(np.float32),
    )


def estimate_source_from_probability_map(
    probability_map: SourceProbabilityMap,
) -> tuple[float, float, float]:
    max_index = np.unravel_index(
        int(np.argmax(probability_map.probabilities)),
        probability_map.probabilities.shape,
    )
    return (
        float(probability_map.latitudes[max_index]),
        float(probability_map.longitudes[max_index]),
        float(probability_map.probabilities[max_index]),
    )


def save_probability_map(
    probability_map: SourceProbabilityMap,
    output_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        latitudes=probability_map.latitudes,
        longitudes=probability_map.longitudes,
        probabilities=probability_map.probabilities,
    )
    LOGGER.info("Saved source probability map to %s", output_path)


@dataclass(frozen=True)
class SourceUncertainty:
    """Uncertainty quantification for a PINN source estimate."""
    confidence: float          # 0-1, higher = more peaked probability surface
    stddev_meters: float       # spatial spread of probability mass around peak
    peak_probability: float    # raw max probability value
    effective_cells: float     # inverse participation ratio (1 = perfect peak)


def compute_source_uncertainty(
    probability_map: SourceProbabilityMap,
) -> SourceUncertainty:
    """Derive confidence and spatial standard deviation from a probability map.

    Confidence is based on the *inverse participation ratio* (IPR):
        IPR = 1 / sum(p_i^2)
    For a perfectly peaked map (all mass on one cell), IPR = 1.
    For a uniform map over N cells, IPR = N.
    We normalize:  confidence = 1 - (IPR - 1) / (N - 1), clipped to [0, 1].

    The spatial standard deviation is the probability-weighted RMS distance
    (in meters) from the peak coordinate.
    """
    probs = probability_map.probabilities.ravel().astype(np.float64)
    lats = probability_map.latitudes.ravel().astype(np.float64)
    lons = probability_map.longitudes.ravel().astype(np.float64)
    n_cells = len(probs)

    # --- Effective cells (IPR) ---
    sum_p2 = float(np.sum(probs ** 2))
    if sum_p2 <= 0.0 or not np.isfinite(sum_p2):
        return SourceUncertainty(
            confidence=0.0,
            stddev_meters=float("inf"),
            peak_probability=0.0,
            effective_cells=float(n_cells),
        )

    ipr = 1.0 / sum_p2
    # Normalize to [0, 1]: 1 when ipr=1 (perfect peak), 0 when ipr=N (uniform)
    confidence = max(0.0, min(1.0, 1.0 - (ipr - 1.0) / max(n_cells - 1.0, 1.0)))

    # --- Peak ---
    peak_idx = int(np.argmax(probs))
    peak_lat = lats[peak_idx]
    peak_lon = lons[peak_idx]
    peak_prob = float(probs[peak_idx])

    # --- Spatial stddev (meters) ---
    # Approximate meter offsets from peak using simple lat/lon scaling
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(peak_lat))
    dy = (lats - peak_lat) * meters_per_deg_lat
    dx = (lons - peak_lon) * meters_per_deg_lon
    dist_sq = dx ** 2 + dy ** 2
    weighted_var = float(np.sum(probs * dist_sq))
    stddev_m = math.sqrt(max(weighted_var, 0.0))

    LOGGER.info(
        "Source UQ: confidence=%.4f stddev=%.1f m effective_cells=%.1f peak_prob=%.8f",
        confidence, stddev_m, ipr, peak_prob,
    )
    return SourceUncertainty(
        confidence=confidence,
        stddev_meters=round(stddev_m, 1),
        peak_probability=peak_prob,
        effective_cells=round(ipr, 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlumeTrace PINN source inversion model.")
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs.")
    parser.add_argument("--collocation-points", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.10)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("plumetrace_source_probability_map.npz"))
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Path to a .pt checkpoint to load (skips training).")
    parser.add_argument("--save-checkpoint", type=Path, default=None, help="Path to save the trained .pt checkpoint.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_data=args.lambda_data,
        lambda_physics=args.lambda_physics,
        collocation_points=args.collocation_points,
        random_seed=args.seed,
    )
    model_config = ModelConfig()

    set_reproducibility(config.random_seed)
    device = get_device()
    sector = CitySector()

    try:
        model = PlumeInversionPINN(model_config).to(device)

        if args.load_checkpoint and args.load_checkpoint.exists():
            LOGGER.info("Loading model checkpoint from %s (bypassing training)", args.load_checkpoint)
            state_dict = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
        else:
            dataset = generate_synthetic_sensor_data(sector, config, device)
            model, history = train_inversion_model(model, dataset, config, device)
            
            if args.save_checkpoint:
                LOGGER.info("Saving trained model to %s", args.save_checkpoint)
                args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict()}, args.save_checkpoint)

        probability_map = evaluate_source_probability_grid(
            model,
            sector,
            device,
            grid_size=args.grid_size,
            source_time=0.0,
        )
        predicted_latitude, predicted_longitude, predicted_probability = estimate_source_from_probability_map(probability_map)
        save_probability_map(probability_map, args.output)

        LOGGER.info("Known source latitude=%.6f longitude=%.6f", sector.source_latitude, sector.source_longitude)
        LOGGER.info("Estimated source latitude=%.6f longitude=%.6f probability=%.8f", predicted_latitude, predicted_longitude, predicted_probability)
        
    except Exception as exc:
        LOGGER.exception("PINN inversion run failed: %s", exc)
        raise

if __name__ == "__main__":
    main()
