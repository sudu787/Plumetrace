"""
PlumeTrace PINN - Configuration System
Research-grade configuration using dataclasses with full type hints.
All hyperparameters are centralized here for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Tuple


@dataclass(frozen=True)
class SensorStation:
    """Deterministic virtual sensor location inside the industrial sector."""
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
    """Neural network architecture configuration."""
    arch_type: Literal[
        'mlp', 'residual', 'fourier_residual', 'siren', 'multiscale'
    ] = 'fourier_residual'
    hidden_units: int = 128
    hidden_layers: int = 9          # 9-layer deep network (was 8, now odd for proper skip pairing)
    fourier_features: int = 256     # Fourier feature dimension (was 128 in script, now 256)
    fourier_bands: int = 128        # Number of Fourier frequency bands
    fourier_sigma: float = 1.0      # Gaussian std for Fourier frequency sampling
    siren_omega0_first: float = 30.0
    siren_omega0_hidden: float = 30.0
    use_adaptive_activation: bool = True
    use_layer_norm: bool = True
    use_attention: bool = False     # Optional attention block after residual layers
    dropout: float = 0.0
    # FIX: Gradient checkpointing properly implemented via config flag
    use_grad_checkpoint: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and physics settings for the inversion experiment."""
    # ── Epochs & Stages ─────────────────────────────────────────────
    adamw_epochs: int = 2000
    lbfgs_epochs: int = 100
    lbfgs_max_iter: int = 20        # Internal L-BFGS iterations per step
    lbfgs_lr: float = 0.5
    lbfgs_history_size: int = 50

    # ── Learning Rate ────────────────────────────────────────────────
    learning_rate: float = 2.0e-3
    warmup_epochs: int = 100
    min_lr_ratio: float = 0.01    # Cosine annealing minimum ratio
    use_reduce_lr_on_plateau: bool = True
    reduce_lr_patience: int = 150
    reduce_lr_factor: float = 0.5

    # ── Loss Weights (Base; adaptive methods override dynamically) ──
    lambda_data: float = 1.0
    lambda_physics: float = 0.10
    lambda_boundary: float = 0.05
    lambda_initial: float = 0.0   # NEW: Initial condition loss weight
    lambda_source: float = 0.0      # NEW: Source localization prior loss

    # ── Adaptive Weighting ────────────────────────────────────────────
    adapt_every: int = 25
    adapt_method: Literal['softadapt', 'gradnorm', 'ntk', 'fixed'] = 'softadapt'
    softadapt_beta: float = 0.1
    softadapt_floor: float = 0.05
    softadapt_ceil: float = 5.0
    softadapt_warmup_epochs: int = 20
    gradnorm_alpha: float = 1.5

    # ── Architecture ────────────────────────────────────────────────
    hidden_layers: int = 9
    hidden_units: int = 128
    fourier_features: int = 256
    fourier_sigma: float = 1.0
    use_grad_checkpoint: bool = True
    gradient_clip_norm: float = 5.0

    # ── Sampling Points ─────────────────────────────────────────────
    collocation_points: int = 8192
    boundary_points: int = 1024
    initial_points: int = 512     # NEW: Points for t=0 initial condition
    source_prior_points: int = 256  # NEW: Points for source localization prior

    # ── RAR (Residual Adaptive Refinement) ──────────────────────────
    rar_every: int = 200
    rar_eval_points: int = 50000
    rar_add_points: int = 1000
    rar_max_points: int = 32000
    rar_fraction: float = 0.25
    rar_pool_multiplier: int = 4
    rar_warmup_epochs: int = 200

    # ── Logging & Checkpointing ───────────────────────────────────
    log_every: int = 50
    checkpoint_every: int = 500
    early_stop_patience: int = 300
    ema_decay: float = 0.999
    tensorboard_dir: str = './tensorboard'
    output_dir: str = './outputs'

    # ── Grid & Evaluation ─────────────────────────────────────────
    grid_size: int = 220
    eval_time_samples: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    probability_temperature: float = 0.010

    # ── Physics Base ────────────────────────────────────────────────
    wind_u: float = 0.16
    wind_v: float = -0.06
    diffusion: float = 0.006
    sensor_time_samples: int = 96
    random_seed: int = 2026

    # ── Data Augmentation ───────────────────────────────────────────
    # NEW: Randomized physics parameters for curriculum learning
    augment_wind_std: float = 0.03
    augment_diffusion_std: float = 0.001
    augment_diffusion_min: float = 0.004
    augment_diffusion_max: float = 0.010
    sensor_noise_std: float = 1.65
    background_so2_ppb: float = 7.5
    event_scale_ppb: float = 180.0
    missing_sensor_prob: float = 0.0   # Probability of dropping a sensor reading
    temporal_jitter_std: float = 0.0     # Std of time perturbation

    # ── Boundary Conditions ─────────────────────────────────────────
    bc_type: Literal['dirichlet', 'neumann', 'robin', 'mixed'] = 'dirichlet'
    bc_dirichlet_value: float = 0.0
    bc_robin_alpha: float = 1.0
    bc_robin_beta: float = 0.0
    hard_bc_enforcement: bool = False    # Transform output to satisfy BCs exactly

    # ── Multi-GPU ────────────────────────────────────────────────────
    use_data_parallel: bool = False
    use_distributed: bool = False
    local_rank: int = 0

    # ── Mixed Precision ─────────────────────────────────────────────
    use_amp: bool = True
    amp_dtype: Literal['float16', 'bfloat16'] = 'bfloat16'

    # ── Reproducibility ─────────────────────────────────────────────
    deterministic: bool = True
    cudnn_benchmark: bool = False   # FIX: Disabled to ensure reproducibility

    # ── Curriculum Learning ─────────────────────────────────────────
    curriculum_epochs: int = 500    # Epochs before full complexity
    curriculum_noise_max: float = 0.5  # Noise multiplier during curriculum

    # ── Ensemble / Uncertainty ────────────────────────────────────────
    mc_dropout_samples: int = 0     # 0 = disabled
    ensemble_size: int = 1          # 1 = disabled

    # ── Validation ────────────────────────────────────────────────────
    val_split: float = 0.2


# ── Default Sensor Stations ─────────────────────────────────────────
SENSOR_STATIONS: Tuple[SensorStation, ...] = (
    SensorStation('industrial_north', 40.7180, -74.0060),
    SensorStation('residential_east', 40.7140, -73.9980),
    SensorStation('park_south', 40.7080, -74.0040),
    SensorStation('river_west', 40.7120, -74.0120),
)

SECTOR = CitySector()
