"""
PlumeTrace PINN - Utilities
Reproducibility, EMA, checkpointing, logging, metrics, and helper functions.
"""
from __future__ import annotations

from collections import deque
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn

LOGGER = logging.getLogger("plumetrace.pinn")


# ── Reproducibility ─────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.
    FIX: Disable cudnn.benchmark to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # FIX: was True, caused non-determinism
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def get_device(local_rank: int = 0) -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)
    return device


# ── Model Helpers ─────────────────────────────────────────────────

def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DistributedDataParallel / Compiled models."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── EMA (Exponential Moving Average) ──────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters and buffers.
    Maintains a shadow copy of state dict that updates slowly.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in unwrap_model(model).state_dict().items()}

    def update(self, model: nn.Module) -> None:
        """Update shadow state dict with EMA formula."""
        for k, v in unwrap_model(model).state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA state dict into the model."""
        unwrap_model(model).load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> Dict[str, Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ── SoftAdapt Dynamic Weighting ───────────────────────────────────

class SoftAdaptWeighter:
    """
    SoftAdapt dynamic loss weighting (Heydari et al., 2019).
    Reweights loss components based on their rate of change over a rolling window.
    """
    def __init__(self, names: list[str], beta: float, floor: float, ceil: float, window_size: int = 20) -> None:
        self.names = names
        self.beta = beta
        self.floor = floor
        self.ceil = ceil
        self.window_size = max(2, window_size)
        self.history: deque[dict[str, float]] = deque(maxlen=self.window_size)

    def compute(self, current: dict[str, float]) -> dict[str, float]:
        self.history.append(dict(current))
        if len(self.history) < 2:
            return {n: 1.0 for n in self.names}
        
        first = self.history[0]
        last = self.history[-1]
        
        rates = []
        for n in self.names:
            first_val = first[n]
            last_val = last[n]
            rate = (last_val - first_val) / (abs(first_val) + 1e-8)
            rates.append(rate)
            
        rates = np.array(rates)
        shifted = rates - rates.max()
        exp = np.exp(self.beta * shifted)
        raw = np.clip(exp / exp.sum() * len(self.names), self.floor, self.ceil)
        return dict(zip(self.names, raw.tolist()))


# ── Checkpointing ─────────────────────────────────────────────────

class CheckpointManager:
    """Handles saving and loading model checkpoints with automatic resume."""

    def __init__(self, output_dir: Path, keep_best: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_score = float('inf')
        self.checkpoint_path = self.output_dir / "latest_checkpoint.pt"
        self.best_path = self.output_dir / "best_model.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        ema: Optional[EMA],
        epoch: int,
        score: float,
        config: Optional[Any] = None,
    ) -> None:
        """Save a checkpoint with all training state."""
        state = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(model).state_dict(),
            'score': score,
        }
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if ema is not None:
            state['ema_state_dict'] = ema.state_dict()
        if config is not None:
            from dataclasses import asdict
            state['config'] = asdict(config)

        torch.save(state, self.checkpoint_path)
        LOGGER.info("Checkpoint saved at epoch %d (score: %.6f)", epoch, score)

        if self.keep_best and score < self.best_score:
            self.best_score = score
            torch.save(state, self.best_path)
            LOGGER.info("New best model saved (score: %.6f)", score)

    def load(self, model: nn.Module, device: torch.device) -> Optional[Dict[str, Any]]:
        """Load checkpoint and return state dict if found."""
        if self.checkpoint_path.exists():
            LOGGER.info("Resuming from checkpoint: %s", self.checkpoint_path)
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            return checkpoint
        return None

    def load_best(self, model: nn.Module, device: torch.device) -> bool:
        """Load best model if it exists."""
        if self.best_path.exists():
            checkpoint = torch.load(self.best_path, map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            LOGGER.info("Loaded best model from %s", self.best_path)
            return True
        return False


# ── Metrics ────────────────────────────────────────────────────────

def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute R² coefficient of determination."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(pred - true)))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two lat/lon points in meters."""
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to a valid probability distribution."""
    clipped = np.clip(scores.astype(np.float64), 0.0, None)
    total = float(clipped.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


# ── Coordinate Transforms ──────────────────────────────────────────

def lat_lon_to_normalized(
    latitude: float | np.ndarray,
    longitude: float | np.ndarray,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude to normalized [0,1] coordinates."""
    lon_arr = np.asarray(longitude, dtype=np.float32)
    lat_arr = np.asarray(latitude, dtype=np.float32)
    x = (lon_arr - lon_min) / (lon_max - lon_min)
    y = (lat_arr - lat_min) / (lat_max - lat_min)
    return x.astype(np.float32), y.astype(np.float32)


def normalized_to_lat_lon(
    x: np.ndarray,
    y: np.ndarray,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized [0,1] coordinates back to latitude/longitude."""
    longitude = lon_min + x * (lon_max - lon_min)
    latitude = lat_min + y * (lat_max - lat_min)
    return latitude.astype(np.float32), longitude.astype(np.float32)


# ── Timer Context Manager ─────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name: str = "Operation") -> None:
        self.name = name
        self.start: Optional[float] = None

    def __enter__(self) -> Timer:
        self.start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.time() - self.start
        LOGGER.info("%s completed in %.3f seconds", self.name, elapsed)


# ── JSON Logger for Experiment Tracking ───────────────────────────

class ExperimentLogger:
    """Log hyperparameters and results to JSON for reproducibility."""
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "experiment_log.json"
        self.records: list[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        self.log_path.write_text(json.dumps(self.records, indent=2), encoding='utf-8')
