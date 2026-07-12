"""Optuna hyperparameter optimization script for PlumeTrace PINN model."""

import json
import logging
import os
import sys
from pathlib import Path
import optuna

# Add project root to python path to support relative imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pinn.config import ModelConfig, TrainingConfig, CitySector
from pinn.utils import seed_everything
from pinn_engine import (
    PlumeInversionPINN,
    train_inversion_model,
    generate_synthetic_sensor_data,
    evaluate_source_probability_grid,
    estimate_source_from_probability_map,
)
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("plumetrace.pinn.tune")


def objective(trial: optuna.Trial) -> float:
    # 1. Suggest hyperparameters
    hidden_layers = trial.suggest_int("hidden_layers", 4, 10)
    hidden_units = trial.suggest_categorical("hidden_units", [64, 128, 256])
    fourier_bands = trial.suggest_categorical("fourier_bands", [4, 8, 16])
    fourier_sigma = trial.suggest_float("fourier_sigma", 0.5, 6.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    
    # 2. Build configs
    model_config = ModelConfig(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        fourier_bands=fourier_bands,
        fourier_sigma=fourier_sigma,
    )
    
    # Run a short training loop (50 epochs) for swift evaluation
    train_config = TrainingConfig(
        learning_rate=learning_rate,
        random_seed=2026,
    )
    
    # 3. Setup device and reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(train_config.random_seed, deterministic=True)
    
    # 4. Generate data and instantiate model
    sector = CitySector()
    dataset = generate_synthetic_sensor_data(sector, train_config, device)
    model = PlumeInversionPINN(model_config).to(device)
    
    # 5. Train model
    try:
        model, history = train_inversion_model(model, dataset, train_config, device, epochs_override=50)
        
        # Evaluate validation loss
        val_loss = history[-1]["val_loss"]
        
        # Evaluate localized peak error (distance in meters)
        prob_map = evaluate_source_probability_grid(model, sector, device, grid_size=60)
        pred_lat, pred_lon, _ = estimate_source_from_probability_map(prob_map)
        
        # Geodesic error offset
        from pinn.utils import haversine_m
        dist_err = haversine_m(sector.source_latitude, sector.source_longitude, pred_lat, pred_lon)
        
        # Objective combines normalized validation loss and localization error
        score = float(val_loss * 100.0 + dist_err / 100.0)
        
        logger.info(
            "Trial %d finished | Score=%.4f (Val Loss=%.6f, Dist Err=%.1fm) | Config: %s",
            trial.number,
            score,
            val_loss,
            dist_err,
            trial.params,
        )
        return score
        
    except Exception as exc:
        logger.warning("Trial %d failed due to numerical instability: %s", trial.number, exc)
        return float("inf")


def main() -> None:
    logger.info("Starting PlumeTrace PINN Optuna hyperparameter study...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, timeout=600)  # Run 5 quick trials
    
    logger.info("Study completed successfully!")
    logger.info("Best trial value: %.6f", study.best_value)
    logger.info("Best parameters: %s", study.best_params)
    
    # Save best parameters to disk
    output_path = Path(__file__).resolve().parent / "pinn_optuna_best_params.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Saved best study parameters to %s", output_path)


if __name__ == "__main__":
    main()
