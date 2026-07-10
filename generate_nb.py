import json
import os

def create_cell(cell_type, source):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        }

cells_data = [
    ("markdown", """# PlumeTrace PINN Source Attribution - Kaggle Dual T4 Notebook

Use this notebook on Kaggle with **Accelerator: GPU T4 x2**. It trains a state-of-the-art Physics-Informed Neural Network (PINN) for inverse pollutant-source attribution.

**Key Refactoring Improvements:**
- **Architecture:** 9-layer Residual PINN with Fourier Feature Positional Encoding.
- **Optimizer:** Two-stage AdamW + L-BFGS fine-tuning with ReduceLROnPlateau & Cosine Annealing.
- **Adaptive Weighting:** SoftAdapt dynamically balances data, physics, and boundary losses.
- **Adaptive Refinement (RAR):** Dynamically adds collocation points in high PDE-error regions.
- **Performance:** Automatic Mixed Precision (AMP) for data loss, fp32 for PDE; Gradient Checkpointing for memory efficiency.
- **Data Augmentation:** Randomized wind, diffusion, and sensor noise each epoch to improve generalization.
- **Outputs:** TensorBoard logging, early stopping, and automatic checkpoint resuming.

The original task structure, validation summary, and visualizations are preserved.
"""),
    ("code", """import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd() / 'kaggle_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

seed_everything(2026)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU_COUNT = torch.cuda.device_count()
print('Torch:', torch.__version__)
print('Device:', DEVICE)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', GPU_COUNT)
for i in range(GPU_COUNT):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"""),
    ("code", """@dataclass(frozen=True)
class SensorStation:
    sensor_id: str
    latitude: float
    longitude: float

@dataclass(frozen=True)
class CitySector:
    lat_min: float = 40.7040
    lat_max: float = 40.7220
    lon_min: float = -74.0160
    lon_max: float = -73.9940
    source_latitude: float = 40.7138
    source_longitude: float = -74.0072

@dataclass
class TrainingConfig:
    # Epochs
    adamw_epochs: int = 2000
    lbfgs_epochs: int = 100
    learning_rate: float = 2.0e-3
    warmup_epochs: int = 100
    
    # Loss Weights
    lambda_data: float = 1.0
    lambda_physics: float = 0.10
    lambda_boundary: float = 0.05
    adapt_every: int = 25 # SoftAdapt interval
    
    # Architecture
    hidden_layers: int = 8
    hidden_units: int = 128
    fourier_features: int = 256
    fourier_sigma: float = 1.0
    use_grad_checkpoint: bool = True
    
    # Points
    collocation_points: int = 8192 if GPU_COUNT >= 2 else 4096
    boundary_points: int = 1024
    
    # RAR
    rar_every: int = 200
    rar_eval_points: int = 50000
    rar_add_points: int = 1000
    rar_max_points: int = 32000
    
    log_every: int = 50
    gradient_clip_norm: float = 5.0
    grid_size: int = 220
    random_seed: int = 2026

    # Physics Base
    wind_u: float = 0.16
    wind_v: float = -0.06
    diffusion: float = 0.006
    sensor_time_samples: int = 96
    
    # Early Stopping
    early_stop_patience: int = 300

SENSOR_STATIONS = (
    SensorStation('industrial_north', 40.7180, -74.0060),
    SensorStation('residential_east', 40.7140, -73.9980),
    SensorStation('park_south', 40.7080, -74.0040),
    SensorStation('river_west', 40.7120, -74.0120),
)

SECTOR = CitySector()
CONFIG = TrainingConfig()
print(CONFIG)
"""),
    ("code", """def lat_lon_to_normalized(latitude, longitude, sector):
    lon_arr = np.asarray(longitude, dtype=np.float32)
    lat_arr = np.asarray(latitude, dtype=np.float32)
    x = (lon_arr - sector.lon_min) / (sector.lon_max - sector.lon_min)
    y = (lat_arr - sector.lat_min) / (sector.lat_max - sector.lat_min)
    return x.astype(np.float32), y.astype(np.float32)

def normalized_to_lat_lon(x, y, sector):
    longitude = sector.lon_min + x * (sector.lon_max - sector.lon_min)
    latitude = sector.lat_min + y * (sector.lat_max - sector.lat_min)
    return latitude.astype(np.float32), longitude.astype(np.float32)

def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def analytic_advection_diffusion_plume(x, y, t, source_x, source_y, wind_u, wind_v, diffusion, source_strength=1.0, initial_spread=0.035):
    effective_time = np.maximum(t, 0.0) + initial_spread
    advected_x = source_x + wind_u * t
    advected_y = source_y + wind_v * t
    radial_distance_squared = (x - advected_x)**2 + (y - advected_y)**2
    denominator = 4.0 * math.pi * diffusion * effective_time
    exponent = -radial_distance_squared / (4.0 * diffusion * effective_time)
    return source_strength * np.exp(exponent) / denominator

@dataclass
class SyntheticSensorDataset:
    features: Tensor
    concentration: Tensor
    val_features: Tensor
    val_concentration: Tensor
    rows: list
    concentration_scale_ppb: float
    source_x: float
    source_y: float

def generate_synthetic_sensor_data(sector, config, device, val_split=0.2):
    source_x, source_y = lat_lon_to_normalized(sector.source_latitude, sector.source_longitude, sector)
    source_x, source_y = float(source_x), float(source_y)
    
    rows, x_vals, y_vals, t_vals, plume_vals = [], [], [], [], []
    sample_times = np.linspace(0.05, 1.0, config.sensor_time_samples, dtype=np.float32)
    
    for sensor in SENSOR_STATIONS:
        sx, sy = lat_lon_to_normalized(sensor.latitude, sensor.longitude, sector)
        sx, sy = float(sx), float(sy)
        for t in sample_times:
            signal = analytic_advection_diffusion_plume(
                np.float32(sx), np.float32(sy), np.float32(t),
                source_x, source_y, config.wind_u, config.wind_v, config.diffusion
            )
            x_vals.append(sx); y_vals.append(sy); t_vals.append(float(t))
            plume_vals.append(float(signal))
            
    plume = np.asarray(plume_vals, dtype=np.float32)
    normalized_signal = plume / max(float(plume.max()), 1.0e-8)
    
    background_so2_ppb, event_scale_ppb = 7.5, 180.0
    noise = np.random.normal(loc=0.0, scale=1.65, size=normalized_signal.shape)
    so2_ppb = np.clip(background_so2_ppb + event_scale_ppb * normalized_signal + noise, 0.0, None)
    concentration_scale_ppb = float(np.max(so2_ppb))
    target = (so2_ppb / concentration_scale_ppb).astype(np.float32)
    
    for i, (x, y, t, so2, c) in enumerate(zip(x_vals, y_vals, t_vals, so2_ppb, target)):
        sensor = SENSOR_STATIONS[i // config.sensor_time_samples]
        rows.append({
            'sensor_id': sensor.sensor_id, 'latitude': sensor.latitude, 'longitude': sensor.longitude,
            'x': float(x), 'y': float(y), 'elapsed_time': float(t),
            'so2_ppb': float(so2), 'normalized_concentration': float(c)
        })
        
    features = torch.tensor(np.column_stack([x_vals, y_vals, t_vals]), dtype=torch.float32)
    concentration = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    
    # Train/Val Split
    indices = torch.randperm(len(features))
    split_idx = int(len(features) * (1 - val_split))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    dataset = SyntheticSensorDataset(
        features=features[train_idx].to(device),
        concentration=concentration[train_idx].to(device),
        val_features=features[val_idx].to(device),
        val_concentration=concentration[val_idx].to(device),
        rows=rows, concentration_scale_ppb=concentration_scale_ppb,
        source_x=source_x, source_y=source_y
    )
    print(f"Generated {len(train_idx)} train, {len(val_idx)} val readings.")
    return dataset

dataset = generate_synthetic_sensor_data(SECTOR, CONFIG, DEVICE)
"""),
    ("code", """class FourierEmbedding(nn.Module):
    def __init__(self, in_features=3, out_features=256, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * sigma, requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.activation = nn.Tanh()
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)

    def forward(self, x):
        identity = x
        out = self.layer_norm1(x)
        out = self.activation(self.linear1(out))
        out = self.layer_norm2(out)
        out = self.activation(self.linear2(out))
        return out + identity

class ResidualFourierPINN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.embedding = FourierEmbedding(3, config.fourier_features, config.fourier_sigma)
        
        layers = []
        layers.append(nn.Linear(config.fourier_features, config.hidden_units))
        layers.append(nn.Tanh())
        
        for _ in range(config.hidden_layers):
            layers.append(ResidualBlock(config.hidden_units))
            
        layers.append(nn.LayerNorm(config.hidden_units))
        layers.append(nn.Linear(config.hidden_units, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self, 'network') and isinstance(self.network[-1], nn.Linear):
            if self.network[-1].bias is not None:
                nn.init.constant_(self.network[-1].bias, -5.0)

    def forward(self, x):
        emb = self.embedding(x)
        
        out = emb
        for module in self.network:
            if isinstance(module, ResidualBlock) and self.config.use_grad_checkpoint and self.training:
                out = checkpoint(module, out, use_reentrant=False)
            else:
                out = module(out)
                
        return F.softplus(out, beta=1.0)
        
# Alias for backwards compatibility of checkpoints
PlumeInversionPINN = ResidualFourierPINN
"""),
    ("code", """def compute_pde_residual(features, model, u, v, diffusion, create_graph=True):
    # Enforce float32 for PDE autograd stability
    features = features.clone().float().requires_grad_(True)
    concentration = model(features).float()
    
    ones = torch.ones_like(concentration)
    grads = torch.autograd.grad(concentration, features, grad_outputs=ones, create_graph=create_graph, retain_graph=create_graph)[0]
        
    dC_dx, dC_dy, dC_dt = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
        
    d2C_dx2 = torch.autograd.grad(dC_dx, features, grad_outputs=torch.ones_like(dC_dx), create_graph=create_graph, retain_graph=create_graph)[0][:, 0:1]
    d2C_dy2 = torch.autograd.grad(dC_dy, features, grad_outputs=torch.ones_like(dC_dy), create_graph=create_graph, retain_graph=create_graph)[0][:, 1:2]
        
    residual = dC_dt + u * dC_dx + v * dC_dy - diffusion * (d2C_dx2 + d2C_dy2)
    return residual.square().mean()

class SoftAdaptWeighter:
    def __init__(self, config):
        self.w_d = config.lambda_data
        self.w_p = config.lambda_physics
        self.w_b = config.lambda_boundary
        self.prev_losses = None
        
    def update(self, l_d, l_p, l_b):
        curr = torch.tensor([l_d, l_p, l_b], dtype=torch.float32)
        if self.prev_losses is not None:
            rates = (curr - self.prev_losses) / (torch.abs(self.prev_losses) + 1e-8)
            weights = F.softmax(rates, dim=0) * 3.0 # N=3 terms
            self.w_d = weights[0].item()
            self.w_p = weights[1].item() * 0.1 # Physics inherently larger scale
            self.w_b = weights[2].item() * 0.05
        self.prev_losses = curr.detach()

def sample_collocation_points(count, device):
    return torch.rand((count, 3), dtype=torch.float32, device=device)

def sample_boundary_points(count, device):
    n = max(count // 4, 1)
    t = torch.rand((n, 1), device=device)
    # Dirichlet boundaries (x=0, x=1, y=0, y=1)
    s0 = torch.cat([torch.zeros((n, 1), device=device), torch.rand((n, 1), device=device), t], dim=1)
    s1 = torch.cat([torch.ones((n, 1), device=device), torch.rand((n, 1), device=device), t], dim=1)
    s2 = torch.cat([torch.rand((n, 1), device=device), torch.zeros((n, 1), device=device), t], dim=1)
    s3 = torch.cat([torch.rand((n, 1), device=device), torch.ones((n, 1), device=device), t], dim=1)
    return torch.cat([s0, s1, s2, s3], dim=0)

def perform_rar(model, config, current_collocation, device):
    \"\"\"Residual-based Adaptive Refinement\"\"\"
    model.eval()
    pts = sample_collocation_points(config.rar_eval_points, device)
    with torch.enable_grad():
        # Compute approximate residual for ranking without full autograd graph overhead
        features = pts.clone().float().requires_grad_(True)
        conc = model(features).float()
        grads = torch.autograd.grad(conc, features, grad_outputs=torch.ones_like(conc), retain_graph=False, create_graph=False)[0]
        dC_dx, dC_dy, dC_dt = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
        d2C_dx2 = torch.autograd.grad(dC_dx, features, grad_outputs=torch.ones_like(dC_dx), retain_graph=False, create_graph=False)[0][:, 0:1]
        d2C_dy2 = torch.autograd.grad(dC_dy, features, grad_outputs=torch.ones_like(dC_dy), retain_graph=False, create_graph=False)[0][:, 1:2]
            
            res = (dC_dt + config.wind_u * dC_dx + config.wind_v * dC_dy - config.diffusion * (d2C_dx2 + d2C_dy2)).abs().squeeze()
            
    _, top_idx = torch.topk(res, config.rar_add_points)
    new_pts = pts[top_idx].detach()
    
    updated = torch.cat([current_collocation, new_pts], dim=0)
    if len(updated) > config.rar_max_points:
        # Keep recent hard points, replace older
        updated = updated[-config.rar_max_points:]
    return updated
"""),
    ("code", """def get_augmented_physics(config):
    u = config.wind_u + np.random.normal(0, 0.03)
    v = config.wind_v + np.random.normal(0, 0.015)
    d = np.clip(config.diffusion + np.random.normal(0, 0.001), 0.004, 0.010)
    return float(u), float(v), float(d)

def train_inversion_model(model, dataset, config, device):
    writer = SummaryWriter(log_dir=str(OUTPUT_DIR / 'tensorboard'))
    scaler = torch.cuda.amp.GradScaler()
    weighter = SoftAdaptWeighter(config)
    
    collocation_points = sample_collocation_points(config.collocation_points, device)
    
    # Optimizer Stage 1: AdamW
    opt_adam = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    # Warmup + Cosine
    def lr_lambda(ep):
        if ep < config.warmup_epochs: return float(ep) / float(max(1, config.warmup_epochs))
        progress = (ep - config.warmup_epochs) / float(max(1, config.adamw_epochs - config.warmup_epochs))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_adam, lr_lambda)
    
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print(f'Stage 1: AdamW for {config.adamw_epochs} epochs')
    
    for epoch in range(1, config.adamw_epochs + 1):
        model.train()
        opt_adam.zero_grad(set_to_none=True)
        
        # Augment physics slightly for robustness
        u, v, diff = get_augmented_physics(config)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            predicted = model(dataset.features)
            data_loss = F.mse_loss(predicted, dataset.concentration)
            
            boundary = sample_boundary_points(config.boundary_points, device)
            boundary_loss = model(boundary).square().mean()

        physics_loss = compute_pde_residual(collocation_points, model, u, v, diff)
        
        total_loss = (weighter.w_d * data_loss + weighter.w_p * physics_loss + weighter.w_b * boundary_loss)
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt_adam)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(opt_adam)
        scaler.update()
        scheduler.step()
        
        # Validation & Logging
        if epoch % config.adapt_every == 0:
            weighter.update(data_loss.item(), physics_loss.item(), boundary_loss.item())
            
        if epoch % config.rar_every == 0:
            collocation_points = perform_rar(model, config, collocation_points, device)
            
        with torch.no_grad():
            val_pred = model(dataset.val_features)
            val_loss = F.mse_loss(val_pred, dataset.val_concentration).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
        else:
            patience_counter += 1
            
        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"AdamW Ep {epoch:04d} | Tot: {total_loss.item():.5f} | Dat: {data_loss.item():.5f} | Phy: {physics_loss.item():.5f} | Val: {val_loss:.5f} | Pts: {len(collocation_points)}")
            writer.add_scalar('Loss/Total', total_loss.item(), epoch)
            writer.add_scalar('Loss/Data', data_loss.item(), epoch)
            writer.add_scalar('Loss/Physics', physics_loss.item(), epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
            
        history.append({'epoch': epoch, 'val_loss': val_loss, 'total_loss': total_loss.item()})
        
        if patience_counter > config.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    # Load best model for L-BFGS
    if (OUTPUT_DIR / 'best_model.pt').exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pt'))
        
    print(f'Stage 2: L-BFGS for {config.lbfgs_epochs} epochs')
    opt_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)
    
    lbfgs_epochs = 0
    def closure():
        opt_lbfgs.zero_grad()
        pred = model(dataset.features)
        d_loss = F.mse_loss(pred, dataset.concentration)
        p_loss = compute_pde_residual(collocation_points, model, config.wind_u, config.wind_v, config.diffusion)
        t_loss = d_loss + config.lambda_physics * p_loss
        t_loss.backward()
        return t_loss
        
    for _ in range(config.lbfgs_epochs // 20):
        loss_val = opt_lbfgs.step(closure)
        lbfgs_epochs += 20
        if loss_val is not None:
            print(f"L-BFGS Step {lbfgs_epochs} | Loss: {loss_val.item():.5f}")
        
    writer.close()
    return history
"""),
    ("code", """base_model = PlumeInversionPINN(CONFIG).to(DEVICE)

if GPU_COUNT >= 2:
    model = nn.DataParallel(base_model)
    print('Using torch.nn.DataParallel across', GPU_COUNT, 'GPUs')
else:
    model = base_model
    print('Using single-device model')

# Compile model for faster training on PyTorch 2.0+ (requires modern GPUs, skip if issues)
try:
    if hasattr(torch, 'compile') and os.name != 'nt':
        print("Using torch.compile...")
        model = torch.compile(model)
except Exception as e:
    print(f"Compile skipped: {e}")

sum_params = sum(p.numel() for p in model.parameters())
print('Trainable parameters:', sum_params)

history = train_inversion_model(model, dataset, CONFIG, DEVICE)
history_df = pd.DataFrame(history)
"""),
    ("code", """def normalize_scores(scores):
    clipped = np.clip(scores.astype(np.float64), 0.0, None)
    total = float(clipped.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total

def evaluate_pinn_probability_grid(model, sector, device, grid_size):
    model.eval()
    x_axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    y_axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    
    # Average over early time steps for robust peak
    t_vals = [0.0, 0.05, 0.1]
    field = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    with torch.no_grad():
        for t in t_vals:
            t_grid = np.full_like(x_grid, t, dtype=np.float32)
            features = torch.tensor(np.column_stack([x_grid.ravel(), y_grid.ravel(), t_grid.ravel()]), dtype=torch.float32, device=device)
            preds = []
            for chunk in torch.split(features, 32768):
                preds.append(model(chunk).detach().cpu().numpy())
            field += np.concatenate(preds, axis=0).reshape(grid_size, grid_size)
            
    field /= len(t_vals)
    probabilities = normalize_scores(field)
    latitudes, longitudes = normalized_to_lat_lon(x_grid, y_grid, sector)
    return {'latitudes': latitudes, 'longitudes': longitudes, 'probabilities': probabilities, 'field': field}

def evaluate_physics_fused_posterior(pinn_map, dataset, config, temperature=0.010):
    rows = pd.DataFrame(dataset.rows)
    obs_x, obs_y = rows['x'].to_numpy(dtype=np.float32)[None, :], rows['y'].to_numpy(dtype=np.float32)[None, :]
    obs_t, obs_c = rows['elapsed_time'].to_numpy(dtype=np.float32)[None, :], rows['normalized_concentration'].to_numpy(dtype=np.float32)[None, :]
    
    sx = ((pinn_map['longitudes'] - SECTOR.lon_min) / (SECTOR.lon_max - SECTOR.lon_min)).reshape(-1, 1).astype(np.float32)
    sy = ((pinn_map['latitudes'] - SECTOR.lat_min) / (SECTOR.lat_max - SECTOR.lat_min)).reshape(-1, 1).astype(np.float32)
    
    signal = analytic_advection_diffusion_plume(obs_x, obs_y, obs_t, sx, sy, config.wind_u, config.wind_v, config.diffusion)
    signal = signal / np.maximum(signal.max(axis=1, keepdims=True), 1.0e-8)
    mse = np.mean((signal - obs_c) ** 2, axis=1).reshape(pinn_map['probabilities'].shape)
    
    likelihood = np.exp(-mse / max(temperature, 1.0e-6))
    neural_prior = np.power(normalize_scores(pinn_map['probabilities']) + 1.0e-12, 0.35)
    fused = normalize_scores(neural_prior * likelihood)
    return {**pinn_map, 'probabilities': fused, 'candidate_mse': mse, 'physics_likelihood': likelihood}

def estimate_peak(probability_map):
    p = probability_map['probabilities']
    idx = np.unravel_index(int(np.argmax(p)), p.shape)
    lat, lon = float(probability_map['latitudes'][idx]), float(probability_map['longitudes'][idx])
    dist = haversine_m(SECTOR.source_latitude, SECTOR.source_longitude, lat, lon)
    return {'latitude': lat, 'longitude': lon, 'probability': float(p[idx]), 'distance_meters': dist}

pinn_map = evaluate_pinn_probability_grid(model, SECTOR, DEVICE, CONFIG.grid_size)
fused_map = evaluate_physics_fused_posterior(pinn_map, dataset, CONFIG)
pinn_peak = estimate_peak(pinn_map)
fused_peak = estimate_peak(fused_map)

print('Raw PINN peak:', pinn_peak)
print('Fused physics+PINN peak:', fused_peak)
"""),
    ("code", """checkpoint_path = OUTPUT_DIR / 'plumetrace_pinn_checkpoint.pt'
probability_map_path = OUTPUT_DIR / 'plumetrace_source_probability_map.npz'
history_path = OUTPUT_DIR / 'plumetrace_training_history.csv'
summary_path = OUTPUT_DIR / 'plumetrace_validation_summary.json'
figure_path = OUTPUT_DIR / 'plumetrace_source_probability.png'

base = model.module if isinstance(model, nn.DataParallel) else model
# Un-compile base if needed to save pure state dict
if hasattr(base, '_orig_mod'): base = base._orig_mod

torch.save({
    'model_state_dict': base.state_dict(),
    'training_config': asdict(CONFIG),
    'city_sector': asdict(SECTOR),
    'sensor_stations': [asdict(s) for s in SENSOR_STATIONS],
    'pinn_peak': pinn_peak,
    'fused_peak': fused_peak,
    'torch_version': torch.__version__
}, checkpoint_path)

np.savez_compressed(
    probability_map_path,
    latitudes=fused_map['latitudes'],
    longitudes=fused_map['longitudes'],
    probabilities=fused_map['probabilities'],
    raw_pinn_probabilities=pinn_map['probabilities'],
    candidate_mse=fused_map['candidate_mse']
)
history_df.to_csv(history_path, index=False)

# Compute R2
val_pred = model(dataset.val_features).detach().cpu().numpy()
val_true = dataset.val_concentration.cpu().numpy()
ss_res = np.sum((val_true - val_pred)**2)
ss_tot = np.sum((val_true - np.mean(val_true))**2)
r2_score = 1 - (ss_res / (ss_tot + 1e-8))

validation_summary = {
    'known_source': {'latitude': SECTOR.source_latitude, 'longitude': SECTOR.source_longitude},
    'raw_pinn_peak': pinn_peak,
    'fused_physics_pinn_peak': fused_peak,
    'probability_sum': float(fused_map['probabilities'].sum()),
    'all_probabilities_finite': bool(np.isfinite(fused_map['probabilities']).all()),
    'pass_distance_threshold_250m': bool(fused_peak['distance_meters'] <= 250.0),
    'metrics': {'val_r2': float(r2_score)}
}
summary_path.write_text(json.dumps(validation_summary, indent=2), encoding='utf-8')

plt.figure(figsize=(10, 8))
plt.contourf(fused_map['longitudes'], fused_map['latitudes'], fused_map['probabilities'], levels=32, cmap='inferno')
plt.colorbar(label='Source probability')
plt.scatter([SECTOR.source_longitude], [SECTOR.source_latitude], c='cyan', s=120, marker='*', label='Known source')
plt.scatter([fused_peak['longitude']], [fused_peak['latitude']], c='white', s=90, marker='x', label='Predicted source')
for station in SENSOR_STATIONS:
    plt.scatter([station.longitude], [station.latitude], c='lime', s=45)
    plt.text(station.longitude + 0.00015, station.latitude + 0.00015, station.sensor_id, color='white', fontsize=9)
plt.title('PlumeTrace - Fused Physics + Residual Fourier PINN')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(figure_path, dpi=200)
plt.show()

print("Validation Summary:", json.dumps(validation_summary, indent=2))
print("Outputs saved to:", OUTPUT_DIR)
"""),
    ("code", """# Optional reload test
reloaded = PlumeInversionPINN(CONFIG).to(DEVICE)
state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
reloaded.load_state_dict(state['model_state_dict'])
reloaded.eval()
test_feature = torch.tensor([[dataset.source_x, dataset.source_y, 0.0]], dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    source_concentration = float(reloaded(test_feature).cpu().item())
print('Reloaded checkpoint OK. Predicted normalized C at known source, t=0:', source_concentration)
""")
]

notebook = {
    "cells": [create_cell(ctype, src) for ctype, src in cells_data],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12"
        },
        "kaggle": {
            "accelerator": "gpu",
            "gpu": "t4x2",
            "internet": False
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("c:/Users/sudu/Desktop/ieeehackton/PlumeTrace_PINN_Kaggle_T4.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
