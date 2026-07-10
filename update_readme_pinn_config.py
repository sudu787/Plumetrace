import re
from pathlib import Path
import sys

# Ensure pinn_engine can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinn_engine import ModelConfig, TrainingConfig

def update_readme():
    model = ModelConfig()
    train = TrainingConfig()
    
    # Calculate dimensionalities
    fourier_dims = model.fourier_bands * 2
    
    # Format the dynamic text
    dynamic_text = f"""- **Network Topology:** 
  - **Input:** $x, y, t$ (normalized spatial and temporal coordinates).
  - **Feature Projection:** Random Fourier features ({fourier_dims} dimensions, $\\sigma={model.fourier_sigma}$) to mitigate the spectral bias typical in coordinate-MLPs.
  - **Hidden Layers:** {model.hidden_layers} residual blocks (width {model.hidden_units}). Incorporates LayerNorm and Adaptive Swish (Swish with a learnable $\\beta$ parameter) to ensure non-zero second-order derivatives for PyTorch autograd.
  - **Output:** Linear layer with Softplus activation to strictly enforce $C > 0$.
- **Optimization Strategy:** 
  - **Stage 1:** AdamW (Learning Rate: {train.learning_rate}, Cosine annealing with a {train.warmup_epochs}-epoch warmup) for {train.epochs} epochs.
  - **Stage 2:** L-BFGS (Learning Rate: {train.lbfgs_lr}, strong Wolfe line search, {train.lbfgs_steps} steps) for refining the PDE residual.
  - **Loss Function:** Weighted sum of data MSE, PDE residual, and boundary conditions. Component weights are dynamically adjusted via `SoftAdaptWeighter` (window_size={train.softadapt_warmup_epochs}, floor={train.softadapt_floor}, ceil={train.softadapt_ceil}, $\\beta$={train.softadapt_beta}).
  - **Sampling:** Collocation points ({train.collocation_points}) are drawn via a Sobol sequence. Residual-based Adaptive Refinement (RAR) is applied every {train.rar_interval} epochs to oversample {int(train.rar_fraction * 100)}% of coordinates exhibiting high PDE error."""

    readme_path = Path(__file__).resolve().parent / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    
    pattern = re.compile(r"(<!-- PINN_CONFIG_START -->).*?(<!-- PINN_CONFIG_END -->)", re.DOTALL)
    
    if not pattern.search(content):
        print("Could not find <!-- PINN_CONFIG_START --> tags in README.md")
        sys.exit(1)
        
    updated_content = pattern.sub(lambda m: f"{m.group(1)}\n{dynamic_text}\n{m.group(2)}", content)
    readme_path.write_text(updated_content, encoding="utf-8")
    print("Successfully updated README.md with dynamic PINN configuration.")

if __name__ == "__main__":
    update_readme()
