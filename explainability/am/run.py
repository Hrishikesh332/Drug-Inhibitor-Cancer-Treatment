import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from dataclasses import asdict
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ActivationMaximizationConfig:
    paper: Literal["biomining", "transynergy"]
    maximize: bool = True
    num_trials: int = 5
    input_bounds: tuple[float, float] = (0, 1)  # Change if needed
    steps: int = 1000
    lr: float = 0.01
    early_stopping: bool = True
    patience: int = 50

def run_activation_maximization(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: Optional[torch.Tensor],
    paper: Literal["biomining", "transynergy"],
):
    config = ActivationMaximizationConfig(paper=paper)
    wandb.init(project=f"EXPLAINABILITY on {config.paper} activation-maximization", config=config.__dict__)
    model.eval()

    input_shape = X[0].shape if config.paper == "biomining" else getattr(model, "input_shape", X[0].shape)

    for trial in range(config.num_trials):
        seed = trial
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_tensor = torch.randn(input_shape, requires_grad=True, device=next(model.parameters()).device)

        optimizer = torch.optim.Adam([input_tensor], lr=config.lr)

        best_val = float("-inf") if config.maximize else float("inf")
        best_input = input_tensor.clone()
        steps_without_improvement = 0

        for step in range(config.steps):
            optimizer.zero_grad()
            output = model(input_tensor.unsqueeze(0))  # add batch dim

            out_val = output.squeeze()
            loss = -out_val if config.maximize else out_val

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                input_tensor.clamp_(*config.input_bounds)

                current_val = out_val.item()
                improved = (current_val > best_val) if config.maximize else (current_val < best_val)
                if improved:
                    best_val = current_val
                    best_input = input_tensor.clone()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

            if config.early_stopping and steps_without_improvement >= config.patience:
                print(f"[Trial {trial}] Early stopping at step {step} with value {best_val:.4f}")
                break

        img_or_tensor = best_input.detach().cpu().numpy()
        label = f"{'max' if config.maximize else 'min'}_seed_{seed}"

        if best_input.dim() == 3:  # image (C, H, W)
            wandb.log({label: wandb.Image(img_or_tensor.transpose(1, 2, 0))})
        else:
            wandb.log({label: wandb.Histogram(img_or_tensor)})

        print(f"[{label}] Best output value: {best_val:.4f}")

    wandb.finish()