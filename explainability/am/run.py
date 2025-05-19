import os
from dataclasses import asdict
from typing import Literal
from logging import Logger

import torch
import wandb
from tqdm import trange

from trans_synergy.utils import set_seed
from explainability.utils import save_with_wandb
from explainability.am.config import ActivationMaximizationConfig

def run_activation_maximization(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    logger: Logger,
    **kwargs
):
    input_min = X.min().item()
    input_max = X.max().item()
    input_bounds = (input_min, input_max)
    
    input_shape = X[0].shape
    
    config = ActivationMaximizationConfig(paper=paper, input_bounds=input_bounds, **kwargs)
    minimax = "max" if config.maximize else "min"
    
    wandb.init(project=f"EXPLAINABILITY on {config.paper} activation-maximization ({minimax})", 
               config=asdict(config),
               name=f"{paper}_{minimax}_activation_maximization_reg_{config.regularization}",)

    device = next(model.parameters()).device

    for trial in range(config.num_trials):
        seed = trial
        set_seed(trial)

        input_tensor = torch.randn(input_shape, requires_grad=True, device=device)
        
        if config.paper == "transynergy":
            input_tensor = input_tensor.view(1, 3, config.cell_drug_feat_len_transynergy).clone().detach().requires_grad_(True)
        elif config.paper == "biomining":
            input_tensor = input_tensor.view(1,  config.cell_drug_feat_len_biomining).clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([input_tensor], lr=config.lr)

        best_val = float("-inf") if config.maximize else float("inf")
        best_input = input_tensor.clone()
        steps_without_improvement = 0

        for step in trange(config.steps, desc=f"Trial {trial+1} steps"):
            optimizer.zero_grad()

            output = model(input_tensor)  # add batch dim
            out_val = output.squeeze()
            loss = -out_val if config.maximize else out_val

            # regularization
            if config.regularization == "l1":
                loss += config.l1_lambda * input_tensor.abs().sum()
            elif config.regularization == "l2":
                loss += config.l2_lambda * input_tensor.norm(p=2)

            loss.backward()
            optimizer.step()
            
            wandb.log({f"trial_{trial}/loss": loss.item(), 
                       f"trial_{trial}/output": out_val,
                       "step": step})

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
                logger.info(f"[Trial {trial}] Early stopping at step {step} with value {best_val:.4f}")
                break
            
        original_best_input = best_input.view(input_shape)
        label = f"{'max' if config.maximize else 'min'}_seed_{seed}"
        img_or_tensor = original_best_input.detach().cpu().numpy()
        
        regularization_method = config.regularization if config.regularization else "none"
        save_path = f"explainability/am/results/{paper}_{minimax}_reg_{regularization_method}/best_input_trial_{trial}.pt"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(original_best_input, save_path)
        wandb.log({label: wandb.Histogram(img_or_tensor)})
        
        save_with_wandb(save_path, name_of_object=f"{paper}_method_am_{minimax}_trial_{trial}.pt")

        logger.info(f"[{label}] Best output value: {best_val:.4f}")

    wandb.finish()