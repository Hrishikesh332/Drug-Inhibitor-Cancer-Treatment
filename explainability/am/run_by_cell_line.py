import os
from dataclasses import asdict
from typing import Literal
from logging import Logger

import torch
import numpy as np
import wandb
from tqdm import trange, tqdm 

from trans_synergy.utils import set_seed
from explainability.utils import save_with_wandb
from explainability.am.config import ActivationMaximizationConfig
from explainability.data_utils import load_biomining_cell_line_data, load_transynergy_cell_line_data


epsilon = 1e-6
def run_activation_maximization_for_cell_line(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    logger: Logger,
    cell_line: str,
    **kwargs
):
    """
    AM for data of a particular cell_line (features of cell line are also masked)
    """
    input_min = X.min().item()
    input_max = X.max().item()
    input_bounds = (input_min, input_max)
    
    input_shape = X[0].shape
    
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    # Regularize to the input space 
    real_mean = X.mean(0, keepdim=True)
    real_std = X.std(0, keepdim=True)
    
    config = ActivationMaximizationConfig(paper=paper, input_bounds=input_bounds, **kwargs)
    minimax = "max" if config.maximize else "min"
    
    wandb.init(project=f"EXPLAINABILITY on {config.paper} (cell line -{cell_line}) activation-maximization ({minimax})", 
               config=asdict(config),
               name=f"{paper}_{minimax}_activation_maximization_reg_{config.regularization}",)

    device = next(model.parameters()).device

    for trial in range(config.num_trials):
        seed = trial
        set_seed(trial)

        input_tensor = torch.randn(input_shape, requires_grad=True, device=device)
        
        if config.paper == "transynergy":
            input_tensor = input_tensor.view(1, 3, config.feature_length).clone().detach().requires_grad_(True)
            real_mean = real_mean.view(1, 3, config.feature_length)
            real_std = real_std.view(1, 3, config.feature_length) + epsilon
            mask = torch.ones_like(input_tensor, dtype=torch.bool, device=device) 
            mask[0, 2, frozen_indices] = False
        elif config.paper == "biomining":
            input_tensor = input_tensor.view(1, config.feature_length).clone().detach().requires_grad_(True)
            real_mean = real_mean.view(1, config.feature_length)
            real_std = real_std.view(1, config.feature_length) + epsilon
            frozen_indices = torch.arange(26, 33)  # those define the cell-line!
            mask = torch.ones_like(input_tensor, dtype=torch.bool, device=device) 
            mask[:, frozen_indices] = False

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
                loss += config.l1_lambda * input_tensor[mask].abs().sum()
            elif config.regularization == "l2":
                loss += config.l2_lambda * input_tensor[mask].norm(p=2)
            elif config.regularization == "l2_input":
                diff = (input_tensor - real_mean) / real_std
                loss += config.l2_lambda * diff[mask].pow(2).mean()

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
        save_path = f"explainability/am/results/{paper}_{cell_line}_{minimax}_reg_{regularization_method}/best_input_trial_{trial}.pt"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(original_best_input, save_path)
        wandb.log({label: wandb.Histogram(img_or_tensor)})
        
        save_with_wandb(save_path, name_of_object=f"{paper}_method_am_{minimax}_trial_{trial}.pt")

        logger.info(f"[{label}] Best output value: {best_val:.4f}")

    wandb.finish()
    
def run_activation_maximization_by_cell_line(
    model,
    paper,
    X,
    logger=None,
    regularization=None,
    maximize=None,
    split='train'
):
    if split != 'train':
        raise ValueError(f"Activation maximization is only supported for the 'train' split, but got '{split}'.")
    if paper == "biomining":
        cell_lines_names = load_biomining_cell_line_data(split=split)
    elif paper == "transynergy":
        cell_lines_names = load_transynergy_cell_line_data(split=split)
    else:
        raise ValueError(f"Unknown paper: {paper}")
    
    # get unique cell_line names
    unique_cell_line_names = set(cell_lines_names)
    for cell_line_name in tqdm(unique_cell_line_names, desc="Processing Cell Lines"):
        cell_line_mask = cell_lines_names == cell_line_name
        X_cell_line = X[cell_line_mask]
        
        run_activation_maximization_for_cell_line(
            model,
            paper,
            X_cell_line,
            logger,
            cell_line_name,
            regularization=regularization,
            maximize=maximize
        )
            
        
        