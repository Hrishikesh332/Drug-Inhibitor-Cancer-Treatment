import torch
import wandb
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from logging import Logger
from explainability.utils import save_with_wandb
from trans_synergy.utils import set_seed
import numpy as np
from tqdm import tqdm
import math
import os
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

@dataclass
class IntegratedGradientsConfig:
    paper: Literal["biomining", "transynergy"]
    n_steps: int = 50
    baseline_type: str = "zero"
    cell_drug_feat_len_transynergy: int = 2402
    cell_drug_feat_len_biomining: int = 33
    batch_size: int = 32

def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int,
    device: torch.device,
    logger: Logger
) -> torch.Tensor:
        
    ig = IntegratedGradients(model)
    
    try:
        attributions = ig.attribute(
            inputs=input_tensor,
            baselines=baseline,
            n_steps=n_steps,
            return_convergence_delta=False 
        )
        logger.debug(f"Attributions shape: {attributions.shape}, mean: {attributions.mean().item():.4f}, abs mean: {attributions.abs().mean().item():.4f}")
    except Exception as e:
        logger.error(f"Failed to compute Integrated Gradients with Captum: {str(e)}")
        raise
    
    
    if torch.all(attributions == 0):
        logger.warning("All attributions are zero, which may indicate an issue with the model or inputs")
    
    return attributions

def run_integrated_gradients(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    Y: torch.Tensor,
    logger: Logger,
    **kwargs
):
       
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()
    
    config = IntegratedGradientsConfig(paper=paper, **kwargs)
    
    wandb.init(
        project=f"EXPLAINABILITY on {config.paper} integrated-gradients",
        config=asdict(config),
        name=f"{paper}_integrated_gradients",
    )
    
    device = next(model.parameters()).device
    model.eval()
    
    
    X = X.to(device)
    Y = Y.to(device)
    
    
    if config.baseline_type == "zero":
        baseline = torch.zeros_like(X[0], device=device)
    else:
        raise ValueError(f"Unsupported baseline type: {config.baseline_type}")
    
    if paper == "transynergy":
        baseline = baseline.view(1, 3, config.cell_drug_feat_len_transynergy)
    elif paper == "biomining":
        baseline = baseline.view(1, config.cell_drug_feat_len_biomining)
    
    
    all_attributions = []
    num_samples = X.shape[0]
    
    total_batches = math.ceil(num_samples / config.batch_size)
    
    
    for i in tqdm(range(0, num_samples, config.batch_size), desc="Processing batches"):
        batch_X = X[i:i + config.batch_size].to(device)
        logger.debug(f"Processing batch {i//config.batch_size + 1}, size: {batch_X.shape[0]}, shape: {batch_X.shape}")
        
        batch_attributions = []
        
        for j, input_tensor in enumerate(batch_X):
            if paper == "transynergy":
                input_tensor = input_tensor.view(1, 3, config.cell_drug_feat_len_transynergy)
            elif paper == "biomining":
                input_tensor = input_tensor.view(1, config.cell_drug_feat_len_biomining)
            
            logger.debug(f"Sample {i+j} input shape: {input_tensor.shape}")
            
            
            attributions = compute_integrated_gradients(
                model,
                input_tensor,
                baseline,
                config.n_steps,
                device,
                logger
            )
            batch_attributions.append(attributions.detach().cpu())
        
        batch_attributions = torch.stack(batch_attributions)
        
        all_attributions.append(batch_attributions)
        
        batch_mean_attr = batch_attributions.abs().mean().item()
        batch_num = i // config.batch_size + 1
        log_dict = {
            "batch_mean_attribution": batch_mean_attr,
            "sample_index": i,
        }
        try:
            log_dict[f"batch_attributions_histogram_{batch_num}"] = wandb.Histogram(batch_attributions.numpy())
            logger.debug(f"Logged batch_attributions_histogram_{batch_num} to wandb")
        except Exception as e:
            logger.warning(f"Failed to log batch_attributions_histogram_{batch_num}: {str(e)}")
        
    
    all_attributions = torch.cat(all_attributions, dim=0)
    logger.info(f"All attributions concatenated, final shape: {all_attributions.shape}, "
                f"mean: {all_attributions.mean().item():.4f}, "
                f"abs mean: {all_attributions.abs().mean().item():.4f}, "
                f"std: {all_attributions.std().item():.4f}, "
                f"min: {all_attributions.min().item():.4f}, "
                f"max: {all_attributions.max().item():.4f}")
    
    
    save_path = f"explainability/ig/results/{paper}_integrated_gradients.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_attributions, save_path)
    
    log_dict = {}
    try:
        log_dict["attributions_histogram"] = wandb.Histogram(all_attributions.numpy())
    except Exception as e:
        logger.error(f"Failed to log attributions_histogram: {str(e)}")
    
    try:
        log_dict["mean_attribution"] = all_attributions.abs().mean().item()
        log_dict["std_attribution"] = all_attributions.std().item()
        log_dict["min_attribution"] = all_attributions.min().item()
        log_dict["max_attribution"] = all_attributions.max().item()
        logger.debug("Logged attribution statistics to wandb")
    except Exception as e:
        logger.error(f"Failed to log attribution statistics: {str(e)}")
    
    if log_dict:
        wandb.log(log_dict)
    
    save_with_wandb(save_path, name_of_object=f"{paper}_integrated_gradients.pt")
    wandb.finish()