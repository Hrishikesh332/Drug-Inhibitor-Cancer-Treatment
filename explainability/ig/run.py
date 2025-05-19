import torch
import wandb
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from logging import Logger
from explainability.utils import save_with_wandb
import os
import numpy as np
from tqdm import tqdm

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
    
    logger.info(f"Computing Integrated Gradients for input shape {input_tensor.shape}")
    
    
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1, 1)
    if input_tensor.dim() == 2:  # biomining case
        alphas = alphas.view(-1, 1)
    
    interpolated_inputs = baseline + alphas * (input_tensor - baseline)
    logger.debug(f"Interpolated inputs shape: {interpolated_inputs.shape}")
    
    interpolated_inputs.requires_grad_(True)
    
    # Compute gradients
    outputs = model(interpolated_inputs)
    logger.debug(f"Model outputs shape: {outputs.shape}, min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
    
    grads = torch.autograd.grad(outputs.sum(), interpolated_inputs)[0]
    logger.debug(f"Gradients shape: {grads.shape}, mean: {grads.mean().item():.4f}, std: {grads.std().item():.4f}")
    
    if torch.all(grads == 0):
        logger.warning("All gradients are zero, which may indicate an issue with the model or inputs")
    
    # Average gradients across interpolation steps
    avg_grads = grads.mean(dim=0)
    logger.debug(f"Average gradients shape: {avg_grads.shape}, mean: {avg_grads.mean().item():.4f}")
    
    # Compute integrated gradients
    integrated_grads = (input_tensor - baseline) * avg_grads
    logger.info(f"Integrated Gradients computed, shape: {integrated_grads.shape}, mean: {integrated_grads.mean().item():.4f}, abs mean: {integrated_grads.abs().mean().item():.4f}")
    
    return integrated_grads

def run_integrated_gradients(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    Y: torch.Tensor,
    logger: Logger,
    **kwargs
):
    
    logger.info(f"Starting Integrated Gradients for {paper} model")
    
    # Convert inputs to tensors if they are NumPy arrays
    if isinstance(X, np.ndarray):
        logger.info(f"Converting X from NumPy array (shape: {X.shape}) to PyTorch tensor")
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        logger.info(f"Converting Y from NumPy array (shape: {Y.shape}) to PyTorch tensor")
        Y = torch.from_numpy(Y).float()
    
    logger.info(f"Input X shape: {X.shape}, dtype: {X.dtype}")
    logger.info(f"Input Y shape: {Y.shape}, dtype: {Y.dtype}")
    
    config = IntegratedGradientsConfig(paper=paper, **kwargs)
    
    wandb.init(
        project=f"EXPLAINABILITY on {config.paper} integrated-gradients",
        config=asdict(config),
        name=f"{paper}_integrated_gradients",
    )
    
    device = next(model.parameters()).device
    model.eval()
    logger.info(f"Model set to evaluation mode on device: {device}")
    
    # Move inputs to device
    X = X.to(device)
    Y = Y.to(device)
    
    # Set baseline
    if config.baseline_type == "zero":
        baseline = torch.zeros_like(X[0], device=device)
    else:
        raise ValueError(f"Unsupported baseline type: {config.baseline_type}")
    
    if paper == "transynergy":
        baseline = baseline.view(1, 3, config.cell_drug_feat_len_transynergy)
    elif paper == "biomining":
        baseline = baseline.view(1, config.cell_drug_feat_len_biomining)
    logger.info(f"Baseline shape: {baseline.shape}")
    
    # Process inputs in batches
    all_attributions = []
    num_samples = X.shape[0]
    
    logger.info(f"Processing {num_samples} samples in batches of {config.batch_size}")
    
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
            
            # Compute integrated gradients for this input
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
        logger.info(f"Batch {i//config.batch_size + 1} attributions shape: {batch_attributions.shape}, mean: {batch_attributions.mean().item():.4f}")
        
        all_attributions.append(batch_attributions)
        
        # Log batch statistics
        batch_mean_attr = batch_attributions.abs().mean().item()
        wandb.log({
            "batch_mean_attribution": batch_mean_attr,
            "sample_index": i
        })
    
    # Concatenate all attributions
    all_attributions = torch.cat(all_attributions, dim=0)
    logger.info(f"All attributions concatenated, final shape: {all_attributions.shape}, mean: {all_attributions.mean().item():.4f}, abs mean: {all_attributions.abs().mean().item():.4f}")
    
    save_path = f"explainability/ig/results/{paper}_integrated_gradients.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"Saving attributions to {save_path}")
    torch.save(all_attributions, save_path)
    
    
    log_dict = {}
    try:
        log_dict["attributions_histogram"] = wandb.Histogram(all_attributions.numpy())
        logger.info("Successfully logged final attributions_histogram to wandb")
    except Exception as e:
        logger.error(f"Failed to log attributions_histogram: {str(e)}")
    
    try:
        log_dict["mean_attribution"] = all_attributions.abs().mean().item()
        logger.debug("Logged mean_attribution to wandb")
    except Exception as e:
        logger.error(f"Failed to log mean_attribution: {str(e)}")
    
    if log_dict:
        wandb.log(log_dict)

    save_with_wandb(save_path, name_of_object=f"{paper}_integrated_gradients.pt")
    wandb.log({
        "attributions_histogram": wandb.Histogram(all_attributions.numpy()),
        "mean_attribution": all_attributions.abs().mean().item()
    })
    
    logger.info(f"Integrated Gradients completed successfully. Results saved to {save_path}")
    wandb.finish()