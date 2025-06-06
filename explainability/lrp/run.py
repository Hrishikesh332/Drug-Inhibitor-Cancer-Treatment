from typing import Literal, Tuple, Optional
from logging import Logger
import os

import numpy as np
import torch
import wandb
from dataclasses import asdict

from trans_synergy.utils import set_seed
from explainability.lrp.config import LRPExplanationConfig
from explainability.lrp.rules import (
    explain_transynergy,
    explain_transynergy_relative
)
from explainability.utils import select_representative_samples, reshape_transynergy_input


def run_lrp_explanation(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    logger: Logger,
    baseline: Optional[torch.Tensor] = None,
    relative: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run LRP explanation on the specified model.
    
    Args:
        model: The model to explain
        paper: Which paper's model to explain ("biomining" or "transynergy")
        X_train: Training data tensor
        X_test: Test data tensor
        logger: Logger instance
        baseline: Optional baseline tensor for relative explanations
        relative: Whether to compute explanations relative to a baseline
        **kwargs: Additional arguments passed to create_transynergy_composite
        
    Returns:
        Tuple of (model outputs, relevance scores) for test data
    """
    config = LRPExplanationConfig(paper=paper, baseline = baseline, relative = relative, **kwargs)
    device = next(model.parameters()).device
    set_seed(config.seed)
    
    wandb.init(
        project=f"EXPLAINABILITY on {config.paper} LRP",
        config=asdict(config),
        name=f"{paper}_baseline_{config.baseline}_relative_{config.relative}_lrp",
    )
    
    model.eval()
    model = model.to(device)
    
    logger.info(f"Running LRP explanation for model: {config.paper}")
    results_dir = f"results/{config.paper}/{config.baseline}_{config.relative}"
    
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    
    if config.subsample:
        X_train_sample = select_representative_samples(X_train, config.num_samples)
        X_test_sample = select_representative_samples(X_test, config.num_samples)
        
    if config.paper == "biomining":
        raise NotImplementedError("Biomining LRP not implemented yet")
    
    elif config.paper == "transynergy":
        X_train_sample = reshape_transynergy_input(X_train_sample, logger, "X_train")
        
        if relative:
            logger.info("Computing relative LRP explanations")
            _, attribution, relevance  = explain_transynergy_relative(
                model=model,
                train_inputs=X_train_sample,
                test_inputs=X_test_sample,
                baseline=baseline
            )
        else:
            logger.info("Computing absolute LRP explanations")
            _, attribution, relevance  = explain_transynergy(
                model=model,
                inputs=X_test_sample
            )
        
        logger.info("Successfully computed LRP relevance scores")
    else:
        raise ValueError(f"Paper {config.paper} not supported")
    
    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/relevances.npy", relevance.cpu().numpy())
    np.save(f"{results_dir}/attributions.npy", attribution.cpu().numpy())
    
    
    