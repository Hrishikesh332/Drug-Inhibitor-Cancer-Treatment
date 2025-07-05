from typing import Literal, Tuple, Optional
from logging import Logger
import os

import numpy as np
import torch
import wandb
from dataclasses import asdict

from trans_synergy.utils import set_seed
from explainability.lrp.config import LRPExplanationConfig
from explainability.lrp.rules_biomining import explain_biomining
from explainability.lrp.rules_transynergy import explain_transynergy
from explainability.data_utils import select_representative_samples, reshape_transynergy_input


def run_lrp_explanation(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    logger: Logger,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run LRP explanation on the specified model.
    
    Args:
        model: The model to explain
        paper: Which paper's model to explain ("biomining" or "transynergy")
        X_train: Training data tensor
        X_test: Test data tensor
        logger: Logger instance
        **kwargs: Additional arguments passed to create_transynergy_composite
        
    Returns:
        Tuple of (model outputs, relevance scores) for test data
    """
    config = LRPExplanationConfig(paper=paper, **kwargs)
    device = next(model.parameters()).device
    set_seed(config.seed)
    
    wandb.init(
        project=f"EXPLAINABILITY on {config.paper} LRP",
        config=asdict(config),
        name=f"{paper}_lrp",
    )
    
    model.eval()
    model = model.to(device)
    
    logger.info(f"Running LRP explanation for model: {config.paper}")
    results_dir = f"explainability/lrp/results/{config.paper}_subsample_{config.subsample}"
    
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    
    if config.subsample:
        X_train = select_representative_samples(X_train, config.num_samples)
        X_test = select_representative_samples(X_test, config.num_samples)
        
    if config.paper == "biomining":
        logger.info("Computing absolute LRP explanations")
        relevance  = explain_biomining(
                model=model,
                inputs=X_test
            )
    elif config.paper == "transynergy":
        X_train = reshape_transynergy_input(X_train, logger, "X_train")
        X_test = reshape_transynergy_input(X_test, logger, "X_test")
        
        relevance  = explain_transynergy(
                model=model,
                inputs=X_test
            )
    else:
        raise ValueError(f"Paper {config.paper} not supported")
    
    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/relevances.npy", relevance.detach().cpu().numpy())
    
    
    