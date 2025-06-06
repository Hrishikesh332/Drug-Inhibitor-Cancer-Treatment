from typing import Literal, Tuple, Optional
from logging import Logger

import torch
import zennit

from explainability.lrp.config import LRPExplanationConfig
from explainability.lrp.rules import (
    create_transynergy_composite,
    explain_transynergy,
    explain_transynergy_relative
)


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
    config = LRPExplanationConfig(paper=paper, **kwargs)
    device = next(model.parameters()).device
    model.eval()
    model = model.to(device)
    
    logger.info(f"Running LRP explanation for model: {config.paper}")
    
    if config.paper == "biomining":
        raise NotImplementedError("Biomining LRP not implemented yet")
    elif config.paper == "transynergy":
        # Create composite with rules
        composite = create_transynergy_composite(**kwargs)
        
        if relative:
            logger.info("Computing relative LRP explanations")
            # Run relative explanation on test data
            outputs, relevance = explain_transynergy_relative(
                model=model,
                inputs=X_test,
                baseline=baseline,
                composite=composite,
                **kwargs
            )
        else:
            logger.info("Computing absolute LRP explanations")
            # Run standard explanation on test data
            outputs, relevance = explain_transynergy(
                model=model,
                inputs=X_test,
                composite=composite,
                **kwargs
            )
        
        logger.info("Successfully computed LRP relevance scores")
        return outputs, relevance
    else:
        raise ValueError(f"Paper {config.paper} not supported")
    
    
    