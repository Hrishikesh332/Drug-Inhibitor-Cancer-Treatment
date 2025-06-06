from typing import Dict, List, Optional, Tuple, Union

import torch
import zennit
from zennit.attribution import Gradient
from zennit.core import Composite, Hook, stabilize
from zennit.rules import (
    Epsilon, Gamma, ZBox, ZPlus, ZPlusFast, Norm, Pass, ReLUDeconvNet, ReLUGuidedBackprop, ReLUInverse, ReLUPass, ReLUZBox, ReLUZPlus, ReLUZPlusFast
)
from zennit.types import Composite as CompositeType
from zennit.layer import Sum

def create_transynergy_composite(
    epsilon: float = 1e-6,
    gamma: float = 0.25,
    stabilizer: Optional[float] = None,
) -> CompositeType:
    """Create a composite for LRP rules specific to the TransSynergy model.
    
    Args:
        epsilon: Small constant for Epsilon rule to avoid division by zero
        gamma: Gamma parameter for the Gamma rule
        stabilizer: Optional stabilizer value for numerical stability
        
    Returns:
        Composite with appropriate LRP rules for TransSynergy model
    """
    # Define rules for different layer types
    rules = [
        # Linear layers - use ZPlus rule with gamma
        (torch.nn.Linear, Gamma(gamma=gamma)),
        
        # Transformer attention layers - use ZPlus rule
        (torch.nn.MultiheadAttention, ZPlus()),
        
        # Normalization layers - use Epsilon rule
        (torch.nn.LayerNorm, Epsilon(epsilon=epsilon)),
        (torch.nn.BatchNorm1d, Epsilon(epsilon=epsilon)),
        
        # ReLU activations - use ZPlus rule
        (torch.nn.ReLU, ZPlus()),
        
        # Dropout layers - pass through
        (torch.nn.Dropout, Pass()),
        
        # Sum operations (used in attention) - use ZPlus rule
        (Sum, ZPlus()),
    ]
    
    # Create composite with rules
    composite = Composite(
        rule_map=rules,
        stabilizer=stabilizer if stabilizer is not None else stabilize,
    )
    
    return composite

def explain_transynergy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    composite: Optional[CompositeType] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run LRP explanation on TransSynergy model for regression task.
    
    Args:
        model: The TransSynergy model
        inputs: Input tensor (drug features, cell line features)
        composite: Optional pre-created composite. If None, will create default one
        **kwargs: Additional arguments passed to create_transynergy_composite
        
    Returns:
        Tuple of (model output, relevance scores)
    """
    if composite is None:
        composite = create_transynergy_composite(**kwargs)
    
    model.eval()
    
    with composite.context(model) as modified_model:
        # Forward pass
        output = modified_model(inputs)
        
        # For regression, we use the output value directly as attribution
        # This means we're explaining the contribution to the predicted synergy score
        attribution = output.clone()
        
        # Compute relevance scores
        relevance = modified_model.relevance
    
    return output, relevance

def explain_transynergy_relative(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None, # TODO; research rajan's baseline for transynergy
    composite: Optional[CompositeType] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run LRP explanation on TransSynergy model relative to a baseline.
    This is useful for understanding how inputs contribute to changes in synergy
    relative to a baseline state (e.g., no drug combination).
    
    Args:
        model: The TransSynergy model
        inputs: Input tensor (drug features, cell line features)
        baseline: Optional baseline input tensor. If None, uses zero tensor
        composite: Optional pre-created composite. If None, will create default one
        **kwargs: Additional arguments passed to create_transynergy_composite
        
    Returns:
        Tuple of (model output, relevance scores relative to baseline)
    """
    if composite is None:
        composite = create_transynergy_composite(**kwargs)
    
    # TODO: implement baseline as mean (total) of all inputs, random noise, zero, and also mean per cell-line! 
    
    if baseline is None:
        # Use zero tensor as baseline if none provided
        baseline = torch.zeros_like(inputs)
    
    model.eval()
    
    with composite.context(model) as modified_model:
        # Forward pass for actual input
        output = modified_model(inputs)
        
        # Forward pass for baseline
        baseline_output = modified_model(baseline)
        
        # Compute difference from baseline
        output_diff = output - baseline_output
        
        # Use the difference as attribution
        # This explains how inputs contribute to changes from baseline
        attribution = output_diff.clone()
        
        # Compute relevance scores
        relevance = modified_model.relevance
    
    return output, relevance 