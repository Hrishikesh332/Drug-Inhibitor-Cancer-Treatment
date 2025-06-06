from typing import Optional, Tuple

import torch
from zennit.core import Composite
from zennit.rules import (
    Epsilon, Gamma, ZPlus, Pass
)
from zennit.layer import Sum
CompositeType = Composite 

def default_module_map(ctx, name, module):
    if isinstance(module, torch.nn.Linear):
        return Gamma(gamma=0.25)
    elif isinstance(module, torch.nn.MultiheadAttention):
        return ZPlus()
    elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
        return Epsilon(epsilon=1e-6)
    elif isinstance(module, torch.nn.ReLU):
        return ZPlus()
    elif isinstance(module, torch.nn.Dropout):
        return Pass()
    elif isinstance(module, Sum):
        return ZPlus()
    return None


def explain_transynergy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
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
    composite = Composite(
            module_map=default_module_map,
            canonizers=[] # canonizers are for complex layers - indicating how to handle them
        )

    model.eval()
    
    with composite.context(model) as modified_model:
        output = modified_model(inputs)
        attribution = output.clone()
        relevance = modified_model.relevance
    
    return output, attribution, relevance 


def explain_transynergy_relative(
    model: torch.nn.Module,
    test_inputs: torch.Tensor,
    train_inputs: torch.Tensor,
    baseline: str,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run LRP explanation on TransSynergy model relative to a baseline.
    This is useful for understanding how inputs contribute to changes in synergy
    relative to a baseline state (e.g., no drug combination).
    
    Args:
        model: The TransSynergy model
        test_inputs: Input tensor (drug features, cell line features)
        baseline: Optional baseline input tensor. If None, uses zero tensor
        composite: Optional pre-created composite. If None, will create default one
        **kwargs: Additional arguments passed to create_transynergy_composite
        
    Returns:
        Tuple of (model output, relevance scores relative to baseline)
    """
    composite = Composite(
            module_map=default_module_map,
            canonizers=[]
        )
        
    if baseline == 'zero':
        baseline = torch.zeros_like(test_inputs)
    elif baseline == 'mean':
        baseline = torch.mean(train_inputs, dim=0)
        baseline = baseline.unsqueeze(0).repeat(test_inputs.shape[0], 1)
    elif baseline == 'random':
        baseline = torch.rand_like(test_inputs)
    elif baseline == 'mean_per_cell_line':
        baseline = torch.mean(train_inputs, dim=0) # TODO: implement this
    
    model.eval()
    
    with composite.context(model) as modified_model:
        output = modified_model(train_inputs)
        baseline_output = modified_model(baseline)
        output_diff = output - baseline_output
        attribution = output_diff.clone()
        relevance = modified_model.relevance
    
    return output, attribution, relevance 