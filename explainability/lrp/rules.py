from typing import  Tuple

import torch
import torch.nn as nn
from tqdm import tqdm 

from zennit.core import Composite
from zennit.rules import Epsilon, Gamma, Pass
from zennit.canonizers import SequentialMergeBatchNorm
CompositeType = Composite 


def lrp_rule_for_biomining(ctx, name, module): # inspired by https://iphome.hhi.de/samek/pdf/MonXAI19.pdf
    if isinstance(module, nn.Linear):
        if name == "net.20": # LRP-0
            return Epsilon(epsilon=0)
        elif name == "net.16" or name == 'net.12' or name == 'net.8': # Lrp-epsilon 
            return Epsilon(epsilon=1e-6)
        elif name == "net.0" or name == 'net.8': # Lrp-gamma
            return Gamma()
    else:
        # activations are ignore just as https://zennit.readthedocs.io/en/0.4.4/how-to/use-rules-composites-and-canonizers.html 
        Pass()


def explain_biomining(
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
            module_map=lrp_rule_for_biomining,
            canonizers=[SequentialMergeBatchNorm()]
        )

    model.eval()
    relevances = []
    with composite.context(model) as modified_model:
        for i in tqdm(range(inputs.shape[0]), desc="LRP samples"):
            inp = inputs[i, :].unsqueeze(0).clone().detach().requires_grad_(True)
            out = model(inp)
            out.backward()

            relevances.append(inp.grad.detach().cpu())
        
    tensor_relevances= torch.cat(relevances, dim=0)
    return tensor_relevances