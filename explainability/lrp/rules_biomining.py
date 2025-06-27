import torch
import torch.nn as nn
from tqdm import tqdm 

from zennit.core import Composite
from zennit.rules import  Gamma, Pass
from zennit.canonizers import NamedMergeBatchNorm
CompositeType = Composite 


def lrp_rule_for_biomining(ctx, name, module):
    if isinstance(module, nn.Linear):
        return Gamma(zero_params =['bias'])
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.Tanh) or isinstance(module, nn.Dropout) or isinstance(module, nn.ReLU):  # for all other layers, e.g. BatchNorm1d
        # activations are ignore just as https://zennit.readthedocs.io/en/0.4.4/how-to/use-rules-composites-and-canonizers.html 
        return Pass()
    else:
        pass
        
def explain_biomining(
    model: torch.nn.Module,
    inputs: torch.Tensor
) -> torch.Tensor:
    """Run LRP explanation on Biomining model for regression task.
    
    Args:
        model: The Biomining model
        inputs: Input tensor (drug features, cell line features)
        composite: Optional pre-created composite. If None, will create default one
        
    Returns:
        Tuple of relevance scores
    """
    composite = Composite(module_map=lrp_rule_for_biomining,
                          canonizers=[NamedMergeBatchNorm([(['net.0'], 'net.1'),
                                                            (['net.4'], 'net.5'),
                                                            (['net.8'], 'net.9'),
                                                            (['net.12'], 'net.13'),
                                                            (['net.16'], 'net.17')])] # depends on architecture used 
                          )

    model.eval()
    relevances = []
    with composite.context(model) as modified_model:
        for i in tqdm(range(inputs.shape[0]), desc="LRP samples"):
            inp = inputs[i, :].unsqueeze(0).clone().detach().requires_grad_(True)
            out = modified_model(inp)
            out.backward()

            relevances.append(inp.grad.detach().cpu())
        
    tensor_relevances= torch.cat(relevances, dim=0)
    return tensor_relevances