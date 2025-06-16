import torch
import torch.nn as nn
from tqdm import tqdm 

from zennit.core import Composite
from zennit.rules import Epsilon, Gamma, Pass, ZBox, ReLUGuidedBackprop
from zennit.canonizers import NamedMergeBatchNorm
from zennit.core import BasicHook, stabilize
CompositeType = Composite 


class GMontavonEpsilon(BasicHook): # taken from https://github.com/rodrigobdz/lrp/blob/2089fda5e74e1255ae062b6c8a0b281661690c75/zennit-lrp-tutorial.ipynb
    def __init__(self, epsilon=1e-6, delta=0.25):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param, _: param],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0] + delta * (outputs[0] ** 2).mean() ** .5, epsilon)),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )

def lrp_rule_for_biomining(ctx, name, module): # inspired by https://iphome.hhi.de/samek/pdf/MonXAI19.pdf
    if isinstance(module, nn.Linear):
        if name == "net.20": # LRP-0
            return Epsilon(epsilon=0)
        elif name == "net.16" or name == 'net.12' or name == 'net.8': # Lrp-epsilon 
            return GMontavonEpsilon(epsilon=1e-9, delta=0.25)
        elif  name == 'net.4': # Lrp-gamma
            return Gamma()
        elif name == "net.0": # a bit unsure about this one https://github.com/rodrigobdz/lrp/blob/2089fda5e74e1255ae062b6c8a0b281661690c75/zennit-lrp-tutorial.ipynb
            low = torch.tensor([8.2900e-06, 8.2900e-06, 2.5200e-03, 2.5200e-03, 2.3800e-04, 2.3800e-04,
        3.4200e-06, 3.4200e-06, 1.1800e-04, 1.1800e-04, 5.7900e-09, 5.7900e-09,
        2.1800e-06, 2.1800e-06, 1.4300e-06, 1.4300e-06, 8.5900e-08, 8.5900e-08,
        1.7200e-04, 1.7200e-04, 6.4200e-05, 6.4200e-05, 5.9700e-03, 5.9700e-03,
        3.8800e-01, 3.8800e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00]).unsqueeze(0) # minimum values for each feature (on test dataset)
            high = torch.tensor([1.0000, 1.0000, 0.9980, 0.9980, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 0.9970, 0.9970, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 2.0000, 1.0000, 2.0000]).unsqueeze(0) # maximum values for each feature (on test dataset)
            return ZBox(low=low, high=high)
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
    composite = Composite(
            Composite(module_map=lrp_rule_for_biomining,
                          canonizers=[NamedMergeBatchNorm([(['net.0'], 'net.1'),
                                                            (['net.4'], 'net.5'),
                                                            (['net.8'], 'net.9'),
                                                            (['net.12'], 'net.13'),
                                                            (['net.16'], 'net.17')])]
                          )
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