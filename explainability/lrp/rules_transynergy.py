import torch
from tqdm import tqdm
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules


from lxt.efficient import monkey_patch, monkey_patch_zennit
from explainability.lrp.lxt.transynergy import attnLRP
from explainability.lrp.lxt.transynergy import TransposeMultiTransformersPlusLinear as TransposeMultiTransformersPlusLinearLxt
import trans_synergy
from trans_synergy.models.trans_synergy import attention_model

setting = trans_synergy.settings.get()

monkey_patch(attention_model, patch_map=attnLRP, verbose=True)
monkey_patch_zennit(verbose=True)


def explain_transynergy(
    model: torch.nn.Module,
    inputs: torch.Tensor
    ) -> torch.Tensor:
    """Run LRP explanation on Transynergy model for regression task.
    
    Args:
        model: The Transynergy model
        inputs: Input tensor (drug features, cell line features)
        composite: Optional pre-created composite. If None, will create default one
        
    Returns:
        Tuple of relevance scores
    """

    lxt_model = TransposeMultiTransformersPlusLinearLxt(
        d_input_list=[2402],
        d_model_list=[setting.d_model],
        n_feature_type_list=setting.n_feature_type,
        N=setting.n_layers,
        heads=setting.attention_heads,
        dropout=0, # LXT, just in case
    )
    
    lxt_model.load_state_dict(state_dict = model.state_dict(), strict = False) # FIXME:I think the layernorms do not work!
    # not strict since some params are unsued and not reflected in the lxt version!
    
    zennit_comp = LayerMapComposite([
        (torch.nn.Linear, z_rules.Gamma()),
    ])
    zennit_comp.register(lxt_model)
    
    relevances = []
    lxt_model.eval()
    for i in tqdm(range(inputs.shape[0]), desc="LRP samples"):
        inp = inputs[i, :].unsqueeze(0).clone().detach().requires_grad_(True)
        out = lxt_model(inp)
        out.backward()

        relevances.append(inp.grad.detach().cpu())
        
    tensor_relevances= torch.cat(relevances, dim=0)
    return tensor_relevances

