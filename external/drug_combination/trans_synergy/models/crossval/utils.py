import torch
import trans_synergy.settings
from trans_synergy.models.trans_synergy.attention_model import (
    TransposeMultiTransformersPlusLinear
    )
from trans_synergy.utils import set_seed

setting = trans_synergy.settings.get()
random_seed = 913 # todo, get from settings?


def get_model_with_params(
    inputs_lengths,
    input_masks=None,
    drugs_on_the_side=False,
    classifier=False,
    **params
):

    d_model_param = params.get("d_model", setting.d_model)
    n_feature_type_param = params.get("n_feature_type", setting.n_feature_type)
    n_layers = params.get("n_layers", setting.n_layers)
    attention_heads = params.get("attention_heads", setting.attention_heads)
    attention_dropout = params.get("attention_dropout", setting.attention_dropout)

    if not isinstance(d_model_param, list):
        d_models = [d_model_param] * len(inputs_lengths)
    else:
        d_models = d_model_param

    if not isinstance(n_feature_type_param, list):
        n_feature_types = [n_feature_type_param] * len(inputs_lengths)
    else:
        n_feature_types = n_feature_type_param

    for d_model in d_models:
        assert (
            d_model % attention_heads == 0
        ), "d_model must be divisible by number of attention heads"
    assert attention_dropout < 1, "dropout must be less than 1"

    if not isinstance(input_masks, list):
        input_masks = [input_masks] * len(inputs_lengths)

    final_inputs_lengths = [
        inputs_lengths[i] // n_feature_types[i] for i in range(len(inputs_lengths))
    ]

    model = TransposeMultiTransformersPlusLinear(
        final_inputs_lengths,
        d_models,
        n_feature_types,
        n_layers,
        attention_heads,
        attention_dropout,
        input_masks,
        linear_only=False,
        drugs_on_the_side=drugs_on_the_side,
        classifier=classifier,
    )

    return model


def setup_model_and_optimizer_with_params(reorder_tensor, params):
    set_seed(random_seed)
    drug_model =  get_model_with_params(reorder_tensor.get_reordered_slice_indices(), 
                                      input_masks=None,
                                      drugs_on_the_side=False,
                                      params=params)
    optimizer = torch.optim.Adam(
        drug_model.parameters(), lr=setting.start_lr,
        weight_decay=setting.lr_decay, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    return drug_model, optimizer, scheduler