import torch
import copy
import wandb
from frozendict import frozendict

import trans_synergy
import trans_synergy.settings
from trans_synergy.models.trans_synergy.attention_model import (
    TransposeMultiTransformersPlusLinear,
    )
from trans_synergy.models.trans_synergy.attention_main import (
    init_wandb,
    train_loop,
    prepare_splitted_dataloaders,
    evaluate,
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

def train_model_and_eval(
        eval_idx,
        partition,
        X,
        Y,
        std_scaler,
        reorder_tensor,
        drug_model,
        best_drug_model,
        optimizer,
        scheduler,
        use_wandb,
        slice_indices,
        model_params,
        save_model = False,
        testing = False,
        n_epochs = 100,
        patience = 50,
        fold_col_name=None
):
    """
    Cross-validation method that builds model from params using get_multi_models.
    """
    if use_wandb:
        init_wandb(eval_idx, crossval=True, testing = testing, fold_col_name = fold_col_name)

    train_idx, test1_idx, test2_idx, eval1_idx, eval2_idx = partition

    std_scaler.fit(Y[train_idx])

    if setting.y_transform:
        Y_scaled = std_scaler.transform(Y) * 100
    else:
        Y_scaled = Y

    (
        training_generator,
        _,
        validation_generator,
        test_generator,
        all_data_generator,
        _,
    ) = prepare_splitted_dataloaders(
        {
            "train": train_idx,
            "test1": test1_idx,
            "test2": test2_idx,
            "eval1": eval1_idx,
            "eval2": eval2_idx,
        },
        Y_scaled.reshape(-1),
        X,
    )
    if model_params is not None:
        wandb.config.update(model_params)

    if best_drug_model is None:
        best_drug_model = copy.deepcopy(drug_model)

    best_model = train_loop(
        model=drug_model,
        best_model=best_drug_model,
        train_loader=training_generator,
        val_loader=validation_generator,
        optimizer=optimizer,
        scheduler=scheduler,
        reorder_tensor=reorder_tensor,
        std_scaler=std_scaler,
        use_wandb=use_wandb,
        slice_indices=slice_indices,
        n_epochs=n_epochs,
        patience=patience
    )
    results = evaluate(
                best_model,
                test_generator,
                reorder_tensor,
                std_scaler,
                slice_indices,
                use_wandb,
            )
    if save_model:
        save_model(best_model, setting.run_dir, eval_idx)

    if use_wandb:
        wandb.finish()

    return results

def make_hashable(d):
    if isinstance(d, dict):
        return frozendict({k: make_hashable(v) for k, v in d.items()})
    elif isinstance(d, list):
        return tuple(make_hashable(x) for x in d)
    else:
        return d
    
def unhashable(d):
    """Recursively convert frozendicts and tuples back to dicts and lists."""
    if isinstance(d, frozendict):
        return {k: unhashable(v) for k, v in d.items()}
    elif isinstance(d, tuple):
        return [unhashable(x) for x in d]
    else:
        return d