import logging
from os import environ, makedirs, path, sep
import copy
import random

from frozendict import frozendict
import numpy as np
import wandb
from sklearn.model_selection import ParameterGrid

import trans_synergy
from trans_synergy.models.trans_synergy.attention_main import (
    init_wandb,
    setup_data,
    setup_tensor_reorganizer,
    setup_model_and_optimizer,
    train_loop,
    prepare_splitted_dataloaders,
    save_model,
    evaluate,
    test_best_model,
)
from trans_synergy.models.crossval.utils import setup_model_and_optimizer_with_params, get_model_with_params

setting = trans_synergy.settings.get()

logger = logging.getLogger(__name__)
random_seed = 913


def crossvalidate(
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
        save_model = False

):
    """
    Cross-validation method that builds model from params using get_multi_models.
    """
    if use_wandb:
        init_wandb(eval_idx, crossval=True)

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
        all_data_loader=all_data_generator,
        n_epochs=setting.n_epochs
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

def run(use_wandb: bool = True, 
        fold_col_name: str = "fold",
        n_params_in_grid: int = 10,
        test_fold = 4):
    """
    Crossval with the set test fold as 4th index.
    """
    #todo: remove from here
    n_epochs_in_cv = 100 # cite badanie 
    
    if not use_wandb:
        environ["WANDB_MODE"] = "dryrun"

    std_scaler, X, Y, drug_features_length, cellline_features_length = setup_data()
    reorder_tensor = setup_tensor_reorganizer(
        drug_features_length, cellline_features_length
    )
    slice_indices = (
        drug_features_length + drug_features_length + cellline_features_length
    )

    split_cv_func = (
        trans_synergy.data.trans_synergy_data.DataPreprocessor.cv_train_eval_test_split_generator
    )

    param_grid = [
        {
            "attention_heads": [1, 2, 4],
            "d_model": [400, 512],
            "n_layers": [1, 2, 3],
            "attention_dropout": [0.1, 0.2, 0.3],
            "n_feature_type": [1, 3],
            "fc_hidden_units": [[2000, 1000, 1], [1000, 500, 1]],
            "classifier": [False, True],
            "drugs_on_the_side": [False, True],
        }
    ]
    
    
    all_combinations = list(ParameterGrid(param_grid))
    sampled_combinations = random.sample(all_combinations, k=min(n_params_in_grid, len(all_combinations)))


    current_results = {}
    for params in sampled_combinations:
        hashable_params = make_hashable(params)
        current_results[hashable_params] = []
        for eval_idx, partition in enumerate(split_cv_func(fold = fold_col_name, test_fold=test_fold)):
            init_wandb(
                fold_idx=eval_idx,
                crossval=True,
            )
            drug_model, optimizer, scheduler = (
                setup_model_and_optimizer_with_params(reorder_tensor, params)
            )
            best_drug_model = copy.deepcopy(drug_model)
            partition_indices = {
                "train": partition[0],
                "test1": partition[1],
                "test2": partition[2],
                "eval1": partition[3],
                "eval2": partition[4],
            }

            std_scaler.fit(Y[partition_indices["train"]])
            if setting.y_transform:
                Y = std_scaler.transform(Y) * 100

            training_generator, _, validation_generator, val_test_generator, _, _ = (
                prepare_splitted_dataloaders(partition_indices, Y.reshape(-1), X)
            )
            

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
                n_epochs=n_epochs_in_cv,
            )

            results = evaluate(
                best_model,
                val_test_generator,
                reorder_tensor,
                std_scaler,
                slice_indices,
                use_wandb,
            )
            current_results[hashable_params].append(results['mse'])
            wandb.finish()

    for params in current_results:
        current_results[params] = np.mean(current_results[params], axis=0)

    best_params = max(current_results, key=lambda x: current_results[x])
    best_params = unhashable(best_params)
    
    drug_model, best_drug_model, optimizer, scheduler = setup_model_and_optimizer(
        reorder_tensor, best_params
    )

    split_func = (
        trans_synergy.data.trans_synergy_data.DataPreprocessor.regular_train_eval_test_split
    )

    eval_idx = 0
    partition = split_func(fold_col_name="fold", test_fold=test_fold, evaluation_fold=eval_idx)
    training_generator, validation_generator, test_generator, _, _ = crossvalidate(
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
        best_params,
    )
    test_best_model(
        best_drug_model,
        test_generator,
        reorder_tensor,
        std_scaler,
        slice_indices,
        use_wandb,
        crossval = True,
    )
