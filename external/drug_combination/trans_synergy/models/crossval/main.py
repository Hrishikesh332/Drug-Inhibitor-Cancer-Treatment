import logging
from os import environ
import copy
import random

import numpy as np
import wandb
from sklearn.model_selection import ParameterGrid

import trans_synergy
from trans_synergy.models.trans_synergy.attention_main import (
    init_wandb,
    setup_data,
    setup_tensor_reorganizer,
    setup_model_and_optimizer,
)
from trans_synergy.models.crossval.utils import setup_model_and_optimizer_with_params, make_hashable, unhashable, train_model_and_eval

setting = trans_synergy.settings.get()

logger = logging.getLogger(__name__)
random_seed = 913


def run(use_wandb: bool = True, 
        fold_col_name: str = "fold",
        n_params_in_grid: int = 10,
        test_fold = 4,
        epochs_in_cv: int = 100):
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
            results = train_model_and_eval(
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
                params,
                save_model = False,
                testing = False,
                n_epochs = n_epochs_in_cv
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
    results = train_model_and_eval(
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
        n_epochs = setting.n_epochs,
        save_model=True,
        testing=True
    )
    wandb.finish()
