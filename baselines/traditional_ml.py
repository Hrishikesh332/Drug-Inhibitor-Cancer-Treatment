import multiprocessing
import multiprocessing.connection
import os
import tempfile
from typing import Any, Tuple, Type

import joblib
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

import wandb

RegressorClass = Type[Any]

def _train_model_in_subprocess(
            result_sender: multiprocessing.connection.Connection,
            model_class: RegressorClass,
            params: dict,
            X_train: np.ndarray,
            y_train: np.ndarray,
        ) -> None:
    try:
        model = model_class(**params)
        model.fit(X_train, y_train)
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        joblib.dump(model, path)
        result_sender.send(path)
    except Exception as e:
        result_sender.send(e)
    finally:
        result_sender.close()
        

def train_model_with_timeout(model_class: BaseEstimator,
                             params: dict, 
                             X_train: np.ndarray,
                             y_train: np.ndarray, 
                             timeout: int=60):
    result_receiver, result_sender = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_train_model_in_subprocess,
        args=(result_sender, model_class, params, X_train, y_train),
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("Model training exceeded timeout.")
    result = result_receiver.recv()
    if isinstance(result, Exception):
        raise result
    model = joblib.load(result)
    os.remove(result)
    return model



def init_wandb(model_name: str, hyperam_suffix: str,  fold_idx: int = None, paper: str = None):
    wandb.init(
            project=f"Drug combination baselines",
            name= f"{paper}_{model_name}_{hyperam_suffix}_fold_{fold_idx}",
        )
    wandb.define_metric("Train Loss", step_metric="Epoch")
    wandb.define_metric("Validation Loss", step_metric="Epoch")
    table_val = wandb.Table(columns=["Model Name", "Best Mean Val Score", "Sample Val Pearson Correlation", "Sample Val Spearman Correlation", "Best Val Params"])
    table_test = wandb.Table(columns=["Model Name", "Test Score", "Test Pearson Correlation", "Test Spearman Correlation", "Test Params"])

    return table_val, table_test

def get_model(name: str) -> Tuple[RegressorClass, dict]:
    if name == 'catboost':
        return CatBoostRegressor, {
            'learning_rate': [0.03, 0.06, 0.1],
            'depth': [3, 6, 9],
            'l2_leaf_reg': [2, 4, 6],
            'boosting_type': ['Ordered', 'Plain']
        }
    if name == 'random_forest':
        return RandomForestRegressor, {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20],
            'max_features': ['log2', 'sqrt', 0.5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    elif name == 'svm':
        return SVR, {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.5]
        }
    elif name == 'decision_tree':
        return DecisionTreeRegressor, {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': [None, 'sqrt', 'log2']
        }
    elif name == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge, {'alpha': [0.1, 1.0, 10.0]}
    elif name == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor, {'n_neighbors':[3,5,10]}
    else:
        raise ValueError(f"Unknown model name: {name}")
    


def train_and_eval_model(
    X_trains,
    y_trains,
    X_vals=None,
    y_vals=None,
    X_test=None,
    y_test=None,
    fold_idx: int = None,
    model_name='random_forest',
    output_path=None,
    scoring='mean_squared_error',
    logger=None, 
    n_iter = 15,
    paper = None,
    timeout = 120,
):
    print(f"Training and evaluating {model_name}...")
    model_class, param_grid = get_model(model_name)
    table, test_table = init_wandb(model_name, "hyperparam_tuning", fold_idx=fold_idx, paper = paper)
    
    best_model = None
    best_val_score = None

    if scoring == 'mean_squared_error':
        eval_score = mean_squared_error
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

    sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    for params in tqdm(sampled_params):
        val_scores_per_model = []
        folds = list(zip(X_trains, y_trains, X_vals, y_vals))
        for X_train, y_train, X_val, y_val in tqdm(folds, desc="Inner folds", total=len(folds)):
            try:
                model = train_model_with_timeout(model_class, params, X_train, y_train, timeout=timeout)
            except TimeoutError:
                print(f"Skipping params due to timeout: {params}")
                continue
            except Exception as e:
                print(f"Skipping params due to error: {params} — {e}")
                continue
            
            val_preds = model.predict(X_val)
            val_score = eval_score(y_val, val_preds)
            spearman_corr_val = spearmanr(y_val, val_preds)[0]
            pearson_corr_val = pearsonr(y_val, val_preds)[0]
            
            val_scores_per_model.append(val_score)
        
        if len(val_scores_per_model) == 0:
            print(f"Skipping params due to no valid scores: {params}")
            continue
        
        mean_val_score = sum(val_scores_per_model) / len(val_scores_per_model)
        table.add_data(
            model_name,
            mean_val_score,
            pearson_corr_val,
            spearman_corr_val,
            str(params)
        )
            
        if (best_val_score is None) or (mean_val_score < best_val_score) or (best_model is None):
            X_all = np.concatenate([X_trains[0], X_vals[0]], axis=0)
            y_all = np.concatenate([y_trains[0], y_vals[0]], axis=0)
            
            best_val_score = mean_val_score
            try:
                best_model = train_model_with_timeout(model_class, params, X_all, y_all, timeout=timeout*20)
                best_params = params
            except Exception as e:
                pass
        
            if output_path:
                joblib.dump(model, f"{output_path}/{model_name}_{hash(frozenset(params.items()))}.pkl")

            wandb.log({
                f"{model_name} Best Mean Val Score ": mean_val_score,
                f"{model_name} Best Params ": best_params,
            })
            wandb.log({f"Hyperpaarmeter Tuning Table": table})
  
    final_val_score = None
    if (X_val is not None) and (y_val is not None) and (best_model is not None):
        final_val_score = eval_score(y_val, best_model.predict(X_val))

    test_score = None
    if (X_test is not None) and (y_test is not None) and (best_model is not None):
        test_preds = best_model.predict(X_test)
        test_score = eval_score(y_test, test_preds)
        spearman_corr_test = spearmanr(y_test, test_preds)[0]
        pearson_corr_test = pearsonr(y_test, test_preds)[0]
        if logger:
            logger.info(f"[{model_name}] Test Score: {test_score}")
            
        test_table.add_data(
            model_name,
            test_score,
            pearson_corr_test,
            spearman_corr_test,
            str(best_params)  # Convert params to string for logging
        )
        
        wandb.log({f"Test Results Table": test_table})

    if output_path and best_model is not None:
        joblib.dump(best_model, output_path)

    wandb.finish()
    
    return best_model, final_val_score, test_score