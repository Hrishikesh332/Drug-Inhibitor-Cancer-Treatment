from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm
import joblib
import wandb


def init_wandb(model_name: str, hyperam_suffix: str,  fold_idx: int = None, paper: str = None):
    wandb.init(
            project=f"Drug combination baselines",
            name= f"{paper}{model_name}_{hyperam_suffix}_{fold_idx}",
        )
    wandb.define_metric("Train Loss", step_metric="Epoch")
    wandb.define_metric("Validation Loss", step_metric="Epoch")

    return wandb

def get_model(name):
    if name == 'random_forest':
        return RandomForestRegressor, {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20],
            'max_features': ['auto', 'sqrt', 0.5],
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
    else:
        raise ValueError(f"Unknown model name: {name}")
    


def run_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    fold_idx: int = None,
    model_name='random_forest',
    output_path=None,
    scoring='mean_squared_error',
    logger=None, 
    scaler = None,
    n_iter = 30,
    paper = None,
):
    model_class, param_grid = get_model(model_name)
    init_wandb(model_name, "hyperparam_tuning", fold_idx=fold_idx)

    best_model = None
    best_val_score = None

    if scoring == 'mean_squared_error':
        eval_score = mean_squared_error
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

    sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    for params in tqdm(sampled_params):
        model = model_class(**params)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_score = eval_score(y_val, val_preds)

        spearman_corr_val = spearmanr(y_val, val_preds)[0]
        pearson_corr_val = pearsonr(y_val, val_preds)[0]

        wandb.log({
                f"{model_name} Val Score": val_score,
                f"{model_name} Spearman Correlation": spearman_corr_val,
                f"{model_name} Pearson Correlation": pearson_corr_val
            })

            
        if best_val_score is None or val_score < best_val_score:
            best_val_score = val_score
            best_model = model
            best_params = params
            best_spearman_corr = spearman_corr_val
            best_pearson_corr = pearson_corr_val

            if output_path:
                joblib.dump(model, f"{output_path}/{model_name}_{hash(frozenset(params.items()))}.pkl")

            wandb.log({
                f"{model_name} Best Val Score": best_val_score,
                f"{model_name} Best Pearson Correlation": best_pearson_corr,
                f"{model_name} Best Spearman Correlation": best_spearman_corr,
                f"{model_name} Best Params": best_params,
            })
            
    final_val_score = None
    if X_val is not None and y_val is not None and best_model is not None:
        final_val_score = eval_score(y_val, best_model.predict(X_val))


    test_score = None
    if X_test is not None and y_test is not None and best_model is not None:
        test_preds = best_model.predict(X_test)
        test_score = eval_score(y_test, test_preds)
        spearman_corr_test = spearmanr(y_test, test_preds)[0]
        spearman_corr_test = pearsonr(y_test, test_preds)[0]
        if logger:
            logger.info(f"[{model_name}] Test Score: {test_score}")
        wandb.log({
                f"{model_name} Test Score": test_score,
                f"{model_name} Test Pearson Correlation": spearman_corr_test,
                f"{model_name} Test Spearman Correlation": spearman_corr_test,
                f"{model_name} Test Params": best_params,
            })

    if output_path and best_model is not None:
        joblib.dump(best_model, output_path)

    wandb.finish()

    return best_model, final_val_score, test_score