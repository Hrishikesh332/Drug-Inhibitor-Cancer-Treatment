import fire
import joblib
import numpy as np
from external.predicting_synergy_nn.src.utils.data_loader import CVDatasetHandler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from trans_synergy.data.trans_synergy_data import (
    DataPreprocessor as DataPreprocessorTranSynergy,
)
from trans_synergy.models.trans_synergy.attention_main import (
    prepare_splitted_datasets as prepare_splitted_datasets_transynergy,
)
from trans_synergy.models.trans_synergy.attention_main import (
    setup_data as setup_data_transynergy,
)

import catboost

BEST_PARAMS_BIOMINING = {
    "learning_rate": 0.06,
    "l2_leaf_reg": 4,
    "depth": 9,
    "boosting_type": "Plain",
}
BEST_PARAMS_TRANSYNERGY = {
    "learning_rate": 0.06,
    "l2_leaf_reg": 6,
    "depth": 9,
    "boosting_type": "Plain",
}


def train_and_eval_biomining_catboost(
    biomining_data_dir_path: str = "external/predicting_synergy_nn/data",
):
    data_loader = CVDatasetHandler(data_dir=biomining_data_dir_path)

    model = catboost.CatBoostRegressor(**BEST_PARAMS_BIOMINING)

    X_train, y_train = data_loader.get_dataset(type="alltrain")
    X_test, y_test = data_loader.get_dataset(type="test")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)[0]
    pearson = pearsonr(y_test, y_pred)[0]
    print(f"Mean Squared Error on test set: {mse:.4f}")
    print(f"Pearson on test set: {pearson:.4f}")
    print(f"Spearman on test set: {spearman:.4f}")

    return model


def train_eval_save_biomining_catboost(
    biomining_data_dir_path: str = "external/predicting_synergy_nn/data",
    model_output_path: str = "baselines/catboost/models/best_catboost_biomining.pkl",
):
    model = train_and_eval_biomining_catboost(biomining_data_dir_path)
    joblib.dump(model, model_output_path)


def train_and_eval_transynergy_catboost():
    _, X, Y, _, _ = setup_data_transynergy()

    split_func = DataPreprocessorTranSynergy.regular_train_eval_test_split
    partition = split_func(fold_col_name="fold", test_fold=4, evaluation_fold=0)
    partition_indices = {
        "train": partition[0],
        "test1": partition[1],
        "test2": partition[2],
        "eval1": partition[3],
        "eval2": partition[4],
    }

    # dataloaders, but we just extract csvs
    training_set, _, validation_set, test_set, _ = (
        prepare_splitted_datasets_transynergy(partition_indices, Y.reshape(-1), X)
    )
    X_train, y_train = training_set.data_cache, training_set.data_cache_y
    X_val, y_val = validation_set.data_cache, validation_set.data_cache_y
    X_test, y_test = test_set.data_cache, test_set.data_cache_y

    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    model = catboost.CatBoostRegressor(**BEST_PARAMS_BIOMINING)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)[0]
    pearson = pearsonr(y_test, y_pred)[0]
    print(f"Mean Squared Error on test set: {mse:.4f}")
    print(f"Pearson on test set: {pearson:.4f}")
    print(f"Spearman on test set: {spearman:.4f}")

    return model


def train_eval_and_save_transynergy_catboost(
    model_output_path: str = "baselines/catboost/models/best_catboost_transynergy.pkl",
):
    model = train_and_eval_transynergy_catboost()
    joblib.dump(model, model_output_path)


if __name__ == "__main__":
    fire.Fire()
