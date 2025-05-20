import argparse

from tqdm import tqdm
from trans_synergy.models.trans_synergy.attention_main import (
    prepare_splitted_datasets as prepare_splitted_datasets_transynergy,
)
from trans_synergy.models.trans_synergy.attention_main import (
    setup_data as setup_data_transynergy,
)
from trans_synergy.data.trans_synergy_data import (
    DataPreprocessor as DataPreprocessorTranSynergy,
)
from trans_synergy.utils import set_seed
from baselines.traditional_ml import train_and_eval_model
from external.predicting_synergy_nn.src.utils.data_loader import CVDatasetHandler

BASELINE_MODELS = ["catboost", "random_forest", "svm", "decision_tree", "ridge", "knn"]


def run_transynergy(timeout=120, n_iter=15, models: list[str] | None = None):
    if models is None:
        baseline_models = BASELINE_MODELS
    else:
        baseline_models = models

    _, X, Y, _, _ = setup_data_transynergy()

    split_func = (
        DataPreprocessorTranSynergy.regular_train_eval_test_split
    )  # only this needs to be changed to do full crossval
    fold_idx = 0
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

    for baseline_model in baseline_models:
        train_and_eval_model(
            [X_train],
            [y_train],
            X_vals=[X_val],
            y_vals=[y_val],
            X_test=X_test,
            y_test=y_test,
            fold_idx=fold_idx,
            model_name=baseline_model,
            paper="transynergy",
            timeout=timeout,
            n_iter=n_iter,
        )


def run_biomining(
    fold: int = 1, n_iter=15, timeout=120, models: list[str] | None = None
):
    handler = CVDatasetHandler(
        data_dir="external/predicting_synergy_nn/data", outer_fold=fold
    )
    if models is None:
        baseline_models = BASELINE_MODELS
    else:
        baseline_models = models

    for baseline_model in tqdm(baseline_models):
        X_trains = []
        y_trains = []
        X_vals = []
        y_vals = []

        X_test, y_test = handler.get_dataset(type="test", fold=fold)

        for inner_fold in range(1, 4):

            X_train, y_train = handler.get_dataset(type="train", fold=inner_fold)
            X_val, y_val = handler.get_dataset(type="val", fold=inner_fold)

            X_trains.append(X_train)
            y_trains.append(y_train)
            X_vals.append(X_val)
            y_vals.append(y_val)

        train_and_eval_model(
            X_trains,
            y_trains,
            X_vals=X_vals,
            y_vals=y_vals,
            X_test=X_test,
            y_test=y_test,
            fold_idx=fold,
            model_name=baseline_model,
            paper="biomining",
            timeout=timeout,
            n_iter=n_iter,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TransSynergy or Biomining experiments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transynergy", "biomining"],
        default="transynergy",
        help="Choose whether to run TransSynergy or Biomining.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=15,
        help="Number of iterations for hyperparameter tuning.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Timeout for model training in seconds."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number for biomining (only applicable for biomining mode).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=False,
        choices=BASELINE_MODELS,
        help=f"Name of model to train and evaluate. If omitted, will run every model.",
    )

    args = parser.parse_args()

    set_seed(142)

    if args.model is not None:
        models = [args.model]
    else:
        models = None

    if args.mode == "transynergy":
        run_transynergy(timeout=args.timeout, n_iter=args.n_iter, models=models)
    elif args.mode == "biomining":
        run_biomining(
            fold=args.fold, timeout=args.timeout, n_iter=args.n_iter, models=models
        )
