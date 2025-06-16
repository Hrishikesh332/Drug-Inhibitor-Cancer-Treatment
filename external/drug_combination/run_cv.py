# -----------------------------------------------------------------------------
# Flexible Cross-Validation Runner 
#
# This function exposes a highly configurable cross-validation pipeline via CLI,
# thanks to Python Fire. You can easily tweak your experiment setup without
# touching the code:
#
#   --settings_choice   Choose which data setting to use (gene_dependency, gene_expression, netexpress)
#   --fold_col_name     Specify the column name for fold assignment (default: "fold")
#   --use_wandb         Enable or disable Weights & Biases logging (default: True)
#   --n_params_in_grid  Number of hyperparameter combinations to try (default: 10)
#   --test_fold         Which fold to use as the test set (default: 4)
#   --epochs_in_cv      Number of epochs for each cross-validation run (default: 100)
#
# Example usage:
#   python run_cv.py --settings_choice=gene_expression --use_wandb=True --n_params_in_grid=20
#
# All arguments are optional and have sensible defaults, making experimentation fast and reproducible!
# -----------------------------------------------------------------------------

import logging

import fire

from trans_synergy import settings
from trans_synergy.models.crossval import run


def configure_logging(log_file_path: str):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh = logging.FileHandler(log_file_path, mode='w+')
    fh.setFormatter(fmt=formatter)
    logger = logging.getLogger("Drug Combination")
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


_CLI_INPUT_TO_CONFIG_MAP = {
    "gene_dependency": settings.gene_dependency_setting,
    "gene_expression": settings.gene_expression_setting,
    "netexpress":      settings.net_setting,
}

def run_crossvalidation(
        settings_choice: str = "gene_dependency",
        fold_col_name: str = "fold",
        use_wandb: bool = True,
        n_params_in_grid: int = 10,
        test_fold: int = 4,
        epochs_in_cv: int = 10
):
    if settings_choice not in _CLI_INPUT_TO_CONFIG_MAP:
        raise ValueError(f"The setting {settings_choice} is not recognised. Please choose one of: {_CLI_INPUT_TO_CONFIG_MAP.keys()}")
    settings_obj = _CLI_INPUT_TO_CONFIG_MAP[settings_choice]
    settings.configure(settings_obj)
    configure_logging(settings_obj.logfile)
    run(use_wandb, 
        fold_col_name=fold_col_name,
        n_params_in_grid=n_params_in_grid,
        test_fold=test_fold,
        epochs_in_cv=epochs_in_cv)

if __name__ == "__main__":
    fire.Fire(run_crossvalidation)