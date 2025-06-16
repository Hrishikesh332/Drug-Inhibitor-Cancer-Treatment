"""
TranSynergy Training Launcher

This script provides a flexible command-line interface for training the
TranSynergy model with different data settings and fold configurations.

Thanks to Python Fire, you can easily customize your experiment by passing
arguments directly from the command line.

CLI Arguments:
    --settings_choice: Select the data setting.
    --use_wandb: Enable or disable Weights & Biases logging.
    --fold_col_name: Column name for fold assignment.
    --test_fold: Fold index used as the test set.
    --eval_fold: Fold index used for validation/evaluation.

Example:
    python main.py --settings_choice=gene_expression --use_wandb=False --fold_col_name=new_drug --test_fold=2 --eval_fold=1

All arguments are optional and have sensible defaults, making it easy to run
and reproduce experiments with different configurations.
"""
import logging

import fire

from trans_synergy import settings
from trans_synergy.models.trans_synergy.attention_main import run
from commons import configure_logging

_CLI_INPUT_TO_CONFIG_MAP = {
    "gene_dependency": settings.gene_dependency_setting,
    "gene_expression": settings.gene_expression_setting,
    "netexpress":      settings.net_setting,
}

def train_trans_synergy(
        settings_choice: str = "gene_dependency",
        use_wandb: bool = True,
        fold_col_name = "fold",
        test_fold = 4,
        eval_fold = 0
):
    """
    Launches training of the TranSynergy model with configurable settings.

    Args:
        settings_choice (str): Dataset setting to use ('gene_dependency', 'gene_expression', 'netexpress').
        use_wandb (bool): Whether to log the training run to Weights & Biases.
        fold_col_name (str): Column name used to define folds in the data.
        test_fold (int): Fold index to use as the test set.
        eval_fold (int): Fold index to use as the evaluation/validation set.

    Raises:
        ValueError: If the provided settings_choice is not supported.
    """
    if settings_choice not in _CLI_INPUT_TO_CONFIG_MAP:
        raise ValueError(f"The setting {settings_choice} is not recognised. Please choose one of: {_CLI_INPUT_TO_CONFIG_MAP.keys()}")
    settings_obj = _CLI_INPUT_TO_CONFIG_MAP[settings_choice]
    settings.configure(settings_obj)
    configure_logging(settings_obj.logfile)
    run(use_wandb,
        fold_col_name=fold_col_name,
        test_fold=test_fold,
        eval_fold=eval_fold)

if __name__ == "__main__":
    fire.Fire(train_trans_synergy)