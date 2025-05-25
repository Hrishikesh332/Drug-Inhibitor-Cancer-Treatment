# -----------------------------------------------------------------------------
# TranSynergy Training Launcher 
#
# This script provides a flexible command-line interface for training the
# TranSynergy model with various data settings and cross-validation options.
# Thanks to Python Fire, you can easily customize your experiment by passing
# arguments directly from the command line:
#
#   --settings_choice   Select the data setting (gene_dependency, gene_expression, netexpress)
#   --use_wandb         Enable or disable Weights & Biases logging (default: True)
#   --fold_col_name     Specify the column name for fold assignment (default: "fold")
#   --test_fold         Choose which fold to use as the test set (default: 4)
#   --eval_fold         Choose which fold to use for evaluation (default: 0)
#
# Example usage:
#   python main.py --settings_choice=gene_expression --use_wandb=False --fold_col_name=new_drug --test_fold=2 --eval_fold=1
#
# All arguments are optional and have sensible defaults, making it easy to run
# and reproduce experiments with different configurations!
# -----------------------------------------------------------------------------

import logging

import fire

from trans_synergy import settings
from trans_synergy.models.trans_synergy.attention_main import run


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

def train_trans_synergy(
        settings_choice: str = "gene_dependency",
        use_wandb: bool = True,
        fold_col_name = "fold",
        test_fold = 4,
        eval_fold = 0
):
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