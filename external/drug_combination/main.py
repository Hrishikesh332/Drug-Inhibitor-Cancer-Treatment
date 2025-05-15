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
):
    if settings_choice not in _CLI_INPUT_TO_CONFIG_MAP:
        raise ValueError(f"The setting {settings_choice} is not recognised. Please choose one of: {_CLI_INPUT_TO_CONFIG_MAP.keys()}")
    settings_obj = _CLI_INPUT_TO_CONFIG_MAP[settings_choice]
    settings.configure(settings_obj)
    configure_logging(settings_obj.logfile)
    run(use_wandb)

if __name__ == "__main__":
    fire.Fire(train_trans_synergy)