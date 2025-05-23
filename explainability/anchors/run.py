from logging import Logger
from dataclasses import asdict
from typing import Literal


import torch
import wandb
from alibi.explainers import AnchorTabular

from explainability.anchors.utils import compute_bins, explain_sample_set, sample_data, build_predict_fn
from trans_synergy.utils import set_seed
from explainability.anchors.config import AnchorConfig


def run_anchors(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    logger: Logger,
    **kwargs
):
    config = AnchorConfig(paper=paper, **kwargs)
    set_seed(config.seed)

    wandb.init(
        project=f"EXPLAINABILITY on {config.paper} anchors",
        config=asdict(config),
        name=f"{paper}_num_exp_{config.num_explanations}_anchors_{config.threshold}",
    )

    if torch.cuda.is_available():
        logger.info("Using GPU for model inference.")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU for model inference.")
        device = torch.device("cpu")
    model.to(device)
    
    X_train_np = sample_data(X_train, Y_train, config.fraction_explained, num_samples=config.num_explanations)
    X_test_np = sample_data(X_test, Y_test, config.fraction_explained, num_samples=config.num_explanations)
    
    bins, bins_names = compute_bins(
        Y_train, config.lowest_quantile_for_binning, config.highest_quantile_for_binning
    )

    predict_fn = build_predict_fn(model, device, config, bins)

    explainer = AnchorTabular(predict_fn, 
                              config.feature_names)
    explainer.fit(X_train_np)

    explain_sample_set(X_train_np, explainer, config, logger, bins_names, 
                       save_dir=f"explainability/anchors/results_train/sample_{config.fraction_explained}_thres_{config.threshold}", 
                       suffix_progress_bar="train")
    explain_sample_set(X_test_np, explainer, config, logger, bins_names, 
                       save_dir=f"explainability/anchors/results_test/sample_{config.fraction_explained}_thres_{config.threshold}", 
                       suffix_progress_bar="test")

    wandb.finish()
