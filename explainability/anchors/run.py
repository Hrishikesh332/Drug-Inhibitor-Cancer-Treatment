import os
from logging import Logger
from dataclasses import asdict
from typing import Literal

import numpy as np
import torch
import wandb
from alibi.explainers import AnchorTabular

from explainability.utils import save_with_wandb
from trans_synergy.utils import set_seed
from explainability.anchors.config import AnchorConfig
    
def run_anchors(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    Y: torch.Tensor,
    logger: Logger,
    **kwargs
):
    
    config = AnchorConfig(paper=paper, **kwargs)
    
    wandb.init(project=f"EXPLAINABILITY on {config.paper} anchors", 
               config=asdict(config),
               name=f"{paper}_num_exp_{config.num_explanations}_anchors_{config.threshold}",)


    device = next(model.parameters()).device

    set_seed(config.seed)

    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    
    if config.num_explanations is not None:
        X_sample = X_np[:config.num_explanations]
    else:
        X_sample = X_np
    bins = np.quantile(Y, [config.lowest_quantile_for_binning, config.highest_quantile_for_binning])
    bins_names = [f"lowest_qunatile_{config.lowest_quantile_for_binning}",
                  f"middle_qunatile_{config.lowest_quantile_for_binning}_{config.highest_quantile_for_binning}",
                  f"highest_qunatile_{config.highest_quantile_for_binning}"]
    
    def predict_fn(x: np.ndarray) -> np.ndarray:
        num_samples = x.shape[0]
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        if config.paper == "transynergy":
            x_tensor = x_tensor.view(num_samples, 3, config.cell_drug_feat_len_transynergy).clone().detach().requires_grad_(True)
        elif config.paper == "biomining":
            x_tensor = x_tensor.view(num_samples, config.cell_drug_feat_len_biomining).clone().detach().requires_grad_(True)

        with torch.no_grad():
            out = model(x_tensor)
        return np.digitize(out, bins).flatten()

    feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
    explainer = AnchorTabular(predict_fn, feature_names)
    explainer.fit(X_np)

    explanations = []
    for i, x in enumerate(X_sample):
        logger.info(f"Explaining instance {i}")
        explanation = explainer.explain(x, threshold=config.threshold)
        explanations.append(explanation)
        
        prediction = bins_names[int(explanation.raw['prediction'])]
        precision = int(np.array(explanation.precision).flatten())
        
        wandb.log({
            f"anchor_{i}/precision": precision,
            f"anchor_{i}/coverage": explanation.coverage,
            f"anchor_{i}/anchor": str(explanation.anchor),
            f"anchor_{i}/prediction": prediction,
            f"anchor_{i}/feature_names": str([feature_names[idx] for idx in explanation.raw['feature']])
        })

        logger.info(f"Anchor explanation {i}:")
        logger.info(f"  Anchor: {explanation.anchor}")
        logger.info(f"  Precision: {precision:.2f}")
        logger.info(f"  Coverage: {explanation.coverage:.2f}")
        logger.info(f"  Prediction: {prediction}")

        # Save individual explanation as text
        text = (
            f"Anchor explanation for instance {i}:\n"
            f"Anchor: {explanation.anchor}\n"
            f"Precision: {precision:.2f}\n"
            f"Coverage: {explanation.coverage:.2f}\n"
            f"Prediction: {prediction}\n"
        )
        save_path = f"explainability/anchors/results/{config.paper}_anchor_{i}.txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text)

        save_with_wandb(save_path, name_of_object=f"{config.paper}_anchor_{i}.txt")

    wandb.finish()
