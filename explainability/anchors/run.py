import os
import wandb
import torch
import numpy as np
from alibi.explainers import AnchorTabular
from explainability.utils import save_with_wandb
from trans_synergy.utils import set_seed
from logging import Logger
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from explainability.anchors.config import AnchorConfig
    
def run_anchors(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    logger: Logger,
    **kwargs
):
    
    input_shape = X[0].shape
    config = AnchorConfig(paper=paper, **kwargs)
    
    wandb.init(project=f"EXPLAINABILITY on {config.paper} anchors", 
               config=asdict(config),
               name=f"{paper}_num_exp_{config.num_explainations}_anchors_reg_{config.thereshold}",)


    device = next(model.parameters()).device

    set_seed(config.seed)

    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    X_sample = X_np[:config.num_explanations] # TODO; fix and get indices of points you are interested in!

    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        if config.paper == "transynergy":
            input_tensor = input_tensor.view(1, 3, config.cell_drug_feat_len_transynergy).clone().detach().requires_grad_(True)
        elif config.paper == "biomining":
            input_tensor = input_tensor.view(1,  config.cell_drug_feat_len_biomining).clone().detach().requires_grad_(True)

        with torch.no_grad():
            out = model(x_tensor)
        return (out.sigmoid().cpu().numpy() > 0.5).astype(int)

    feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
    explainer = AnchorTabular(predict_fn, feature_names)
    explainer.fit(X_np)

    explanations = []
    for i, x in enumerate(X_sample):
        logger.info(f"Explaining instance {i}")
        explanation = explainer.explain(x, threshold=config.threshold)
        explanations.append(explanation)

        wandb.log({
            f"anchor_{i}/precision": explanation.precision,
            f"anchor_{i}/coverage": explanation.coverage,
            f"anchor_{i}/anchor": str(explanation.anchor),
            f"anchor_{i}/feature_names": str([feature_names[idx] for idx in explanation.feature])
        })

        logger.info(f"Anchor explanation {i}:")
        logger.info(f"  Anchor: {explanation.anchor}")
        logger.info(f"  Precision: {explanation.precision:.2f}")
        logger.info(f"  Coverage: {explanation.coverage:.2f}")

        # Save individual explanation as text
        text = (
            f"Anchor explanation for instance {i}:\n"
            f"Anchor: {explanation.anchor}\n"
            f"Precision: {explanation.precision:.2f}\n"
            f"Coverage: {explanation.coverage:.2f}\n"
        )
        save_path = f"explainability/anchors/results/{config.paper}_anchor_{i}.txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text)

        save_with_wandb(save_path, name_of_object=f"{config.paper}_anchor_{i}.txt")

    wandb.finish()
