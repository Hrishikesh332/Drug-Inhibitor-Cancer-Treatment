import torch
import shap
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import asdict
from typing import Literal
import wandb
from logging import Logger
from explainability.shap.utils import select_representative_samples, reshape_transynergy_input
from explainability.shap.config import SHAPExplanationConfig


def run_shap_explanation(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    logger: Logger,
    **kwargs
):
    config = SHAPExplanationConfig(paper=paper, **kwargs)
    device = next(model.parameters()).device
    model.eval()
    model = model.to(device)

    wandb.init(
        project=f"EXPLAINABILITY on SHAP",
        config=asdict(config),
        name=f"{paper}_shap_explanation",
    )

    logger.info(f"Running SHAP (GradientExplainer) for model: {paper}")

    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)

    X_train = X_train.to(device)
    X_test = X_test.to(device)

    # sample size is equal to percentage but at least the minimum and no more than the maximum configured limits
    background_size = min(max(config.min_background, int(X_train.shape[0] * config.samples_percentage)), config.max_background) 
    test_size = min(max(config.min_test, int(X_test.shape[0] * config.samples_percentage)), config.max_test)

    if config.paper == "transynergy":
        X_train_sample = X_train.view(X_train.shape[0], -1) 
        X_test_sample = X_test.view(X_test.shape[0], -1)    
    else:
        X_train_sample = X_train
        X_test_sample = X_test
        
    background = select_representative_samples(X_train_sample, background_size)
    test_inputs = select_representative_samples(X_test_sample, test_size)

    if config.paper == "transynergy":
        X_train = reshape_transynergy_input(X_train, logger, "X_train")
        X_test = reshape_transynergy_input(X_test, logger, "X_test")
        background = background.view(background.shape[0], 3, -1)
        test_inputs = test_inputs.view(test_inputs.shape[0], 3, -1)

    explainer = shap.GradientExplainer(model, background)

    shap_values = []
    for i in tqdm(range(test_inputs.shape[0]), desc="SHAP explanation"):
        val = explainer.shap_values(test_inputs[i:i+1])
        shap_values.append(val)

    if config.paper == "transynergy":
        shap_values_matrix = np.array([sample.reshape(-1) for sample in shap_values])
    elif config.paper == "biomining":
        shap_values_matrix = np.vstack([sv[0] for sv in shap_values])

    output_dir = Path(f"explainability/shap/results/{paper}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(test_inputs, torch.Tensor):
        inputs_np = test_inputs.detach().cpu().numpy()
        if inputs_np.ndim == 3:
            inputs_np = inputs_np.reshape(inputs_np.shape[0], -1)
    else:
        inputs_np = np.array(test_inputs)

    npz_path = output_dir / "shap_complete.npz"
    np.savez_compressed(
        npz_path,
        shap_values=shap_values_matrix,
        inputs=inputs_np,
        feature_names=np.array(config.feature_names)
    )

    logger.info(f"Saved complete SHAP data at {output_dir / 'shap_complete.npz'}")

    artifact = wandb.Artifact(f"{paper}_shap_complete_data", type="shap_data")
    artifact.add_file(str(npz_path))
    wandb.log_artifact(artifact)

    wandb.finish()