import shutil
from pathlib import Path
from typing import Literal
from logging import Logger

import torch
import wandb
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from external.predicting_synergy_nn.src.models.architectures import SynergyModel


def load_transynergy_model(model_path: str, map_location: str = 'cpu'):
    """
    Load the model from the specified path.

    Args:
        model_path (str): Path to the saved model file.
        map_location: Device mapping for loading the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = torch.load(model_path, map_location=map_location, weights_only=False)
    model.eval()
    return model


def load_biomining_model(model_path: str, map_location: str = 'cpu'):
    """
    Load the Biomining model from the specified path.

    Args:
        model_path (str): Path to the saved model file.
        map_location: Device mapping for loading the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = SynergyModel(in_dim=33)
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    return model


def save_with_wandb(object_path, name_of_object):
    try:
        wandb.save(object_path, policy="now")
    except OSError as e: # Windows throws OS errors because of symlinks https://github.com/wandb/wandb/issues/1370
        wandb_path = str(Path(wandb.run.dir) / f"{name_of_object}.pt")
        shutil.copy(object_path, wandb_path)
        wandb.save(wandb_path, base_path = wandb.run.dir)
        
        
def reshape_transynergy_input(X: torch.Tensor, logger: Logger, name: str) -> torch.Tensor:
    """
    TranSynergy model expects input as a 3D tensor shaped (batch_size, 3, feature_dim),
    where the three parts represent: Drug A features, Drug B features, Cell line features.

    So we must reshape the input - 2D tensor (batch_size, total_features) - by splitting the last 
    dimension into 3 parts, each of size feature_dim = total_features / 3, to fix the dimension mismatch 
    and allow the model to properly separate drug and cell line features.
    """
    if X.ndim == 2:
        batch_size, total_features = X.shape
        if total_features % 3 != 0:
            raise ValueError(f"Expected {name} input features divisible by 3, got {total_features}")
        feature_dim = total_features // 3
        logger.info(f"Reshaping TranSynergy {name}: ({batch_size}, {total_features}) → ({batch_size}, 3, {feature_dim})")
        X = X.view(batch_size, 3, feature_dim)
    return X

def select_representative_samples(X: torch.Tensor, num_samples: int):
    """
    This method uses PCA to reduce dimensionality and MiniBatchKMeans clustering to find 
    cluster centers. It then selects the closest original samples to these centers as representatives.

    We chose to use these methods, because using all samples is expensive and often redundant. 
    This way, it reduces SHAP computation time while maintaining explanation quality.
    """
    X_np = X.detach().cpu().numpy()
    pca = PCA(n_components=min(50, X_np.shape[1]))
    X_reduced = pca.fit_transform(X_np)

    n_clusters = min(num_samples, X_np.shape[0])

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100, max_iter=100)
    kmeans.fit(X_reduced)

    centers = kmeans.cluster_centers_
    indices = []
    for center in centers:
        distances = np.linalg.norm(X_reduced - center, axis=1)
        idx = np.argmin(distances)
        indices.append(idx)
    indices = list(set(indices))
    selected_samples = X[indices]
    return selected_samples