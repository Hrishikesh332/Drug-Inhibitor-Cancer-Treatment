from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
import torch
from typing import Literal
from pathlib import Path
from logging import Logger

def load_shap_data(
    paper: Literal["biomining", "transynergy"],
    base_dir_path: str = "./explainability/shap/results",
):
    npz_path = Path(base_dir_path) / paper / "shap_complete.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"SHAP complete data file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    shap_values = data["shap_values"]
    inputs = data["inputs"]
    feature_names = data["feature_names"].tolist()
    return shap_values, inputs, feature_names

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
    
