from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
import torch
from typing import Literal

def select_representative_samples(X: torch.Tensor, num_samples: int):
    device = X.device
    X_np = X.detach().cpu().numpy()
    if X_np.ndim == 3:
        X_np = X_np.reshape(X_np.shape[0], -1)

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
    
def load_shap_data(
    paper: Literal["biomining", "transynergy"],
    base_dir_path: str = "./explainability/shap/results",
):
    import numpy as np
    from pathlib import Path

    npz_path = Path(base_dir_path) / paper / "shap_complete.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"SHAP complete data file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    shap_values = data["shap_values"]
    inputs = data["inputs"]
    feature_names = data["feature_names"].tolist()
    return shap_values, inputs, feature_names
