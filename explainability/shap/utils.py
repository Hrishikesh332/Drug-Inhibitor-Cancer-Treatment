from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
import torch

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
