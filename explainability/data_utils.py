from typing import Literal
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from pathlib import Path

import trans_synergy
from trans_synergy.models.trans_synergy.attention_main import setup_data as setup_data_transynergy 
from external.predicting_synergy_nn.src.utils.data_loader import CVDatasetHandler


def load_transynergy_data(split: Literal['train', 'test'] = 'train'):
    """
    Load the TransyNet data from the specified path.

    Args:
        split: Which split to load ('train' or 'test')

    Returns:
        Tuple of (X, Y) data tensors
    """
    
    if split == 'test':
        split = 'test1'
    std_scaler, X, Y, _, _ = setup_data_transynergy ()
    split_func = trans_synergy.data.trans_synergy_data.DataPreprocessor.regular_train_eval_test_split
    partition = split_func(fold_col_name="fold", test_fold=4, evaluation_fold=0)
    partition_indices = {
        'train': partition[0],
        'test1': partition[1],
        'test2': partition[2],
        'eval1': partition[3],
        'eval2': partition[4]
    }
    
    std_scaler.fit(Y[partition_indices['train']])
    Y = std_scaler.transform(Y)
    X_res = X[partition_indices[split]]
    Y_res = Y[partition_indices[split]]
    
    return X_res, Y_res


def load_biomining_data(split: Literal['train', 'test'] = 'train'):
    """
    Load the Biomining data from the specified path.
    """
    handler = CVDatasetHandler(data_dir='external/predicting_synergy_nn/data')
    if split == 'train':
        X_res, Y_res = handler.get_dataset(type='alltrain')
    elif split == 'test':
        X_res, Y_res = handler.get_dataset(type='test')
    return X_res, Y_res

def load_biomining_cell_line_data(split: Literal['train', 'test'] = 'train'):
    """
    Load the Biomining cell line data from the specified path.
    """
    handler = CVDatasetHandler(data_dir='external/predicting_synergy_nn/data')
    if split == 'train':
        cell_lines = handler.all_train_dataset.get_cell_lines() 
    elif split == 'test':
        cell_lines = handler.test_dataset.get_cell_lines()
    return  cell_lines


def reshape_transynergy_input(X: torch.Tensor, logger, name: str) -> torch.Tensor:
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
    This way, it reduces computation time while maintaining explanation quality.
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
