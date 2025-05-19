import shutil
from pathlib import Path
from typing import Literal

import torch
import wandb

import trans_synergy
from trans_synergy.models.trans_synergy.attention_main import setup_data as setup_data_transynergy
from external.predicting_synergy_nn.src.utils.data_loader import CVDatasetHandler
from external.predicting_synergy_nn.src.models.architectures import SynergyModel


def load_transynergy_model(model_path: str, map_location: str = 'cpu'):
    """
    Load the  model from the specified path.

    Args:
        model_path (str): Path to the saved model file.
        map_location: Device mapping for loading the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded  model.
    """
    model = torch.load(model_path, map_location=map_location, weights_only=False)
    model.eval()
    return model
    

def load_biomining_model(model_path: str, map_location: str = 'cpu'):
    """
    Load the Biomining model from the specified path.
    """
    state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
    model = SynergyModel(arch = 'std', in_dim=33)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_transynergy_data(split:Literal['train', 'test'] = 'train'):
    """
    Load the TransyNet data from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        DataFrame: Loaded data.
    """
    if split == 'test':
        split = 'test1'
    std_scaler, X, Y, _, _= setup_data_transynergy()
    split_func = trans_synergy.data.trans_synergy_data.DataPreprocessor.regular_train_eval_test_split
    partition = next(split_func(fold='fold', test_fold=4))
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

def load_biomining_data(split:Literal['train', 'test'] = 'train'):
    """
    Load the Biomining data from the specified path.
    """
    handler = CVDatasetHandler(data_dir='external/predicting_synergy_nn/data')
    if split == 'train':
        X_res, Y_res = handler.get_dataset(type='alltrain')
    elif split == 'test':
        X_res, Y_res = handler.get_dataset(type='test')
    return X_res, Y_res 

def save_with_wandb(object_path, name_of_object):
    try:
        wandb.save(object_path, policy="now")
    except OSError as e: # Windows throws OS errors because of symlinks https://github.com/wandb/wandb/issues/1370
        wandb_path = str(Path(wandb.run.dir) / f"{name_of_object}.pt")
        shutil.copy(object_path, wandb_path)
        wandb.save(wandb_path, base_path = wandb.run.dir)