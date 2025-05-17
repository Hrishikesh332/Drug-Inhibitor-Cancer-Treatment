import torch
import tqdm
import trans_synergy
from trans_synergy.models.trans_synergy.attention_main import setup_data as setup_data_transynergy


def load_transynergy_model(model_path: str, map_location: str = 'cpu'):
    """
    Load the  model from the specified path.

    Args:
        model_path (str): Path to the saved model file.
        map_location: Device mapping for loading the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded  model.
    """
    model = torch.load(model_path, map_location=map_location)
    model.eval()
    return model
    

def load_biomining_model():
    """
    Load the Biomining model from the specified path.
    """
    # Placeholder for actual model loading logic
    return "Biomining Model Loaded"


def load_data_transynergy(data_path: str):
    """
    Load the TransyNet data from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        DataFrame: Loaded data.
    """
    std_scaler, X, Y, _, _ = setup_data_transynergy()
    split_func = trans_synergy.data.trans_synergy_data.DataPreprocessor.regular_train_eval_test_split

    for fold_idx, partition in enumerate(tqdm(split_func(fold='fold', test_fold=4), desc="Folds", total=1)):
        std_scaler.fit(Y[partition['train']])
    Y = std_scaler.transform(Y)
    return X, Y

def load_data_biomining(data_path: str):
    """
    Load the Biomining data from the specified path.
    """
    # Placeholder for actual data loading logic
    return "Biomining Data Loaded"