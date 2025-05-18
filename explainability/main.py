import argparse
from typing import Callable
from dataclasses import dataclass

from explainability.utils import (
    load_transynergy_model,
    load_biomining_model,
    load_transynergy_data,
    load_biomining_data,
)
from explainability.am import run_activation_maximization


@dataclass
class ModelAndDataConfig:
    name: str
    model_loader: Callable
    model_path: str
    data_loader: Callable

MODEL_DATA_REGISTRY = {
    "biomining": ModelAndDataConfig(
        name="biomining",
        model_loader=load_biomining_model,
        model_path="external/predicting_synergy_nn/outputs/models/best_f1.pt",
        data_loader=load_biomining_data
    ),
    "transynergy": ModelAndDataConfig(
        name="transynergy",
        model_loader=load_transynergy_model,
        model_path="external/drug_combination/trans_synergy/data/models/fold_test_model.pt",
        data_loader=load_transynergy_data
    )
}


def load_model(model_name: str):
    if model_name not in MODEL_DATA_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    cfg = MODEL_DATA_REGISTRY[model_name]
    return cfg.model_loader(model_path=cfg.model_path)

def load_data(model_name: str):
    if model_name not in MODEL_DATA_REGISTRY:
        raise ValueError(f"No data loader registered for model: {model_name}")
    cfg = MODEL_DATA_REGISTRY[model_name]
    return cfg.data_loader()


def run_explanation(model, model_name, method, X, Y):
    if method == 'shap':
        raise NotImplementedError("SHAP explainability not yet implemented.")
    elif method == 'anchors':
        raise NotImplementedError("Anchors explainability not yet implemented.")
    elif method == 'activation_max':
        run_activation_maximization(model, model_name, X)
    elif method == 'integrated_gradients':
        raise NotImplementedError("Integrated gradients not yet implemented.")
    else:
        raise ValueError(f"Unknown explanation method: {method}")

def main():
    parser = argparse.ArgumentParser(description="Run explainability on model outputs")
    parser.add_argument('--model', 
                        type=str, 
                        default='transynergy',
                        choices=MODEL_DATA_REGISTRY.keys(),
                        help='Which model to explain')
    parser.add_argument('--method', 
                        type=str, 
                        default='activation_max',
                        choices=['shap', 'anchors', 'activation_max', 'integrated_gradients'],
                        help='Which explainability method to use')

    args = parser.parse_args()

    model = load_model(args.model)
    model.eval()

    X, Y = load_data(args.model)

    run_explanation(model, args.model, args.method, X, Y)


if __name__ == '__main__': 
    main()
