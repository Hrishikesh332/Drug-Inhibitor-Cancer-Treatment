import argparse
from typing import Callable
from dataclasses import dataclass
from explainability.utils import load_transynergy_model, load_biomining_model
from explainability.am import run_activation_maximization


@dataclass
class ModelConfig:
    name: str
    loader: Callable
    path: str

MODEL_REGISTRY = {
    "biomining": ModelConfig(
        name="biomining",
        loader=load_biomining_model,
        path="external/predicting_synergy_nn/outputs/models/best_f1.pt"
    ),
    "transynergy": ModelConfig(
        name="transynergy",
        loader=load_transynergy_model,
        path="external/drug_combination/trans_synergy/data/models/fold_test_model.pt"
    )
}


def load_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = MODEL_REGISTRY[model_name]
    return cfg.loader(model_path=cfg.path)

def run_explanation(model, model_name, method, input_data):
    if method == 'shap':
        raise ValueError(f"A little more work is needed to run SHAP.")
    elif method == 'anchors':
        raise ValueError(f"A little more work is needed to run anchors.")
    elif method == 'activation_max':
        run_activation_maximization(model, model_name)
    elif method == 'integrated_gradients':
        raise ValueError(f"A little more work is needed to run integrated gradients.")
    else:
        raise ValueError(f"Unknown explanation method: {method}")

def main():
    parser = argparse.ArgumentParser(description="Run explainability on model outputs")
    parser.add_argument('--model', type=str, required=True,
                        choices=['transynergy', 'biomining'],
                        help='Which model to explain')
    parser.add_argument('--method', type=str, required=True,
                        choices=['shap', 'anchors', 'activation_max', 'integrated_gradients'],
                        help='Which explainability method to use')

    args = parser.parse_args()

    model = load_model(args.model)
    model.eval()

    input_data = None # TODO: Load your input data here

    if args.method in ['shap', 'anchors', 'activation_max', 'integrated_gradients']:
        run_explanation(model, parser.model, args.method, input_data)
    else:
        run_explanation(model, parser.model, args.method, input_data=None)

if __name__ == '__main__': 
    main()
