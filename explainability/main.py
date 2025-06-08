import argparse
import logging
from typing import Callable
from dataclasses import dataclass

from explainability.utils import (
    load_transynergy_model,
    load_biomining_model,
    load_transynergy_data,
    load_biomining_data,
)
from explainability.am import run_activation_maximization
from explainability.shap import run_shap_explanation
from explainability.anchors import run_anchors

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

def load_data(model_name: str, split: str = 'train'):
    if model_name not in MODEL_DATA_REGISTRY:
        raise ValueError(f"No data loader registered for model: {model_name}")
    cfg = MODEL_DATA_REGISTRY[model_name]
    return cfg.data_loader(split=split)

def run_explanation(model, model_name, method, X_train, Y_train, X_test, Y_test, logger, kernel: bool = False, **kwargs):
    if method == 'shap':
        run_shap_explanation(
                            model=model,
                            paper=model_name,
                            X_train=X_train,
                            X_test=X_test,
                            logger=logger,
                            kernel=kernel, 
                            **kwargs
                        )
    elif method == 'anchors':
        for fraction_explained in ["all", "bottom_10_percent", "top_10_percent", "random"]:
            for threshold in [0.90, 0.95]:
                logger.info(f"Running anchors with fraction_explained={fraction_explained}, threshold={threshold}")
                run_anchors(
                            model = model, 
                            paper = model_name, 
                            X_train = X_train,
                            Y_train = Y_train,
                            X_test = X_test,
                            Y_test = Y_test,
                            logger = logger,
                            threshold=threshold,
                            fraction_explained=fraction_explained,
                            num_explanations=1000,
                        )
    elif method == 'activation_max':
        for regularization in [None, "l2", "l1"]:
            for maximize in [True, False]:
                logger.info(f"Running activation maximization with regularization={regularization}, maximize={maximize}")
                run_activation_maximization(
                    model = model, 
                    paper = model_name, 
                    X = X_train,
                    logger = logger,
                    regularization = regularization,
                    maximize = maximize,
                )
    elif method == 'integrated_gradients':
        raise NotImplementedError("Integrated gradients not yet implemented.")
    else:
        raise ValueError(f"Unknown explanation method: {method}")

def main():
    parser = argparse.ArgumentParser(description="Run explainability on model outputs")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    parser.add_argument('--model', 
                        type=str, 
                        default='transynergy',
                        choices=MODEL_DATA_REGISTRY.keys(),
                        help='Which model to explain')
    parser.add_argument('--method', 
                        type=str, 
                        default='anchors',
                        choices=['shap', 'anchors', 'activation_max', 'integrated_gradients'],
                        help='Which explainability method to use')
    parser.add_argument('--kernel', 
                        action='store_true',
                        help='Use KernelExplainer for SHAP (default is GradientExplainer)')

    args = parser.parse_args()

    model = load_model(args.model)
    model.eval()

    X_train, Y_train = load_data(args.model)
    X_test, Y_test = load_data(args.model, split='test')

    run_explanation(model = model, 
                    model_name = args.model, 
                    method = args.method, 
                    X_train= X_train,
                    Y_train= Y_train,
                    X_test= X_test,
                    Y_test= Y_test,
                    logger=logger,
                    kernel=args.kernel)


if __name__ == '__main__': 
    main()