import argparse
from explainability.utils import load_transynergy_model, load_biomining_model
from explainability.am import run_activation_maximization


def load_model(model_name):
    if model_name == 'transynergy':
        return load_transynergy_model()
    elif model_name == 'biomining':
        return load_biomining_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

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
