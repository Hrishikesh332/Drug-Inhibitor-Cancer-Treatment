import torch
from tqdm import tqdm

from zennit.core import Composite, BasicHook
from zennit.canonizers import NamedMergeBatchNorm

from explainability.lrp.rules import lrp_rule_for_biomining
from explainability.utils import (
    load_biomining_model,
)
from explainability.data_utils import (
    load_biomining_data
)


class ConservationChecker(BasicHook):
    def __init__(self):
        super().__init__(
            # No forward change
            input_modifiers=[lambda x: x],
            output_modifiers=[lambda x: x],
            param_modifiers=[lambda p, _: p],
            gradient_mapper=lambda grad_out, outputs: grad_out,
            reducer=lambda inputs, grads: grads[0],
            name = 'conservation_checker'
        )
        self.rin = None
        self.rout = None

    def backward(self, module, grad_input, grad_output):
        self.rin = grad_output[0].sum().item()
        if grad_output[0] is not None and grad_input[0] is not None:
            self.rout = grad_input[0].sum().item()

def biomining_explain_with_conservation(model, inputs):
    composite = Composite(module_map=lrp_rule_for_biomining,
                          canonizers=[NamedMergeBatchNorm([(['net.0'], 'net.1'),
                                                            (['net.4'], 'net.5'),
                                                            (['net.8'], 'net.9'),
                                                            (['net.12'], 'net.13'),
                                                            (['net.16'], 'net.17')])]
                          )

    model.eval()
    conservation_violations = []
    for i in tqdm(range(inputs.shape[0]), desc="LRP samples"):
        inp = inputs[i].unsqueeze(0).clone().detach().requires_grad_(True)
        
        checkers = {}
        # attach checker to ever layer
        for name, module in model.named_modules():
            checker = ConservationChecker()
            handle = module.register_backward_hook(checker.backward)
            checkers[module] = (name, checker, handle)

        with composite.context(model):
            model.eval()
            out = model(inp)
            torch.autograd.set_detect_anomaly(True)
            out.backward()
            print(f"Howdy Fella, we will now check the conservation property for sample {i}! Enjoy Buddy :D")
            print(f"Sample {i}: Relevance sum: {inp.grad.detach().sum().item()}")

        # inspect conservation per layer
        batch_violations = {}
        for module, (name, checker, handle) in checkers.items():
            diff = checker.rin - checker.rout
            print(f"Layer: {name}, Rin: {checker.rin}, Rout: {checker.rout}, Diff: {diff}")
            if abs(diff) > 1e-5:
                batch_violations[name] = diff
            handle.remove()
            
        # print violations 
        if batch_violations:
            print(f"Sad news buddy D: Sample {i} has conservation violations: {batch_violations}")
        else:
            print(f"Fella! Sample {i} has no conservation violations.")

        if batch_violations:
            conservation_violations.append((i, batch_violations))

    return conservation_violations

if __name__ == "__main__":
    # Example usage
    model = load_biomining_model("external/predicting_synergy_nn/outputs/models/best_f1.pt")  # Load your model here
    X, Y = load_biomining_data(split='test')
    sample = torch.tensor(X[0:5,:],  dtype=torch.float32)
    violations = biomining_explain_with_conservation(model, sample)