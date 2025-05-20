# Explainability

This submodule provides explainability tools for drug synergy prediction models using both **model-agnostic** and **model-specific** techniques. It supports analysis of the following models:

- **TranSynergy** ([Liu and Xie, 2021]; Transformer + attention model)
- **Biomining Neural Network** ([Srithanyarat et al., 2024]; BMC BioData Mining)

## Supported Methods

- **Activation Maximization**
- **Anchors (only for Biomining, too expensive for Transynergy)**

## Running an Explanation

You can run an explainability experiment using:

```bash
python -m explainability.main --model biomining --method activation_max
```
