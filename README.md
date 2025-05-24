# Drug-Inhibitor-Cancer-Treatment

This project explores **drug combination therapies** in cancer treatment using deep learning. It includes multiple submodules focused on **prediction modeling**, **explainability**, and **validation** across different architectures like **TranSynergy** and **Biomining Neural Networks**.

We have incorporated existing repositories as subtrees to retain full code control while preserving their commit histories.

---

## 📦 Submodules Overview

### 🔬 1. TranSynergy (`external/drug_combination`)

A Transformer-based model ([Liu & Xie, 2021]) for predicting cancer drug synergy.

#### 🚀 Motivation

> Tumors often develop resistance to therapies. This model aims to discover effective drug combinations that overcome resistance mechanisms.

#### ⚙️ Environment Setup

```bash
conda create -yn trans_synergy python=3.11
conda activate trans_synergy
# Install PyTorch based on system
conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia   # For CUDA
conda install pytorch==2.4.1 cpuonly -c pytorch                       # For CPU-only
pip install -e .
```

#### 📂 Dataset & Model Weights

- Download synergy scores:
  ```bash
  zenodo_get 10.5281/zenodo.4789936 -o ./data/synergy_score
  ```

- Pull LFS files:
  ```bash
  git lfs install
  git lfs pull
  ```

- Pretrained weights: [Download via Google Drive](https://drive.google.com/file/d/1wZ-1KFirBBdy1egMXxBB7Sgx0uQm6Jue/view?usp=sharing)

#### 🧪 Training

```bash
# Choose one of the following feature types:
python main.py --settings_choice='gene_dependency'
python main.py --settings_choice='gene_expression'
python main.py --settings_choice='netexpress'
```

---

### 💡 2. Explainability (`external/explainability`)

This submodule supports explainability for the drug synergy models using both model-specific and model-agnostic techniques.

#### ✅ Supported Models

- TranSynergy
- Biomining Neural Network

#### 🔍 Methods

- Activation Maximization
- Anchors (only for Biomining due to computational cost)

#### ▶️ Running Explanations

```bash
python -m explainability.main --model biomining --method activation_max
```

---

### 🧠 3. Biomining Neural Network (`external/biomining_synergy`)

A deep learning approach for predicting breast cancer drug synergy based on **target-protein inhibition profiles**.

#### 🧪 Nested Cross-Validation

- Outer loop: 3 folds (training + testing)
- Inner loop: 3 folds (grid search on validation)

#### 📁 Project Structure

```
biomining_synergy/
├── data/
├── src/
├── configs/
├── scripts/
└── outputs/
```

#### ⚙️ Installation

```bash
python setup_venv.py
```

#### ▶️ Usage

**Train on a single fold**  
```bash
python -m src.training.trainer --config configs/base.yaml
```

**Train on multiple folds**  
```bash
python scripts/run_training.py --folds 1,2,3 --config configs/base.yaml
```

**Hyperparameter tuning**  
```bash
python -m src.training.hyperparameter --config configs/grid.yaml
```

**Full Grid Search (multiple folds/splits)**  
```bash
python scripts/run_grid_search.py --folds 1,2,3 --splits 1,2,3 --config configs/grid.yaml
```

---

## 🔧 Git Subtree Setup (for Contributors)

To add a subtree manually:

```bash
git remote add drug_combination_repo https://github.com/qiaoliuhub/drug_combination.git
git fetch drug_combination_repo
git subtree add --prefix=external/drug_combination drug_combination_repo main
```

---

## 📚 References

- Liu and Xie (2021): TranSynergy – Transformer-based drug synergy prediction.
- Srithanyarat et al. (2024): Biomining neural network for synergy prediction in breast cancer.

---
