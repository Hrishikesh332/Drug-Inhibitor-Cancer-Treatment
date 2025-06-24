# Drug-Inhibitor-Cancer-Treatment

This project leverages deep learning to explore drug combination therapies for cancer treatment, focusing on data prepration, prediction modeling and explainaiblity. It integrates submodules like **TranSynergy** and **Biomining**.

## Table of Contents
- [Motivation](#motivation)
- [Submodules](#submodules)
  - [TranSynergy](#transynergy)
  - [Biomining Neural Network](#biomining-neural-network)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Data & Dataset](#data--dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Data Leakage](#data-leakage)
- [Baseline Results](#baseline)
  - [Process Overview](#process-overview)
- [Explainability](#explainability)
- [References](#references)

---

## Motivation
Tumors often develop resistance to single-drug therapies, limiting treatment efficiency. This project aims to identify effective drug combinations using machine learning and deep learning models.
## Submodules

### TranSynergy
**Location**: `external/drug_combination`  
A Transformer-based model ([Liu & Xie, 2021]) for predicting drug synergy in cancer treatment. It uses features like:
- **Gene dependency**: Measures cell line reliance on specific genes.
- **Gene expression**: Quantifies gene activity levels.
- **Netexpress**: Network-based expression integrating interaction data.

### Biomining Neural Network
**Location**: `external/biomining_synergy`  
A deep learning model for predicting drug synergy in breast cancer based on target-protein inhibition profiles, using nested cross-validation for robust evaluation.

---

## Prerequisites
- **Tools**: `conda`, `git`, `git-lfs`, `wandb` (for baseline results)
- **Hardware**: GPU with CUDA support (optional, for faster training)
- **Python**: Version 3.11  

---

## Setup
For TranSynergy:
1. **Create and activate a conda environment**:
   ```bash
   conda create -n trans_synergy python=3.11
   conda activate trans_synergy
   ```
2. **Install PyTorch** (choose based on your system):
   ```bash
   # For GPU (CUDA 12.4)
   conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   # For CPU-only
   conda install pytorch==2.4.1 cpuonly -c pytorch
   ```
3. **Install project dependencies**:
   ```bash
   pip install -e .  # Installs the project in editable mode
   ```
4. **For baseline results**:
   ```bash
   pip install -r baselines/requirements.txt  # Installs additional dependencies (e.g., scikit-learn, catboost)
   ```

For Biomining Neural Network:
   ```bash
   python setup_venv.py  # Sets up the virtual environment and dependencies
   ```

---

## Data & Dataset
This section provides instructions for downloading the synergy score dataset used for training and evaluation. 

**Transynsergy**
Drug data is derived from DrugBank (drug-target interactions), ChEMBL (bioactivity data), and STRING (protein-protein interaction networks), encoding each drug as a binary vector representing interactions with 2401 selected genes. Cell line data is obtained from CCLE and GDSC (gene expression profiles) and Achilles and Sanger CRISPR (gene dependency scores), providing gene expression or dependency values for the same 2401 genes. Drug combination data, sourced from Merck drug synergy data, is structured as a matrix with three columns: two for drug profiles (binary vectors) and one for cell line features (gene expression or dependency, with a potential fourth column if both are included). The output is a continuous synergy score, indicating whether drug pairs are synergistic, additive, or antagonistic. This dataset enables the TranSynergy transformer-based model to predict synergy and deconvolve pathways via attention mechanisms.

1. **Download synergy scores** (drug combination efficacy data):
   ```bash
   zenodo_get 10.5281/zenodo.4789936 -o ./data/synergy_score
   ```
2. **Pull large files with Git LFS**:
   ```bash
   git lfs install
   git lfs pull
   ```
3. **Pretrained model weights**:  
   Download from [Google Drive](https://drive.google.com/file/d/1wZ-1KFirBBdy1egMXxBB7Sgx0uQm6Jue/view) and place in `external/drug_combination/data/weights/`.

**Biomining**
* Source: DrugComb database, containing 24,145 drug combination pairs tested on five breast cancer cell lines (MCF-7, T-47D, MDA-MB-468, BT-549, MDA-MB-231) with 98 unique drugs.divided into three folds in the outer loop of the nested CV, with one fold serving as the test dataset while the other two folds were further divided into three folds in the inner loop. During each round of the inner loop, two folds were used as a training dataset, while the other fold was used as a validation set in a grid search for the best hyperparameter set.
* Features: Concatenated inhibition profiles for 13 cancer-related proteins (e.g., EGFR, PIK3CA) per drug pair (26 features) plus cell line genomic features, generated using graph convolutional neural networks from DrugBank and PubChem data.
* Output: Continuous synergy score (e.g., Bliss score) for each drug pair-cell line combination. data can be found at this location `external/predicting_synergy_nn/data`.

ForBiomining data can be found at this location `external/predicting_synergy_nn/data`.

---

## Usage

### Training
**TranSynergy**:

Run training with different feature types:
```bash
python main.py --settings_choice='gene_dependency'  # Uses gene dependency features
python main.py --settings_choice='gene_expression'  # Uses gene expression profiles
python main.py --settings_choice='netexpress'       # Uses network-based expression
```

Here is how you can check your result: [Transynergy Result](./external/drug_combination/README.md)

**Biomining Neural Network**:
- Single fold:
  ```bash
  python -m src.training.trainer --config configs/base.yaml
  ```
- Multiple folds:
  ```bash
  python scripts/run_training.py --folds 1,2,3 --config configs/base.yaml
  ```

---

### Hyperparameter Tuning

Hyperparameter tuning is performed using a grid search script (hyperparameter.py) with a configuration file (grid.yml), optimizing parameters such as learning rates ([1e-5, 5e-5, 1e-4, 5e-4]), dropout rates ([0.3, 0.5]), and batch sizes ([32, 64, 100]) across ten folds to ensure robust synergy predictions. This dataset enables biomining of critical protein targets driving drug synergy in breast cancer treatment, with Integrated Gradients used to interpret deep neural network predictions by attributing synergy scores to input features. You can do hyperparameter tuninh using the below script.

Run grid search for hyperparameter optimization:
```bash
python -m src.training.hyperparameter --config configs/grid.yaml
```
For multiple folds and splits:
```bash
python scripts/run_grid_search.py --folds 1,2,3 --splits 1,2,3 --config configs/grid.yaml
```

---

## Data Leakage
At the start of the project, data leakage checks were conducted for both Biomining Neural Network and TranSynergy submodules to ensure robust model performance and prevent overfitting. These checks confirmed that training, validation, and test sets were properly separated, free from unintended overlaps or biases, and any identified issues were resolved.

- **Biomining Neural Network**:  
  - Checks for **correlation of data** to identify redundant or highly correlated features that could lead to leakage.  
  - Verifies **semantic integrity** to ensure data annotations (e.g., drug or protein profiles) are consistent and meaningful.  
  - Examines **overlapping** between train, validation, and test sets to prevent data leakage across folds in nested cross-validation.

- **TranSynergy**:  
  - Performs **structural integrity checks** to ensure the data (e.g., gene dependency, gene expression, or netexpress features) is correctly formatted and free from errors that could introduce leakage.  
  - Conducts **correlation analysis** to detect and mitigate highly correlated features that might bias predictions.

---

## Baseline:
We implemented traditional machine learning models in main.py and traditional_ml.py for drug synergy prediction in breast cancer. The baseline models used for benchmark are **CatBoost, Random Forest, SVM, Decision Tree, Ridge Regression, and k-NN** for deep learning approaches. The main.py script supports two modes: TranSynergy (using a fixed train-validation-test split with test fold 4 and evaluation fold 0) and Biomining (using nested cross-validation with inner folds 1–3 on outer fold 1). The traditional_ml.py script trains these models by iterating over a parameter grid defined in its get_model method (e.g., learning rates, tree depth), performing up to 15 iterations with a 60-second timeout per model for efficient fitting. Performance is evaluated using mean squared error, Pearson, and Spearman correlations, logged via Weights & Biases..

### Process Overview
Baseline models are trained by iterating over a parameter grid defined in `baselines/traditional_ml.py` (see `get_model` method)

To replicate baseline results:
1. Install dependencies for both `drug_combination` and `predicting_synergy_nn` projects.
2. Install baseline-specific dependencies:
   ```bash
   pip install -r baselines/requirements.txt
   ```
3. Run one of the following:
   ```bash
   python -m baselines.main --mode transynergy --n_iter 15 --timeout 60  # For TranSynergy
   python -m baselines.main --mode biomining --n_iter 15 --timeout 60    # For Biomining
   ```
4. To train a specific model (e.g., CatBoost):
   ```bash
   python -m baselines.main --mode transynergy --n_iter 100 --timeout 600 --model catboost
   ```

You can check more about result table and summary here : [Baseline Results](./baselines/BASELINES.md)

---

## Explainability

Explainability in machine learning refers to the set of techniques and tools used to understand, interpret, and visualize the decisions made by complex models—especially deep learning models. In the context of drug synergy prediction, explainability helps researchers and practitioners understand **why** a model makes a certain prediction, **which features** influence it most, and **how** reliable or generalizable those decisions might be. This transparency is particularly important in biomedical applications where model outputs may guide experimental or clinical decisions.

This section describes how to generate explanations for the drug synergy models using both **model-specific** and **model-agnostic** techniques.

**Supported Explainability Methods**:

- **Activation Maximization**:  
  A **model-specific** technique that generates synthetic input samples which maximize or minimize the activation of a particular output neuron. By optimizing the input space, we can visualize what kinds of patterns lead the model to predict strong synergy or antagonism. This method is useful for **understanding what the model has learned**, and is applicable to both TranSynergy and Biomining models.

- **Anchors**:  
  A **model-agnostic** technique that produces **IF-THEN rules** to explain a specific prediction. It identifies a set of key features (anchors) such that, when they are present, the model’s output is almost always the same. Anchors provide **high-precision, human-interpretable explanations** and are especially useful when decision stability is important. Due to computational cost, Anchors are currently used only with the Biomining model.

- **SHAP (SHapley Additive exPlanations)**:  
  A popular **model-agnostic** method based on cooperative game theory. SHAP assigns each feature an importance value—its **Shapley value**—representing how much it contributes to the prediction compared to a baseline. SHAP is valuable for both **global** and **local** interpretability and has been applied to both TranSynergy and Biomining.

- **Integrated Gradients**:  
  A **model-specific** technique designed for differentiable models like neural networks. It computes the gradient of the model’s output with respect to the input features, integrated over a straight-line path from a baseline (e.g., all zeros) to the actual input. Integrated Gradients provide **attribution scores** that highlight the relative importance of each feature in driving the prediction. Applicable to both TranSynergy and Biomining models.

Here is the location to check result of these meathods : `explainability\notebooks`.


**Running Explanations (Example)**:
```bash
python -m explainability.main --model biomining --method activation_max
python -m explainability.main --model transynergy --method integrated_gradients
```

**To run GSEA after running Shapley (Example)**
```bash
python -m explainability.shapley.gsea --paper biomining
```

---

## References
- Liu and Xie (2021): *TranSynergy – Transformer-based drug synergy prediction.* [Link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008653)
- Srithanyarat et al. (2024): *Biomining neural network for synergy prediction in breast cancer.* [Link](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-024-00359-z)
