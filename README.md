# Drug-Inhibitor-Cancer-Treatment

This project leverages deep learning to explore drug combination therapies for cancer treatment, focusing on data prepration, prediction modeling and explainaiblity. It integrates submodules like **TranSynergy** and **Biomining**.

## Table of Contents
- [Motivation](#motivation)
- [Submodules](#submodules)
  - [TranSynergy](#transynergy)
  - [Biomining Neural Network](#biomining-neural-network)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Data Leakage](#data-leakage)
- [Baseline Results](#baseline-results)
  - [Process Overview](#process-overview)
  - [Results Summary](#results-summary)
  - [Results Table](#results-table)
  - [Generate Markdown Results Table](#generate-markdown-results-table)
- [Explainability](#explainability)
- [Git Subtree Setup](#git-subtree-setup)
- [References](#references)

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


## Prerequisites
- **Tools**: `conda`, `git`, `git-lfs`, `wandb` (for baseline results)
- **Hardware**: GPU with CUDA support (optional, for faster training)
- **Python**: Version 3.11  

## Setup
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

## Dataset
This section provides instructions for downloading the synergy score dataset used for training and evaluation.

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
   Download from [Google Drive](https://drive.google.com/file/d/1wZ-1KFirBBdy1egMXxBB7Sgx0uQm6Jue/view) and place in `data/weights/`.

## Usage

### Training
**TranSynergy**:

Run training with different feature types:
```bash
python main.py --settings_choice='gene_dependency'  # Uses gene dependency features
python main.py --settings_choice='gene_expression'  # Uses gene expression profiles
python main.py --settings_choice='netexpress'       # Uses network-based expression
```

**Biomining Neural Network**:
- Single fold:
  ```bash
  python -m src.training.trainer --config configs/base.yaml
  ```
- Multiple folds:
  ```bash
  python scripts/run_training.py --folds 1,2,3 --config configs/base.yaml
  ```

### Hyperparameter Tuning
Run grid search for hyperparameter optimization:
```bash
python -m src.training.hyperparameter --config configs/grid.yaml
```
For multiple folds and splits:
```bash
python scripts/run_grid_search.py --folds 1,2,3 --splits 1,2,3 --config configs/grid.yaml
```

## Data Leakage
At the start of the project, data leakage checks were conducted for both Biomining Neural Network and TranSynergy submodules to ensure robust model performance and prevent overfitting. These checks confirmed that training, validation, and test sets were properly separated, free from unintended overlaps or biases, and any identified issues were resolved.

- **Biomining Neural Network**:  
  - Checks for **correlation of data** to identify redundant or highly correlated features that could lead to leakage.  
  - Verifies **semantic integrity** to ensure data annotations (e.g., drug or protein profiles) are consistent and meaningful.  
  - Examines **overlapping** between train, validation, and test sets to prevent data leakage across folds in nested cross-validation.

- **TranSynergy**:  
  - Performs **structural integrity checks** to ensure the data (e.g., gene dependency, gene expression, or netexpress features) is correctly formatted and free from errors that could introduce leakage.  
  - Conducts **correlation analysis** to detect and mitigate highly correlated features that might bias predictions.

## Baseline Results
This section summarizes baseline results for traditional machine learning models compared against **TranSynergy** and **Biomining Neural Network**. Baselines for TranSynergy are evaluated on a stable validation set (using the same train, validation, and test splits as the original code). For Biomining, baselines are evaluated via inner cross-validation on outer fold 1.

### Process Overview
Baseline models are trained by iterating over a parameter grid defined in `baselines/traditional_ml.py` (see `get_model` method). A maximum of 15 iterations are performed, with a 60-second timeout per model to ensure efficient fitting.

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

### Results Summary
The chart below compares baseline model performance based on Test Score (mean squared error, lower is better) for TranSynergy and Biomining. CatBoost outperforms other models in both submodules, with Test Scores of 262.073 (TranSynergy) and 16.1133 (Biomining).

![Baseline Results Chart](https://github.com/Hrishikesh332/Drug-Inhibitor-Cancer-Treatment/tree/main/baselines/baseline_result.png)

### Results Table
The table below summarizes baseline model performance. Test scores represent mean squared error (lower is better). Pearson and Spearman correlations measure predictive performance (higher is better).

| Paper       | Model Name    | Test Score | Test Pearson Correlation | Test Spearman Correlation | Test Parameters                                                                                                   |
|:------------|:--------------|-----------:|-------------------------:|--------------------------:|:---------------------------------------------------------------------------------------------------------------|
| transynergy | knn           | 520.798   | 0.327595                | 0.347093                 | {'n_neighbors': 10}                                                                                             |
| transynergy | ridge         | 401.544   | 0.489785                | 0.533125                 | {'alpha': 10.0}                                                                                                 |
| transynergy | decision_tree | 458.543   | 0.37063                 | 0.407311                 | {'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 5}                          |
| transynergy | svm           | -         | -                       | -                        | -                                                                                                               |
| transynergy | random_forest | 324.0     | 0.624637                | 0.632665                 | {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None} |
| transynergy | catboost      | 262.073   | 0.710145                | 0.693605                 | {'learning_rate': 0.06, 'l2_leaf_reg': 6, 'depth': 9, 'boosting_type': 'Plain'}                                |
| biomining   | knn           | 23.501    | 0.416087                | 0.416025                 | {'n_neighbors': 10}                                                                                             |
| biomining   | ridge         | 26.2676   | 0.27162                 | 0.289326                 | {'alpha': 10.0}                                                                                                 |
| biomining   | decision_tree | 22.0338   | 0.495622                | 0.484003                 | {'min_samples_split': 5, 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 15}                         |
| biomining   | svm           | 23.1375   | 0.440991                | 0.461077                 | {'kernel': 'rbf', 'gamma': 'auto', 'epsilon': 0.1, 'C': 10}                                                    |
| biomining   | random_forest | 17.8277   | 0.609501                | 0.58001                  | {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None} |
| biomining   | catboost      | 16.1133   | 0.65723                 | 0.640797                 | {'learning_rate': 0.06, 'l2_leaf_reg': 4, 'depth': 9, 'boosting_type': 'Plain'}                                |

### Generate Markdown Results Table
To regenerate the results table:
1. Download test result tables from Weights & Biases (wandb).
2. Run the following script with the CSV path and paper name:
   ```bash
   python baselines/wandb_markdown_table.py --wandb_csv_path="./baselines/test_results.csv" --paper=transynergy
   ```
   The script outputs Markdown to the console, which can be copied to update the table above.

## Explainability
This section describes how to generate explanations for the drug synergy models using both model-specific and model-agnostic techniques.

**Supported Models**:
- TranSynergy
- Biomining Neural Network

**Supported Methods**:
- **Activation Maximization**: Activation maximization is an explainability technique that generates synthetic input samples to maximize (or minimize) the output of a neural network model. By optimizing the input to strongly activate a particular neuron or output, we can gain insights into what patterns or features the model has learned to associate with high or low predictions. Applicable to both TranSynergy and Biomining.
- **Anchors**: Anchors are a model-agnostic method that generates simple, high-precision IF-THEN rules to explain a model's predictions by identifying critical features that ensure the same outcome. Only performed for Biomining due to computational cost.
-  **SHAP**: SHAP estimates how much each input feature contributes to a prediction by assigning an importance value called a Shapley value. This value shows how much the feature pushes the model’s output away from or toward a baseline (usually the average prediction). Performed SHAP on both Transynergy and Biomining.





**Running Explanations(Example)**:
```bash
python -m explainability.main --model biomining --method activation_max
```

## Git Subtree Setup (for Contributors)
To add a subtree (e.g., `drug_combination`):
```bash
git remote add drug_combination_repo https://github.com/qiaoliuhub/drug_combination.git
git fetch drug_combination_repo
git subtree add --prefix=external/drug_combination drug_combination_repo main
```

## References
- Liu and Xie (2021): *TranSynergy – Transformer-based drug synergy prediction.* [Link]
- Srithanyarat et al. (2024): *Biomining neural network for synergy prediction in breast cancer.* [Link]
