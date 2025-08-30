# Drug-Inhibitor-Cancer-Treatment

This project leverages deep learning and XAI to explore drug combination therapies for cancer treatment. 
It builds upon previous modelling approaches, namely: **TranSynergy** and **Biomining**.

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
Tumors often develop resistance to single-drug therapies, limiting treatment efficiency. This project aims to identify effective drug combinations using machine learning models and XAI methods.
The idea is to build a model that predicts the synergistic effect of two drugs when battling cancer in a particular cell line. Then by using XAI techniques, we can 
extract information about what patterns the model has learned. These patterns may uncover previously unknown information, and reveal insights for further research.

## Submodules

### TranSynergy
**Location**: `external/drug_combination`  
A Transformer-based model ([Liu & Xie, 2021]) for predicting drug synergy in cancer treatment. It uses features like:
- **Gene dependency**: Measures cell line reliance on specific genes.
- **Gene expression**: Quantifies gene activity levels.
- **Netexpress**: Network-based expression integrating interaction data.

### Biomining Neural Network
**Location**: `external/predicting_synergy_nn`  
A deep learning model for predicting drug synergy in breast cancer based on target-protein inhibition profiles, using nested cross-validation for robust evaluation.

---

## Prerequisites
- **Tools**: `conda`, `git`, `git-lfs`, `wandb` 
- **Hardware**: GPU with CUDA support (optional, for faster training)
- **Python**: Version 3.11  

---

## Setup
0. **Create and activate a conda environment**:
   ```bash
   conda create -n cancer_ml python=3.11
   conda activate cancer_ml
   ```
1. **Install PyTorch**<br>
Visit https://pytorch.org/get-started/locally/ and install based on your system. <br>
Some example installations:
   ```bash
   # For GPU (CUDA 12.4)
   conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   ```
   ```bash
   # For CPU-only
   conda install pytorch==2.4.1 cpuonly -c pytorch
   ```

2. **Install TransSynergy**:
   ```bash
   pip install external/drug_combination 
   ```

3. **Install BioMining**:
   ```bash
   pip install -r external/predicting_synergy_nn/requirements.txt
   ```
   
4. **Install 'Baselines' requirements**:
   ```bash
   pip install -r baselines/requirements.txt  
   ```
   
5. **Install 'Explainability' requirements**:
   ```bash
   pip install -r explainability/requirements.txt  
   ```
---

## Data & Dataset
This section provides instructions for downloading the synergy score dataset used for training and evaluation. 

### Transynsergy
Drug data is derived from DrugBank (drug-target interactions), ChEMBL (bioactivity data), and STRING (protein-protein interaction networks), encoding each drug as a binary vector representing interactions with 2401 selected genes. Cell line data is obtained from CCLE and GDSC (gene expression profiles) and Achilles and Sanger CRISPR (gene dependency scores), providing gene expression or dependency values for the same 2401 genes. Drug combination data, sourced from Merck drug synergy data, is structured as a matrix with three columns: two for drug profiles (binary vectors) and one for cell line features (gene expression or dependency, with a potential fourth column if both are included). The output is a continuous synergy score, indicating whether drug pairs are synergistic, additive, or antagonistic. This dataset enables the TranSynergy transformer-based model to predict synergy and deconvolve pathways via attention mechanisms.
#### Data Access Instructions
0. **Navigate to TranSynergy root**
   ```bash
   cd external/drug_combination
   ```
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

### Biomining
* Source: DrugComb database, containing 24,145 drug combination pairs tested on five breast cancer cell lines (MCF-7, T-47D, MDA-MB-468, BT-549, MDA-MB-231) with 98 unique drugs.divided into three folds in the outer loop of the nested CV, with one fold serving as the test dataset while the other two folds were further divided into three folds in the inner loop. During each round of the inner loop, two folds were used as a training dataset, while the other fold was used as a validation set in a grid search for the best hyperparameter set.
* Features: Concatenated inhibition profiles for 13 cancer-related proteins (e.g., EGFR, PIK3CA) per drug pair (26 features) plus cell line genomic features, generated using graph convolutional neural networks from DrugBank and PubChem data.
* Output: Continuous synergy score (e.g., Bliss score) for each drug pair-cell line combination. data can be found at this location `external/predicting_synergy_nn/data`.

For Biomining, the data can be found at the following location: `external/predicting_synergy_nn/data`.

---

## Usage

### TranSynergy
*Note: You need to run this from the root directory* <br>
Run training with different feature types: 
```bash
python external/drug_combination/main.py --settings_choice='gene_dependency'  # Uses gene dependency features
python external/drug_combination/main.py --settings_choice='gene_expression'  # Uses gene expression profiles
python external/drug_combination/main.py --settings_choice='netexpress'       # Uses network-based expression
```

Here is how you can check your result: [Transynergy Result](./external/drug_combination/README.md)

### Biomining Neural Network

Navigate to the [Biomining Repo Root](./external/predicting_synergy_nn) (all code should be run from there) and read the [Biomining README](./external/predicting_synergy_nn/README.md).

---

## Data Leakage
At the start of the project, data leakage checks were conducted for both Biomining Neural Network and TranSynergy submodules to ensure robust model performance and prevent overfitting. These checks confirmed that training, validation, and test sets were properly separated, free from unintended overlaps or biases, and any identified issues were resolved.
You can find the relevant scripts under [./data_leakage_analysis](./data_leakage_analysis)

- **Biomining Neural Network**:  
  - Checks for **correlation of data** to identify redundant or highly correlated features that could lead to leakage.  
  - Verifies **semantic integrity** to ensure data annotations (e.g., drug or protein profiles) are consistent and meaningful.  
  - Examines **overlapping** between train, validation, and test sets to prevent data leakage across folds in nested cross-validation.

- **TranSynergy**:  
  - Performs **structural integrity checks** to ensure the data (e.g., gene dependency, gene expression, or netexpress features) is correctly formatted and free from errors that could introduce leakage.  
  - Conducts **correlation analysis** to detect and mitigate highly correlated features that might bias predictions.

**Running Explanations (Example) from the root directory**:
```bash
cd data_leakage_analysis/transynergy && python check_for_overlap.py

cd data_leakage_analysis/biomining/test && python check_correlation_data.py

cd data_leakage_analysis/biomining/validation && python check_dataset_structure.py
```
---

## Baseline:
We implemented traditional machine learning models for drug synergy prediction, in order to validate the Deep Learning apporaches. The baseline models used for benchmark are **CatBoost, Random Forest, SVM, Decision Tree, Ridge Regression, and k-NN**. The main.py script supports two modes: TranSynergy (using a fixed train-validation-test split with test fold 4 and evaluation fold 0) and Biomining (using nested cross-validation with inner folds 1–3 on outer fold 1). The traditional_ml.py script trains these models by iterating over a parameter grid defined in its get_model method (e.g., learning rates, tree depth), performing up to 15 iterations with a 60-second timeout per model for efficient fitting. Performance is evaluated using mean squared error, Pearson, and Spearman correlations, logged via Weights & Biases..

### Process Overview
Baseline models are trained by iterating over a parameter grid defined in `baselines/traditional_ml.py` (see `get_model` method).
The n_iter parameter determines the number of hyperparameter search iterations. The timeout parameter determines the timeout (maximum possible time) for training a model.
The mode parameter determines which paper/data to use (transynergy | biomining).  <br>
The training is tracked via **weights and biases**. Follow the wandb URL outputted in stdout for training details.

To replicate baseline results:
#### Running all baselines
   ```bash
   python -m baselines.main --mode transynergy --n_iter 15 --timeout 60  # For TranSynergy
   python -m baselines.main --mode biomining --n_iter 15 --timeout 60    # For Biomining
   ```
#### Running a specific model (e.g CatBoost)
   ```bash
   python -m baselines.main --mode transynergy --n_iter 100 --timeout 600 --model catboost
   ```

You can check more about result table and summary here: [Baseline Results](./baselines/BASELINES.md)

---

## Explainability

Explainability in machine learning refers to the set of techniques and tools used to understand, interpret, and visualize the decisions made by complex models—especially deep learning models. In the context of drug synergy prediction, explainability helps researchers and practitioners understand **why** a model makes a certain prediction, **which features** influence it most, and **how** reliable or generalizable those decisions might be. This transparency is particularly important in biomedical applications where model outputs may guide experimental or clinical decisions.

This section describes how to generate explanations for the drug synergy models using both **model-specific** and **model-agnostic** techniques.

**Supported Explainability Methods**:

- **Activation Maximization**:  
  A **model-specific** technique that generates synthetic input samples which maximize or minimize the activation of a particular output neuron. By optimizing the input space, we can visualize what kinds of patterns lead the model to predict strong synergy or antagonism. This method is useful for **understanding what the model has learned**, and is applicable to both TranSynergy and Biomining models.

- **Anchors**:  
  A **model-agnostic** technique that produces **IF-THEN rules** to explain a specific prediction. It identifies a set of key features (anchors) such that, when they are present, the model’s output is almost always the same. Anchors provide **high-precision, human-interpretable explanations** and are especially useful when decision stability is important. Due to computational cost, Anchors are currently used only with the Biomining model and wasn't further analysed.

- **SHAP (SHapley Additive exPlanations)**:  
  A popular **model-agnostic** method based on cooperative game theory. SHAP assigns each feature an importance value—its **Shapley value**—representing how much it contributes to the prediction compared to a baseline. SHAP is valuable for both **global** and **local** interpretability and has been applied to both TranSynergy and Biomining.

- **Integrated Gradients**:  
  A **model-specific** technique designed for differentiable models like neural networks. It computes the gradient of the model’s output with respect to the input features, integrated over a straight-line path from a baseline (e.g., all zeros) to the actual input. Integrated Gradients provide **attribution scores** that highlight the relative importance of each feature in driving the prediction. Applicable to both TranSynergy and Biomining models.

- **GSEA**:
   A model-agnostic analysis method that interprets feature attributions from SHAP in terms of biological pathways or gene sets. Instead of examining single genes in isolation, GSEA identifies whether groups of genes that share common biological functions are collectively enriched among the most important features driving a model’s prediction. This helps translate raw feature importance into biologically meaningful insights, making the explanations more actionable in a biomedical context.

Use this path to check the results of all methods : `explainability\notebooks`. To replicate, some large files will need to be downloaded as they could't be uploaded directly to github. Check below.

To download the pretrained model checkpoints required for running Integrated Gradients [file](https://drive.google.com/drive/folders/1Xk0onniJZI51SM-d1crSXWr7O0wwAJ5w?usp=sharing).


### Running Explanation Methods
The CLI interface entrypoint for XAI methods is the `explainability.main` module. <br>
Supported arguments for `--model` are `[biomining, transynergy]` <br>
Supported arguments for `--method` are `[anchors, activation_max, lrp, integrated_gradients, shap]` <br>
Every method will save results in its own subdirectory under `explainability`, e.g `explainability/shapley/results`. <br>
Under `explainability/notebooks` you can find jupyter notebooks which interact with the explanation results.

**Examples:**
```bash
python -m explainability.main --model biomining --method activation_max
python -m explainability.main --model transynergy --method integrated_gradients
```

**To run GSEA after running Shapley (Example)**
```bash
python -m explainability.shapley.gsea --paper biomining
```

**To inspect CatBoost feature importances and feature interaction strength, use the notebook `explainability/catboost/catboost_xai.ipynb`.**

### ⚠️ **Caution: Large File Notice** ⚠️  
Some result files are too large to upload via Git. For those, please use the provided Google Drive links instead.

**TransSynergy LRP Relevances**  
Paste the downloaded file under:  
`explainability/lrp/results/transynergy_subsample_False`

🔗 [Download from Google Drive](https://drive.google.com/file/d/15CLzOmeJk0-FpqtSNumrjJMNG1JBFKDq/view?usp=sharing)

**TransSynergy IG Attributions**  
Paste the downloaded file under:  
`explainability/ig/results`

🔗 [Download from Google Drive](https://drive.google.com/file/d/1ftC-7CVpCgf-bFOTiVH6OKr3_t1sj7BU/view?usp=sharing)

**TransSynergy SHAP Attributions**  
Paste the downloaded file under:  
`explainability/shapley/results/transynergy`

🔗 [Download from Google Drive](https://drive.google.com/file/d/1faQvvnRxIAXBC7t6J4GRz3LRLKo0RUb0/view?usp=sharing)

---

## References
- Liu and Xie (2021): *TranSynergy – Transformer-based drug synergy prediction.* [Link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008653)
- Srithanyarat et al. (2024): *Biomining neural network for synergy prediction in breast cancer.* [Link](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-024-00359-z)
