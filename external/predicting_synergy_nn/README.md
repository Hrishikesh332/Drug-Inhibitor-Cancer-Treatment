# Biomining folder structure 

```
predicting_synergy_nn/
├── data/
│   ├── fold1/
│   │   ├── fold1_alltrain.csv
│   │   ├── fold1_test.csv
│   │   └── validation/
│   │       ├── fold1_train1.csv
│   │       ├── fold1_valid1.csv
│   │       ├── fold1_train2.csv
│   │       ├── fold1_valid2.csv
│   │       ├── fold1_train3.csv
│   │       └── fold1_valid3.csv
│   ├── fold2/  # Similar structure to fold1
│   └── fold3/  # Similar structure to fold1
├── src/
│   ├── models/
│   │   ├── architectures.py  
│   │   └── metrics.py   
│   ├── utils/
│   │   ├── data_loader.py   
│   │   └── visualization.py
│   └── training/
│       ├── trainer.py        
│       └── hyperparameter.py
├── configs/
│   ├── base.yaml      
│   └── grid.yaml        
├── scripts/
│   ├── run_training.py       
│   └── run_grid_search.py    
└── outputs/                
    ├── models/               # Saved models
    ├── logs/                 
    ├── results/            
    └── grid_results/         
```

## Installation

To set up the virtual environment and install dependencies

1. **Create and activate a conda environment**:
   ```bash
   conda create -n biomining python=3.11
   conda activate biomining
   ```
2. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1 Running Training for a Specific Fold

```bash
python -m src.training.trainer --config configs/base.yaml
```

Make sure to update the `base.yaml` file with the correct configuration for specific fold.

### 2 Running Training for Multiple Folds

```bash
python scripts/run_training.py --folds 1,2,3 --config configs/base.yaml
```

### 3 Hyperparameter Search

```bash
python -m src.training.hyperparameter --config configs/grid.yaml
```

### 4 Running Grid Search for Multiple Folds and Splits


```bash
python scripts/run_grid_search.py --folds 1,2,3 --splits 1,2,3 --config configs/grid.yaml
```

### 5 Run training/grid search

# For Training
python scripts/run_training.py --folds 1,2,3 --config config/cv.yaml

# For Grid search
python scripts/run_grid_search.py --folds 1,2,3 --splits 1,2,3 --config configs/cv.yaml

### In cv.yaml

# Random CV (default)
cv_strategy: random

# Leave-cell-line-out CV  
cv_strategy: cell_line
cell_line_col: CELL_LINE

# Leave-drug-out CV (True Drug Out: holds out any row where the drug appears in DRUG1 or DRUG2)
cv_strategy: drug
# drug_col is ignored for true drug out; both DRUG1 and DRUG2 are used automatically

## Cross-Validation

This project supports three types of cross-validation for model evaluation:

- **Random CV** - Randomly splits the data into train/validation splits.
- **Cell line out CV** - Holds out all samples from specific cell lines in each split, testing generalization to unseen cell lines.
- **Drug out CV** - Holds out all samples containing a specific drug in either the DRUG1 or DRUG2 column in each split, testing generalization to unseen drugs in any position of the combination.

### Running Cross-Validation

To run cross-validation (for fold 2, with 3 splits):

```bash
python scripts/run_cv.py --config configs/cv.yaml
```

- Edit `cv.yaml` to set the desired strategy and columns:

```yaml
# Random CV (default)
cv_strategy: random

# Leave-cell-line-out CV  
cv_strategy: cell_line
cell_line_col: CELL_LINE

# Leave-drug-out CV (True Drug Out: holds out any row where the drug appears in DRUG1 or DRUG2)
cv_strategy: drug
# drug_col is ignored for true drug out; both DRUG1 and DRUG2 are used automatically
```

- The code will print validation metrics for each split and the mean across splits.
- For full experiment tracking, set `use_wb: true` in your config to enable Weights & Biases logging.

### Notes
- The cross-validation code is implemented in `src/training/cross_valid.py`.
- Only fold 2 is currently supported for cross-validation.
- For true drug out, the grouping is done on the set of all drugs in each row (both DRUG1 and DRUG2).

## Configuration Files

* `configs/base.yaml`