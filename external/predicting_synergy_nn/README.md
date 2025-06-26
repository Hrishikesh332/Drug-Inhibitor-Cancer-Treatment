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

# Leave-drug-out CV
cv_strategy: drug
drug_col: DRUG_ID


## Configuration Files

* `configs/base.yaml`
 This file contains the configuration for the model training, including the model architecture, batch size, learning rate, etc

* `configs/grid.yaml`

This file contains the configuration for hyperparameter tuning, including the parameters to search over during grid search.


