# Biomining project

## Folder structure

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
│   ├── grid.yaml      
│   └── cv.yaml   
├── cli/
│   ├── run_grid_search.py       
│   └── run_k_fold_cross_validation.py    
└── outputs/                
    ├── models/               # Saved models
    ├── logs/                 
    ├── results/            
    └── grid_results/         
```

## Installation

To set up the virtual environment and install dependencies

1. **Create and activate a conda environment (if not yet existing)**:
   ```bash
   conda create -n biomining python=3.11
   conda activate biomining
   ```
2. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Before running any command, set `PYTHONPATH=src`.
The data is split up into 3 train/test folds by the authors.

### 1 Running k-fold Cross Validation
The data is split up into 3 folds. You can choose which to run on using the `--folds` argument, and the script will 
report the final CV scores.

```bash
PYTHONPATH=src python cli/run_k_fold_cross_validation.py --folds 1,2,3 --config configs/base.yaml
```

### 2 Hyperparameter Search

```bash
PYTHONPATH=src python cli/run_grid_search.py --folds 1,2,3 --splits 1,2,3 --config configs/grid.yaml
```

### 3 Training the Final Model
Trains model on one fold.
```bash
PYTHONPATH=src python cli/train_model.py --config configs/base.yaml --fold 1
```

## Configuration Files

* `configs/base.yaml`
This file contains the configuration for the model training, including the model architecture, batch size, learning rate, etc

* `configs/grid.yaml`
This file contains the configuration for hyperparameter tuning, including the parameters to search over during grid search.

*  `configs/cv.yaml`
Cross-validation configuration for fold2 data only (fold1/fold3 contain identical data). Supports three strategies -

- **Random**: Standard KFold splitting
- **Cell Line Out**: GroupKFold by CELL_LINE (no cell line overlap)
- **Drug Out**: GroupKFold by drug combinations (no drug pair overlap)

**Quick Usage**:
```bash
cd external/predicting_synergy_nn

# Run random CV (default)
python src/training/cross_valid.py

# Run cell line out CV
python src/training/cross_valid.py --strategy cell_line

# Run drug out CV  
python src/training/cross_valid.py --strategy drug

# Run with custom number of splits
python src/training/cross_valid.py --strategy drug --splits 5

# Run with custom config file
python src/training/cross_valid.py --strategy random --config configs/cv.yaml
```

**Strategy Configuration** -
- Edit `cv_strategy` in `cv.yaml`: `random`, `cell_line`, or `drug`
- For cell_line: set `cell_line_col: CELL_LINE`
- For drug: ensure `drug_col: null` (uses default DRUG1/DRUG2 columns)


