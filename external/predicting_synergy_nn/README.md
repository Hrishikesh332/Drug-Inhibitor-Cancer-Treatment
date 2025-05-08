# Predicting drug synergy in breast cancer with deep learning using target-protein inhibition profiles

In this study, a 3×3 nested cross-validation (CV) method was used. The initial set of drug combinations, which consisted of 24145 pairs, were divided into three folds in the outer loop of the nested CV, with one fold serving as the test dataset while the other two folds were further divided into three folds in the inner loop. During each round of the inner loop, two folds were used as a training dataset, while the other fold was used as a validation set in a grid search for the best hyperparameter set. The datasets and Python codes required for this task can be found in the 'identify_best_parameters' folder. To perform a grid search on each round of the inner loops, execute GridSearch.py in each subfolder.

The best hyperparameter set (identified based on the average Pearson correlation coefficients obtained across each round of the three inner loops).


## Project Structure

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

```bash
python setup_venv.py
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

## Configuration Files

* `configs/base.yaml`
 This file contains the configuration for the model training, including the model architecture, batch size, learning rate, etc

* `configs/grid.yaml`

This file contains the configuration for hyperparameter tuning, including the parameters to search over during grid search.


