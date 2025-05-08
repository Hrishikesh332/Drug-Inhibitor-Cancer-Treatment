# Motivation

In many patients, a tumor’s innate or acquired resistance to a given therapy will render the treatment ineffective. To increase therapeutic options and to overcome drug resistance, cancer researchers have been actively investigating drug combinations.
The repository was originally set up for two gpu cards, however we made it so that only one cuda device or none is allowed.

## Build environment (dependencies)

### 1. Create environment
```bash
conda create -yn trans_synergy python=3.11
```
### 2. Activate environment
```bash
conda activate trans_synergy
```
### 3. Install PyTorch
Based on your environment, install a compatible PyTorch version: [PyTorch Get Started](https://pytorch.org/get-started/locally/). 
Visit the official PyTorch page for previous versions:  
[PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)<br>
If you use CUDA, you can run the following command to check your installed CUDA version:
```bash
nvidia-smi      
```
Example Setup for CUDA 12.4:
```bash
conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia 
```
Example Setup for CPU-only:
```bash
conda install pytorch==2.4.1 cpuonly -c pytorch
```

### 4. Install package
```bash
pip install -e .
```

## Dataset

### Download LFS files

Files under `./data` are managed using [Git Large File Storage (Git LFS)](https://git-lfs.github.com/). 
Follow the instructions below to fetch the LFS data:

1. **Install Git LFS**  
   Follow the instructions at: https://git-lfs.github.com/

2. **Initialize Git LFS (run once per system)**  
   ```bash
   git lfs install
    ```
3. **Pull files**
   ```bash
   git lfs pull
    ```

### Download processed synergy score files
```bash
zenodo_get 10.5281/zenodo.4789936 -o ./data/synergy_score
```

## Check model performance with 


## Run training
You can choose between the following cell line features (gene dependencies, gene expression and netexpress scores.
```bash
python main.py --settings_choice='gene_dependency'
```
```bash
python main.py --settings_choice='gene_expression'
```
```bash
python main.py --settings_choice='netexpress'
```

#### Check results
check the logfile in the newest ```_run_*****``` folder

## Drug combination Synergy scores

#### An Unbiased Oncology Compound Screen to Identify Novel Combination Strategies. (O'Neil J et. al)

```
Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
5-FU_ABT-888_A375,5-FU,ABT-888,A375,-1.1985054379
5-FU_ABT-888_A427,5-FU,ABT-888,A427,2.5956844375
5-FU_ABT-888_CAOV3,5-FU,ABT-888,CAOV3,-5.1399712212
5-FU_ABT-888_DLD1,5-FU,ABT-888,DLD1,1.9351271188
```

## Cell line gene-level dependencies score

```
index,genes,CAL148,HT1197,A2780,MCF7,HCC1395, ..., ENTREZID
CDH2 (1000),CDH2,0.1709,0.2125,0.1651,-0.0103,-0.2448, ..., 1000
AKT3 (10000),AKT3,-0.0400,-0.0137,0.2180,-0.0194],-0.0503644515648, ..., 10000
```

## Drug features
Drug target profile

## Gene networks

```
Entrezid Entrezid posterior prob.
5988    53905   0.137373
5988    286234  0.116511
5988    277     0.104127
5988    387856  0.114427
5988    90317   0.115751
5988    100287366       0.100036
5988    100287362       0.105938
```
