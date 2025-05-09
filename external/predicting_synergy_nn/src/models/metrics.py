import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

def calc_pearson(pred, true):
    x = pred.flatten()
    y = true.flatten()
    
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    
    xd, yd = x - x_m, y - y_m
    
    num = torch.mean(xd * yd)
    den = torch.std(xd) * torch.std(yd)
    
    return num / den

def calc_spearman(pred, true):
    x = pred.detach().cpu().numpy().flatten()
    y = true.detach().cpu().numpy().flatten()
    
    corr, _ = spearmanr(x, y)
    
    if np.isnan(corr):
        return torch.tensor(0.0).to(pred.device)
    
    return torch.tensor(corr).to(pred.device)