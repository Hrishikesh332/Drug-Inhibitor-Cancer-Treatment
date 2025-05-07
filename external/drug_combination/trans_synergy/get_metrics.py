import pandas as pd
from torch import load
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='Get_stats_from_prediction')
parser.add_argument('--prediction_file')
args = parser.parse_args()

prediction_file = args.prediction_file

pred = load(prediction_file)
new_pred = pd.DataFrame(pred, columns = ['name', 'prediction', 'ground_truth'])
new_pred['prediction'] = new_pred['prediction'].astype(float)
new_pred['ground_truth'] = new_pred['ground_truth'].astype(float)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_metrics(k):
    pred_ls = sigmoid(new_pred['prediction'] * k)
    y_label = (new_pred['ground_truth'] > 30).astype(int)
    return roc_auc_score(y_label, pred_ls), average_precision_score(y_label, pred_ls)

def print_best_metrics():
    start, end = 0, 7
    count = 0
    while count < 6:
        res_ls = []
        res_auc = []
        for k in np.linspace(start, end, 100):
            roc, pr = get_metrics(k)
            res_ls.append(pr)
            res_auc.append(roc)
        max_idx = np.argmax(res_ls)
        start, end = np.linspace(start, end, 100)[max_idx - 1], np.linspace(start, end, 100)[max_idx + 1]
        count += 1
        print(res_ls[max_idx], res_auc[max_idx])

print_best_metrics()
