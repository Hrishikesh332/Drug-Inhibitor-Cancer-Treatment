import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# Define paths
SAVE_PATH = "results/biomining_integrated_gradients.pt"
BATCH_SAVE_DIR = "results/biomining_batch_attributions"
HIST_SAVE_DIR = "results/biomining_histograms"
os.makedirs(HIST_SAVE_DIR, exist_ok=True)

feature_cols = [
    "ABL", "ABLb", "CSF1R", "CSF1Rb", "EGFR", "EGFRb", "FLT1", "FLT1b", "FLT4", "FLT4b",
    "KDR", "KDRb", "KIT", "KITb", "MCL1", "MCL1b", "NR1I2", "NR1I2b", "PDGFRB", "PDGFRBb",
    "RET", "Retb", "TOP2", "TOP2b", "TUB1", "TUB1b", "GATA3", "NF1", "NF2", "P53", "PI3K", "PTEN", "RAS"
]

all_attributions = torch.load(SAVE_PATH, weights_only=False)

mean_attributions = all_attributions.mean(dim=0).squeeze().numpy()  # [1, 33] -> [33]

indices = np.argsort(mean_attributions)
top_negative = indices[:5]  # Lowest 5
top_positive = indices[-5:]  # Highest 5
top_indices = np.concatenate([top_negative, top_positive])
feature_values = mean_attributions[top_indices]
feature_labels = [feature_cols[i] for i in top_indices]

plt.figure(figsize=(12, 8))
plt.barh(feature_labels, feature_values, color=np.where(np.array(feature_values) < 0, 'red', 'green'))
plt.title("Top-5 Positive and Negative Feature Attributions (Biomining)")
plt.xlabel("Mean Attribution")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.savefig(os.path.join(HIST_SAVE_DIR, "top_k_features.png"))
plt.close()
