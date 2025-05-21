import matplotlib.pyplot as plt
import torch
import numpy as np
import os

SAVE_PATH = "results/transynergy_integrated_gradients.pt"
BATCH_SAVE_DIR = "results/transynergy_batch_attributions"
HIST_SAVE_DIR = "results/transynergy_histograms"
os.makedirs(HIST_SAVE_DIR, exist_ok=True)

all_attributions = torch.load(SAVE_PATH, weights_only=False) 
subset = all_attributions.numpy().flatten()  
MEAN_ATTRIBUTION = all_attributions.mean().item()
MIN_ATTRIBUTION = all_attributions.min().item()
MAX_ATTRIBUTION = all_attributions.max().item()


print("Generating top-k features bar plot...")
mean_attributions = all_attributions.mean(dim=(0, 1, 2)).numpy()  
indices = np.argsort(mean_attributions)
top_negative = indices[:5]  
top_positive = indices[-5:]
feature_indices = np.concatenate([top_negative, top_positive])
feature_values = mean_attributions[feature_indices]
plt.figure(figsize=(10, 6))
plt.barh([f"Feature {i}" for i in feature_indices], feature_values, color=np.where(feature_values < 0, 'red', 'green'))
plt.title("Top-5 Positive and Negative Feature Attributions (Transynergy)")
plt.xlabel("Mean Attribution")
plt.ylabel("Feature Index")
plt.tight_layout()
plt.savefig(os.path.join(HIST_SAVE_DIR, "top_k_features.png"))
plt.close()
print(f"Saved top-k features plot to {HIST_SAVE_DIR}/top_k_features.png")

