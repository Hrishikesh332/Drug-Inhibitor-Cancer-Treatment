import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

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

# op-K Features Bar Plot 
mean_attributions = all_attributions.mean(dim=0).squeeze().numpy()  
indices = np.argsort(mean_attributions)
top_negative = indices[:5]  
top_positive = indices[-5:]  
top_indices = np.concatenate([top_negative, top_positive])
feature_values = mean_attributions[top_indices]
feature_labels = [feature_cols[i] for i in top_indices]

plt.figure(figsize=(12, 8))
plt.barh(feature_labels, feature_values, color=np.where(np.array(feature_values) < 0, 'red', 'green'))
plt.title("Top-5 Positive and Negative Feature Attributions (Biomining)")
plt.xlabel("Mean Attribution")
plt.ylabel("Feature Name")
plt.tight_layout()
top_k_path = os.path.join(HIST_SAVE_DIR, "top_k_features.png")
plt.savefig(top_k_path)
plt.close()

attributions_np = all_attributions.squeeze().numpy()  

variances = np.var(attributions_np, axis=0)
valid_indices = np.where(variances > 0)[0]
attributions_np_filtered = attributions_np[:, valid_indices]
feature_cols_filtered = [feature_cols[i] for i in valid_indices]

correlation_matrix = np.corrcoef(attributions_np_filtered, rowvar=False)  

sns.clustermap(
    correlation_matrix,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    xticklabels=feature_cols_filtered,
    yticklabels=feature_cols_filtered,
    figsize=(12, 10),
    cbar_kws={'label': 'Correlation'},
    method='average',
    metric='euclidean',
)
plt.title("Feature Correlation Heatmap (Biomining Attributions)", pad=100)
corr_heatmap_path = os.path.join(HIST_SAVE_DIR, "feature_correlation_heatmap.png")
plt.savefig(corr_heatmap_path, bbox_inches='tight')
plt.close()

mean_abs_attributions = np.abs(all_attributions.squeeze().numpy()).mean(axis=0)  # Shape: [33]

plt.figure(figsize=(12, 4))
sns.heatmap(
    mean_abs_attributions.reshape(1, -1),
    cmap='Reds',
    xticklabels=feature_cols,
    yticklabels=['Mean Abs Attribution'],
    cbar_kws={'label': 'Mean Absolute Attribution'},
)
plt.title("Mean Absolute Feature Attributions (Biomining)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
mean_abs_path = os.path.join(HIST_SAVE_DIR, "mean_abs_attribution_heatmap.png")
plt.savefig(mean_abs_path)
plt.close()
