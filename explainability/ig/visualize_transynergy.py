import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
import pandas as pd

# Define paths
SAVE_PATH = "results/transynergy_integrated_gradients.pt"
BATCH_SAVE_DIR = "results/transynergy_batch_attributions"
HIST_SAVE_DIR = "results/transynergy_histograms"
os.makedirs(HIST_SAVE_DIR, exist_ok=True)

gene_symbols = [f"gene_{str(i).zfill(4)}" for i in range(1, 2402)]  
drug_feature_names = gene_symbols + ['pIC50']
cellline_feature_names = gene_symbols
feature_names = (
    [f"drugA_{name}" for name in drug_feature_names] +
    [f"drugB_{name}" for name in drug_feature_names] +
    [f"cellline_{name}" for name in cellline_feature_names]
)

feature_names = feature_names + ["unknown_feature_7206"]  
def get_feature_name(index, channel):
    channel_start = channel * 2402
    return feature_names[channel_start + index]

all_attributions = torch.load(SAVE_PATH, weights_only=False)
MEAN_ATTRIBUTION = all_attributions.mean().item()
MIN_ATTRIBUTION = all_attributions.min().item()
MAX_ATTRIBUTION = all_attributions.max().item()
print(f"Attributions shape: {all_attributions.shape}")

if len(all_attributions.shape) == 4:  # Expected [N, 1, 3, 2402]
    N = all_attributions.shape[0]
    attributions_reshaped = all_attributions.squeeze(1).reshape(N, -1)  

    attributions_np = attributions_reshaped.numpy()
    correlation_matrix = np.corrcoef(attributions_np, rowvar=False)  
    
    mean_attributions = np.abs(attributions_np).mean(axis=0)
    top_k_indices = np.argsort(mean_attributions)[-50:]  # Select top 50 features
    top_k_features = [feature_names[i] for i in top_k_indices]
    correlation_subset = correlation_matrix[top_k_indices][:, top_k_indices]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_subset,
        xticklabels=top_k_features,
        yticklabels=top_k_features,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
    )
    plt.title("Correlation Heatmap of Top 50 Features (Transynergy Attributions)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_filename = os.path.join(HIST_SAVE_DIR, "feature_correlation_heatmap.png")
    plt.savefig(heatmap_filename)
    plt.close()
    
    mean_attributions_per_channel = all_attributions.mean(dim=(0, 1)).numpy() 
    channel_names = ["Drug A", "Drug B", "Cell Line"]
    for channel in range(3):
        channel_attributions = mean_attributions_per_channel[channel]  
        indices = np.argsort(channel_attributions)
        top_negative = indices[:5]
        top_positive = indices[-5:]
        top_indices = np.concatenate([top_negative, top_positive])
        feature_values = channel_attributions[top_indices]
        feature_labels = [get_feature_name(idx, channel) for idx in top_indices]

        plt.figure(figsize=(12, 8))
        plt.barh(feature_labels, feature_values, color=np.where(np.array(feature_values) < 0, 'red', 'green'))
        plt.title(f"Top-5 Positive and Negative Feature Attributions for {channel_names[channel]} (Transynergy)")
        plt.xlabel("Mean Attribution")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plot_filename = os.path.join(HIST_SAVE_DIR, f"top_k_features_channel_{channel}.png")
        plt.savefig(plot_filename)
        plt.close()
else:
    raise ValueError(f"Unexpected shape for all_attributions: {all_attributions.shape}. Expected [N, 1, 3, 2402].")