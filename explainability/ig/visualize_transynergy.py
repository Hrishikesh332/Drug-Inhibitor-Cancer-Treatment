import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# Define paths
SAVE_PATH = "results/transynergy_integrated_gradients.pt"
BATCH_SAVE_DIR = "results/transynergy_batch_attributions"
HIST_SAVE_DIR = "results/transynergy_histograms"
os.makedirs(HIST_SAVE_DIR, exist_ok=True)

gene_symbols = [f"gene_{str(i).zfill(4)}" for i in range(1, 2402)]  # Placeholder: replace with actual gene names
drug_feature_names = gene_symbols + ['pIC50']
cellline_feature_names = gene_symbols
feature_names = (
    [f"drugA_{name}" for name in drug_feature_names] +
    [f"drugB_{name}" for name in drug_feature_names] +
    [f"cellline_{name}" for name in cellline_feature_names]
)

feature_names = feature_names + ["unknown_feature_7206"]  # Placeholder for the 7206th feature
def get_feature_name(index, channel):
    channel_start = channel * 2402
    return feature_names[channel_start + index]

all_attributions = torch.load(SAVE_PATH, weights_only=False)

if len(all_attributions.shape) == 4:  # Expected [N, 1, 3, 2402]
    mean_attributions_per_channel = all_attributions.mean(dim=(0, 1)).numpy()  # Shape: [3, 2402]

    channel_names = ["Drug A", "Drug B", "Cell Line"]
    for channel in range(3):
        channel_attributions = mean_attributions_per_channel[channel]  # Shape: [2402]
        
        indices = np.argsort(channel_attributions)
        top_negative = indices[:5]  # Lowest 5
        top_positive = indices[-5:]  # Highest 5
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

elif len(all_attributions.shape) == 3:  
    mean_attributions = all_attributions.mean(dim=0).squeeze().numpy()  

   
    num_features = mean_attributions.shape[0]
    feature_names = feature_names[:num_features]  

    indices = np.argsort(mean_attributions)
    top_negative = indices[:5]  # Lowest 5
    top_positive = indices[-5:]  # Highest 5
    top_indices = np.concatenate([top_negative, top_positive])
    feature_values = mean_attributions[top_indices]
    feature_labels = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(12, 8))
    plt.barh(feature_labels, feature_values, color=np.where(np.array(feature_values) < 0, 'red', 'green'))
    plt.title("Top-5 Positive and Negative Feature Attributions (Transynergy)")
    plt.xlabel("Mean Attribution")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plot_filename = os.path.join(HIST_SAVE_DIR, "top_k_features_flattened.png")
    plt.savefig(plot_filename)
    plt.close()

else:
    raise ValueError(f"Unexpected shape for all_attributions: {all_attributions.shape}. Expected [N, 1, 3, 2402] or [N, 1, M].")

