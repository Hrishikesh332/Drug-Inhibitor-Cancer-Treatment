import os
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm


def compute_bins(Y: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, list[str]]:
    bins = np.quantile(Y, [low_q, high_q])
    names = [
        f"lowest_quantile_{low_q}",
        f"middle_quantile_{low_q}_{high_q}",
        f"highest_quantile_{high_q}"
    ]
    return bins, names


def explain_sample_set(X_sample, explainer, config, logger, bins_names, save_dir, suffix_progress_bar=""):
    """Run Anchor explanations on a sample set and log results."""
    explanations = []

    for i, x in tqdm(enumerate(X_sample), desc=f"Explaining samples {suffix_progress_bar}", total=len(X_sample)):
        explanation = explainer.explain(x, threshold=config.threshold, batch_size=1000)
        explanations.append({
            "instance": i,
            "anchor": explanation.anchor,
            "feature_names": [config.feature_names[idx] for idx in explanation.raw['feature']],
            "precision": float(explanation.precision),
            "coverage": float(explanation.coverage),
            "prediction": bins_names[int(explanation.raw["prediction"].flatten())],
        })

    df = pd.DataFrame(explanations)
    save_results(df, config, save_dir, logger, suffix_progress_bar)


def save_results(df, config, save_dir, logger, suffix=""):
    """Save CSV and log metrics to Weights & Biases."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{config.paper}_anchors.csv")
    df.to_csv(csv_path, index=False)

    wandb.log({
        f"precision_histogram_{suffix}": wandb.Histogram(df["precision"]),
        f"coverage_histogram_{suffix}": wandb.Histogram(df["coverage"]),
        f"mean_precision_{suffix}": df["precision"].mean(),
        f"mean_coverage_{suffix}": df["coverage"].mean(),
        f"num_empty_anchors_{suffix}": (df["anchor"].apply(len) == 0).sum(),
        f"csv_summary_{suffix}": wandb.Table(dataframe=df)
    })

    logger.info(f"Saved anchor summary with {len(df)} instances to {csv_path}")


def build_predict_fn(model, device, config, bins):
    """Return a prediction function that prepares input and bins output."""
    def predict_fn(x: np.ndarray) -> np.ndarray:
        num_samples = x.shape[0]
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        if config.paper == "transynergy":
            x_tensor = x_tensor.view(num_samples, 3, config.feature_length)
        elif config.paper == "biomining":
            x_tensor = x_tensor.view(num_samples, config.feature_length)

        x_tensor = x_tensor.clone().detach().requires_grad_(True)

        with torch.no_grad():
            out = model(x_tensor)

        return np.digitize(out.cpu().numpy(), bins).flatten()

    return predict_fn


def sample_data(X, Y, strategy: str, num_samples: int | None = None):
    """Return a subsample of the data according to the strategy."""

    n = len(Y)
    indices = None

    if strategy == 'bottom_10_percent':
        indices = np.argsort(Y)[-int(0.1 * n):]
    elif strategy == 'top_10_percent':
        indices = np.argsort(Y)[:int(0.1 * n)]
    elif strategy == 'random':
        if n < num_samples:
            raise ValueError(f"Cannot sample {num_samples} items from {n} available")
        indices = np.random.choice(n, size=num_samples, replace=False)
    elif strategy == 'all':
        indices = np.arange(n)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    sampled_X = X[indices]
    
    if num_samples is not None:
        sampled_X = sampled_X[:num_samples]

    return sampled_X

