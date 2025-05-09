import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(tr_loss, vl_loss, tr_pearson, vl_pearson, 
                tr_spearman, vl_spearman, out_path=None):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(tr_loss)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 3, 4)
    plt.plot(vl_loss)
    plt.title('Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 3, 2)
    plt.plot(tr_pearson)
    plt.title('Train Pearson')
    plt.xlabel('Epoch')
    plt.ylabel('Corr')
    
    plt.subplot(2, 3, 5)
    plt.plot(vl_pearson)
    plt.title('Val Pearson')
    plt.xlabel('Epoch')
    plt.ylabel('Corr')
    
    plt.subplot(2, 3, 3)
    plt.plot(tr_spearman)
    plt.title('Train Spearman')
    plt.xlabel('Epoch')
    plt.ylabel('Corr')
    
    plt.subplot(2, 3, 6)
    plt.plot(vl_spearman)
    plt.title('Val Spearman')
    plt.xlabel('Epoch')
    plt.ylabel('Corr')
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
    
    plt.close()

def plot_grid_results(res, x_col='lr', y_col='val_spearman', 
                     group_col='arch', out_path=None):
    plt.figure(figsize=(10, 7))
    
    for g in res[group_col].unique():
        group = res[res[group_col] == g]
        plt.scatter(
            group[x_col], 
            group[y_col],
            label=g,
            alpha=0.7,
            s=80
        )
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Grid Search Results')
    
    if x_col == 'lr':
        plt.xscale('log')
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    best_idx = res[y_col].idxmax()
    best = res.iloc[best_idx]
    
    plt.annotate(
        f"Best: {best[y_col]:.4f}",
        xy=(best[x_col], best[y_col]),
        xytext=(best[x_col]*1.2 if x_col == 'lr' else best[x_col]+0.5, 
                best[y_col]*0.98),
        arrowprops=dict(color='black', width=1.5, headwidth=8),
        fontsize=12
    )
    
    if out_path:
        plt.savefig(out_path)
    
    plt.close()

def plot_preds(true, pred, title="Predictions", out_path=None):
    plt.figure(figsize=(8, 6))
    
    true = true.flatten()
    pred = pred.flatten()
    
    corr = np.corrcoef(true, pred)[0, 1]
    
    plt.scatter(true, pred, alpha=0.5)
    
    min_v = min(true.min(), pred.min())
    max_v = max(true.max(), pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--')
    
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(f"{title}, Corr: {corr:.4f}")
    plt.grid(alpha=0.3)
    
    if out_path:
        plt.savefig(out_path)
    
    plt.close()