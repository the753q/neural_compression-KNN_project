import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(log_path):
    df = pd.read_csv(log_path)

    metrics = [c for c in df.columns if c not in ["epoch", "step"]]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    
    if len(metrics) == 1:
        axes = [axes]

    for ax, col in zip(axes, metrics):
        epoch_means = df.groupby("epoch")[col].mean()
        ax.plot(epoch_means.index, epoch_means.values)
        ax.set_title(col)
        ax.set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(log_path.replace(".csv", ".png"))
    plt.show()

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python plot_metrics.py [path to log.csv]"
    plot_metrics(sys.argv[1])