import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import gaussian_filter1d


COLORS = ['tab:pink', 'tab:green', 'tab:blue', 'r', 'tab:orange', 'c', 'y', 'tab:purple', 
          'tab:brown', 'tab:olive']


def load_tensorboard_files(data_files: list[str], n_epochs: int = 1000, 
                           normalize_y: bool = False) -> list[pd.DataFrame]:
    dfs = []
    for fp in data_files:
        df = pd.read_csv(fp)
        e_min = df['Step'].min()
        df['epoch'] = ((df['Step'] - e_min) / (df['Step'].max() - e_min) * n_epochs).astype(int)
        if normalize_y:
            v_min = df['Value'].min()
            df['Value'] = (df['Value'] - v_min) / (df['Value'].max() - v_min)
        dfs.append(df)
    return dfs


def plot_training_progress(train_data_files: list[str], val_data_files: list[str], labels: list[str], 
                           title: str, n_epochs: int = 1000, figsize: tuple[int, int] = (10, 4),
                           normalize_y: bool = False, smooth: int = 4, y_max: float = 0.2, y_min: float = 0., 
                           alpha: float = 0.3):
    dfs_train = load_tensorboard_files(train_data_files, n_epochs, normalize_y)
    dfs_val = load_tensorboard_files(val_data_files, n_epochs, normalize_y)
    dfs = [dfs_train, dfs_val]
    titles = ['Training Loss', 'Validation Loss']
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for i, dfs in enumerate(dfs):
        for j, df in enumerate(dfs):
            axs[i].plot(df['epoch'], gaussian_filter1d(df['Value'], smooth), label=labels[j], color=COLORS[j])
            axs[i].plot(df['epoch'], df['Value'], color=COLORS[j], alpha=alpha)
        axs[i].set_ylim((y_min, y_max))
        axs[i].set_title(titles[i])
    fig.suptitle(title)
    fig.tight_layout()
    plt.legend()
    plt.show()
