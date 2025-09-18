import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


COLORS = ['tab:pink', 'tab:green', 'tab:blue', 'r', 'tab:orange', 'c', 'y', 'tab:purple', 
          'tab:brown', 'tab:olive']


def get_cmap(n, name='hsv'):
    return plt.get_cmap(name, n + 1)


def get_tensorboard_data(base_path: str, folders: list[str], labels: list[str]) -> pd.DataFrame:
    events_df = pd.DataFrame()
    for folder, label in zip(folders, labels):
        event_acc = EventAccumulator(base_path + folder)
        event_acc.Reload()
        df = pd.DataFrame(event_acc.Scalars('val_loss'))
        df['run'] = label
        df_epoch = pd.DataFrame(event_acc.Scalars('epoch'))
        df_epoch['epoch'] = df_epoch['value']
        df = df.merge(df_epoch[['step', 'epoch']], on='step', how='left')
        events_df = pd.concat([events_df, df], ignore_index=True)
    
    # fix missing epochs
    steps_per_epoch = (events_df.groupby('run')
                                .apply(lambda x: (x['step'] / x['epoch']).dropna().iloc[-1].astype(int), include_groups=False)
                                .rename('steps/epoch'))
    events_df = events_df.merge(steps_per_epoch, left_on='run', right_index=True)
    events_df['epoch'] = (events_df['step'] / events_df['steps/epoch']).astype(int)
    return events_df.drop(columns='steps/epoch')


def _tensorboard_walltimes_to_run_hours(walltimes: pd.Series) -> pd.Series:
    return round((walltimes.max() - walltimes.min()) / 3600, 2)


def _count_epochs_no_decrease(losses: pd.Series) -> int:
    cummin = losses.cummin()
    decreasing = [True] + list(cummin[1:].values < cummin[:-1].values)
    counter = np.array(decreasing).cumsum()
    epochs_no_decrease = pd.DataFrame({'losses': losses, 'counter': counter}).groupby('counter').count()['losses']
    return epochs_no_decrease.max()


def get_tensorboard_statistics(tensorboard_data: pd.DataFrame) -> pd.DataFrame:
    return tensorboard_data.groupby('run').agg(wall_time=pd.NamedAgg('wall_time', _tensorboard_walltimes_to_run_hours), 
                                               epoch=pd.NamedAgg('epoch', 'max'), 
                                               min_loss=pd.NamedAgg('value', 'min'), 
                                               min_loss_epoch=pd.NamedAgg('value', lambda x: tensorboard_data.loc[x.argmin(), 'epoch']),
                                               max_epochs_no_decrease=pd.NamedAgg('value', _count_epochs_no_decrease))


def plot_loss_progress(tensorboard_data: pd.DataFrame, labels: list[str], figsize: tuple[int, int] = (8, 4),
                       smooth: int = 4, y_max: float = 0.2, y_min: float = 0., alpha: float = 0.3, log: bool = False):
    # group events_df by 'run' and iterate through groups
    grouped_df = tensorboard_data.groupby('run')
    cmap = get_cmap(grouped_df.ngroups, 'hsv')
    plt.figure(figsize=figsize)
    for i, (label, group) in enumerate(grouped_df):
        plt.plot(group['epoch'], gaussian_filter1d(group['value'], smooth), label=label, color=cmap(i))
        plt.plot(group['epoch'], group['value'], color=cmap(i), alpha=alpha)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if log:
        plt.yscale('log')
    else:
        plt.ylim((y_min, y_max))
    plt.title('Validation Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
