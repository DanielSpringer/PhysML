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
        df = df.groupby('epoch').agg('last').reset_index()
        events_df = pd.concat([events_df, df], ignore_index=True)
    
    # fix missing epochs
    steps_per_epoch = (events_df.groupby('run')
                                .apply(lambda x: (x['step'] / x['epoch']).dropna().iloc[-1].astype(int), include_groups=False)
                                .rename('steps/epoch'))
    events_df = events_df.merge(steps_per_epoch, left_on='run', right_index=True)
    events_df['epoch'] = (events_df['step'] / events_df['steps/epoch']).astype(int)
    return events_df.drop(columns='steps/epoch')


def _tensorboard_walltimes_to_run_hours(walltimes: pd.Series) -> float:
    return round((walltimes.max() - walltimes.min()) / 3600, 2)


def _agg_tensorboard_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    epoch = df.loc[df['value'].argmin(), 'epoch']
    sub_df = df.loc[:df[df['epoch'] == epoch].index[0]].copy()
    sub_df['cummin'] = sub_df['value'].cummin()
    sub_df['decreasing'] = sub_df['cummin'].diff().fillna(0) < 0
    sub_df['counter'] = sub_df['decreasing'].cumsum()
    g = sub_df[['value', 'epoch', 'counter']].groupby('counter').agg({'value': 'count', 'epoch': ['min', 'max']}).droplevel(0, axis=1)
    m = g.loc[g['count'].idxmax()]
    return pd.Series({
        'wall time [hours]': _tensorboard_walltimes_to_run_hours(df['wall_time']), 
        'epochs': df['epoch'].max(), 
        'min. loss': df['value'].min(), 
        'epoch of min. loss': epoch,
        'max. epochs w/o decrease': m['max'] - m['min'] + 1,
        'epoch of min. loss (10% threshold)': df[df['value'] <= (df['value'].min() * 1.1)]['epoch'].min()
    })


def get_tensorboard_statistics(tensorboard_data: pd.DataFrame) -> pd.DataFrame:
    return (tensorboard_data.groupby(['run', 'ld', 's', 'seed']).apply(_agg_tensorboard_data, include_groups=False)
                            .astype({'epochs': int, 'epoch of min. loss': int, 'max. epochs w/o decrease': int, 
                                     'epoch of min. loss (10% threshold)': int}))


def tensorboard_statistics_to_latex(stats_df: pd.DataFrame, ld: int = 32, s: int = 24000, seed: int = 123) -> str:
    columns = ['run', 'wall time [hours]', 'min. loss', 'epoch of min. loss', 'max. epochs w/o decrease', 
               'epoch of min. loss (10% threshold)']
    df = stats_df.reset_index()
    df = df[(df['ld'] == ld) & (df['s'] == s) & (df['seed'] == seed)]
    df = df[columns]
    df[columns[0]] = df[columns[0]].str.replace('_', '-')
    df[columns[1]] = df[columns[1]].map('{:.0f}'.format)
    df[columns[2]] = df[columns[2]].map('{:.6f}'.format)
    df[columns[3]] = df[columns[3]].map('{:,.0f}'.format)
    df[columns[4]] = df[columns[4]].map('{:,.0f}'.format)
    df[columns[5]] = df[columns[5]].map('{:,.0f}'.format)
    values_str = '\n        '.join([' & '.join(row) + r' \\' for row in df.values])
    latex_str = (r"""\begin{table}
    \centering
    \begin{tabular}{lccccc}
        \toprule
        run & \begin{tabular}[c]{@{}c@{}} wall time\\{[hours]}\end{tabular} & min. loss & \begin{tabular}[c]{@{}c@{}}epoch of\\min. loss\end{tabular} & \begin{tabular}[c]{@{}c@{}}epoch of min. loss\\(10\% threshold)\end{tabular} & \begin{tabular}[c]{@{}c@{}}max. epochs\\w/o decrease\end{tabular}\\
        \midrule
        """ + values_str +
        r"""
        \bottomrule
    \end{tabular}
    \caption{Statistics for the training of each scenario (with latent space dimension of 32 and subsamples per vertex of 24,000).}
    \label{tab:ae_converge}
\end{table}""")
    return latex_str


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
