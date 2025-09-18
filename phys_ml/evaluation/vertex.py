import asyncio
import nest_asyncio
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm

from .. import metrics
from ..config import Vertex24x6Config
from ..load_data.vertex import *
from ..trainer import TrainerModes
from ..trainer.vertex import VertexTrainer, VertexTrainer24x6
from ..visualization import base as vis, vertex_visualization as vertvis
from ..util import is_notebook



# ----------------------------------------------------------------------------------------------
# VERTEX CORRELATION MATRIX
# ----------------------------------------------------------------------------------------------
def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped


@background
def process_vertex(cor_mat: np.ndarray, i: int, fp2_idcs: list[list[int]], 
                   paths_or_vertices: list[str]|list[np.ndarray], pre_load_vertices: bool = False):
    cor_mat[i, i] = 1
    if pre_load_vertices:
        vertex1 = paths_or_vertices[i]
    else:
        vertex1 = AutoEncoderVertexDataset.load_from_file(paths_or_vertices[i])
    for j in tqdm(fp2_idcs[i], leave=False):
        if pre_load_vertices:
            vertex2 = paths_or_vertices[j]
        else:
            vertex2 = AutoEncoderVertexDataset.load_from_file(paths_or_vertices[j])
        # v1n, v2n = vertex1 - vertex1.mean(), vertex2 - vertex2.mean()
        # cor_mat[i, j] = cor_mat[j, i] = np.sum(v1n * v2n) / np.sqrt(np.sum(v1n**2) * np.sum(v2n**2))
        z1, z2 = vertex1.flatten(), vertex2.flatten()
        cor_mat[i, j] = cor_mat[j, i] = np.dot((z1 / np.linalg.norm(z1)), (z2 / np.linalg.norm(z2)))
    del vertex1


def vertex_correlation(vertex_dir: str, paths_or_vertices: list[np.ndarray]|list[str]|None = None, n_workers: int = -1, 
                       pre_load_vertices: bool = False, vertex_file_ending: Literal['h5', 'npy'] = 'h5', save_suffix: str = '') -> np.ndarray:
    """ pre_load_vertices: Load all vertices into memory. Requires ca. 100 GB of memory for all 51 vertices."""
    nest_asyncio.apply()
    if paths_or_vertices is None:
        paths_or_vertices = sorted(glob.glob(os.path.join(vertex_dir, f'*.{vertex_file_ending}')))
        if pre_load_vertices:
            if vertex_file_ending == 'h5':
                paths_or_vertices = [AutoEncoderVertexDataset.load_from_file(fp) for fp in tqdm(paths_or_vertices, desc='load files')]
            elif vertex_file_ending == 'npy':
                paths_or_vertices = [np.load(fp) for fp in tqdm(paths_or_vertices, desc='load files')]
    s = len(paths_or_vertices)
    cor_mat = np.empty((s, s))
    fp2_idcs = [range(i, s) for i in range(1, s + 1)]
    idx_ranges = ([range(s)] if n_workers == -1 
                  else [range(*(i, min(i + n_workers, s))) for i in range(0, s, n_workers)])
    for r in idx_ranges:
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(*[process_vertex(cor_mat, i, fp2_idcs, paths_or_vertices, pre_load_vertices) 
                                for i in r])
        loop.run_until_complete(looper)

    # save result
    fname = 'cor_mat'
    if save_suffix:
        fname += f'_{save_suffix}'
    np.save(f'{fname}.npy', cor_mat)
    return cor_mat



# ----------------------------------------------------------------------------------------------
# VERTEX ANALYSIS
# ----------------------------------------------------------------------------------------------
def plot_statistics(data, nbins: int = 60, log: bool = False, alpha: float = 0.25, figsize: tuple[int, int] = (12, 3)):
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    col_iter = iter(colors)
    plt.figure(figsize=figsize)
    for phase in data.keys():
        color = next(col_iter)
        plt.hist(data[phase], bins=nbins, density=True, label=phase, log=log, histtype='step', color=color)
        plt.hist(data[phase], bins=nbins, density=True, log=log, alpha=alpha, color=color)
    plt.xlim(-32, 32)
    plt.legend()
    plt.show()


def vertex_statistics(data_dir: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    filepath_dict = AutoEncoder24x6InfoNCEDataset.get_filepaths(data_dir, subset=None, subset_shuffle=False)[0]
    phases = [p.upper() for p in filepath_dict.keys()]
    data = {'AFM': None, 'SC': None, 'FM': None}
    df = pd.DataFrame(index=phases, columns=['min', 'max', 'mean', 'sum'])
    for phase in phases:
        filepaths = filepath_dict[phase.lower()]
        phase_vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(filepaths)
        data[phase] = np.ravel(list(phase_vertices.values()))
        values = []
        for fp in tqdm(filepaths, leave=False, desc='Analyse vertices'):
            vertex = phase_vertices[fp]
            values.append((vertex.min(), vertex.max(), vertex.mean(), vertex.sum()))
        values = np.array(values)
        df.loc[phase] = [values[:, 0].min(), values[:, 1].max(), values[:, 2].mean(), values[:, 3].sum()]
    return df, data



# ----------------------------------------------------------------------------------------------
# TRAINING INFO DICTIONARY
# ----------------------------------------------------------------------------------------------
def load_info_dict(info_fn: str) -> list[dict[str, Any]]:
    try:
        info_dict = pickle.load(open(info_fn, 'rb'))
    except:
        info_dict = []
    return info_dict


def backup_info(info_fn: str) -> None:
    # back_up existing info_files
    info_name = info_fn.split('.')[0]
    files = sorted(glob.glob(f'{info_name}*.pkl'))
    files = [f for f in files if re.match(rf'{info_name}_?\d*.pkl', f)]
    if len(files) > 0:
        last_i = int(files[-1].split('.')[0].split('_')[-1]) if len(files) > 1 else 0
        os.rename(f'{info_name}.pkl', f'{info_name}_{last_i + 1:02d}.pkl')



# ----------------------------------------------------------------------------------------------
# TRAIN AUTOENCODER
# ----------------------------------------------------------------------------------------------
def make_dataset(vertices: dict[str, np.ndarray], sample_count_per_vertex: int, dataset_kwargs: dict[str, Any], subset: int|None = None, 
                 subset_type: str|list[str]|None = None, file_paths: list[str]|None = None, 
                 dataset_class: type[AutoEncoderVertex24x6Dataset] = AutoEncoderVertex24x6Dataset,
                 return_filepaths: bool = True) -> AutoEncoderVertex24x6Dataset:
    config = Vertex24x6Config(subset=subset, subset_type=subset_type, sample_count_per_vertex=sample_count_per_vertex, **dataset_kwargs)
    return dataset_class(config, vertices, file_paths=file_paths, return_filepaths=return_filepaths)


def get_test_filepaths(path_train: str, train_dataset: AutoEncoderVertex24x6Dataset) -> list[str]:
    if train_dataset.__class__.__name__ == 'AutoEncoderVertex24x6Dataset':
        file_paths = train_dataset.file_paths
    elif train_dataset.__class__.__name__ == 'AutoEncoder24x6InfoNCEDataset':
        file_paths = train_dataset.file_paths_by_phase.melt().drop_duplicates()['value'].tolist()
    else:
        raise ValueError(f"Unknown dataset type: {type(train_dataset)}")
    return [fp for fp in [Path(fp).resolve().as_posix() for fp in glob.glob(f"{path_train}/*.h5")] if fp not in file_paths]


def make_test_from_train_dataset(vertices: dict[str, np.ndarray], path_train: str, dataset_kwargs: dict[str, Any], 
                                 sample_count_per_vertex: int, train_dataset: AutoEncoderVertex24x6Dataset) -> AutoEncoderVertex24x6Dataset:
    file_paths = get_test_filepaths(path_train, train_dataset)
    return make_dataset(vertices, sample_count_per_vertex, dataset_kwargs, subset=None, subset_type=None, file_paths=file_paths, 
                        dataset_class=type(train_dataset), return_filepaths=train_dataset.return_filepaths)


def init_trainer(config_kwargs: dict[str, Any], dataset: AutoEncoderVertex24x6Dataset|None = None, dataset_kwargs: dict[str, Any] = {},
                 subset_type: list[str]|str|None = None, device_type: Literal['cpu', 'gpu'] = 'gpu', 
                 load_from: str|None = None, **kwargs) -> VertexTrainer24x6:
    kws = config_kwargs.copy()
    kws.update(dataset_kwargs)
    kws['device_type'] = device_type
    for k, v in kwargs.items():
        kws[k] = v
    if dataset:
        kws['subset_type'] = dataset.config.subset_type
    else:
        kws['subset_type'] = subset_type
    trainer = VertexTrainer24x6(project_name='vertex_24x6', config_name='confmod_auto_encoder.json', 
                                subconfig_name='AUTO_ENCODER_VERTEX_24X6', dataset=dataset, load_from=load_from,
                                config_kwargs=kws)
    return trainer


def eval_train(trainer: VertexTrainer, info_dict: list[dict[str, Any]], info_filename: str, hidden_dims: list, 
               resume: bool = False, path: str|None = None, version: int|None = None):
    if resume:
        assert path or version, 'If resuming, either `path` or `version` must be provided.'
        if not path:
            if version:
                path = (sorted(trainer.get_full_save_path().glob('*'))[-1] / f'version_{version}').as_posix()
            else:
                path = sorted(trainer.get_full_save_path().glob('*/*'))[-1].as_posix()
        trainer.config.resume = 'best'
        trainer.config.save_path = path

    trainer.config.hidden_dims = hidden_dims
    trainer.train(train_mode=TrainerModes.JUPYTER)

    if info_dict is None:
        info_dict = load_info_dict(info_filename)
    info_dict.append({'hidden_dims': hidden_dims, 'latent_dim': hidden_dims[-1], 
                      'save_path': Path(trainer.config.save_path).as_posix()})
    with open(info_filename, 'wb') as f:
        pickle.dump(info_dict, f)
    print(f">>> dim: {info_dict[-1]['latent_dim']}\n>>> save_path: '{info_dict[-1]['save_path']}'")



# ----------------------------------------------------------------------------------------------
# CONVERGENCE
# ----------------------------------------------------------------------------------------------
def _create_loss_plot(ax: Axes, tensorboard_data: pd.DataFrame, labels: list[str], y_max: float = 0.2, 
                      smooth: int = 4, alpha: float = 0.3, log: bool = False):
    grouped_df = tensorboard_data.groupby(['run', 'ld', 's'])
    cmap = vis.get_cmap(grouped_df.ngroups, 'hsv')
    for i, ((run_id, ld, s), group) in enumerate(grouped_df):
        ax.plot(group['epoch'], gaussian_filter1d(group['value'], smooth), label=labels[i], color=cmap(i))
        ax.plot(group['epoch'], group['value'], color=cmap(i), alpha=alpha)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    if log:
        ax.set_yscale('log')
    else:
        ax.set_ylim((.0, y_max))
    ax.legend(loc='upper right')


def plot_loss_progress(tensorboard_data: pd.DataFrame, y_maxs: tuple[float, float, float, float], 
                       figsize: tuple[int, int] = (8, 4), smooth: int = 4, alpha: float = 0.3, log: bool = False):
    data = tensorboard_data.sort_values(by=['run', 'ld', 's'])

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    subset = data[(data['ld'] == 32) & (data['s'] == 24000)]
    plot_data = subset[subset['run'].str.match(r'(1_1)|(2_._1$)|(3_.*)')]
    _create_loss_plot(axs[0], plot_data, labels=sorted(plot_data['run'].unique()), 
                      y_max=y_maxs[0], smooth=smooth, alpha=alpha, log=log)
    plot_data = subset[subset['run'].str.match(r'(1_2)|(2_._2$)')]
    _create_loss_plot(axs[1], plot_data, labels=sorted(plot_data['run'].unique()),
                      y_max=y_maxs[1], smooth=smooth, alpha=alpha, log=log)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    subset = data[data['run'] == '2_1_1']
    plot_data = subset[subset['s'] == 24000]
    _create_loss_plot(axs[0], plot_data, labels=sorted(plot_data['ld'].unique()),
                      y_max=y_maxs[2], smooth=smooth, alpha=alpha, log=log)
    plot_data = subset[subset['ld'] == 32]
    _create_loss_plot(axs[1], plot_data, labels=sorted(plot_data['s'].unique()),
                      y_max=y_maxs[3], smooth=smooth, alpha=alpha, log=log)
    plt.tight_layout()
    plt.show()



# ----------------------------------------------------------------------------------------------
# EVALUATE AUTOENCODER
# ----------------------------------------------------------------------------------------------
def evaluate_prediction(save_path: str, test_filename: str, trainer: VertexTrainer, target: np.ndarray, 
                        hidden_dims: list[int], target_slice: np.ndarray, 
                        predict_func: Callable[...,np.ndarray]|None = None, 
                        load_func: Callable[...,np.ndarray]|None = None, 
                        slice_at: int|tuple[int,...]|None = None, axis: int|None = None, 
                        **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
    assert predict_func is not None or load_func is not None, \
        'Either `predict_func` or `load_func` must be provided.'
    if load_func is None:
        trainer.config.hidden_dims = hidden_dims
        pred = predict_func(test_filename, new_vertex=target, train_mode=TrainerModes.JUPYTER, 
                            load_from=save_path, **kwargs)
    else:
        pred = load_func(save_path)
    if len(target.shape) == 3:
        pred = AutoEncoderVertex24x6Dataset.to_3d_vertex(pred)
    dim = len(pred.shape)
    if dim in [3, 6]:
        pred_slice = vertvis.get_mat_slice(pred, axis, slice_at)
        rmse = metrics.rmse(target, pred)
    else:
        if dim == 4:
            pred_slice = pred.reshape((AutoEncoderVertexDataset.length,) * 2, order='F')
        else:
            pred_slice = pred
        rmse = metrics.rmse(target_slice, pred_slice)
    eigvec = metrics.vertex.get_dominant_eigenvector(pred_slice)
    return rmse, eigvec, pred_slice


def evaluate_all_models(train_results: list[dict[str, Any]], test_filename: str, trainer: VertexTrainer, 
                        target: np.ndarray, slice_at: int|tuple[int,...]|None, axis: int, keys: list[str],
                        predict_func: Callable[...,np.ndarray]|None = None, 
                        load_func: Callable[...,np.ndarray]|None = None, 
                        **kwargs) -> tuple[dict[int, tuple[float, np.ndarray, np.ndarray]], np.ndarray]:
    target_slice = vertvis.get_mat_slice(target, axis, slice_at)
    results = {key: 
               evaluate_prediction(mod_info['save_path'], test_filename, trainer, target, mod_info['hidden_dims'], 
                                   target_slice, predict_func, load_func, slice_at, axis, **kwargs)
               for mod_info, key in zip(train_results, keys)}
    return results, target_slice


def report_results(results: dict[int, tuple[float, np.ndarray, np.ndarray]], target_slice: np.ndarray, 
                   slice_at: int|tuple[int,...]|None, axis: int, nrows:int, ncols: int):
    assert nrows * ncols >= len(results) + 1, \
        f"`{nrows=}`and `{ncols=}` not enough for {len(results + 1)} items to plot in `train_info` + target."
    
    res_print = '\n   '.join([f'latent_dim={k}: RMSE={v[0]:.4f}' for k, v in results.items()])
    print(f"RESULTS:\n   {res_print}")
    target_eigvec = metrics.vertex.get_dominant_eigenvector(target_slice)

    # vertex visualisation
    if slice_at is None:
        params_str = str(axis)
    elif isinstance(slice_at, int):
        params_str = str(slice_at)
    elif isinstance(slice_at, tuple):
        if isinstance(axis, tuple):
            slice_k = (set(range(1,4)) - set(axis)).pop()
            params_str = f'$k_{slice_k}={slice_at[0] * 24 + slice_at[1]}$'
        elif isinstance(axis, int):
            other_ks = set(range(1,4)) - {(axis + 1) // 2}
            params_str = ', '.join([f'$k_{{{k}_{c}}}={sl}$' 
                                    for (k, c), sl in zip([(k, c) for k in other_ks for c in ['x', 'y']], slice_at)])
    plot_data = {'target': target_slice}
    plot_data.update({k: v[2] for k, v in results.items()})
    vertvis.plot_compare_grid(plot_data, nrows, ncols, axis, None, figsize=(8, 6), 
                              title=f'Visualization of reconstructed vertices at ({params_str})')

    plot_data = {k: np.square(v - target_slice) for k, v in plot_data.items()}
    vertvis.plot_compare_grid(plot_data, nrows, ncols, axis, None, figsize=(8, 6),
                              title=f'Squared errors to target of reconstructed vertices at ({params_str})')

    # plot rmses
    pred_rmses = {k: v[0] for k, v in results.items()}
    vertvis.lineplot([pred_rmses.keys()], [pred_rmses.values()], title='Root mean squared error', 
                     ylabel='RMSE', xlabel='latent dimension', xticks=list(pred_rmses.keys()))

    # plot eigenvectors
    pred_eigvecs = {k: v[1] for k, v in results.items()}
    vertvis.lineplot_compare(pred_eigvecs, target=target_eigvec, title='Eigenvector', ylabel='Eigenvector', 
                             xlabel='k', xticks=[])


def evaluate_and_report(train_results: dict[str, Any], test_filename: str, trainer: VertexTrainer, 
                        target: np.ndarray, slice_at: int|tuple[int,...]|None, axis: int, keys: list[str],
                        nrows:int, ncols: int, predict_func: Callable[...,np.ndarray]|None = None, 
                        load_func: Callable[...,np.ndarray]|None = None, 
                        **kwargs):
    assert nrows * ncols >= len(train_results) + 1, \
        f"`{nrows=}`and `{ncols=}` not enough for {len(train_results) + 1} items to plot in `train_info` + target."
    
    results, target_slice = evaluate_all_models(train_results, test_filename, trainer, target, slice_at, axis, keys,
                                                predict_func, load_func, **kwargs)
    report_results(results, target_slice, slice_at, axis, nrows, ncols)


def predict_all(file_paths: list[str], vertices: dict[str, np.ndarray], save_path: str, 
                config_kwargs: dict[str, Any], dataset_kwargs: dict[str, Any], encode_only: bool, 
                subset_type: str|list[str]|None = None, device_type: Literal['cpu', 'gpu'] = 'gpu',
                train_mode: TrainerModes = TrainerModes.JUPYTER) -> list[np.ndarray]:
    trainer = init_trainer(config_kwargs, dataset_kwargs=dataset_kwargs, subset_type=subset_type, device_type=device_type, 
                           load_from=save_path)
    _ = trainer.init_trainer(train_mode=train_mode, load_from=save_path)
    preds = []
    iterator = file_paths
    if is_notebook():
        iterator = tqdm(iterator)
    for fp in iterator:
        vertex = vertices[fp]
        pred = trainer.predict(fp, vertex, encode_only=encode_only)
        preds.append(pred)
    return preds


def mean_rmse(vertices: dict[str, np.ndarray], save_path: str, plot: bool = False) -> dict[float, float]:
    true_fps = sorted(vertices.keys())
    pred_dir = f'{save_path}/predictions'

    if plot:
        i = 18
        axis = 3
        nrows, ncols = 1, 2
        slice_at = (i, i, i, i)
        other_ks = set(range(1,4)) - {(axis + 1) // 2}
        params_str = ', '.join([f'$k_{{{k}_{c}}}={sl}$' for (k, c), sl 
                                in zip([(k, c) for k in other_ks for c in ['x', 'y']], slice_at)])

    rmses: dict[float, float] = {}
    iterator = true_fps
    if is_notebook():
        iterator = tqdm(iterator, desc='Computing RMSE', leave=False)
    for fp in iterator:
        fn = Path(fp).stem
        tp = float(fn[2:6])
        true = vertices[fp]
        pred = np.load(f'{pred_dir}/{fn}.npy')

        if plot:
            true_slice = vertvis.get_mat_slice(true, axis, slice_at)
            pred_slice = vertvis.get_mat_slice(pred, axis, slice_at)
            plot_data = {'true': true_slice, 'reconstruction': pred_slice}
            vertvis.plot_compare_grid(plot_data, nrows, ncols, axis, None, figsize=(6, 4), 
                                      title=f'Reconstruction of vertex({fn}) at ({params_str})', vmin=0.0, vmax=22.5)
        
        rmses[tp] = metrics.rmse(true, pred)
    return rmses


def plot_rmse_against_tp(rmse_df: pd.DataFrame, run_name: str, figsize: tuple[int, int] = (8,4), font_size: int = 14):
    errors = rmse_df['rmse']
    mean_rmse = np.mean(errors)
    train_df = rmse_df[rmse_df['train_data']]
    test_df = rmse_df[~rmse_df['train_data']]
    train_rmses = train_df['rmse']
    test_rmses = test_df['rmse']
    train_mean = np.mean(train_rmses)
    test_mean = np.mean(test_rmses)
    print(f'mean: {mean_rmse}, min: {min(errors)}, max: {max(errors)}')
    if not train_df.empty:
        print(f'training data only - mean: {train_mean}, min: {min(train_rmses)}, max: {max(train_rmses)}')
    if not test_df.empty:
        print(f'test data only - mean: {test_mean}, min: {min(test_rmses)}, max: {max(test_rmses)}')
    plt.figure(figsize=figsize)
    plt.plot(rmse_df['tp'], errors, color='tab:blue', zorder=0)
    plt.scatter(train_df['tp'], train_rmses, marker='o', color='tab:blue', label='train data')
    plt.scatter(test_df['tp'], test_rmses, marker='o', color='tab:pink', label='test data')
    plt.axhline(y=mean_rmse, color='tab:orange', linestyle='--', label='mean')
    plt.axhline(y=train_mean, color='tab:blue', linestyle='--', label='train mean')
    plt.axhline(y=test_mean, color='tab:pink', linestyle='--', label='test mean')
    plt.xlim(-0.02, 0.52)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.title(f'reconstruction RMSE for {run_name}', fontsize=font_size)
    plt.xlabel('tp', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.grid()
    plt.show()


def _create_rmse_boxplot(ax: Axes, rmses: pd.DataFrame, xlabel: str, xtick_labels: list[str], alpha: float = 0.4, width: float = 0.5):
    # Sort run_ids and lds for consistent plotting
    # big_gap = 2 * width  # gap between run_id groups
    # pos = 0
    # small_gap = width * 1.5  # gap within group of boxplots
    # for run_id in run_ids:
    #     for i, subgroup in enumerate(subgroups):
    #         subset = rmses[(rmses['run_id'] == run_id) & (rmses[subgrouping] == subgroup)]['rmse']
    #         if not subset.empty:
    #             data_to_plot.append(subset)
    #             positions.append(pos)
    #             labels.append(f"{run_id}_{subgrouping}{subgroup}")
    #             color_list.append(vis.COLORS[i])
    #             pos += small_gap
    #     pos += big_gap  # add gap after each run_id group
    # box = ax.boxplot(data_to_plot, positions=positions, widths=width, patch_artist=True)
    # for path_patch, color in zip(box['boxes'], color_list):
    #     path_patch.set_facecolor(color)
    #     path_patch.set_alpha(alpha)

    positions = []
    data_to_plot = []
    pos = 0
    gap = width * 1.5
    for (run_id, ld, s), group in rmses.groupby(['run_id', 'ld', 's']):
        data_to_plot.append(group['rmse'])
        positions.append(pos)
        pos += gap
    box = ax.boxplot(data_to_plot, positions=positions, widths=width, patch_artist=True)
    for path_patch in box['boxes']:
        path_patch.set_facecolor(vis.COLORS[0])
        path_patch.set_alpha(alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('RMSE')
    ax.set_xticks(positions, xtick_labels, rotation=90)


def plot_rmse_boxplots(rmses: pd.DataFrame, train_data: bool = True, figsize: tuple[int, int] = (6,4), 
                       alpha: float = 0.4, width: float = 0.5):
    rmses = rmses.sort_values(by=['run_id', 'ld', 's'])
    if not train_data:
        rmses = rmses[rmses['run_id'].str.startswith('1_') | (rmses['train_data'] == False)]

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    subset = rmses[(rmses['ld'] == 32) & (rmses['s'] == 24000)]
    plot_data = subset[subset['run_id'].str.match(r'(1_1)|(2_._1$)|(3_.*)')]
    _create_rmse_boxplot(axs[0], plot_data, 'scenario', sorted(plot_data['run_id'].unique()), alpha, width)
    plot_data = subset[subset['run_id'].str.match(r'(1_2)|(2_._2$)')]
    _create_rmse_boxplot(axs[1], plot_data, 'scenario', sorted(plot_data['run_id'].unique()), alpha, width)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    subset = rmses[rmses['run_id'] == '2_1_1']
    plot_data = subset[subset['s'] == 24000]
    _create_rmse_boxplot(axs[0], plot_data, 'latent space dimension', sorted(plot_data['ld'].unique()), alpha, width)
    plot_data = subset[subset['ld'] == 32]
    _create_rmse_boxplot(axs[1], plot_data, 'subsamples per vertex', sorted(plot_data['s'].unique()), alpha, width)
    plt.tight_layout()
    plt.show()



# ----------------------------------------------------------------------------------------------
# EVALUATE PHASE CLASSIFIER
# ----------------------------------------------------------------------------------------------
def _plot_classification(ax: Axes, classifications: pd.DataFrame, xlabel: str, xtick_labels: list[str], 
                         alpha: float = 0.4, width: float = 0.5):
    classifications = classifications.set_index(['run_id', 'ld', 's'])
    positions = []
    pos = 0
    gap = width * 1.5
    col = vis.COLORS[0]
    for (run_id, ld, s), row in classifications.iterrows():
        ax.bar(pos, row['f1'], width=width, color=col, alpha=alpha)
        positions.append(pos)
        pos += gap
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('f1 score')
    ax.set_xticks(positions, xtick_labels)
    ax.tick_params(axis='x', rotation=90, which='both', length=0)
    ax.set_ylim(int(classifications['f1'].min() * 20) / 20 - 0.06, 1.01)

    # sg_name = 'latent dimension' if subgrouping == 'ld' else 'sample count'
    # pos = 0
    # positions = []
    # labels = []
    # for i, (run_id, group) in enumerate(classifications.groupby('run_id')):
    #     subpos = []
    #     for j, (_, row) in enumerate(group.iterrows()):
    #         label = f"{subgrouping}={row[subgrouping]}" if i == 0 else None
    #         ax.bar(pos, row['f1'], width=width, label=label, color=vis.COLORS[j], alpha=alpha)
    #         subpos.append(pos)
    #         pos += width
    #     positions.append(np.mean(subpos))
    #     labels.append(f"run {run_id}")
    #     pos += width / 4
    # ax.set_xlabel('run ID')
    # ax.set_ylabel('f1 score')
    # ax.set_title(f'Classification results for {sg_name}s and runs')
    # ax.legend(title=sg_name, loc='lower right')
    # ax.set_xticks(positions, labels)
    # ax.tick_params(axis='x', rotation=90, which='both', length=0)
    # ax.set_ylim(int(classifications['f1'].min() * 20) / 20 - 0.06, 1.01)


def plot_classification_results(classifications: pd.DataFrame, figsize: tuple[int, int] = (6,4), alpha: float = 0.4, 
                                width: float = 0.5):
    classifications = classifications.sort_values(by=['run_id', 'ld', 's']).drop(columns='conf_mat')

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    subset = classifications[(classifications['ld'] == 32) & (classifications['s'] == 24000)]
    plot_data = subset[subset['run_id'].str.match(r'(1_1)|(2_._1$)|(3_.*)')]
    _plot_classification(axs[0], plot_data, 'scenario', sorted(plot_data['run_id'].unique()), alpha, width)
    plot_data = subset[subset['run_id'].str.match(r'(1_2)|(2_._2$)')]
    _plot_classification(axs[1], plot_data, 'scenario', sorted(plot_data['run_id'].unique()), alpha, width)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    subset = classifications[classifications['run_id'] == '2_1_1']
    plot_data = subset[subset['s'] == 24000]
    _plot_classification(axs[0], plot_data, 'latent space dimension', sorted(plot_data['ld'].unique()), alpha, width)
    plot_data = subset[subset['ld'] == 32]
    _plot_classification(axs[1], plot_data, 'subsamples per vertex', sorted(plot_data['s'].unique()), alpha, width)
    plt.tight_layout()
    plt.show()



# ----------------------------------------------------------------------------------------------
# FULL EVALUATION
# ----------------------------------------------------------------------------------------------
from sklearn import ensemble
from phys_ml.analysis.vertex import PhaseClassification


def predict_for_all_models(run_id: str, ld: int = 32, s: int = 24000, run_dir_name: str = 'run_results'):
    # load vertices
    path_train = '/gpfs/data/fs71925/shepp123/frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=None, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)

    # autoencoder
    hidden_dims = {
        8: [128, 64, 32, 8],
        16: [128, 64, 32, 16],
        20: [128, 64, 32, 20],
        24: [128, 64, 24],
        32: [128, 64, 32],
    }
    config_kwargs = {
        'hidden_dims': None,
        'epochs': 1000,
        'test_ratio': 0.2, 
        'devices': 1, 
        'num_dataloader_workers': 2, 
        'strategy': 'auto', 
        'batch_size': 8192 * 2,
    }
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': False, 
    }
    pred_configs = {
        '1_1': {},
        '1_2': {},
        '2_1_1': {'subset_type': ['afm', 'fm']},
        '2_1_2': {'subset_type': ['afm', 'fm']},
        '2_2_1': {'subset_type': ['sc', 'fm']},
        '2_2_2': {'subset_type': ['sc', 'fm']},
        '2_3_1': {'subset_type': ['afm', 'sc']},
        '2_3_2': {'subset_type': ['afm', 'sc']},
        '3_1': {'subset_type': 'sc'},
        '3_2': {'subset_type': 'afm'},
        '3_3': {'subset_type': 'fm'},
    }
    pred_config = pred_configs[run_id]
    pref = f'ld{ld}' if s == 24000 else f's{s}'
    save_path = f'/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/{run_dir_name}/{run_id}_{pref}'
    if os.path.exists(save_path):
        config_kwargs['hidden_dims'] = hidden_dims[ld]
        preds = predict_all(file_paths, vertices, save_path, config_kwargs, dataset_kwargs, 
                            encode_only=False, train_mode=TrainerModes.SLURM, **pred_config)


def evaluate_all(run_id: str, ld: int = 32, s: int = 24000, run_dir_name: str = 'run_results'):
    # load vertices
    path_train = '/gpfs/data/fs71925/shepp123/frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=None, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)

    # autoencoder
    seed = 123
    train_samples_per_vertex = 24000
    nce_train_samples = train_samples_per_vertex // 4
    test_samples_per_vertex = 2000
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': True, 
    }

    # phase classifier
    pc_models = [
        # svm.SVC(verbose=True, random_state=seed + 1), 
        ensemble.RandomForestClassifier(n_jobs=-1, verbose=0, random_state=seed + 2),
    ]

    # general train sets
    nce_train_dataset = make_dataset(vertices, nce_train_samples, dataset_kwargs, file_paths=file_paths, 
                                     dataset_class=AutoEncoder24x6InfoNCEDataset)

    # reconstruction filepaths
    ex_sc_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                           subset_seed=seed, subset_type=['afm', 'fm'], file_paths=file_paths)[0]
    ex_sc_fps = list(set(file_paths) - set(ex_sc_fps))
    ex_afm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                            subset_seed=seed, subset_type=['sc', 'fm'], file_paths=file_paths)[0]
    ex_afm_fps = list(set(file_paths) - set(ex_afm_fps))
    ex_fm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                           subset_seed=seed, subset_type=['afm', 'sc'], file_paths=file_paths)[0]
    ex_fm_fps = list(set(file_paths) - set(ex_fm_fps))
    sc_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type='sc', file_paths=file_paths)[0]
    sc_fps = list(set(file_paths) - set(sc_fps))
    afm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type='afm', file_paths=file_paths)[0]
    afm_fps = list(set(file_paths) - set(afm_fps))
    fm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type='fm', file_paths=file_paths)[0]
    fm_fps = list(set(file_paths) - set(fm_fps))

    # test sets
    test_dataset_full = make_dataset(vertices, test_samples_per_vertex, dataset_kwargs, file_paths=file_paths, 
                                     dataset_class=AutoEncoder24x6InfoNCEDataset)

    # run info
    base_path = f'/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/{run_dir_name}/'
    run_info = {
        '1_1': [],
        '1_2': [],
        '2_1_1': ex_sc_fps, 
        '2_1_2': ex_sc_fps,
        '2_2_1': ex_afm_fps,
        '2_2_2': ex_afm_fps,
        '2_3_1': ex_fm_fps,
        '2_3_2': ex_fm_fps,
        '3_1': sc_fps,
        '3_2': afm_fps,
        '3_3': fm_fps,
    }
    save_path = '/gpfs/data/fs71925/shepp123/PhysML/notebooks/vertex/'
    rmse_df = pd.DataFrame(columns=['run_id', 'ld', 's', 'tp', 'rmse', 'train_data'])
    classification_df = pd.DataFrame(columns=['run_id', 'ld', 's', 'f1', 'conf_mat'])

    # evaluate models
    def eval(run_id: str, run_name: str, ld: int, s: int, recon_files: list[str]):
        model_path = base_path + run_name
        if os.path.exists(model_path):
            # reconstruction
            if len(rmse_df[(rmse_df['run_id'] == run_id) 
                        & (rmse_df['ld'] == ld) 
                        & (rmse_df['s'] == s)]) < len(file_paths):
                rmses = mean_rmse(vertices, model_path)
                is_train_data = [fp not in recon_files for fp in file_paths]
                for is_td, (tp, rmse) in zip(is_train_data, rmses.items()):
                    rmse_df.loc[len(rmse_df)] = [run_id, ld, s, tp, rmse, is_td]
                rmse_df.to_csv(save_path + f'reconstruction_results_{run_name}.csv', index=False)

            # classification
            if classification_df[(classification_df['run_id'] == run_id)
                                & (classification_df['ld'] == ld) 
                                & (classification_df['s'] == s)].empty:
                pc = PhaseClassification(model_path, pc_models, run_name)
                pc.train(nce_train_dataset)
                model_scores = pc.evaluate_classifiers(test_dataset_full, print_conf_mat=False)
                pc_results = list(model_scores.values())[0]
                classification_df.loc[len(classification_df)] = [run_id, ld, s, pc_results[0]['f1'], pc_results[1]]
                classification_df.to_pickle(save_path + f'classification_results_{run_name}.pkl')

    pref = f'ld{ld}' if s == 24000 else f's{s}'
    run_name = f'{run_id}_{pref}'
    eval(run_id, run_name, ld, s, run_info[run_id])
