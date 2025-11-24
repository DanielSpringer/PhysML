import asyncio
import nest_asyncio
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
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


RUN_ID_MAPPING = {
    '1_1': '$S_{[AFM+SC+FM]}$', 
    '1_2': '$S_{[AFM+SC+FM]}$',
    '2_1_1': '$S_{[AFM+FM]}$', 
    '2_1_1_step': 'next $t\'$',
    '2_1_2': '$S_{[AFM+FM]}$',
    '2_2_1': '$S_{[SC+FM]}$',
    '2_2_2': '$S_{[SC+FM]}$',
    '2_3_1': '$S_{[AFM+SC]}$',
    '2_3_2': '$S_{[AFM+SC]}$',
    '3_1': '$S_{SC}$',
    '3_2': '$S_{AFM}$',
    '3_3': '$S_{FM}$',
    'no_encoding': 'original vertex',
}


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
def _create_loss_plot(tensorboard_data: pd.DataFrame, category_key: str, y_max: float = 0.2, smooth: int = 4, 
                      alpha: float = 0.3, log: bool = False, figsize: tuple[int, int] = (4, 3), 
                      plot_dir: Path = Path(), plot_name: str|None = None):
    grouped_df = tensorboard_data.groupby(['run', 'ld', 's'])
    labels = sorted(tensorboard_data[category_key].unique())
    if category_key == 'run':
        labels = [RUN_ID_MAPPING.get(x, x) for x in labels]
    cmap = vis.get_cmap(grouped_df.ngroups, 'hsv')
    fig, ax = plt.subplots(figsize=figsize)
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
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_loss_progress(tensorboard_data: pd.DataFrame, y_maxs: tuple[float, float, float, float, float], 
                       figsize: tuple[int, int] = (4, 3), smooth: int = 4, alpha: float = 0.3, log: bool = False,
                       plot_dir: Path = Path(), plot_name: str = ''):
    data = tensorboard_data.sort_values(by=['run', 'ld', 's', 'seed'])

    subset = data[(data['ld'] == 32) & (data['s'] == 24000) & (data['seed'] == 123)]
    plot_data = subset[subset['run'].str.match(r'(1_1)|(2_._1$)|(3_.*)|(no_*)')]
    _create_loss_plot(plot_data, 'run', y_maxs[0], smooth, alpha, log, 
                      figsize, plot_dir, plot_name + '_mse')
    plot_data = subset[subset['run'].str.match(r'(1_2)|(2_._2$)')]
    _create_loss_plot(plot_data, 'run', y_maxs[1], smooth, alpha, log, 
                      figsize, plot_dir, plot_name + '_contr')

    subset = data[data['seed'] == 123]
    plot_data = subset[(subset['run'] == '2_1_1') & (subset['s'] == 24000)]
    _create_loss_plot(plot_data, 'ld', y_maxs[2], smooth, alpha, log, 
                      figsize, plot_dir, plot_name + '_ld')
    plot_data = subset[(subset['run'] == '2_3_1') & (subset['ld'] == 32)]
    _create_loss_plot(plot_data, 's', y_maxs[3], smooth, alpha, log, 
                      figsize, plot_dir, plot_name + '_s')
    
    subset = data[(data['run'] == '2_3_1') & (data['ld'] == 32) & (data['s'] == 24000)]
    grouped_runs = subset.groupby('seed')
    fig, ax = plt.subplots(figsize=figsize)
    for i, (seed, group) in enumerate(grouped_runs):
        ax.plot(group['epoch'], gaussian_filter1d(group['value'], smooth), color='tab:blue', alpha=2*alpha)
        ax.plot(group['epoch'], group['value'], color='blue', alpha=alpha)
    
    mean_values = subset.groupby('epoch')['value'].mean()
    ax.plot(mean_values.index, gaussian_filter1d(mean_values, smooth), label='mean', color='red')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    if log:
        ax.set_yscale('log')
    else:
        ax.set_ylim((.0, y_maxs[4]))
    ax.legend(loc='upper right')
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}_subsets.png', bbox_inches='tight', dpi=300)
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
    _ = trainer.init_trainer(train_mode=train_mode, load_ckpt='best')
    preds = []
    iterator = file_paths
    if is_notebook():
        iterator = tqdm(iterator)
    for fp in iterator:
        vertex = vertices[fp]
        pred = trainer.predict(fp, vertex, encode_only=encode_only)
        preds.append(pred)
    return preds


def mean_rmse(vertices: dict[str, np.ndarray], save_path: str, train_files: list[str], plot: bool = False) -> pd.DataFrame:
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

    rmse_data = []
    iterator = true_fps
    if is_notebook():
        iterator = tqdm(iterator, desc='Computing RMSE', leave=False)
    for fp in iterator:
        fn = Path(fp).stem
        tp = float(fn[2:6])
        true = vertices[fp]
        try:
            pred = np.load(f'{pred_dir}/{fn}.npy')

            if plot:
                true_slice = vertvis.get_mat_slice(true, axis, slice_at)
                pred_slice = vertvis.get_mat_slice(pred, axis, slice_at)
                plot_data = {'true': true_slice, 'reconstruction': pred_slice}
                vertvis.plot_compare_grid(plot_data, nrows, ncols, axis, None, figsize=(6, 4), 
                                        title=f'Reconstruction of vertex({fn}) at ({params_str})', vmin=0.0, vmax=22.5)
            
            rmse_data.append((tp, metrics.rmse(true, pred), fp in train_files))
        except FileNotFoundError:
            print(f'Prediction file not found for {fn}, skipping.')
            continue
    return pd.DataFrame(rmse_data, columns=['tp', 'rmse', 'train_data'])


def plot_rmse_against_tp(rmse_df: pd.DataFrame, run_name: str, figsize: tuple[int, int] = (6, 3), font_size: int = 14, 
                         legend: bool = False, ymin: float|None = None, ymax: float|None = None, plot_dir: Path = Path(), 
                         plot_name: str|None = None):
    errors = rmse_df['rmse']
    mean_rmse = np.mean(errors)
    train_df = rmse_df[rmse_df['train_data']]
    test_df = rmse_df[~rmse_df['train_data']]
    train_rmses = train_df['rmse']
    test_rmses = test_df['rmse']
    train_mean = np.mean(train_rmses)
    test_mean = np.mean(test_rmses)
    print(f'reconstruction RMSE for {run_name}')
    print(f'mean: {mean_rmse}, min: {min(errors)}, max: {max(errors)}')
    if not train_df.empty:
        print(f'training data only - mean: {train_mean}, min: {min(train_rmses)}, max: {max(train_rmses)}')
    if not test_df.empty:
        print(f'test data only - mean: {test_mean}, min: {min(test_rmses)}, max: {max(test_rmses)}')
    plt.figure(figsize=figsize)
    plt.plot(rmse_df['tp'], errors, color='tab:blue', zorder=0)
    plt.scatter(train_df['tp'], train_rmses, marker='o', color='tab:blue', label='train data')
    plt.scatter(test_df['tp'], test_rmses, marker='o', color='tab:pink', label='test data')
    
    # statistics lines
    plt.axhline(y=mean_rmse, color='tab:orange', linestyle='--', label='mean')
    plt.axhline(y=train_mean, color='tab:blue', linestyle='--', label='train mean')
    plt.axhline(y=test_mean, color='tab:pink', linestyle='--', label='test mean')

    # phase borders
    plt.axvline(x=0.2, color='k', alpha=0.4, linestyle=':', label='phase boundary\n(AFM-SC-FM)')
    plt.axvline(x=0.33, color='k', alpha=0.4, linestyle=':')

    # layout
    plt.grid(axis='y')
    plt.xlim(-0.02, 0.52)
    plt.xticks(fontsize=font_size-2)
    if ymin:
        plt.ylim(bottom=ymin)
    if ymax:
        plt.ylim(top=ymax)
    plt.yticks(fontsize=font_size-2)
    # plt.title(f'reconstruction RMSE for {run_name}', fontsize=font_size)
    plt.xlabel('tp', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)

    # phase labels
    ymin, ymax = plt.ylim()
    y_pos = (ymax - ymin) * 0.5 + ymin
    text_kwargs = {'alpha': 0.2, 'fontsize': font_size + 32, 'ha': 'center', 'va': 'center', 'rotation': 90}
    plt.text(0.2 / 2, y_pos, 'AFM', **text_kwargs)
    plt.text((0.33 - 0.2) / 2 + 0.2, y_pos, 'SC', **text_kwargs)
    plt.text((0.5 - 0.33) / 2 + 0.33, y_pos, 'FM', **text_kwargs)

    if legend:
        plt.legend(fontsize=font_size-2, bbox_to_anchor=(1, 1))
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


def _create_rmse_boxplot(rmses: pd.DataFrame, xlabel: str, category_key: str, alpha: float = 0.4, width: float = 0.5, 
                         figsize: tuple[int, int] = (4, 3), plot_dir: Path = Path(), plot_name: str|None = None):
    xtick_labels = sorted(rmses[category_key].unique())
    if category_key == 'run_id':
        xtick_labels = [RUN_ID_MAPPING.get(label, label) for label in xtick_labels]
    fig, ax = plt.subplots(figsize=figsize)
    positions = []
    data_to_plot = []
    pos = 0
    gap = width * 1.5
    for (run_id, ld, s, seed), group in rmses.groupby(['run_id', 'ld', 's', 'seed']):
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
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_rmse_boxplots(rmses: pd.DataFrame, train_data: bool = True, figsize: tuple[int, int] = (4, 3), 
                       alpha: float = 0.4, width: float = 0.5, plot_dir: Path = Path(), plot_name: str = ''):
    plot_dir = Path(plot_dir) if plot_dir else plot_dir
    rmses = rmses.sort_values(by=['run_id', 'ld', 's', 'seed'])
    if not train_data:
        rmses = rmses[rmses['run_id'].str.startswith('1_') | ~rmses['train_data']]

    subset = rmses[(rmses['run_id'] == '2_3_1') & (rmses['ld'] == 32) & (rmses['s'] == 24000)]
    _create_rmse_boxplot(subset, 'scenario', 'seed', alpha, width, figsize, 
                         plot_dir, plot_name + '_subsets')

    subset = rmses[(rmses['ld'] == 32) & (rmses['s'] == 24000) & (rmses['seed'] == 123)]
    plot_data = subset[~subset['contrastive']]
    _create_rmse_boxplot(plot_data, 'scenario', 'run_id', alpha, width, figsize, 
                         plot_dir, plot_name + '_mse')
    
    plot_data = subset[(~subset['run_id'].str.startswith('3_')) & (~subset['contrastive'])]
    _create_rmse_boxplot(plot_data, 'scenario', 'run_id', alpha, width, figsize, 
                         plot_dir, plot_name + '_mse_c')
    
    plot_data = subset[subset['contrastive']]
    _create_rmse_boxplot(plot_data, 'scenario', 'run_id', alpha, width, figsize, 
                         plot_dir, plot_name + '_contr')

    subset = rmses[rmses['seed'] == 123]
    plot_data = subset[(subset['run_id'] == '2_1_1') & (subset['s'] == 24000)]
    _create_rmse_boxplot(plot_data, 'latent space dimension', 'ld', alpha, width, figsize, 
                         plot_dir, plot_name + '_ld')
    
    plot_data = subset[(subset['run_id'] == '2_3_1') & (subset['ld'] == 32)]
    _create_rmse_boxplot(plot_data, 'subsamples per vertex', 's', alpha, width, figsize, 
                         plot_dir, plot_name + '_s')

    plot_data = subset[(subset['run_id'] == '2_1_1') & (subset['ld'] == 32)]
    _create_rmse_boxplot(plot_data, 'subsamples per vertex', 's', alpha, width, figsize, 
                         plot_dir, plot_name + '_s2')
    
    plot_data = rmses[(rmses['run_id'] == '2_1_1_step') 
                      | ((rmses['run_id'] == '2_1_1') & (rmses['ld'] == 32) & (rmses['s'] == 24000))]
    _create_rmse_boxplot(plot_data, 'scenarios', 'run_id', alpha, width, figsize, 
                         plot_dir, plot_name + '_step')



# ----------------------------------------------------------------------------------------------
# ANALYZE RECONSTRUCTION ERROR
# ----------------------------------------------------------------------------------------------
def load_vertex(save_path: Path, tp: float) -> np.ndarray:
    pref = f'tp{tp:.2f}'
    fn = next(save_path.glob(f'{pref}*.h5'))
    return AutoEncoderVertex24x6Dataset.load_from_file(fn)


def load_prediction(save_path: Path, tp: float) -> np.ndarray:
    pref = f'tp{tp:.2f}'
    fn = next((save_path / 'predictions').glob(f'{pref}*.npy'))
    return np.load(fn)


def load_vertices(plot_data: list[tuple[str, list[float]]], data_dir: Path, 
                  base_dir: Path) -> list[tuple[np.ndarray, np.ndarray, str, float]]:
    vertex_data = []
    for run_id, tps in plot_data:
        save_path = base_dir / run_id
        for tp in tps:
            vertex = load_vertex(data_dir, tp).flatten()
            pred = load_prediction(save_path, tp).flatten()
            sort_idcs = np.argsort(vertex)
            vertex = vertex[sort_idcs]
            pred = pred[sort_idcs]
            vertex_data.append((vertex, pred, run_id, tp))
    return vertex_data


def plot_vertex_error(vertex: np.ndarray, pred: np.ndarray, bars: bool, nbins: int = 12, width_factor: float = 0.45, 
                      figsize: tuple[int, int] = (4, 3), legend: bool = False, plot_dir: Path = Path(), 
                      plot_name: str|None = None) -> pd.DataFrame:
    bin_edges = np.linspace(vertex.min(), vertex.max(), nbins + 1)
    v_means = []
    p_means = []
    for i in range(nbins):
        if i == 0:
            idcs = vertex < bin_edges[i+1]
        elif i == nbins - 1:
            idcs = vertex >= bin_edges[i]
        else:
            idcs = (vertex >= bin_edges[i]) & (vertex < bin_edges[i+1])
        v_means.append(vertex[idcs].mean())
        p_means.append(pred[idcs].mean())
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=figsize)
    if bars:
        width = (bin_edges[1] - bin_edges[0]) * width_factor
        plt.bar(bin_centers - width / 2, v_means, width=width, label='true vertex', color=vis.COLORS[0], align='center')
        plt.bar(bin_centers + width / 2, p_means, width=width, label='reconstruction', color=vis.COLORS[1], align='center')
    else:
        plt.plot(bin_centers, v_means, label='true vertex', color=vis.COLORS[0])
        plt.plot(bin_centers, p_means, label='reconstruction', color=vis.COLORS[1])
    if legend:
        plt.legend()
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.show()



# ----------------------------------------------------------------------------------------------
# EVALUATE PHASE CLASSIFIER
# ----------------------------------------------------------------------------------------------
def _plot_classification(classifications: pd.DataFrame, xlabel: str, category_key: str, y_min: float, alpha: float = 0.4,
                         figsize: tuple[int, int] = (4, 3), plot_dir: Path = Path(), plot_name: str|None = None):
    col = vis.COLORS[0]
    categories = classifications[category_key].astype(str).to_list()
    if category_key == 'run_id':
        categories = [RUN_ID_MAPPING.get(x, x) for x in categories]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, classifications['f1'], color=col, alpha=alpha)

    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_ylabel('f1 score')
    ax.set_ylim(y_min, 1.)

    y_pos = 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
    for bar, label in zip(bars, categories):
        x = bar.get_x() + bar.get_width() / 2.0
        ax.text(x, y_pos, label, rotation=90, rotation_mode='anchor', va='center', ha='left', clip_on=True)
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_classification_results(classifications: pd.DataFrame, figsize: tuple[int, int] = (4, 3), alpha: float = 0.4, y_factor: int = 200,
                                plot_dir: Path = Path(), plot_name: str = ''):
    labels_mapping = {
        'NeuralNetClassifier(batch_size=4096, num_workers=2, max_epochs=1000)': 'Neural Network',
        'PolynomialRegression(method=spline, degree=3)': 'Spline-polynomial\nregressor (knots=10)',
        'PolynomialRegression(method=spline, degree=3, n_knots=4)': 'Spline-polynomial\nregressor (knots=4)',
        'PolynomialRegression(method=spline, degree=3, n_knots=100)': 'Spline-polynomial\nregressor (knots=100)',
        'PolynomialRegression(method=spline, degree=3, n_knots=1000)': 'Spline-polynomial\nregressor (knots=1000)',
        'RandomForestClassifier(n_jobs=-1, random_state=108)': 'Random Forest\nclassifier',
        'RandomForestClassifier(n_jobs=-1, random_state=125)': 'Random Forest\nclassifier',
        'RandomForestClassifier(n_jobs=-1, random_state=142)': 'Random Forest\nclassifier',
        'RandomForestClassifier(n_jobs=-1, random_state=159)': 'Random Forest\nclassifier',
        'RandomForestClassifier(n_jobs=-1, random_state=91)': 'Random Forest\nclassifier',
        'RandomForestRegressor(n_jobs=-1, random_state=125)': 'Random Forest\nregressor',
        'StandardSVC()': 'SVM',
        'GaussianProcessClassifier': 'Gaussian Process',
        'PolynomialRegression': 'Polynomial\nregressor',
        }

    plot_dir = Path(plot_dir) if plot_dir else plot_dir
    classifications = classifications.sort_values(by=['run_id', 'ld', 's', 'seed']).drop(columns='conf_mat')
    y_min = (int(classifications[classifications['model'].str.startswith('RandomForestClassifier')]['f1'].min() * y_factor) - 1) / y_factor

    subset = classifications[classifications['model'].str.startswith('RandomForestClassifier') 
                             & (classifications['ld'] == 32) & (classifications['s'] == 24000) & (classifications['seed'] == 123)]
    plot_data = subset[subset['run_id'].str.match(r'(1_1)|(2_._1$)|(3_.*)|(no_*)')]
    _plot_classification(plot_data, 'scenario', 'run_id', y_min, alpha, figsize, plot_dir, plot_name + '_mse')

    plot_data = subset[subset['run_id'].str.match(r'(1_1)|(2_._1$)|(no_*)')]
    _plot_classification(plot_data, 'scenario', 'run_id', y_min, alpha, figsize, plot_dir, plot_name + '_mse_c')

    plot_data = subset[subset['run_id'].str.match(r'(1_2)|(2_._2$)|(no_*)')]
    _plot_classification(plot_data, 'scenario', 'run_id', y_min, alpha, figsize, plot_dir, plot_name + '_contr')

    subset = classifications[(classifications['run_id'] == '2_1_1') & (classifications['s'] == 24000)]
    plot_data = subset[(subset['ld'] == 32)].copy()
    plot_data['model'] = plot_data['model'].apply(lambda x: labels_mapping.get(x, x.split('(')[0]))
    y_min_models = (int(plot_data['f1'].min() * y_factor) - 1) / y_factor
    _plot_classification(plot_data, 'model', 'model', y_min_models, alpha, figsize, plot_dir, plot_name + '_models')

    plot_data = subset[subset['model'].str.startswith('RandomForestClassifier')]
    _plot_classification(plot_data, 'latent space dimension', 'ld', y_min, alpha, figsize, plot_dir, plot_name + '_ld')

    subset = classifications[(classifications['run_id'] == '2_3_1') & (classifications['ld'] == 32)]
    plot_data = subset[(subset['s'] == 24000)]
    _plot_classification(plot_data, 'subset random seed', 'seed', y_min, alpha, figsize, plot_dir, plot_name + '_subsets')

    plot_data = subset[(subset['seed'] == 123)]
    _plot_classification(plot_data, 'subsamples per vertex', 's', y_min, alpha, figsize, plot_dir, plot_name + '_s')



# ----------------------------------------------------------------------------------------------
# CLASSIFIER ANALYSIS
# ----------------------------------------------------------------------------------------------
def analyse_regression(regression_df: pd.DataFrame, figsize: tuple[int, int] = (4, 3), alpha: float = 0.2, 
                       plot_dir: Path = Path(), plot_name: str = ''):
    df = regression_df[(regression_df['run_id'] == '2_1_1') & (regression_df['ld'] == 32) 
                       & (regression_df['s'] == 24000) & (regression_df['seed'] == 123)]
    df = (df.groupby(['model', 'run_id', 'ld', 's', 'seed', 'tp']).agg(['min', 'max', 'mean'])
            .sort_values(['model', 'tp']).reset_index())

    for i, model in enumerate(df['model'].unique()):
        print(str(model))
        df_model = df[df['model'] == model]
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df_model['tp'], df_model[('pred', 'mean')], color=vis.COLORS[i])
        ax.fill_between(df_model['tp'], df_model[('pred', 'min')], df_model[('pred', 'max')], color=vis.COLORS[i], alpha=alpha)
        
        tmin, tmax = df_model['tp'].min(), df_model['tp'].max()
        ax.plot([tmin, tmax], [tmin, tmax], 'r--', linewidth=1)
        
        ax.set_xlabel("true $t'$")
        ax.set_ylabel("predicted $t'$")
        plt.tight_layout()
        if plot_dir and plot_name:
            plt.savefig(plot_dir / f'{plot_name}_{i}.png', bbox_inches='tight', dpi=300)
        plt.show()


def analyse_regression_box(df: pd.DataFrame, figsize: tuple[int, int] = (8, 3), alpha: float = 0.4, 
                           plot_dir: Path = Path(), plot_name: str = ''):
    pred_data = df.groupby(['tp'])['pred']
    tps = sorted(df['tp'].unique())
    plot_data = pred_data.apply(list)
    plot_data = [plot_data[tp] for tp in tps]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(plot_data, positions=tps, widths=0.008, patch_artist=True, boxprops={'facecolor': vis.COLORS[2], 'alpha': alpha},
               flierprops={'marker': '.', 'mec': 'grey', 'mfc': 'grey', 'alpha': 0.2})
    ax.fill_between(tps, pred_data.min(), pred_data.max(), color=vis.COLORS[2], alpha=alpha / 3)
    ax.set_xlim((-0.01, 0.51))

    tps = np.arange(min(tps), max(tps) + 0.01, 0.1)
    ax.set_xticks(tps)
    ax.set_xticklabels([f'{tp:.2f}' for tp in tps])

    tmin, tmax = df['tp'].min(), df['tp'].max()
    ax.plot([tmin, tmax], [tmin, tmax], 'r--', linewidth=0.75, label='perfect prediction')

    # phase borders
    pb_kwargs = {'color': 'k', 'alpha': 0.75, 'linestyle': ':'}
    ax.axvline(x=0.2, **pb_kwargs, label='phase boundaries\n(AFM-SC-FM)')
    ax.axvline(x=0.33, **pb_kwargs)
    ax.axhline(y=0.2, **pb_kwargs)
    ax.axhline(y=0.33, **pb_kwargs)

    ax.set_xlabel("true $t'$")
    ax.set_ylabel("predicted $t'$")
    plt.legend()
    if plot_dir and plot_name:
        plt.savefig(plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class FeatureImportanceAnalysis:
    def __init__(self, data_dir: str, plot_dir: Path = Path()):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        self.vertices: dict[str, np.ndarray] = None
        self.file_paths: list[str] = None
        self.dataset: AutoEncoder24x6InfoNCEDataset = None
    
    def load_vertices(self):
        self.file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(self.data_dir, subset=None, subset_shuffle=False)[0]
        self.vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(self.file_paths)

    def create_dataset(self, sample_count_per_vertex: int):
        dataset_kwargs = {'path_train': self.data_dir, 'subset_shuffle': True}
        self.dataset = make_dataset(self.vertices, sample_count_per_vertex, dataset_kwargs, file_paths=self.file_paths, 
                                    dataset_class=AutoEncoder24x6InfoNCEDataset)
    
    def load_test_data(self, encoder_path: str|None) -> tuple[np.ndarray, np.ndarray]:
        pc = PhaseClassification(encoder_path, None, None, device='cpu')
        X_test, y_test, _ = pc.load_data(self.dataset)
        return X_test.astype(np.float32), np.array(y_test).astype(np.float32)
    
    @staticmethod
    def load_rf(rf_path: str) -> RandomForestClassifier:
        with open(rf_path, 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def get_feat_importances(rf_model: RandomForestClassifier) -> tuple[pd.Series, np.ndarray]:
        importances = rf_model.feature_importances_
        importances = np.array([tree.feature_importances_ for tree in rf_model.estimators_])
        return importances

    @staticmethod
    def compute_perm_importances(rf_model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, 
                                 n_repeats: int = 1) -> np.ndarray:
        pi_res = permutation_importance(rf_model, X_test, y_test, n_repeats=n_repeats, random_state=42)
        return pi_res.importances.flatten()

    def plot_feat_importances(self, importances: np.ndarray, box: bool = False, figsize: tuple[int, int] = (6, 3), 
                              ymax: float = 0.06, alpha: float = 0.4, plot_name: str = '') -> None:
        indices = np.arange(1, importances.shape[1] + 1)
        imp_mean = importances.mean(axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(indices, imp_mean)
        if box:
            ax.boxplot(importances, patch_artist=True, boxprops={'facecolor': vis.COLORS[0], 'alpha': alpha},
                       flierprops={'marker': '.', 'mec': 'grey', 'mfc': 'grey', 'alpha': 0.2})
            ax.set_ylim(0, ymax)
        else:
            imp_std = np.std(importances, axis=0)
            ax.fill_between(indices, imp_mean - imp_std, imp_mean + imp_std, alpha=0.2)
        ax.set_title("Feature importances")
        ax.set_ylabel("Mean decrease in impurity")
        plt.tight_layout()
        if self.plot_dir and plot_name:
            plt.savefig(self.plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_perm_importances(self, importances: pd.Series, X_test: np.ndarray, scatter: bool = False, 
                              figsize: tuple[int, int] = (6, 3), plot_name: str = '') -> None:
        fig, ax = plt.subplots(figsize=figsize)
        if scatter:
            ax.scatter(np.arange(X_test.shape[1]), importances)
        else:
            ax.plot(np.arange(X_test.shape[1]), importances)
        ax.axhline(y=0, color="r", linestyle="--", lw=1)
        ax.set_ylabel("Decrease in accuracy score")
        ax.set_title("Permutation Importances")
        fig.tight_layout()
        if self.plot_dir and plot_name:
            plt.savefig(self.plot_dir / f'{plot_name}.png', bbox_inches='tight', dpi=300)
        plt.show()



# ----------------------------------------------------------------------------------------------
# FULL EVALUATION
# ----------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from phys_ml.analysis.vertex import PhaseClassification, NeuralNetClassifier, PolynomialRegression, StandardSVC


def predict_for_all_models(run_id: str, ld: int = 32, s: int = 24000, run_dir_name: str = 'run_results', 
                           seed: int = 123, project_name: str = 'vertex_24x6', device: Literal['cpu', 'gpu'] = 'gpu'):
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
        'sample_seed': seed,
        'subset_seed': seed,
    }
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': False, 
    }
    pred_configs = {
        '1_1': {},
        '1_2': {},
        '2_1_1': {'subset_type': ['afm', 'fm']},
        '2_1_1_step': {'subset_type': ['afm', 'fm']},
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
    if s != 24000:
        pref = f's{s}'
    elif seed != 123:
        pref = f'r{seed}'
    else:
        pref = f'ld{ld}'
    save_path = f'/gpfs/data/fs71925/shepp123/PhysML/saves/{project_name}/{run_dir_name}/{run_id}_{pref}'
    if os.path.exists(save_path):
        config_kwargs['hidden_dims'] = hidden_dims[ld]
        preds = predict_all(file_paths, vertices, save_path, config_kwargs, dataset_kwargs, 
                            encode_only=False, train_mode=TrainerModes.SLURM, device_type=device, **pred_config)


def evaluate_all(run_id: str, ld: int = 32, s: int = 24000, run_dir_name: str|None = 'run_results', 
                 seed: int = 123, models: Literal['rf', 'svc', 'gp', 'nn', 'poly', 'spline', 'spline_k10', 'rfr'] = ['rf'],
                 subset: int|float|None = None, train_samples_per_vertex: int = 24000, test_samples_per_vertex: int = 4800,
                 device: Literal['cpu', 'cuda']|None = None):
    # load vertices
    path_train = '/gpfs/data/fs71925/shepp123/frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=subset, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)

    # autoencoder
    nce_train_samples = train_samples_per_vertex // 4
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': True, 
    }

    # phase classifier
    pc_models = []
    for m in models:
        match m:
            case None:
                pc_models.append(None)
            case 'rf':
                pc_models.append(RandomForestClassifier(n_jobs=-1, random_state=seed + 2))
            case 'gp':
                pc_models.append(GaussianProcessClassifier(n_jobs=-1, random_state=seed + 2))
            case 'svc':
                pc_models.append(StandardSVC())
            case 'nn':
                pc_models.append(NeuralNetClassifier(n_layers=3, out_dim=3, batch_size=4096, num_workers=2, max_epochs=1000))
            case 'poly':
                pc_models.append(PolynomialRegression.new(method='poly', degree=3))
            case x if x.startswith('spline'):
                s = x.split('_k')
                k = int(s[1]) if len(s) > 1 else 4
                pc_models.append(PolynomialRegression.new(method='spline', degree=3, n_knots=k))
            case 'rfr':
                pc_models.append(RandomForestRegressor(n_jobs=-1, random_state=seed + 2))
            case _:
                raise ValueError(f'Unknown model type: {m}')

    # general train sets
    nce_train_dataset = make_dataset(vertices, nce_train_samples, dataset_kwargs, file_paths=file_paths, 
                                     dataset_class=AutoEncoder24x6InfoNCEDataset)

    # reconstruction filepaths
    match run_id:
        case '1_1' | '1_2' | 'no_encoding':
            train_files = file_paths
        case '2_1_1' | '2_1_2' | '2_1_1_step':
            train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                                     subset_seed=seed, subset_type=['afm', 'fm'], file_paths=file_paths)[0]
        case '2_2_1' | '2_2_2':
            train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                                     subset_seed=seed, subset_type=['sc', 'fm'], file_paths=file_paths)[0]
        case '2_3_1' | '2_3_2':
            train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                                     subset_seed=seed, subset_type=['afm', 'sc'], file_paths=file_paths)[0]
        case '3_1':
            train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                                     subset_seed=seed, subset_type='sc', file_paths=file_paths)[0]
        case '3_2':
            train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                                     subset_seed=seed, subset_type='afm', file_paths=file_paths)[0]
        case '3_3':
            train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                                     subset_seed=seed, subset_type='fm', file_paths=file_paths)[0]
        case _:
            raise ValueError(f'Unknown run ID: {run_id}')

    # test sets
    test_dataset_full = make_dataset(vertices, test_samples_per_vertex, dataset_kwargs, file_paths=file_paths, 
                                     dataset_class=AutoEncoder24x6InfoNCEDataset)

    # run info
    base_path = Path('/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6')
    save_path = '/gpfs/data/fs71925/shepp123/PhysML/notebooks/vertex/eval_results/'

    # evaluate models
    def eval(run_id: str, run_name: str, ld: int, s: int, seed: int, train_files: list[str], device: Literal['cpu', 'cuda']|None = None):
        model_path = base_path / run_dir_name / run_name if run_dir_name else None
        if model_path and os.path.exists(model_path):
            # reconstruction
            rmse_df = mean_rmse(vertices, model_path, train_files)
            rmse_df[['run_id', 'ld', 's', 'seed']] = [run_id, ld, s, seed]
            rmse_df.to_csv(save_path + f'reconstruction_results_{run_name}.csv', index=False)

        # classification
        if pc_models:
            pc = PhaseClassification(model_path, pc_models, run_name, device=device)
            classification_df, regression_df = pc.evaluate_classifiers(test_dataset_full, nce_train_dataset, print_conf_mat=False)
            classification_df.to_pickle(save_path + f'classification_results_{run_name}.pkl')
            regression_df.to_csv(save_path + f'regression_results_{run_name}.csv', index=False)

    if s != 24000:
        pref = f's{s}'
    elif seed != 123:
        pref = f'r{seed}'
    else:
        pref = f'ld{ld}'
    run_name = f'{run_id}_{pref}'
    eval(run_id, run_name, ld, s, seed, train_files, device)
