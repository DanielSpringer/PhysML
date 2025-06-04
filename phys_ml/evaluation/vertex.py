import asyncio
import nest_asyncio
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from tqdm.notebook import tqdm

from .. import metrics
from ..config import Vertex24x6Config
from ..load_data.vertex import AutoEncoderVertexDataset, AutoEncoderVertex24x6Dataset
from ..trainer import TrainerModes
from ..trainer.vertex import VertexTrainer, VertexTrainer24x6
from ..visualization import vertex_visualization as vertvis



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
        cor_mat[i, j] = cor_mat[j, i] = (z1 / np.sqrt(z1 * z1)) * (z2 / np.sqrt(z2 * z2))
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
    fname = '_'.join(['cor_mat_vertex24x6', save_suffix])
    np.save(f'{fname}.npy', cor_mat)
    return cor_mat



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
    if str(type(train_dataset)) == 'phys_ml.load_data.vertex.AutoEncoderVertex24x6Dataset':
        file_paths = train_dataset.file_paths
    elif str(type(train_dataset)) == 'phys_ml.load_data.vertex.AutoEncoder24x6InfoNCEDataset':
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
                 device_type: Literal['cpu', 'gpu'] = 'gpu', load_from: str|None = None, **kwargs) -> VertexTrainer24x6:
    kws = config_kwargs.copy()
    kws.update(dataset_kwargs)
    kws['device_type'] = device_type
    for k, v in kwargs.items():
        kws[k] = v
    kws['subset_type'] = dataset.config.subset_type
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


def predict_all(dataset: AutoEncoderVertex24x6Dataset, file_paths:tuple[list[str], int], vertices: dict[str, np.ndarray], 
                path_train: str, save_path: str, config_kwargs: dict[str, Any], dataset_kwargs: dict[str, Any], 
                encode_only: bool, all_vertices: bool = False) -> list[np.ndarray]:
    trainer = init_trainer(config_kwargs, dataset, dataset_kwargs, load_from=save_path)
    ckpt_path = trainer.init_trainer(train_mode=TrainerModes.JUPYTER, load_from=save_path)
    preds = []
    fps = file_paths if all_vertices else get_test_filepaths(path_train, dataset)
    for fp in tqdm(fps):
        vertex = vertices[fp]
        pred = trainer.predict(fp, vertex, encode_only=encode_only)
        preds.append(pred)
    return preds


def mean_rmse(file_paths:tuple[list[str], int], vertices: dict[str, np.ndarray], path_train: str, save_path: str, 
              dataset: AutoEncoderVertex24x6Dataset|None = None, plot: bool = False) -> dict[float, float]:
    true_fps = file_paths if dataset is None else get_test_filepaths(path_train, dataset)
    true_dir = Path(file_paths[0]).parent.as_posix()
    pred_dir = f'{save_path}/predictions'
    filenames = [Path(fp).stem for fp in sorted(true_fps)]

    if plot:
        i = 18
        axis = 3
        nrows, ncols = 1, 2
        slice_at = (i, i, i, i)
        other_ks = set(range(1,4)) - {(axis + 1) // 2}
        params_str = ', '.join([f'$k_{{{k}_{c}}}={sl}$' for (k, c), sl 
                                in zip([(k, c) for k in other_ks for c in ['x', 'y']], slice_at)])

    rmses: dict[float, float] = {}
    for fn in tqdm(filenames):
        tp = float(fn[2:6])
        true = vertices[f'{true_dir}/{fn}.h5']
        pred = np.load(f'{pred_dir}/{fn}.npy')

        if plot:
            true_slice = vertvis.get_mat_slice(true, axis, slice_at)
            pred_slice = vertvis.get_mat_slice(pred, axis, slice_at)
            plot_data = {'true': true_slice, 'reconstruction': pred_slice}
            vertvis.plot_compare_grid(plot_data, nrows, ncols, axis, None, figsize=(6, 4), 
                                      title=f'Reconstruction of vertex({fn}) at ({params_str})', vmin=0.0, vmax=22.5)
        
        rmses[tp] = metrics.rmse(true, pred)
    return rmses


def print_rmses(rmses: dict[float, float]):
    errors = list(rmses.values())
    mean_rmse = np.mean(errors)
    print(f'mean: {mean_rmse}, min: {min(errors)}, max: {max(errors)}')
    plt.figure(figsize=(8, 4))
    plt.plot(list(rmses.keys()), errors, marker='o')
    plt.axhline(y=mean_rmse, color='r', linestyle='--', label='mean')
    plt.title('RMSE of vertex reconstruction')
    plt.xlabel('tp')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()
