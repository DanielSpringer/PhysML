import glob
import os
import pickle

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .. import metrics
from ..visualization import vertex_visualization as vertvis
from ..trainer import TrainerModes
from ..trainer.vertex import VertexTrainer


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
        pred = trainer.dataset.to_3d_vertex(pred)
    dim = len(pred.shape)
    if dim in [3, 6]:
        pred_slice = vertvis.get_mat_slice(pred, axis, slice_at)
        rmse = metrics.rmse(target, pred)
    else:
        if dim == 4:
            pred_slice = pred.reshape((vertvis.AutoEncoderVertexDataset.length,) * 2, order='F')
        else:
            pred_slice = pred
        rmse = metrics.rmse(target_slice, pred_slice)
    eigvec = metrics.vertex.get_dominant_eigenvector(pred_slice)
    return rmse, eigvec, pred_slice


def evaluate_all_models(train_results: list[dict[str, Any]], test_filename: str, trainer: VertexTrainer, 
                        target: np.ndarray, slice_at: int|tuple[int,...]|None, axis: int, 
                        predict_func: Callable[...,np.ndarray]|None = None, 
                        load_func: Callable[...,np.ndarray]|None = None, 
                        **kwargs) -> tuple[dict[int, tuple[float, np.ndarray, np.ndarray]], np.ndarray]:
    target_slice = vertvis.get_mat_slice(target, axis, slice_at)
    results = {mod_info['latent_dim']: 
               evaluate_prediction(mod_info['save_path'], test_filename, trainer, target, mod_info['hidden_dims'], 
                                   target_slice, predict_func, load_func, slice_at, axis, **kwargs)
               for mod_info in train_results}
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
            other_ks = set(range(1,4)) - {axis}
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
                        target: np.ndarray, slice_at: int|tuple[int,...]|None, axis: int, 
                        nrows:int, ncols: int, predict_func: Callable[...,np.ndarray]|None = None, 
                        load_func: Callable[...,np.ndarray]|None = None, 
                        **kwargs):
    assert nrows * ncols >= len(train_results) + 1, \
        f"`{nrows=}`and `{ncols=}` not enough for {len(train_results + 1)} items to plot in `train_info` + target."
    
    results, target_slice = evaluate_all_models(train_results, test_filename, trainer, target, slice_at, axis, 
                                                predict_func, load_func, **kwargs)
    report_results(results, target_slice, slice_at, axis, nrows, ncols)


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
    if len(files) > 0:
        last_i = files[-1].split('.')[0][-1] if len(files) > 1 else 0
        os.rename(f'{info_name}.pkl', f'{info_name}{last_i + 1}.pkl')


def eval_train(trainer: VertexTrainer, info_dict: list[dict[str, Any]], info_filename: str, hidden_dims: list, 
               resume: bool = False, path: str|None = None, version: int|None = None):
    if resume:
        assert path or version, 'If resuming, either `path` or `version` must be provided.'
        if not path:
            if version:
                path = (sorted(trainer.get_full_save_path().glob('*'))[-1] / f'version_{version}').as_posix()
            else:
                path = sorted(trainer.get_full_save_path().glob('*/*'))[-1].as_posix()
        trainer.config.resume = True
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
