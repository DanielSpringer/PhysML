from collections.abc import Callable
from typing import Any

import numpy as np

from .. import metrics
from ..visualization import vertex_visualization as vertvis


def evaluate_prediction(save_path: str, test_filename: str, target: np.ndarray, target_slice: np.ndarray,
                        predict_func: Callable[...,np.ndarray], **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
    pred_slice = predict_func(test_filename, new_vertex=target, save_path=save_path, **kwargs)
    if len(pred_slice.shape) == 4:
        pred_slice = pred_slice.reshape((vertvis.AutoEncoderVertexDataset.length,) * 2, order='F')
    rmse = metrics.rmse(target_slice, pred_slice)
    eigvec = metrics.vertex.get_dominant_eigenvector(pred_slice)
    return rmse, eigvec, pred_slice


def evaluate_all_models(train_results: list[dict[str, Any]], test_filename: str, target: np.ndarray, 
                        slice_at: int|tuple[int,...]|None, axis: int, predict_func: Callable[...,np.ndarray], 
                        **kwargs) -> tuple[dict[int, tuple[float, np.ndarray, np.ndarray]], np.ndarray]:
    target_slice = vertvis.get_mat_slice(target, axis, slice_at)
    results = {mod_info['latent_dim']: evaluate_prediction(mod_info['save_path'], test_filename, target, 
                                                           target_slice, predict_func, **kwargs)
               for mod_info in train_results}
    return results, target_slice


def report_results(results: dict[int, tuple[float, np.ndarray, np.ndarray]], target_slice: np.ndarray, 
                   slice_at: int|tuple[int,...]|None, axis: int, nrows:int, ncols: int):
    assert nrows * ncols >= len(results) + 1, \
        f"`{nrows=}`and `{ncols=}` not enough for {len(results + 1)} items to plot in `train_info` + target."
    
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


def evaluate_and_report(train_results: dict[str, Any], test_filename: str, target: np.ndarray, 
                        slice_at: int|tuple[int,...]|None, axis: int, nrows:int, ncols: int,
                        predict_func: Callable[...,np.ndarray], **kwargs):
    assert nrows * ncols >= len(train_results) + 1, \
        f"`{nrows=}`and `{ncols=}` not enough for {len(train_results + 1)} items to plot in `train_info` + target."
    
    results, target_slice = evaluate_all_models(train_results, test_filename, target, slice_at, axis, 
                                                predict_func, **kwargs)
    print(f"\nRESULTS:\n   {'\n   '.join([f'latent_dim={k}: RMSE={v[0]:.4f}' for k, v in results.items()])}")
    report_results(results, target_slice, slice_at, axis, nrows, ncols)
