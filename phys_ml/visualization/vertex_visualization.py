from collections.abc import Callable
from typing import Iterable

import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt

import numpy as np

from ..load_data.vertex import AutoEncoderVertexDataset


cmap_big = mpl.colormaps['twilight_shifted'].resampled(int(1e3))
#cmap_resc = mplcolors.ListedColormap(cmap_big(np.linspace(0.075, 0.925, 10000)))
cmap_resc = mpl.colormaps['viridis'].resampled(int(1e3))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _plot(data: np.ndarray, ax: axes.Axes, x_label: str, y_label: str, colmap: str|mplcolors.Colormap, 
          vmin: float|None = None, vmax: float|None = None, title: str|None = None):
    cmap = colmap if colmap else cmap_resc
    img = ax.imshow(data, cmap=cmap)
    if vmin is not None and vmax is not None:
        img.set_clim(vmin=vmin, vmax=vmax)
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title)
    return img


def get_mat_slice(mat: np.ndarray, axis: int|tuple[int,int], slice_at: int|tuple[int,...]) -> np.ndarray:
    k_dim, n_freq = AutoEncoderVertexDataset.k_dim, AutoEncoderVertexDataset.n_freq
    length = AutoEncoderVertexDataset.length
    if len(mat.shape) == 4:
        return mat.reshape((length,) * 2, order='F')
    if isinstance(slice_at, int):
        data_slice = [slice(None)] * k_dim
        data_slice[axis - 1] = slice_at
    elif len(slice_at) == 2 and isinstance(axis, tuple):
        if len(mat.shape) != k_dim:
            mat = mat.reshape((length,) * k_dim, order='F')
        k, other_k = axis
        kix, kiy = slice_at
        slice_axis = sum(range(1, k_dim + 1)) - k - other_k - 1
        data_slice = [slice(None)] * k_dim
        data_slice[slice_axis] = kix * n_freq + kiy
    elif len(slice_at) == 4:
        data_slice = list(slice_at)
        k = (axis + 1) // 2
        for i in range(2):
            data_slice.insert(i + 2 * (k - 1), slice(None))
    else:
        raise ValueError("Allowed parameter combinations: `axis` and `slice_at` of type `int`, "
                         "`axis` of type `tuple[int, int]` and `slice_at` of type `tuple[int, int]`, "
                         "`axis` of type `int` and `slice_at` of type `tuple[int, int, int, int]`.")
    return mat[*data_slice]


def _create_plot(ax: axes.Axes, data: np.ndarray, axis: int, colmap: str|mplcolors.Colormap|None = None, 
                 vmin: float|None = None, vmax: float|None = None, title: str|None = None):
    k_dim, n_freq = AutoEncoderVertexDataset.k_dim, AutoEncoderVertexDataset.n_freq
    space_dim, length = AutoEncoderVertexDataset.space_dim, AutoEncoderVertexDataset.length
    dim = len(data.shape)
    if dim == 3 or data.shape == (length, length):
        if isinstance(axis, int):
            assert axis >= 1 and axis <= 3, f"Axis must be in range [1,{k_dim}]"
            axs_range = np.arange(1, k_dim + 1)
            print_axs = axs_range[axs_range != axis]
        elif isinstance(axis, tuple):
            assert len(axis) == 2, "Axis must be a tuple of length 2"
            assert all([ax >= 1 and ax <= 3 for ax in axis]), f"Axis must be in range [1,{k_dim}]"
            print_axs = axis
        x_label, y_label = (f'$k_{pa}$' for pa in print_axs)
    elif dim in [4, 6] or data.shape == (n_freq, n_freq):
        assert axis >= 1 and axis <= 6, f"Axis must be in range [1,{k_dim * space_dim}]"
        k = (axis + 1) // 2
        x_label, y_label = f'$k_{{{k}_x}}$', f'$k_{{{k}_y}}$'
    img = _plot(data, ax, x_label, y_label, colmap, vmin, vmax, title)
    return img


def _prepare_data(data: list[np.ndarray], axis: int|tuple[int,int], 
                  slice_at: int|tuple[int,...]|None) -> tuple[list[np.ndarray], float, float]:
    for i, d in enumerate(data):
        if len(d.shape) != 2 and slice_at is not None:
            data[i] = get_mat_slice(d, axis, slice_at)
    return data, min([d.min() for d in data]), max([d.max() for d in data])


def plot_section(data: np.ndarray, axis: int|tuple[int,int], slice_at: int|tuple[int,...],
                 figsize: tuple[int, int] = (8,6), colmap: str|mplcolors.Colormap|None = None,
                 title: str|None = None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    data, vmin, vmax = _prepare_data([data], axis, slice_at)
    img = _create_plot(ax, data[0], axis, colmap, vmin, vmax, title)
    fig.colorbar(img, ax=ax)


def plot_compare_grid(data: dict[str, np.ndarray], nrows: int, ncols: int, axis: int|tuple[int,int], 
                      slice_at: int|tuple[int,...]|None, figsize: tuple[int, int] = (14,6),  
                      colmap: str|mplcolors.Colormap|None = None, title: str|None = None):
    assert nrows * ncols >= len(data), \
        "Given `nrows` and `ncols` do not yield enough subplots for the length of the given data"
    labels = data.keys()
    data, vmin, vmax = _prepare_data(list(data.values()), axis, slice_at)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, layout='compressed', subplot_kw={'aspect': 'equal'})
    axs = axs.flatten()
    for ax, label, d in zip(axs, labels, data):
        img = _create_plot(ax, d, axis, colmap, vmin, vmax)
        label = label if isinstance(label, str) else f'n={label}'
        ax.set_title(label)
    for ax in axs[len(data):]:
        ax.axis('off')
    fig.colorbar(img, ax=axs)
    if title:
        fig.suptitle(title)


def plot_compare(target: np.ndarray, pred: np.ndarray, axis: int|tuple[int,int], 
                 slice_at: int|tuple[int,...]|None, figsize: tuple[int, int] = (14,6),  
                 colmap: str|mplcolors.Colormap|None = None, title: str|None = None):
    data = {'target': target, 'prediction': pred}
    plot_compare_grid(data, 1, 2, axis, slice_at, figsize, colmap, title)


def _set_lineplot(legend: bool = True, title: str|None = None, lim=None, xlabel: str|None = None, 
                  ylabel: str|None = None, xticks: list|None = None):
    if xticks is not None:
        plt.xticks(xticks)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if lim:
        plt.ylim(lim)
    if title:
        plt.title(title)
    if legend:
        plt.legend()


def lineplot_compare(data: dict[int, np.ndarray], target: np.ndarray|None = None, 
                     title: str|None = None, figSize: tuple[int, int] = (10,4), 
                     label_func: Callable[[int], str] = lambda x: str(x), lim=None, 
                     xlabel: str|None = None, ylabel: str|None = None, xticks: list|None = None):
    plt.figure(figsize=figSize)
    if target is not None:
        plt.plot(target, label="original", color=color_cycle[0])
    for i, (k, v) in enumerate(data.items(), start=1):
        plt.plot(v, label=label_func(k), color=color_cycle[i])
    _set_lineplot(True, title, lim, xlabel, ylabel, xticks)
    plt.show()


def lineplot(x_list: Iterable[Iterable], y_list: Iterable[Iterable], labels: list|None = None,
             title: str|None = None, figSize: tuple[int, int] = (10,4), lim=None, 
             xlabel: str|None = None, ylabel: str|None = None, xticks: list|None = None):
    plt.figure(figsize=figSize)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        plt.plot(x, y, color=color_cycle[i], label=labels[i] if labels else None)
    _set_lineplot(bool(labels), title, lim, xlabel, ylabel, xticks)
    plt.show()


def plot_correlation(cor_mat: np.ndarray, title: str):
    # Create the heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(cor_mat, extent=[0, 0.5, 0, 0.5], cmap='coolwarm', interpolation='nearest', origin="lower")
    plt.colorbar(label='Correlation Coefficient')

    # Add labels
    plt.title(title)
    plt.xlabel('tp')
    plt.ylabel('tp')
    plt.tight_layout()
    plt.show()
