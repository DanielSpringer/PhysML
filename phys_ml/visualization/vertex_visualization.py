import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt

import numpy as np


cmap_big = mpl.colormaps['twilight_shifted'].resampled(int(1e3))
cmap_resc = mplcolors.ListedColormap(cmap_big(np.linspace(0.075, 0.925, 10000)))


def _plot(data: np.ndarray, ax: axes.Axes, max_value: float, x_label: str, y_label: str,
          colmap: str|mplcolors.Colormap, vmin: float|None = None, vmax: float|None = None):
    cmap = colmap if colmap else cmap_resc
    img = ax.imshow(data, cmap=cmap, vmax=max_value)
    if vmin is not None and vmax is not None:
        img.set_clim(vmin=vmin, vmax=vmax)
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    return img


def _create_plot(ax: axes.Axes, data: np.ndarray, k: int, max_value: float, axis: int, 
                 colmap: str|mplcolors.Colormap|None = None):
    data_slice = [slice(None)] * 3
    data_slice[axis - 1] = k
    d = data[*data_slice]
    
    axs_range = np.arange(1,4)
    printed_axs = axs_range[axs_range != axis]
    x_label, y_label = (f'$k_{pa}$' for pa in printed_axs)
    img = _plot(d, ax, max_value, x_label, y_label, colmap)
    return img


def _create_plot_24x6(ax: axes.Axes, data: np.ndarray, k: int, kix: int, kiy: int, kjx: int, kjy: int, 
                      max_value: float, colmap: str|mplcolors.Colormap|None = None):
    data_slice = [kix, kiy, kjx, kjy]
    [data_slice.insert(i + 2 * (k - 1), slice(None)) for i in range(2)]
    d = data[*data_slice]

    x_label, y_label = f'$k_{{{k}_x}}$', f'$k_{{{k}_y}}$'
    img = _plot(d, ax, max_value, x_label, y_label, colmap)
    return img


def _init_plot(data: np.ndarray, k_axis: int = 3, figsize: tuple[int, int] = (8,6)) -> tuple:
    assert k_axis > 0 and k_axis <= 3, f'k_axis must be within range [1, 3]'
    
    max_value = np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return fig, ax, max_value


def plot_section(data: np.ndarray, k: int, axis: int = 3, figsize: tuple[int, int] = (8,6), 
                 colmap: str|mplcolors.Colormap|None = None):
    fig, ax, max_value = _init_plot(data, axis, figsize)
    img = _create_plot(ax, data, k, max_value, axis, colmap)
    fig.colorbar(img, ax=ax)


def plot_section_24x6(data: np.ndarray, k: int = 3, kix: int = 12, kiy: int = 12, kjx: int = 12, kjy: int = 12, 
                      figsize: tuple[int, int] = (8,6), colmap: str|mplcolors.Colormap|None = None):
    fig, ax, max_value = _init_plot(data, k, figsize)
    img = _create_plot_24x6(ax, data, k, kix, kiy, kjx, kjy, max_value, colmap)
    fig.colorbar(img, ax=ax)


def plot_comparison(target: np.ndarray, pred: np.ndarray, k: int, figsize: tuple[int, int] = (14,6), axis: int = 3, 
                    colmap: str|mplcolors.Colormap|None = None, show_title: bool = True):
    max_value = np.max([target, pred])
    fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={'aspect': 'equal'})
    for i, (ax, d) in enumerate(zip(axs, [target, pred])):
        img = _create_plot(ax, d, k, max_value, axis, colmap)
        if i == 0:
            ax.set_title("Target")
        else:
            ax.set_title("Prediction")
    fig.colorbar(img, ax=axs)
    if show_title:
        fig.suptitle(fr"Reconstruction for $k_{axis}$", fontsize=16)


def plot_comparison_24x6(target: np.ndarray, pred: np.ndarray, kix: int, kiy: int, kjx: int, kjy: int, 
                         figsize: tuple[int, int] = (14,6), k: int = 3, 
                         colmap: str|mplcolors.Colormap|None = None):
    max_value = np.max([target, pred])
    fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={'aspect': 'equal'})
    for i, (ax, d) in enumerate(zip(axs, [target, pred])):
        img = _create_plot_24x6(ax, d, k, kix, kiy, kjx, kjy, max_value, colmap)
        if i == 0:
            ax.set_title("Target")
        else:
            ax.set_title("Prediction")
    fig.colorbar(img, ax=axs)
