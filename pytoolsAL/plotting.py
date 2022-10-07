#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import ipywidgets as widgets


def get_default_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def add_colorbar(ax, mappable=None, shrink=1):
    """
    wrapper function to add a properly scaled colorbar and apply
    styling defaults

    input ax object to add colorplot onto, returns colorbar object
    """
    cax = make_colorbar_axes(ax)
    if mappable is not None:
        cb = plt.colorbar(cax=cax, mappable=mappable, shrink=shrink)
    else:
        cb = plt.colorbar(cax=cax, shrink=shrink)
    cb = apply_colorbar_defaults(cb)
    return cb


def add_colorbar_space(ax):
    """
    since adding a colorbar to just one plot makes subplots uneven,
    add the same empty 'space' to other axes if desired
    """
    cax = make_colorbar_axes(ax)
    cax.axis('off')
    return cax


def apply_colorbar_defaults(cb):
    """
    since the default style sheet is kinda annoying for colorbars
    pass the plt.colorbar() object to style it better
    Args:
        cb: input colorbar (created with cb = plt.colorbar())

    Returns:
        cb: style-adjusted colorbar

    """
    cb.outline.set_visible(False)
    cb.ax.tick_params(size=0, which='both')
    return cb


def make_colorbar_axes(ax, pos='right', size='3%', pad=0.15):
    """
    return an ax object to contain the colorbar
    properly scaled based on the ax input to add it to
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    return cax


def apply_image_defaults(ax):
    """
    since the default style sheet is ugly for images
    pass the ax object to style it better
    Args:
        ax: matplotlib axis object where image is plotted

    Returns:
        ax: style-adjusted image axis

    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def apply_heatmap_defaults(ax):
    """
    since the default style sheet is ugly for heatmaps
    pass the ax object to style it better
    Args:
        ax: matplotlib axis object where heatmap is plotted

    Returns:
        ax: style-adjusted heatmap axis

    """
    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def apply_multiple_locator(ax, multiple_dict):
    """
    quick wrapper for setting tick locator
    Args:
        ax: matplotlib axis object
        multiple: int or list
        which: 'x', 'y', 'both'

    Returns:
        ax: matplotlib axis object

    """
    for axis in multiple_dict.keys():
        if multiple_dict[axis] is not None:
            if axis == 'x':
                xmult = multiple_dict[axis]
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xmult))
            if axis == 'y':
                ymult = multiple_dict[axis]
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ymult))

    return ax


def get_plottable_positions(positions, jitter_factor=1):
    """
    plots neurons by their positions on neuropixels probe
    discretizes x positions (such that there's a smaller gap between shanks)
    jitters neurons that share single channel

    Args:
        positions: array of x/z positions in shape (neurons, 2)

    Returns:
        pos_jittered: array to use for scatter
        x_um: original x positions for tick labels
        x_plot: real positions of x coordinates
    """
    x_um = np.unique(positions[:, 0])

    # typically 8 for 4 shank probes, assume/hardcode this case for now
    if len(x_um == 8):
        x_plot = [0, 1, 3, 4, 6, 7, 9, 10]

    pos_descr = np.copy(positions)
    for i, x_coord in enumerate(x_um):
        x_ix = np.argwhere(positions[:, 0] == x_coord).ravel()
        pos_descr[x_ix, 0] = x_plot[i]

    pos_jittered = np.copy(pos_descr)
    pos, p_counts = np.unique(positions, axis=0, return_counts=True)

    # i honestly do not know how *not* to hardcode this
    for i, count in enumerate(p_counts):
        duplicate_pos = pos[i]
        neurs_p = np.where((positions == duplicate_pos).all(axis=1))[0]
        if count % 2 == 1:
            c_range = count - 1
            cmin, cmax = -c_range/2, c_range/2
            jitters = np.arange(cmin, cmax+1)*0.3*jitter_factor

        if count % 2 == 0:
            c_range = np.arange(count).astype('float64')
            c_range -= np.mean(c_range)
            jitters = c_range*0.3*jitter_factor

        for iN, neuron in enumerate(neurs_p):
            pos_jittered[neuron] += jitters[iN]

    return pos_jittered, x_um, x_plot


def anim_to_file(anim, savepath, rewrite=False, fps=10, bitrate=-1):
    """
    Saves matplotlib animation object to file

    Args:
        anim: matplotlib animation object
        savepath: path to save - include filetype
        rewrite: bool, whether to rewrite if exists
        fps: framerate, default 10
        bitrate: bitrate, default -1 (lets writer choose)

    Returns:

    """
    if os.path.isfile(savepath):
        if rewrite:
            os.remove(savepath)
        else:
            raise FileExistsError(f'{savepath} already exists. delete or pass rewrite=True')

    fileformat = savepath[-3:]

    if fileformat == 'gif':
        writer = mpl.animation.PillowWriter(fps=fps, bitrate=bitrate)

    elif fileformat in ['avi', 'mp4']:
        writer = mpl.animation.FFMpegWriter(fps=fps, bitrate=bitrate)

    else:
        raise ValueError(f'incompatible file format found: {fileformat}. check savepath?')

    anim.save(savepath, writer=writer, dpi=300)


def cmap_bbr():
    """
    blue black red diverging colormap
    """
    cmap_path = Path(__file__).resolve().parent / '../styles/cmap_blueblackred.npy'
    cmap_bbr = mpl.colors.LinearSegmentedColormap.from_list(
        'bbr', np.load(cmap_path))
    return cmap_bbr

def cmap_gcamp():
    """
    black/green/bright green cmap for GCaMP
    """
    colors = ['black', 'green', 'lime']
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'gcamp', colors)
    return cmap


def heatmap(x, y, **kwargs):
    """
    from https://github.com/dylan-profiler/heatmaps/blob/master/heatmap/heatmap.py
    """
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    if 'cmap' in kwargs:
        cm = mpl.cm.ScalarMappable(cmap=kwargs['cmap'])
    else:
        cm = mpl.cm.ScalarMappable(cmap='coolwarm')
    cm.set_clim(-1, 1)

    def value_to_color(val):
        return cm.to_rgba(val)

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale

    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'xlabel', 'ylabel'
    ]}

    plt.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax = plt.gca()
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=90, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor', color='white')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#ececec')

    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))

    cb = add_colorbar(ax, mappable=cm)
    return ax, cb


def corrplot(data, size_scale=500, marker='s', cmap='coolwarm'):
    """
    from https://github.com/dylan-profiler/heatmaps/blob/master/heatmap/heatmap.py
    """
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    ax, cb = heatmap(
            corr['x'], corr['y'],
            color=corr['value'], color_range=[-1, 1],
            size=corr['value'].abs(), size_range=[0,1],
            marker=marker,
            x_order=data.columns,
            y_order=data.columns[::-1],
            size_scale=size_scale,
            cmap=cmap
            )
    return ax, cb


def sliderplot(x_list):
    """
    idk make it better later
    """
    length = len(x_list[0])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def update(xslider):
        f = plt.figure(figsize=(4, 3))
        ax = plt.gca()

        ax.set_yticks([])
        xmin = int(xslider[0])
        xmax = int(xslider[1])

        for ix, x in enumerate(x_list):
            ax2 = ax.twinx()
            plt.plot(x[xmin:xmax], color=colors[ix], alpha=0.7)
            ax2.spines['right'].set_position(('axes', 1+0.1*ix))
            ax2.tick_params(axis='y', which='both', colors=colors[ix])

        ax2.relim()
        ax2.autoscale_view(scalex=False)

        plt.show()

    w = widgets.interactive(update, xslider=widgets.FloatRangeSlider(
                            value=[0, length], min=0, max=length,
                            step=1, layout=widgets.Layout(width='50%'),
                            description='x range',
                            style={'description_width': 'initial'}))
    w.update()
    return w
