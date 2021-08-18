#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os


def add_colorbar(ax):
    """
    """
    cax = make_colorbar_axes(ax)
    cb = plt.colorbar(cax=cax)
    cb = apply_colorbar_defaults(cb)
    return cb


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


def make_colorbar_axes(ax, pos='right', size='4%', pad=0.15):
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

    anim.save(savepath, writer=writer)
