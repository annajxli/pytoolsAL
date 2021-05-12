#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os


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


def apply_mulitple_locator(ax, multiple, which='both'):
    """
    quick wrapper for setting tick locator
    Args:
        ax: matplotlib axis object
        multiple: int or list
        which: 'x', 'y', or 'both'

    Returns:
        ax:

    """
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