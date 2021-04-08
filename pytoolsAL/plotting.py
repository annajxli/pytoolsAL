#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_colorbar_defaults(cb):
    """
    since the default style sheet is kinda annoying for colorbars
    pass a cb object to style it better
    Args:
        cb: input colorbar (created with cb = plt.colorbar())

    Returns:
        cb: style-adjusted colorbar

    """
    cb.outline.set_visible(False)
    cb.ax.tick_params(size=0, which='both')
    return cb
