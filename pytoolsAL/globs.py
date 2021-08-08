#!/usr/bin/env python3
from pathlib import Path
import matplotlib as mpl
import numpy as np


def datadir():
    """
    quick way to access data directory path
    Returns: default data directory path

    """
    dd = Path('/Users/anna/Data/steinmetzlab')
    return dd


def mplstylesheet():
    """
    quick way to access matplotlib stylesheet
    Returns: default matplotlib stylesheet path

    """
    mss = Path('/Users/anna/Repositories/pytoolsAL/styles/ALpres.mplstyle')
    return mss


def cmap_blueblackred():
    bbr = mpl.colors.LinearSegmentedColormap.from_list('bbr', np.load(
        '/Users/anna/Repositories/pytoolsAL/styles/cmap_blueblackred.npy'))
    return bbr
