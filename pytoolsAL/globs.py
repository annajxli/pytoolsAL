#!/usr/bin/env python3
from pathlib import Path
import matplotlib as mpl
import numpy as np


class Globs:
    def __init__(self, machine):
        self.machine = machine

        if self.machine == 'labpc':
            self.datadir = Path(r'C:\Users\anna\Data')
            self.dataserver = Path(r'Z:\Subjects')
            self.stylesheet = Path(r'C:\Users\anna\Repositories\pytoolsAL\styles\ALpres.mplstyle')

        elif self.machine == 'macbook':
            self.datadir = Path('/Users/anna/Data/steinmetzlab')
            self.stylesheet = Path('/Users/anna/Repositories/pytoolsAL/styles/ALpres.mplstyle')

        else:
            raise ValueError('not a valid machine, try labpc or macbook')
