import importlib

from . import _version

__version__ = _version.__version__

# This module uses anaconda, and so assumes numpy, scipy etc are installed.

# Don't load some modules if dependencies are missing
for tN in ['image', 'globs', 'math', 'neuropixels', 'plotting']:
    try:
        globals()[tN] = importlib.import_module('.' + tN, __name__)
    except ImportError as e:
        print("pytoolsAL: Modules missing.  Not loading {mod}.  Message: {msg}" \
              .format(mod=tN, msg=str(e)))
