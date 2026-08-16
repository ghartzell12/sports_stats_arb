"""nflfm — an NFL fantasy football projection model.

Layers, bottom up:

* :mod:`nflfm.data`     — fetch and load public nflverse feeds
* :mod:`nflfm.scoring`  — league rules as data, applied to box scores
* :mod:`nflfm.features` — leak-free lag/rolling features
* :mod:`nflfm.models`   — projectors and walk-forward backtesting
"""

__version__ = "0.1.0"
