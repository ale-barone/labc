import numpy as np
import pathlib

def _get_extension(file):
    """Get the extension of a file."""
    path = pathlib.Path(file)
    return path.suffix

def concatenate_stats(mean, err, bins):
    out = np.array([mean, err])
    out = np.concatenate([out, bins], axis=0)
    return out