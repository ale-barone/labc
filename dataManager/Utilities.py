import numpy as np
import pathlib

def _get_extension(file):
    """Get the extension of a file."""
    path = pathlib.Path(file)
    return path.suffix

