import sys
import os
from os.path import dirname, abspath


class Path:
    """Path manager for analysis."""

    def __init__(self, work_dir=None):
        if work_dir is None:
            self.work_dir = dirname(dirname((dirname(abspath(__file__))))) + "/"
        else:
            self.work_dir = work_dir

        self._folder_raw = self.work_dir + 'raw/'
        self._folder_data = self.work_dir + 'data/'
        self._folder_plot = self.work_dir + 'plot/'

    # SET CUSTOM PATH
    def set_folder_raw(self, path):
        Path.make_folder(path)
        self._folder_raw = path

    def set_folder_data(self, path):
        Path.make_folder(path)
        self._folder_data = path

    def set_folder_plot(self, path):
        Path.make_folder(path)
        self._folder_plot = path

    # GET PATH
    def folder_raw(self, ensID=''):
        out = self._folder_raw + f'{ensID}/'
        Path.make_folder(out)
        return out
    
    def folder_data(self, ensID=''):
        out = self._folder_data + f'{ensID}/'
        Path.make_folder(out)
        return out

    def folder_plot(self, ensID=''):
        out = self._folder_plot + f'{ensID}/'
        Path.make_folder(out)
        return out

    # UTILITIES
    @staticmethod
    def make_folder(path_to_folder):
        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)

    @staticmethod
    def add_module(path_to_module):
        """Add module to the path."""
        if not path_to_module in sys.path:
            sys.path.append(path_to_module)
