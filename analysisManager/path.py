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

        self._folder_raw = None
        self._folder_gauge = None
        self._folder_data = None
        self._folder_plot = None


    @property
    def folder_raw(self):
        """Path to folder containing raw data
        (raw gauge configurations from the computation)."""
        return self._folder_raw

    @folder_raw.setter
    def folder_raw(self, folder_raw_path):
        self._folder_raw = folder_raw_path

    @property
    def folder_gauge(self):
        """Path to folder containing reshaped raw data
        (raw gauge configurations)."""
        return self._folder_gauge

    @folder_gauge.setter
    def folder_gauge(self, folder_gauge_path):
        self._folder_gauge = folder_gauge_path
        self.make_folder(folder_gauge_path)

    @property
    def folder_data(self):
        """Path to folder containing analysed data."""
        return self._folder_data

    @folder_data.setter
    def folder_data(self, folder_data_path):
        self._folder_data = folder_data_path
        self.make_folder(folder_data_path)

    @property
    def folder_plot(self):
        """Path to folder containing plots of the analysis."""
        return self._folder_plot

    @folder_plot.setter
    def folder_plot(self, folder_plot_path):
        self._folder_plot = folder_plot_path
        self.make_folder(folder_plot_path)

    @staticmethod
    def make_folder(path_to_folder):
        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)

    @staticmethod
    def add_module(path_to_module):
        """Add module to the path."""
        if not path_to_module in sys.path:
            sys.path.append(path_to_module)
