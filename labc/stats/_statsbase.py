import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from .. import data as dM
from scipy.linalg import block_diag

# DataStats is imported only during type checking (e.g. mypy, Pyright) and not
# at runtime. This avoids a circular import: labc.data imports labc.stats, so
# importing labc.data here at runtime would create a cycle. The string
# annotations ('DataStats') in the signatures below are resolved lazily and
# never evaluated at runtime, so the guard is safe.
if TYPE_CHECKING:
    from ..data import DataStats


class Istats(ABC):
    """Base class for managing statistics."""

    @abstractmethod
    def generate_bins(self, array_raw):
        """It generates resampled bins from raw data."""

    @abstractmethod
    def err_func(self, array_bins):
        """It computes the error from resampled bins."""

    @abstractmethod
    def generate_stats(self, array_raw):
        """It computes the 'stats' version returning (array_mean, array_err, array_bins)."""

    @abstractmethod
    def cov(self, *arrays):
        """General covariance matrix possibly for different input arrays."""

    @abstractmethod
    def corr(self, *arrays):
        """General correlation matrix possibly for different input arrays."""


class StatsBase(Istats):
    """Base class for statistical analysis.

    Provides shared infrastructure for error estimation, covariance and
    correlation computation, and the full ``generate_stats`` workflow.
    Concrete subclasses implement the resampling strategy
    via :meth:`generate_bins`.

    Parameters
    ----------
    num_config : int or None, optional
        Number of raw gauge configurations. May be ``None`` when working
        with pre-computed bins only (bins-only workflow via ``StatsType('Jack')``
        or ``StatsType('Boot')``).
    num_bins : int or None, optional
        Number of resampled bins. ``None`` in the bins-only workflow.
    seed : int or None, optional
        Random seed for resampling methods that require it.
        Ignored in the bins-only workflow.
    """

    def __init__(self, num_config: int | None = None,
                 num_bins: int | None = None,
                 seed: int | None = None):
        self.num_config = num_config
        self.num_bins = num_bins
        self.seed = seed
        self._prefactor_func = None
        self.ID = None

    def __str__(self):
        out = (
            f"StatsType = '{self.ID}':"
            f"\n -num_config = {self.num_config}"
            f"\n -num_bins = {self.num_bins}"
            f"\n -seed = {self.seed}"
        )
        return out

    def __repr__(self):
        out = (
            f"StatsType.{self.ID}("
            f"num_config={self.num_config}, "
            f"num_bins={self.num_bins}, "
            f"seed={self.seed})"
        )
        return out

    def generate_bins(self, _array_raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_num_bins(self, array_bins: np.ndarray) -> int:
        assert(array_bins.ndim == 2)
        return len(array_bins)

    def _get_prefactor(self, array_bins: np.ndarray) -> float:
        return self._prefactor_func(self._get_num_bins(array_bins))

    def _assert_bins_size(self, array: np.ndarray):
        if self.num_bins is not None:
            if self.num_bins != len(array):
                raise ValueError(
                    f"array length {len(array)} does not match "
                    f"num_bins={self.num_bins} of this StatsType object"
                )

    def err_func(self, array_mean: np.ndarray, array_bins: np.ndarray) -> np.ndarray:
        """Compute the statistical error from resampled bins.

        Parameters
        ----------
        array_mean : np.ndarray
            Sample mean, shape ``(T,)``.
        array_bins : np.ndarray
            Resampled bins, shape ``(num_bins, T)``.

        Returns
        -------
        np.ndarray
            Statistical errors, shape ``(T,)``.

        Notes
        -----
        The error is computed as:

        .. math::

            \\sigma_i = \\sqrt{f \\cdot \\frac{1}{N_\\mathrm{bins}}
                        \\sum_b \\left(b_i - \\bar{x}_i\\right)^2}

        where :math:`f` is the prefactor defined by the concrete subclass.
        """
        diff2 = (array_bins - array_mean)**2
        self._assert_bins_size(diff2)
        prefactor = self._get_prefactor(diff2)
        err2 = prefactor * np.mean(diff2, 0)
        return np.sqrt(err2)

    def generate_stats(self, array_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean, error, and resampled bins from raw configurations.

        Parameters
        ----------
        array_raw : np.ndarray
            Raw configurations, shape ``(num_config, T)``.

        Returns
        -------
        mean : np.ndarray
            Sample mean, shape ``(T,)``.
        err : np.ndarray
            Statistical error, shape ``(T,)``.
        bins : np.ndarray
            Resampled bins, shape ``(num_bins, T)``.
        """
        mean = np.mean(array_raw, 0)
        bins = self.generate_bins(array_raw)
        err = self.err_func(mean, bins)
        return mean, err, bins

    def cov(self, data_x: 'DataStats', data_y: 'DataStats | None' = None) -> np.ndarray:
        """Compute the covariance matrix of one or two ``DataStats`` objects.

        Parameters
        ----------
        data_x : DataStats
            First dataset.
        data_y : DataStats, optional
            Second dataset. If ``None``, the auto-covariance of ``data_x``
            is returned.

        Returns
        -------
        np.ndarray
            Covariance matrix, shape ``(len(data_x), len(data_y))``.

        Notes
        -----
        Slicing for fit ranges or thinning should be applied to the inputs
        before calling this method.
        """
        if data_y is None:
            data_y = data_x
        num_bins = data_x.num_bins()
        cov = np.empty(shape=(num_bins, len(data_x), len(data_y)))
        for b in range(num_bins):
            vec_x = data_x.bins[b] - data_x.mean
            vec_y = data_y.bins[b] - data_y.mean
            cov[b] = np.outer(vec_x, vec_y)
        prefactor = self._get_prefactor(data_x.bins)
        return prefactor * np.mean(cov, 0)

    def cov_blocks(self, *data: 'DataStats') -> np.ndarray:
        """Covariance matrix of multiple ``DataStats`` objects merged into one.

        All datasets are concatenated along the observable axis before the
        covariance is computed, preserving cross-correlations between them.

        Parameters
        ----------
        *data : DataStats
            Datasets to merge.

        Returns
        -------
        np.ndarray
            Full covariance matrix of the concatenated dataset.
        """
        return self.cov(dM.merge(*data))

    def cov_blocks_diag(self, *data: 'DataStats') -> np.ndarray:
        """Block-diagonal covariance matrix of multiple ``DataStats`` objects.

        Computes the covariance of each dataset independently and assembles
        them into a block-diagonal matrix, assuming no cross-correlations.

        Parameters
        ----------
        *data : DataStats
            Datasets for each diagonal block.

        Returns
        -------
        np.ndarray
            Block-diagonal covariance matrix.
        """
        return block_diag(*[self.cov(d) for d in data])

    def corr(self, data_x: 'DataStats', data_y: 'DataStats | None' = None) -> np.ndarray:
        """Compute the correlation matrix of one or two ``DataStats`` objects.

        Parameters
        ----------
        data_x : DataStats
            First dataset.
        data_y : DataStats, optional
            Second dataset. If ``None``, the auto-correlation of ``data_x``
            is returned. Diagonal entries are exactly 1 in the auto-correlation
            case.

        Returns
        -------
        np.ndarray
            Correlation matrix, shape ``(len(data_x), len(data_y))``.

        Notes
        -----
        Slicing for fit ranges or thinning should be applied to the inputs
        before calling this method.
        """
        if data_y is None:
            data_y = data_x
        cov = self.cov(data_x, data_y)
        return np.diag(1/data_x.err) @ cov @ np.diag(1/data_y.err)