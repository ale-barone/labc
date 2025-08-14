import math
import warnings
import numpy as np
from ._statsbase import StatsBase


class StatsJack(StatsBase):
    """Jackknife resampling for statistical error estimation.

    Supports optional rebinning: neighbouring configurations are averaged
    into blocks of size ``rebin`` before the leave-one-out procedure.
    This reduces the effective autocorrelation length of the ensemble.
    Instantiate via :meth:`~labc.stats.StatsType.Jack`.

    Parameters
    ----------
    num_config : int
        Number of raw gauge configurations.
    rebin : int, optional
        Number of consecutive configurations to average into a single block
        before jackknifing. Must be between ``1`` and ``_MAX_REBIN`` (= ``3``)
        inclusive. Default is ``1`` (no rebinning).

    Raises
    ------
    ValueError
        If ``rebin > _MAX_REBIN``. For larger rebinning factors, pre-average
        your configurations manually before constructing this object.

    Warns
    -----
    UserWarning
        If ``rebin == _MAX_REBIN``, since this is unusual and may indicate
        significant autocorrelations in the data.

    Notes
    -----
    The prefactor used in error and covariance estimation is
    :math:`f = N_\\mathrm{bins} - 1`.
    """

    _MAX_REBIN = 3

    @staticmethod
    def _prefactor_func(num_bins):
        return num_bins - 1

    def __init__(self, num_config: int, rebin: int = 1):
        if rebin > self._MAX_REBIN:
            raise ValueError(
                f"rebin={rebin} is too large; maximum allowed is {self._MAX_REBIN}. "
                f"Consider pre-averaging your configurations manually."
            )
        if rebin == self._MAX_REBIN:
            warnings.warn(
                f"rebin={rebin} is unusual and may indicate autocorrelation issues. "
                f"Make sure this is intentional.",
                UserWarning, stacklevel=2
            )
        num_bins = math.ceil(num_config / rebin) if num_config is not None else None
        super().__init__(num_config, num_bins)
        self._prefactor_func = StatsJack._prefactor_func
        self.rebin = rebin
        self.ID = 'Jack'

    def generate_bins(self, array_raw: np.ndarray) -> np.ndarray:
        """Generate jackknife bins from raw configurations.

        Parameters
        ----------
        array_raw : np.ndarray
            Raw configurations, shape ``(num_config, T)``.

        Returns
        -------
        np.ndarray
            Jackknife bins, shape ``(num_bins, T)``, where
            ``num_bins = ceil(num_config / rebin)``.

        Notes
        -----
        If ``num_config`` is not divisible by ``rebin``, the last configuration
        is repeated to pad to the nearest multiple. Rebinning averages
        consecutive blocks of ``rebin`` configurations before the leave-one-out
        jackknife is applied to the ``num_bins`` block averages.
        """
        num_config = np.size(array_raw, 0)

        # pad if needed so num_config is divisible by rebin
        remainder = num_config % self.rebin
        if remainder != 0:
            n_pad = self.rebin - remainder
            padding = np.repeat(array_raw[[-1]], n_pad, axis=0)
            array_raw = np.concatenate([array_raw, padding], axis=0)

        # average blocks of `rebin` neighbouring configs
        array_rebinned = np.mean(
            array_raw.reshape(self.num_bins, self.rebin, -1), axis=1
        )

        # jackknife leave-one-out on the rebinned configs
        bins = np.array([
            np.delete(np.arange(self.num_bins), b, 0) for b in range(self.num_bins)
        ])
        bins = np.mean(array_rebinned[bins], axis=1)
        return bins