import numpy as np
from ._statsbase import StatsBase


class StatsBoot(StatsBase):
    """Bootstrap resampling for statistical error estimation.

    Instantiate via :meth:`~labc.stats.StatsType.Boot`.

    Parameters
    ----------
    num_config : int
        Number of raw gauge configurations.
    num_bins : int
        Number of bootstrap samples to generate.
    seed : int, optional
        Seed for the random number generator. Default is ``0``.

    Attributes
    ----------
    num_config : int
        Number of raw gauge configurations.
    num_bins : int
        Number of bootstrap samples.
    seed : int
        Random seed for reproducible resampling.
    ID : str
        Always ``'Boot'``.

    Notes
    -----
    The prefactor used in error and covariance estimation is :math:`f = 1`.
    """

    @staticmethod
    def _prefactor_func(_):
        return 1

    def __init__(self, num_config: int, num_bins: int, seed: int = 0):
        super().__init__(num_config, num_bins)
        self._prefactor_func = StatsBoot._prefactor_func
        self.seed = seed
        self.ID = 'Boot'

    def generate_bins(self, array_raw: np.ndarray) -> np.ndarray:
        """Generate bootstrap bins from raw configurations.

        Parameters
        ----------
        array_raw : np.ndarray
            Raw configurations, shape ``(num_config, T)``.

        Returns
        -------
        np.ndarray
            Bootstrap bins, shape ``(num_bins, T)``.

        Raises
        ------
        ValueError
            If the number of configurations in ``array_raw`` does not match
            ``self.num_config``.

        Notes
        -----
        Each bootstrap sample is the mean of ``num_config`` configurations
        drawn with replacement. The random state is seeded with ``self.seed``
        so results are fully reproducible.
        """
        num_config = np.size(array_raw, 0)
        if num_config != self.num_config:
            raise ValueError(
                f"data has {num_config} configurations but this object "
                f"was created with num_config={self.num_config}"
            )

        rng = np.random.default_rng(self.seed)
        bins = rng.integers(0, num_config, size=(self.num_bins, num_config))
        bins = np.mean(array_raw[bins], axis=1)
        return bins