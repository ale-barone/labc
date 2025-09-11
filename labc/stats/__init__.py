"""Jackknife and bootstrap resampling for statistical error estimation,
covariance matrices, and correlation matrices.
"""
from ._statsbase import StatsBase
from ._statsjack import StatsJack
from ._statsboot import StatsBoot

__all__ = ['StatsType', 'StatsBase', 'StatsJack', 'StatsBoot']

_REGISTRY = {
    'Jack': StatsJack,
    'Boot': StatsBoot,
}


class StatsType(StatsBase):
    """Base class for statistical analysis.

    The standard usage is through the static methods :meth:`StatsType.Jack`
    and :meth:`StatsType.Boot`, which return fully initialised
    :class:`~labc.stats.StatsJack` and :class:`~labc.stats.StatsBoot` objects::

        stats = StatsType.Jack(num_config=100)
        mean, err, bins = stats.generate_stats(raw_data)

        stats = StatsType.Boot(num_config=100, num_bins=500)
        mean, err, bins = stats.generate_stats(raw_data)

    ``StatsType`` can also be instantiated directly for a **bins-only workflow**,
    when only pre-computed bins are available and raw configurations are not.
    For example, with jackknife bins::

        stats = StatsType('Jack')
        mean = np.mean(bins, axis=0)
        err = stats.err_func(mean, bins)

    Parameters
    ----------
    statsID : str
        Identifier for the resampling method.
        Known values: ``'Jack'``, ``'Boot'`` (see :attr:`_KNOWN_IDS`).

    Raises
    ------
    ValueError
        If ``statsID`` is not in :attr:`_KNOWN_IDS`.

    Attributes
    ----------
    ID : str
        Resampling strategy identifier (``'Jack'`` or ``'Boot'``).
    num_config : None
        Always ``None`` in the bins-only workflow.
    num_bins : None
        Always ``None`` in the bins-only workflow.
    seed : None
        Always ``None`` in the bins-only workflow.
    """

    _KNOWN_IDS = set(_REGISTRY)

    def __init__(self, statsID: str):
        if statsID not in self._KNOWN_IDS:
            raise ValueError(f"unknown statsID {statsID!r}, known: {self._KNOWN_IDS}")
        super().__init__()
        self.ID = statsID
        self._prefactor_func = _REGISTRY[statsID]._prefactor_func

    @staticmethod
    def Jack(*, num_config: int, rebin: int = 1) -> 'StatsJack':
        """Jackknife factory — see :class:`~labc.stats.StatsJack` for full documentation.

        Parameters
        ----------
        num_config : int
            Number of raw gauge configurations.
        rebin : int, optional
            Rebinning factor. Default is ``1``.

        Returns
        -------
        StatsJack
        """
        if num_config is None:
            raise ValueError(
                "num_config is required for StatsType.Jack. "
                "For the bins-only workflow use StatsType('Jack') instead."
            )
        return StatsJack(num_config, rebin)

    @staticmethod
    def Boot(*, num_config: int, num_bins: int, seed: int = 0) -> 'StatsBoot':
        """Bootstrap factory — see :class:`~labc.stats.StatsBoot` for full documentation.

        Parameters
        ----------
        num_config : int
            Number of raw gauge configurations.
        num_bins : int
            Number of bootstrap samples to generate.
        seed : int, optional
            Random seed. Default is ``0``.

        Returns
        -------
        StatsBoot
        """
        return StatsBoot(num_config=num_config, num_bins=num_bins, seed=seed)