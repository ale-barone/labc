from ._statsbase import StatsBase as _StatsBase
from ._statsjack import StatsJack as _StatsJack
from ._statsboot import StatsBoot as _StatsBoot

_REGISTRY = {
    'Jack': _StatsJack,
    'Boot': _StatsBoot,
}


class StatsType(_StatsBase):

    _KNOWN_IDS = set(_REGISTRY)

    def __init__(self, statsID):
        if statsID not in self._KNOWN_IDS:
            raise ValueError(f"unknown statsID {statsID!r}, known: {self._KNOWN_IDS}")
        super().__init__()
        self.ID = statsID
        self._prefactor_func = _REGISTRY[statsID]._prefactor_func

    class Jack:
        def __new__(cls, *, num_config, rebin=1):
            if num_config is None:
                raise ValueError(
                    "num_config is required for StatsType.Jack. "
                    "For the bins-only workflow use StatsType('Jack') instead."
                )
            return _StatsJack(num_config, rebin)

    class Boot:
        def __new__(cls, *, num_config, num_bins, seed=0):
            return _StatsBoot(num_config=num_config, num_bins=num_bins, seed=seed)
