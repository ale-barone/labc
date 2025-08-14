from ._statsbase import StatsBase
from ._statsjack import StatsJack
from ._statsboot import StatsBoot

__all__ = ['StatsType', 'StatsBase', 'StatsJack', 'StatsBoot']

_REGISTRY = {
    'Jack': StatsJack,
    'Boot': StatsBoot,
}


class StatsType(StatsBase):

    _KNOWN_IDS = set(_REGISTRY)

    def __init__(self, statsID):
        if statsID not in self._KNOWN_IDS:
            raise ValueError(f"unknown statsID {statsID!r}, known: {self._KNOWN_IDS}")
        super().__init__()
        self.ID = statsID
        self._prefactor_func = _REGISTRY[statsID]._prefactor_func

    @staticmethod
    def Jack(*, num_config, rebin=1):
        if num_config is None:
            raise ValueError(
                "num_config is required for StatsType.Jack. "
                "For the bins-only workflow use StatsType('Jack') instead."
            )
        return StatsJack(num_config, rebin)

    @staticmethod
    def Boot(*, num_config, num_bins, seed=0):
        return StatsBoot(num_config=num_config, num_bins=num_bins, seed=seed)