from ._statsbase import StatsBase as _StatsBase
from ._statsjack import StatsJack as _StatsJack
from ._statsboot import StatsBoot as _StatsBoot


class StatsType:

    class Jack:
        def __new__(cls, *, num_config):
            return _StatsJack(num_config)
    
    class Boot:
        def __new__(cls, *, num_config, num_bins, seed=0):
            return _StatsBoot(num_config=num_config, num_bins=num_bins, seed=seed)
