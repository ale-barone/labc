import numpy as np
import pytest
from labc.stats import StatsType

# TODO: change this with data on disk (maybe pion correlator?)
def make_data(num_config, T, seed=0):
    return np.random.default_rng(seed).standard_normal((num_config, T))


class TestStatsType:

    def test_invalid_id_raises(self):
        with pytest.raises(ValueError):
            StatsType('Invalid')
