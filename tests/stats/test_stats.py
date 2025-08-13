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

class TestStatsJack:

    def test_generate_bins_shape(self):
        num_config, T = 20, 8
        data = make_data(num_config, T)
        jack = StatsType.Jack(num_config=num_config)
        bins = jack.generate_bins(data)
        assert bins.shape == (num_config, T)

    def test_generate_stats_shapes(self):
        num_config, T = 20, 8
        data = make_data(num_config, T)
        jack = StatsType.Jack(num_config=num_config)
        mean, err, bins = jack.generate_stats(data)
        assert mean.shape == (T,)
        assert err.shape == (T,)
        assert bins.shape == (num_config, T)
    

    def test_err_zero_for_constant_data(self):
        num_config, T = 20, 8
        data = np.ones((num_config, T)) * 3.14
        jack = StatsType.Jack(num_config=num_config)
        _, err, _ = jack.generate_stats(data)
        np.testing.assert_allclose(err, 0.0)

    # TODO: add more tests for err_func?

    def test_jackknife_mean(self):
        num_config, T = 20, 8
        data = make_data(num_config, T)
        jack = StatsType.Jack(num_config=num_config)
        mean, _, _ = jack.generate_stats(data)
        expected_mean = np.mean(data, axis=0)
        np.testing.assert_allclose(mean, expected_mean)

    @pytest.mark.parametrize("rebin", [2, 3])
    @pytest.mark.filterwarnings("ignore::UserWarning")  # rebin=3 intentionally warns
    def test_rebin_equivalent_to_repeated_configs(self, rebin):
        """Repeating each config k times then using rebin=k must equal plain jackknife on original.

        np.repeat(data, k, axis=0) duplicates each row k times consecutively, so rebin=k
        averages them back to the original config — giving the same bins as rebin=1 on data.
        """
        num_config, T = 10, 8
        data = make_data(num_config, T)
        data_repeated = np.repeat(data, rebin, axis=0)  # shape(num_config*rebin, T)

        bins_original = StatsType.Jack(num_config=num_config).generate_bins(data)
        bins_rebin = StatsType.Jack(num_config=num_config*rebin, rebin=rebin).generate_bins(data_repeated)

        np.testing.assert_allclose(bins_rebin, bins_original)

    # FIXME: here I want num_config=None not to be acceptable, and use
    # StatsType('Jack') instead...should I change it?
    def test_num_config_none_raises(self):
        with pytest.raises(ValueError):
            StatsType.Jack(num_config=None)

    def test_bins_average(self):
        num_config, T = 20, 8
        data = make_data(num_config, T)
        jack = StatsType.Jack(num_config=num_config)
        bins = jack.generate_bins(data)
        for i in range(T):
            expected = np.mean(np.delete(data, i, axis=0), axis=0)
            np.testing.assert_allclose(bins[i], expected)

    def test_rebin3_warns(self):
        """rebin=3 should emit a UserWarning."""
        with pytest.warns(UserWarning):
            StatsType.Jack(num_config=9, rebin=3)

    def test_rebin4_raises(self):
        """rebin >= 4 should raise a ValueError."""
        with pytest.raises(ValueError):
            StatsType.Jack(num_config=20, rebin=4)
