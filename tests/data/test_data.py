import numpy as np
import pytest
from labc.stats import StatsType
from labc.data import DataBins, DataStats


NUM_CONFIG = 100
NUM_BOOT_BINS = 500
T = 16


def make_raw(seed=0):
    return np.random.default_rng(seed).standard_normal((NUM_CONFIG, T))


# ── StatsType fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def jack():
    return StatsType.Jack(num_config=NUM_CONFIG)


@pytest.fixture
def boot():
    return StatsType.Boot(num_config=NUM_CONFIG, num_bins=NUM_BOOT_BINS)


# ── DataStats fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def jack_datastats(jack):
    mean, _, bins = jack.generate_stats(make_raw(seed=0))
    return DataStats(mean, bins, jack)


@pytest.fixture
def jack_datastats2(jack):
    mean, _, bins = jack.generate_stats(make_raw(seed=1))
    return DataStats(mean, bins, jack)


@pytest.fixture
def boot_datastats(boot):
    mean, _, bins = boot.generate_stats(make_raw(seed=0))
    return DataStats(mean, bins, boot)


@pytest.fixture
def boot_datastats2(boot):
    mean, _, bins = boot.generate_stats(make_raw(seed=1))
    return DataStats(mean, bins, boot)


# ── plain DataBins fixtures ─────────────────────────────────────────────────

@pytest.fixture
def plain_bins_jack():
    """DataBins whose num_bins matches the jackknife fixtures (== NUM_CONFIG)."""
    rng = np.random.default_rng(42)
    return DataBins(rng.standard_normal(T), rng.standard_normal((NUM_CONFIG, T)))


@pytest.fixture
def plain_bins_boot():
    """DataBins whose num_bins matches the bootstrap fixtures (== NUM_BOOT_BINS)."""
    rng = np.random.default_rng(42)
    return DataBins(rng.standard_normal(T), rng.standard_normal((NUM_BOOT_BINS, T)))


################################################################################
# DataBins arithmetic (no statsType involved)
################################################################################

class TestDataBinsArithmetic:

    @pytest.mark.parametrize("op", [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ])
    def test_databins_op_databins_returns_databins(self, plain_bins_jack, op):
        rng = np.random.default_rng(99)
        other = DataBins(rng.standard_normal(T), rng.standard_normal((NUM_CONFIG, T)))
        assert type(op(plain_bins_jack, other)) is DataBins

    def test_add_values(self, plain_bins_jack):
        rng = np.random.default_rng(99)
        other = DataBins(rng.standard_normal(T), rng.standard_normal((NUM_CONFIG, T)))
        result = plain_bins_jack + other
        np.testing.assert_allclose(result.mean, plain_bins_jack.mean + other.mean)
        np.testing.assert_allclose(result.bins, plain_bins_jack.bins + other.bins)

    def test_scalar_mul_returns_databins(self, plain_bins_jack):
        result = plain_bins_jack * 3.0
        assert type(result) is DataBins
        np.testing.assert_allclose(result.mean, plain_bins_jack.mean * 3.0)
        np.testing.assert_allclose(result.bins, plain_bins_jack.bins * 3.0)

    def test_scalar_add_returns_databins(self, plain_bins_jack):
        result = plain_bins_jack + 2.0
        assert type(result) is DataBins
        np.testing.assert_allclose(result.mean, plain_bins_jack.mean + 2.0)

    def test_neg_returns_databins(self, plain_bins_jack):
        result = -plain_bins_jack
        assert type(result) is DataBins
        np.testing.assert_allclose(result.mean, -plain_bins_jack.mean)
        np.testing.assert_allclose(result.bins, -plain_bins_jack.bins)


################################################################################
# DataStats arithmetic (same statsType)
################################################################################

class TestDataStatsArithmetic:

    @pytest.mark.parametrize("s1,s2", [
        ("jack_datastats", "jack_datastats2"),
        ("boot_datastats", "boot_datastats2"),
    ])
    @pytest.mark.parametrize("op", [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ])
    def test_datastats_op_datastats_returns_datastats(self, request, s1, s2, op):
        a = request.getfixturevalue(s1)
        b = request.getfixturevalue(s2)
        assert type(op(a, b)) is DataStats

    @pytest.mark.parametrize("s1,s2", [
        ("jack_datastats", "jack_datastats2"),
        ("boot_datastats", "boot_datastats2"),
    ])
    def test_add_values(self, request, s1, s2):
        a = request.getfixturevalue(s1)
        b = request.getfixturevalue(s2)
        result = a + b
        np.testing.assert_allclose(result.mean, a.mean + b.mean)
        np.testing.assert_allclose(result.bins, a.bins + b.bins)

    @pytest.mark.parametrize("s,statstype", [
        ("jack_datastats", "jack"),
        ("boot_datastats", "boot"),
    ])
    def test_statstype_preserved(self, request, s, statstype):
        datastats = request.getfixturevalue(s)
        expected_statstype = request.getfixturevalue(statstype)
        assert (datastats + datastats).statsType is expected_statstype

    def test_scalar_mul_returns_datastats(self, jack_datastats):
        result = jack_datastats * 2.0
        assert type(result) is DataStats
        np.testing.assert_allclose(result.mean, jack_datastats.mean * 2.0)
        np.testing.assert_allclose(result.bins, jack_datastats.bins * 2.0)

    def test_scalar_add_returns_datastats(self, jack_datastats):
        result = jack_datastats + 1.0
        assert type(result) is DataStats
        np.testing.assert_allclose(result.mean, jack_datastats.mean + 1.0)

    def test_neg_returns_datastats(self, jack_datastats):
        result = -jack_datastats
        assert type(result) is DataStats
        np.testing.assert_allclose(result.mean, -jack_datastats.mean)


################################################################################
# Mixed DataStats / DataBins arithmetic
################################################################################

class TestMixedArithmetic:

    @pytest.mark.parametrize("s,b", [
        ("jack_datastats", "plain_bins_jack"),
        ("boot_datastats", "plain_bins_boot"),
    ])
    def test_datastats_add_databins_returns_datastats(self, request, s, b):
        assert type(request.getfixturevalue(s) + request.getfixturevalue(b)) is DataStats

    @pytest.mark.parametrize("s,b", [
        ("jack_datastats", "plain_bins_jack"),
        ("boot_datastats", "plain_bins_boot"),
    ])
    def test_databins_add_datastats_returns_datastats(self, request, s, b):
        assert type(request.getfixturevalue(b) + request.getfixturevalue(s)) is DataStats

    @pytest.mark.parametrize("s,b", [
        ("jack_datastats", "plain_bins_jack"),
        ("boot_datastats", "plain_bins_boot"),
    ])
    def test_datastats_add_databins_values(self, request, s, b):
        stats = request.getfixturevalue(s)
        bins = request.getfixturevalue(b)
        result = stats + bins
        np.testing.assert_allclose(result.mean, stats.mean + bins.mean)
        np.testing.assert_allclose(result.bins, stats.bins + bins.bins)

    @pytest.mark.parametrize("s,b", [
        ("jack_datastats", "plain_bins_jack"),
        ("boot_datastats", "plain_bins_boot"),
    ])
    def test_databins_add_datastats_values(self, request, s, b):
        stats = request.getfixturevalue(s)
        bins = request.getfixturevalue(b)
        result = bins + stats
        np.testing.assert_allclose(result.mean, bins.mean + stats.mean)
        np.testing.assert_allclose(result.bins, bins.bins + stats.bins)

    @pytest.mark.parametrize("s,b", [
        ("jack_datastats", "plain_bins_jack"),
        ("boot_datastats", "plain_bins_boot"),
    ])
    def test_add_commutativity(self, request, s, b):
        stats = request.getfixturevalue(s)
        bins = request.getfixturevalue(b)
        r1 = stats + bins
        r2 = bins + stats
        np.testing.assert_allclose(r1.mean, r2.mean)
        np.testing.assert_allclose(r1.bins, r2.bins)

    @pytest.mark.parametrize("s,statstype,b", [
        ("jack_datastats", "jack", "plain_bins_jack"),
        ("boot_datastats", "boot", "plain_bins_boot"),
    ])
    def test_statstype_preserved(self, request, s, statstype, b):
        stats = request.getfixturevalue(s)
        bins = request.getfixturevalue(b)
        expected = request.getfixturevalue(statstype)
        assert (stats + bins).statsType is expected
        assert (bins + stats).statsType is expected

    @pytest.mark.parametrize("op,fn", [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
    ])
    def test_datastats_op_databins_values_jack(self, jack_datastats, plain_bins_jack, op, fn):
        result = fn(jack_datastats, plain_bins_jack)
        np.testing.assert_allclose(result.mean, fn(jack_datastats.mean, plain_bins_jack.mean))
        np.testing.assert_allclose(result.bins, fn(jack_datastats.bins, plain_bins_jack.bins))

    @pytest.mark.parametrize("op,fn", [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
    ])
    def test_datastats_op_databins_values_boot(self, boot_datastats, plain_bins_boot, op, fn):
        result = fn(boot_datastats, plain_bins_boot)
        np.testing.assert_allclose(result.mean, fn(boot_datastats.mean, plain_bins_boot.mean))
        np.testing.assert_allclose(result.bins, fn(boot_datastats.bins, plain_bins_boot.bins))
