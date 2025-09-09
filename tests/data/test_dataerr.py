"""Tests for arithmetic between DataErr and DataStats.

Error tolerance: with fixed seeds results are deterministic, but finite-sample
noise in _to_datastats means err only approximates the analytical formula.
RTOL_ERR is set conservatively based on 1/sqrt(2*num_bins) noise estimates:
  jackknife (NUM_CONFIG bins): ~1/sqrt(2*200) ≈ 5%
  bootstrap (NUM_BOOT_BINS bins): ~1/sqrt(2*1000) ≈ 2%
A 15% tolerance gives a comfortable margin.

The mean/sigma parameters are chosen so relative errors are ≈5% for both
objects.  This ensures the first-order error-propagation formula used in the
mul/div tests is accurate to <0.1%, far below RTOL_ERR.  Using large relative
errors (>10%) breaks the formula for division because higher-order 1/Y tail
terms dominate the variance and the ratio distribution becomes heavy-tailed.
"""
import numpy as np
import pytest
from labc.stats import StatsType
from labc.data import DataErr, DataStats


NUM_CONFIG    = 200
NUM_BOOT_BINS = 1000
T             = 1
RTOL_ERR      = 0.15   # tolerance for finite-sample noise in err propagation

MEAN_DS  = 1.2345
SIGMA_DS = 0.0617   # ≈ 5 % of MEAN_DS  — keeps linear formula accurate
MEAN_DE  = 4.97679
SIGMA_DE = 0.2489   # ≈ 5 % of MEAN_DE


# ── helpers ──────────────────────────────────────────────────────────────────

def make_raw(mean_val, sigma, num_config, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(mean_val, sigma * np.sqrt(num_config),
                     size=(num_config, T))
    raw += mean_val - np.mean(raw, 0)
    return raw


def expected_err_add_sub(s1, s2):
    return np.sqrt(s1**2 + s2**2)


def expected_err_mul(m1, s1, m2, s2):
    return abs(m1 * m2) * np.sqrt((s1/m1)**2 + (s2/m2)**2)


def expected_err_div(m1, s1, m2, s2):
    return abs(m1 / m2) * np.sqrt((s1/m1)**2 + (s2/m2)**2)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def jack():
    return StatsType.Jack(num_config=NUM_CONFIG)


@pytest.fixture
def boot():
    return StatsType.Boot(num_config=NUM_CONFIG, num_bins=NUM_BOOT_BINS)


@pytest.fixture
def jack_datastats(jack):
    mean, _, bins = jack.generate_stats(make_raw(MEAN_DS, SIGMA_DS, NUM_CONFIG))
    return DataStats(mean, bins, jack)


@pytest.fixture
def boot_datastats(boot):
    mean, _, bins = boot.generate_stats(make_raw(MEAN_DS, SIGMA_DS, NUM_CONFIG))
    return DataStats(mean, bins, boot)


@pytest.fixture
def de():
    # seed=0 makes _to_datastats deterministic
    return DataErr(np.full(T, MEAN_DE), np.full(T, SIGMA_DE), seed=1)


# ── result type ──────────────────────────────────────────────────────────────

class TestDataErrDataStatsType:

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_de_op_ds_returns_datastats(self, request, de, ds_fix, op):
        ds = request.getfixturevalue(ds_fix)
        result = getattr(de, f"__{op}__")(ds)
        assert type(result) is DataStats

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_ds_op_de_returns_datastats(self, request, de, ds_fix, op):
        ds = request.getfixturevalue(ds_fix)
        result = getattr(ds, f"__{op}__")(de)
        assert type(result) is DataStats


# ── mean propagation (exact) ─────────────────────────────────────────────────

class TestDataErrDataStatsMean:

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_add_mean(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        np.testing.assert_allclose((de + ds).mean, de.mean + ds.mean, rtol=1e-12)
        np.testing.assert_allclose((ds + de).mean, ds.mean + de.mean, rtol=1e-12)

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_sub_mean(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        np.testing.assert_allclose((de - ds).mean, de.mean - ds.mean, rtol=1e-12)
        np.testing.assert_allclose((ds - de).mean, ds.mean - de.mean, rtol=1e-12)

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_mul_mean(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        np.testing.assert_allclose((de * ds).mean, de.mean * ds.mean, rtol=1e-12)

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_div_mean(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        np.testing.assert_allclose((de / ds).mean, de.mean / ds.mean, rtol=1e-12)
        np.testing.assert_allclose((ds / de).mean, ds.mean / de.mean, rtol=1e-12)


# ── error propagation (analytical, within RTOL_ERR) ──────────────────────────

class TestDataErrDataStatsErr:

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_add_err(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        expected = expected_err_add_sub(de.err, ds.err)
        np.testing.assert_allclose((de + ds).err, expected, rtol=RTOL_ERR)
        np.testing.assert_allclose((ds + de).err, expected, rtol=RTOL_ERR)

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_sub_err(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        expected = expected_err_add_sub(de.err, ds.err)
        np.testing.assert_allclose((de - ds).err, expected, rtol=RTOL_ERR)
        np.testing.assert_allclose((ds - de).err, expected, rtol=RTOL_ERR)

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_mul_err(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        expected_de_ds = expected_err_mul(de.mean, de.err, ds.mean, ds.err)
        expected_ds_de = expected_err_mul(ds.mean, ds.err, de.mean, de.err)
        np.testing.assert_allclose((de * ds).err, expected_de_ds, rtol=RTOL_ERR)
        np.testing.assert_allclose((ds * de).err, expected_ds_de, rtol=RTOL_ERR)

    @pytest.mark.parametrize("ds_fix", ["jack_datastats", "boot_datastats"])
    def test_div_err(self, request, de, ds_fix):
        ds = request.getfixturevalue(ds_fix)
        expected_de_ds = expected_err_div(de.mean, de.err, ds.mean, ds.err)
        expected_ds_de = expected_err_div(ds.mean, ds.err, de.mean, de.err)
        np.testing.assert_allclose((de / ds).err, expected_de_ds, rtol=RTOL_ERR)
        np.testing.assert_allclose((ds / de).err, expected_ds_de, rtol=RTOL_ERR)


# ── statsType preserved ───────────────────────────────────────────────────────

class TestDataErrDataStatsType2:

    @pytest.mark.parametrize("ds_fix,st_fix", [
        ("jack_datastats", "jack"),
        ("boot_datastats", "boot"),
    ])
    def test_statstype_preserved(self, request, de, ds_fix, st_fix):
        ds = request.getfixturevalue(ds_fix)
        st = request.getfixturevalue(st_fix)
        assert (de + ds).statsType is st
        assert (ds + de).statsType is st