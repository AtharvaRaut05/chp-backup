import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from backup_model import VARIANTS  # noqa: E402
from evaluate import (  # noqa: E402
    by_group, calibration, concordance, cross_validate, good_labels, per_day, scores, summary,
)
from simple_models import LogNormalAFT, SimpleModel  # noqa: E402


def test_lognormal_aft_recovers_known_effects_despite_censoring():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=(n, 2))
    true_beta = np.array([3.0, 0.5, -0.3])  # intercept, x1, x2
    t = np.exp(true_beta[0] + x @ true_beta[1:] + 0.6 * rng.normal(size=n))
    limit = np.quantile(t, 0.75)            # a quarter of the durations are cut off ("still going")
    censored = t > limit
    observed = np.minimum(t, limit)

    fit = LogNormalAFT(l2=0.0).fit(x, observed, censored)
    assert fit.beta == pytest.approx(true_beta, abs=0.06)
    assert fit.sigma == pytest.approx(0.6, abs=0.05)
    # treating cut-off durations as if they ended there would bias the effects toward zero
    naive = LogNormalAFT(l2=0.0).fit(x, observed, np.zeros(n, bool))
    assert abs(naive.beta[1]) < abs(fit.beta[1]) - 0.05


def test_lognormal_aft_gradient_matches_numeric():
    from scipy.optimize import approx_fprime
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 2))
    t = np.exp(2 + x[:, 0] + rng.normal(size=50))
    cens = rng.random(50) < 0.3
    model = LogNormalAFT(l2=0.5)
    captured = {}
    import simple_models

    real_minimize = simple_models.minimize

    def spy(fun, x0, **kw):
        captured["fun"] = fun
        return real_minimize(fun, x0, **kw)

    simple_models.minimize = spy
    try:
        model.fit(x, t, cens)
    finally:
        simple_models.minimize = real_minimize
    theta = np.array([1.5, 0.3, -0.2, 0.1])
    analytic = captured["fun"](theta)[1]
    numeric = approx_fprime(theta, lambda th: captured["fun"](th)[0], 1e-6)
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-4)


def test_simple_model_finds_the_planted_effects(data):
    v = VARIANTS[1]  # CHP + pre-crash only
    model = SimpleModel(v.simple_features).fit(data)
    effects = model.duration_effects()
    # the fake data makes backups longer with more lanes blocked and more traffic per open lane
    assert effects.loc["lanes_blocked_now", "per_unit_pct"] > 0
    assert effects.loc["demand_vs_capacity", "per_unit_pct"] > 0
    assert model.backup_odds().loc["blocking_now", "odds_change_per_sd_pct"] > 0


def test_concordance():
    t, e = np.array([1, 2, 3]), np.array([1, 1, 1])
    assert concordance(t, e, np.array([1, 2, 3])) == 1.0
    assert concordance(t, e, np.array([3, 2, 1])) == 0.0
    # censored at 1: we only know it outlasted nothing -> pairs (2,3) only, ranked right
    assert concordance(np.array([1, 2, 3]), np.array([0, 1, 1]), np.array([9, 2, 3])) == 1.0
    # tie: an event and a censored row at the same time; the event counts as shorter
    assert concordance(np.array([5, 5]), np.array([1, 0]), np.array([1, 2])) == 1.0
    assert concordance(np.array([5, 5]), np.array([1, 0]), np.array([2, 1])) == 0.0


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_concordance_matches_the_all_pairs_definition(seed):
    rng = np.random.default_rng(seed)
    n = 700
    t = rng.integers(1, 40, n) * 5.0                       # 5-minute steps, lots of ties
    e = rng.integers(0, 2, n)
    p = np.round(rng.normal(size=n), 1)                    # tied predictions too
    ti, tj = t[:, None], t[None, :]
    comparable = (e[:, None] == 1) & ((ti < tj) | ((ti == tj) & (e[None, :] != 1)))
    right = (p[:, None] < p[None, :]) + 0.5 * (p[:, None] == p[None, :])
    assert concordance(t, e, p) == pytest.approx((right * comparable).sum() / comparable.sum(), abs=1e-12)


@pytest.fixture(scope="module")
def oof(data):
    small = data[data["incident_day"].isin(sorted(data["incident_day"].unique())[:5])]
    return cross_validate(small, VARIANTS[1], log=lambda *_: None)


def test_every_row_gets_one_out_of_sample_prediction(data, oof):
    days = sorted(data["incident_day"].unique())[:5]
    small = data[data["incident_day"].isin(days)]
    assert len(oof) == len(small)
    assert not oof.duplicated(["log_id", "landmark_min"]).any()
    for m in ("rule", "boosted", "simple"):
        assert oof[f"{m}_p"].between(0, 1).all() and (oof[f"{m}_min"] > 0).all()


def test_models_learn_the_planted_signal(oof):
    for m in ("boosted", "simple"):
        s = scores(oof, m)
        assert s["auc"] > 0.7, m
        assert s["c_index"] > 0.55, m


def test_reports_have_the_expected_shape(oof):
    overall = summary(oof, n_boot=20)
    assert list(overall.index) == ["rule", "boosted", "simple", "boosted - rule", "simple - rule"]
    assert " to " in overall.loc["simple", "auc"]  # point estimate with an interval
    groups = by_group(oof)
    assert {"5 min", "10 min", "20 min", "30 min"} <= set(groups.index)
    assert calibration(oof, "simple")["rows"].sum() == len(oof)
    assert per_day(oof)["incidents"].sum() == oof["log_id"].nunique()
    assert (good_labels(oof)["label_adjusted_share"] >= 0.8).all()


def test_paired_difference_is_resampled_by_day(oof):
    # identical predictions -> the difference is exactly 0 in every resample
    same = oof.assign(boosted_p=oof["rule_p"], boosted_min=oof["rule_min"])
    assert summary(same, n_boot=10).loc["boosted - rule", "auc"] == "0.000 (0.000 to 0.000)"


def test_a_fold_without_enough_data_is_skipped_not_fatal(data):
    days = sorted(data["incident_day"].unique())[:4]
    small = data[data["incident_day"].isin(days)]
    # make one day nearly empty of backups: holding out any other day leaves too little validation data
    tiny_day = small[small["incident_day"] == days[-1]].head(6)
    small = pd.concat([small[small["incident_day"] != days[-1]], tiny_day])
    logged = []
    out = cross_validate(small, VARIANTS[1], log=logged.append)
    assert any("skipped" in m for m in logged)
    assert out["incident_day"].nunique() < len(days)
