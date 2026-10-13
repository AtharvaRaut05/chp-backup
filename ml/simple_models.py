"""Simple, explainable models on a handful of features.

  will it back up?   logistic regression
  how much longer?   log-normal survival model (accelerated failure time):
                       log(minutes) = b0 + b1*x1 + ... + noise,  noise ~ Normal(0, sigma)
                     fitted by maximum likelihood, so backups still going when the data ended
                     ("at least N minutes") count as what they are: a lower bound.

Each coefficient reads as "this much longer or shorter": exp(b) - 1 is the % change in the
typical (median) remaining duration per unit of the feature.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from backup_model import more_backup


class Standardize:
    """Fill missing values with the training median, then scale to mean 0, SD 1."""

    def fit(self, x: pd.DataFrame) -> Standardize:
        self.columns = list(x.columns)
        self.median = x.median()
        filled = x.fillna(self.median).fillna(0)
        self.mean, self.std = filled.mean(), filled.std().replace(0, 1).fillna(1)
        return self

    def __call__(self, x: pd.DataFrame) -> np.ndarray:
        filled = x[self.columns].fillna(self.median).fillna(0)
        return ((filled - self.mean) / self.std).to_numpy(float)


class LogNormalAFT:
    """Log-normal survival regression with an L2 penalty (not on the intercept)."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, x: np.ndarray, minutes: np.ndarray, censored: np.ndarray) -> LogNormalAFT:
        X = np.column_stack([np.ones(len(x)), x])
        y, cens = np.log(np.asarray(minutes, float)), np.asarray(censored, bool)
        p = X.shape[1]

        def nll(theta):
            beta, log_s = theta[:p], theta[p]
            s = np.exp(log_s)
            z = (y - X @ beta) / s
            # uncensored: log density of log(T); censored: log P(log T > y)
            ll = np.where(cens, norm.logsf(z), norm.logpdf(z) - log_s)
            hazard = np.exp(norm.logpdf(z) - norm.logsf(z))  # pdf / survival, stable in the tails
            d_z = np.where(cens, -hazard, -z)                  # d ll / d z
            grad_beta = -(X.T @ (d_z * (-1 / s)))
            grad_log_s = -np.sum(d_z * (-z) + np.where(cens, 0.0, -1.0))
            pen = self.l2 * np.r_[0.0, beta[1:]]
            return -ll.sum() + 0.5 * self.l2 * np.sum(beta[1:] ** 2), np.r_[grad_beta + pen, grad_log_s]

        start = np.r_[y.mean(), np.zeros(p - 1), np.log(y.std() + 1e-6)]
        res = minimize(nll, start, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        if not np.isfinite(res.fun):
            raise RuntimeError(f"survival model did not fit: {res.message}")
        if not res.success:
            warnings.warn(f"survival model stopped early: {res.message}")
        self.beta, self.sigma = res.x[:p], float(np.exp(res.x[p]))
        return self

    def predict_median(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.beta[0] + x @ self.beta[1:])


class SimpleModel:
    """Logistic regression for "will it back up?" + log-normal survival for "how much longer?"."""

    def __init__(self, columns: list[str], l2: float = 1.0):
        self.columns, self.l2 = list(columns), l2

    def fit(self, train: pd.DataFrame) -> SimpleModel:
        self.prep = Standardize().fit(train[self.columns])
        more = more_backup(train)
        self.classifier = LogisticRegression(C=1.0 / self.l2, max_iter=2000).fit(self.prep(train[self.columns]), more)
        pos = train[more]
        self.survival = LogNormalAFT(self.l2).fit(self.prep(pos[self.columns]),
                                                  pos["remaining_minutes"], pos["is_censored"] == 1)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self.prep(df[self.columns])
        return pd.DataFrame({"p_more_backup": self.classifier.predict_proba(x)[:, 1],
                             "minutes_if_more": self.survival.predict_median(x)}, index=df.index)

    def duration_effects(self) -> pd.DataFrame:
        """% change in the typical remaining duration per unit and per standard deviation of each feature."""
        b = pd.Series(self.survival.beta[1:], index=self.columns)
        out = pd.DataFrame({
            "per_unit_pct": (np.exp(b / self.prep.std) - 1) * 100,
            "per_sd_pct": (np.exp(b) - 1) * 100,
            "unit_sd": self.prep.std,
        })
        return out.reindex(out["per_sd_pct"].abs().sort_values(ascending=False).index).round(1)

    def backup_odds(self) -> pd.DataFrame:
        """How each feature changes the odds of more backup, per standard deviation."""
        b = pd.Series(self.classifier.coef_[0], index=self.columns)
        out = pd.DataFrame({"odds_change_per_sd_pct": (np.exp(b) - 1) * 100})
        return out.reindex(out["odds_change_per_sd_pct"].abs().sort_values(ascending=False).index).round(1)
