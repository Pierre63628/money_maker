"""Statistics that guard against selection bias and luck.

References:
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR, MinTRL).
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
- Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting" (CSCV/PBO).
- Politis & Romano (1994), stationary bootstrap.

All Sharpe ratios here are per period (not annualised) unless stated otherwise.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def _moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else float("nan")
    z = (r - r.mean()) / r.std(ddof=0) if r.std(ddof=0) > 0 else np.zeros(n)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # raw kurtosis (normal = 3)
    return float(sr), skew, kurt, n


def sharpe_std_error(sr: float, n: int, skew: float, kurt: float) -> float:
    """Standard error of the per-period Sharpe estimator under non-normal returns."""
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1)
    return math.sqrt(max(var, 1e-18))


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given the sample, accounting for skew and fat tails."""
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr):
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) / sharpe_std_error(sr, n, skew, kurt)))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` independent zero-skill trials."""
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(sharpe_variance, 0.0))
    return sd * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(returns, n_trials: int, trial_sharpes: np.ndarray | None = None) -> dict:
    """PSR against the Sharpe expected from the best of ``n_trials`` unskilled strategies.

    ``trial_sharpes``: per-period Sharpe of every trial, used to estimate their dispersion.
    Without it, the estimator's own sampling variance is used (a lower bound on dispersion).
    """
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr):
        return {"dsr": float("nan"), "sr": sr, "sr_threshold": float("nan"), "n_trials": n_trials, "n_obs": n}
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.nanvar(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        variance = sharpe_std_error(sr, n, skew, kurt) ** 2
    threshold = expected_max_sharpe(n_trials, variance)
    dsr = float(norm.cdf((sr - threshold) / sharpe_std_error(sr, n, skew, kurt)))
    return {"dsr": dsr, "sr": sr, "sr_threshold": threshold, "n_trials": n_trials, "n_obs": n,
            "skew": skew, "kurtosis": kurt}


def minimum_track_record_length(returns, sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Observations needed for PSR(sr_benchmark) to reach ``confidence``."""
    sr, skew, kurt, _ = _moments(returns)
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("inf")
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (norm.ppf(confidence) / (sr - sr_benchmark)) ** 2


@njit(cache=True)
def _stationary_indices(starts, new_block, n_obs):
    n_samples = starts.shape[0]
    idx = np.empty((n_samples, n_obs), dtype=np.int64)
    for s in range(n_samples):
        cur = 0
        for t in range(n_obs):
            cur = starts[s, t] if new_block[s, t] else (cur + 1) % n_obs
            idx[s, t] = cur
    return idx


def stationary_bootstrap_indices(n_obs: int, n_samples: int, mean_block: float, seed: int = 0) -> np.ndarray:
    """Politis-Romano stationary bootstrap: random blocks with geometric lengths, wrapping around."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs, size=(n_samples, n_obs))
    new_block = rng.random((n_samples, n_obs)) < 1.0 / mean_block
    new_block[:, 0] = True
    return _stationary_indices(starts, new_block, n_obs)


def bootstrap_performance(daily_returns: pd.Series, n_samples: int = 2000, mean_block: float = 10.0,
                          seed: int = 0, periods_per_year: float = 365.0) -> dict:
    """Distribution of annualised Sharpe, CAGR and max drawdown under a stationary bootstrap."""
    r = daily_returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 30:
        raise ValueError("need at least 30 daily returns")
    idx = stationary_bootstrap_indices(n, n_samples, mean_block, seed)
    samples = r[idx]
    mean = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mean / sd * math.sqrt(periods_per_year), np.nan)
    growth = np.cumprod(1 + samples, axis=1)
    years = n / periods_per_year
    cagr = np.where(growth[:, -1] > 0, growth[:, -1] ** (1 / years) - 1, -1.0)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_samples, 1)), growth], axis=1), axis=1)[:, 1:]
    mdd = (growth / peak - 1).min(axis=1)
    q = [0.05, 0.5, 0.95]
    return {
        "sharpe_q05_q50_q95": tuple(np.nanquantile(sharpe, q)),
        "cagr_q05_q50_q95": tuple(np.quantile(cagr, q)),
        "max_drawdown_q05_q50_q95": tuple(np.quantile(mdd, q)),
        "prob_sharpe_le_0": float(np.mean(sharpe <= 0)),
        "prob_loss": float(np.mean(growth[:, -1] < 1)),
    }


def pbo_cscv(performance: pd.DataFrame, n_blocks: int = 16, max_combinations: int = 5000, seed: int = 0) -> dict:
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation.

    ``performance``: rows = periods (e.g. days), columns = candidate configurations,
    values = per-period returns. For each split of the blocks into equal IS/OOS halves,
    pick the best IS configuration and record its relative OOS rank. PBO = P(rank below median).
    """
    m = performance.dropna(how="all").fillna(0.0).to_numpy(dtype=float)
    t, n_cfg = m.shape
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    block_len = t // n_blocks
    if block_len < 5:
        raise ValueError("not enough observations per block")
    blocks = m[: block_len * n_blocks].reshape(n_blocks, block_len, n_cfg)
    sums = blocks.sum(axis=1)
    sq = (blocks ** 2).sum(axis=1)

    all_combos = list(combinations(range(n_blocks), n_blocks // 2))
    if len(all_combos) > max_combinations:
        rng = np.random.default_rng(seed)
        all_combos = [all_combos[i] for i in rng.choice(len(all_combos), max_combinations, replace=False)]

    def sharpe(block_ids):
        cnt = block_len * len(block_ids)
        s = sums[list(block_ids)].sum(axis=0)
        q = sq[list(block_ids)].sum(axis=0)
        mean = s / cnt
        var = (q - cnt * mean ** 2) / (cnt - 1)
        return np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-30)), 0.0)

    logits, is_best, oos_of_best = [], [], []
    everything = set(range(n_blocks))
    for combo in all_combos:
        sr_is = sharpe(combo)
        sr_oos = sharpe(tuple(everything - set(combo)))
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1) + 1
        omega = rank / (n_cfg + 1)
        logits.append(math.log(omega / (1 - omega)))
        is_best.append(sr_is[best])
        oos_of_best.append(sr_oos[best])
    logits = np.array(logits)
    is_best, oos_of_best = np.array(is_best), np.array(oos_of_best)
    slope = float(np.polyfit(is_best, oos_of_best, 1)[0]) if len(set(is_best)) > 1 else float("nan")
    return {
        "pbo": float(np.mean(logits <= 0)),
        "n_splits": len(logits),
        "median_logit": float(np.median(logits)),
        "prob_oos_sharpe_le_0": float(np.mean(oos_of_best <= 0)),
        "is_to_oos_slope": slope,
        "mean_is_sharpe_of_best": float(is_best.mean()),
        "mean_oos_sharpe_of_best": float(oos_of_best.mean()),
    }
