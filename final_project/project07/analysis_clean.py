"""
STATS 531 Final Project analysis.

Dataset: monthly U.S. pneumonia-and-influenza deaths grouped by race
(`Group_By_Race.csv`, from CDC WONDER: Underlying Cause of Death, 1999-2020).

Mirrors the W21 final project "To The Moon or Not" workflow
(https://ionides.github.io/531w21/final_project/project06/blinded.html),
with a seasonal SEIR POMP replacing the Heston stochastic-volatility POMP as
the mechanistic analog appropriate for monthly mortality counts.

Workflow:
  1. EDA (time series, seasonality, ACF/PACF, AR(1) baseline)
  2. ARMA(p,q) AIC grid benchmark + residual diagnostics
  3. GARCH(1,1) and GARCH(4,2) on log-differences
  4. Seasonal SEIR POMP (Breto-style env noise):
        a. simulation study for identifiability
        b. local IF2 search
        c. global IF2 search from randomized starts
        d. simulations + probe diagnostics at MLE
        e. profile likelihood with re-optimization
  5. Model comparison (ARMA vs GARCH vs SEIR-POMP)
  6. Across-race baseline comparison

Run:
    python analysis.py --race "White"
    python analysis.py --quick
    python analysis.py --force-recompute
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
from arch import arch_model
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pypomp as pp
from pypomp import RWSigma
from pypomp import logmeanexp

DATA_FILE = "Group_By_Race.csv"
CACHE_DIR = "cache_race_model"
EPS = 1e-9


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RunConfig:
    race: str = "White"
    quick: bool = False
    force_recompute: bool = False

    @property
    def mif_J(self):
        return 300 if self.quick else 1500

    @property
    def mif_M(self):
        return 20 if self.quick else 100

    @property
    def pf_J(self):
        return 400 if self.quick else 2000

    @property
    def pf_reps(self):
        return 3 if self.quick else 10

    @property
    def n_local(self):
        return 3 if self.quick else 10

    @property
    def n_global(self):
        return 6 if self.quick else 24

    @property
    def profile_grid_n(self):
        return 5 if self.quick else 12

    @property
    def sim_nsim(self):
        return 40 if self.quick else 200


# =============================================================================
# Cache utilities
# =============================================================================

def cache_path(tag: str, race: str) -> str:
    return os.path.join(CACHE_DIR, f"{tag}_{race.replace(' ', '_')}.pkl")


def maybe_load(path: str, force: bool):
    if force or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cache(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def safe_name(race: str) -> str:
    return race.replace(" ", "_").replace("/", "_")


# =============================================================================
# Data handling
# =============================================================================

def load_data(path: str = DATA_FILE):
    df = pd.read_csv(path)
    missing = {"Race", "Month", "Deaths"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    return df


def prepare_race_data(df, race: str):
    x = df.loc[df["Race"] == race, ["Month", "Deaths"]].copy()
    if x.empty:
        avail = sorted(df["Race"].dropna().unique().tolist())
        raise ValueError(f"Race '{race}' not available. Choices: {avail}")

    # Normalize CDC WONDER month strings ("Jan 1999", "Jan., 1999", ...).
    norm_month = (
        x["Month"].astype(str).str.strip()
        .str.replace(r"\.", "", regex=True)
        .str.replace(r",", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    date = pd.to_datetime(norm_month, format="%b %Y", errors="coerce")
    date = date.fillna(pd.to_datetime(norm_month, format="%B %Y", errors="coerce"))
    date = date.fillna(pd.to_datetime(norm_month, errors="coerce"))
    if date.isna().any():
        bad = x["Month"][date.isna()].head().tolist()
        raise ValueError(f"Could not parse month strings: {bad}")
    x["date"] = date

    x = x.sort_values("date").reset_index(drop=True)
    x["time"] = np.arange(1, len(x) + 1, dtype=float)
    x["month_num"] = x["date"].dt.month
    x["year"] = x["date"].dt.year

    ys = x[["Deaths"]].rename(columns={"Deaths": "deaths"}).astype(float)
    ys.index = x["time"].to_numpy(dtype=float)
    return x, ys


# =============================================================================
# 1. EDA
# =============================================================================

def fit_ar1_baseline(logy):
    y1 = logy[1:]
    X = np.column_stack([np.ones(len(logy) - 1), logy[:-1]])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((y1 - X @ beta) ** 2)))
    return {"intercept": float(beta[0]), "phi": float(beta[1]), "rmse": rmse}


def run_eda(race_df, race: str) -> Dict[str, float]:
    y = race_df["Deaths"].to_numpy(dtype=float)
    logy = np.log(np.maximum(y, 1.0))
    dlogy = np.diff(logy)
    acf_log = sm.tsa.stattools.acf(logy, nlags=36, fft=True)
    pacf_log = sm.tsa.stattools.pacf(logy, nlags=36, method="ywm")
    ar1 = fit_ar1_baseline(logy)

    tag = safe_name(race)
    n = len(logy)
    ci = 1.96 / np.sqrt(n)

    # 1. Raw + log time series.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(race_df["date"], y, color="#1f77b4", lw=1.8)
    axes[0].set_ylabel("Monthly deaths")
    axes[0].set_title(f"Monthly pneumonia-and-influenza deaths ({race})")
    axes[1].plot(race_df["date"], logy, color="#d62728", lw=1.4)
    axes[1].set_ylabel("log(deaths)")
    axes[1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"eda_timeseries_{tag}.png", dpi=150)
    plt.close()

    # 2. Seasonality.
    seas_mean = race_df.groupby("month_num")["Deaths"].mean()
    seas_sd = race_df.groupby("month_num")["Deaths"].std()
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        seas_mean.index, seas_mean.values, yerr=seas_sd.values,
        marker="o", capsize=3, color="#1f77b4",
    )
    plt.xticks(range(1, 13))
    plt.title(f"Seasonal mean deaths ({race})")
    plt.xlabel("Month of year")
    plt.ylabel("Mean (+/- 1 SD) deaths")
    plt.tight_layout()
    plt.savefig(f"eda_seasonality_{tag}.png", dpi=150)
    plt.close()

    # 3. ACF/PACF.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, vals, name in [(axes[0], acf_log, "ACF"), (axes[1], pacf_log, "PACF")]:
        ax.stem(np.arange(len(vals)), vals, basefmt=" ")
        ax.axhline(ci, ls="--", color="gray")
        ax.axhline(-ci, ls="--", color="gray")
        ax.set_title(f"{name} of log deaths ({race})")
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel(name)
    plt.tight_layout()
    plt.savefig(f"eda_acf_pacf_{tag}.png", dpi=150)
    plt.close()

    # 4. First-differenced log series.
    plt.figure(figsize=(11, 3.5))
    plt.plot(race_df["date"].iloc[1:], dlogy, color="#2ca02c", lw=1.2)
    plt.axhline(0, color="black", lw=0.6)
    plt.title(f"First-differenced log deaths ({race})")
    plt.xlabel("Date")
    plt.ylabel("Δ log(deaths)")
    plt.tight_layout()
    plt.savefig(f"eda_logdiff_{tag}.png", dpi=150)
    plt.close()

    covid_mask = race_df["date"] >= pd.Timestamp("2020-03-01")
    pre = float(race_df.loc[~covid_mask, "Deaths"].mean()) if (~covid_mask).any() else np.nan
    covid = float(race_df.loc[covid_mask, "Deaths"].mean()) if covid_mask.any() else np.nan

    return {
        "n_months": int(len(y)),
        "mean_deaths": float(y.mean()),
        "sd_deaths": float(y.std()),
        "ar1_phi": ar1["phi"],
        "ar1_rmse": ar1["rmse"],
        "acf12": float(acf_log[12]),
        "pacf1": float(pacf_log[1]),
        "covid_ratio": covid / max(pre, 1.0),
    }


# =============================================================================
# 2. ARMA benchmark
# =============================================================================

def _qq_plot(ax, resid, title):
    rs = np.sort(resid)
    probs = (np.arange(1, len(rs) + 1) - 0.5) / len(rs)
    q_theo = norm.ppf(probs)
    standardized = (rs - rs.mean()) / rs.std()

    ax.plot(q_theo, standardized, "o", ms=3, color="#d62728")
    lim = max(abs(q_theo.min()), abs(q_theo.max()))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Theoretical")
    ax.set_ylabel("Standardized")


def run_arma_benchmark(race_df, race: str):
    tag = safe_name(race)
    logy = np.log(np.maximum(race_df["Deaths"].to_numpy(dtype=float), 1.0))
    y = logy - logy.mean()

    P_MAX, Q_MAX = 4, 4
    aic = np.full((P_MAX + 1, Q_MAX + 1), np.nan)
    best = (np.inf, None, None)
    best_res = None

    for p in range(P_MAX + 1):
        for q in range(Q_MAX + 1):
            try:
                res = sm.tsa.ARIMA(y, order=(p, 0, q), trend="c").fit()
                aic[p, q] = float(res.aic)
                if res.aic < best[0]:
                    best = (float(res.aic), p, q)
                    best_res = res
            except Exception:
                continue

    if best_res is None:
        return None

    aic_df = pd.DataFrame(
        aic,
        index=[f"AR{p}" for p in range(P_MAX + 1)],
        columns=[f"MA{q}" for q in range(Q_MAX + 1)],
    )
    aic_df.to_csv(f"arma_aic_table_{tag}.csv", float_format="%.2f")

    resid = np.asarray(best_res.resid, dtype=float)
    acf_r = sm.tsa.stattools.acf(resid, nlags=36, fft=True)
    ci = 1.96 / np.sqrt(len(resid))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(resid, color="#1f77b4", lw=1.0)
    axes[0, 0].axhline(0, color="k", lw=0.6)
    axes[0, 0].set_title(f"Residuals ARMA({best[1]},{best[2]})")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("residual")

    axes[0, 1].stem(np.arange(len(acf_r)), acf_r, basefmt=" ")
    axes[0, 1].axhline(ci, ls="--", color="gray")
    axes[0, 1].axhline(-ci, ls="--", color="gray")
    axes[0, 1].set_title("ACF of residuals")
    axes[0, 1].set_xlabel("Lag")

    _qq_plot(axes[1, 0], resid, "Normal Q-Q")

    # Inverse roots (inside unit circle -> causal / invertible).
    ar_params = np.r_[1, -np.asarray(best_res.polynomial_reduced_ar[1:])]
    ma_params = np.r_[1, np.asarray(best_res.polynomial_reduced_ma[1:])]
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[1, 1].plot(np.cos(theta), np.sin(theta), "k-", lw=0.8)

    for params, marker, color, label in [
        (ar_params, "o", "#1f77b4", "inverse AR"),
        (ma_params, "x", "#d62728", "inverse MA"),
    ]:
        if len(params) > 1:
            inv = 1 / np.roots(params)
            axes[1, 1].plot(inv.real, inv.imag, marker, color=color, label=label)

    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlim(-1.5, 1.5)
    axes[1, 1].set_ylim(-1.5, 1.5)
    axes[1, 1].axhline(0, color="gray", lw=0.5)
    axes[1, 1].axvline(0, color="gray", lw=0.5)
    axes[1, 1].set_title("Inverse roots")
    axes[1, 1].legend(loc="best", fontsize=8)

    plt.suptitle(f"ARMA({best[1]},{best[2]}) diagnostics ({race})")
    plt.tight_layout()
    plt.savefig(f"arma_diagnostics_{tag}.png", dpi=150)
    plt.close()

    return {
        "p": best[1],
        "q": best[2],
        "aic": float(best[0]),
        "loglik": float(best_res.llf),
        "n_params": int(best[1]) + int(best[2]) + 2,
        "aic_table": aic_df,
    }


# =============================================================================
# 3. GARCH benchmarks
# =============================================================================

def run_garch_benchmarks(race_df, race: str):
    tag = safe_name(race)
    logy = np.log(np.maximum(race_df["Deaths"].to_numpy(dtype=float), 1.0))
    dlog = np.diff(logy)
    r = 100.0 * (dlog - dlog.mean())

    results = {}
    for name, (p, q) in [("GARCH(1,1)", (1, 1)), ("GARCH(4,2)", (4, 2))]:
        try:
            model = arch_model(r, mean="Zero", vol="GARCH", p=p, q=q, dist="Normal")
            res = model.fit(disp="off")
            results[name] = {
                "loglik": float(res.loglikelihood),
                "aic": float(res.aic),
                "n_params": int(res.num_params),
                "params": {k: float(v) for k, v in res.params.items()},
                "std_resid": np.asarray(res.std_resid),
            }
        except Exception as e:
            print(f"[warn] {name} fit failed: {e}")

    if not results:
        return None

    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 7), squeeze=False)
    for j, (name, info) in enumerate(results.items()):
        sr = np.asarray(info["std_resid"], dtype=float)
        acf_sr = sm.tsa.stattools.acf(sr, nlags=24, fft=True)
        ci = 1.96 / np.sqrt(len(sr))

        axes[0, j].stem(np.arange(len(acf_sr)), acf_sr, basefmt=" ")
        axes[0, j].axhline(ci, ls="--", color="gray")
        axes[0, j].axhline(-ci, ls="--", color="gray")
        axes[0, j].set_title(f"{name} std-resid ACF (logLik={info['loglik']:.1f})")
        axes[0, j].set_xlabel("Lag")

        _qq_plot(axes[1, j], sr, f"{name} Q-Q of std-resid")

    plt.suptitle(f"GARCH diagnostics ({race})")
    plt.tight_layout()
    plt.savefig(f"garch_diagnostics_{tag}.png", dpi=150)
    plt.close()

    return {
        name: {k: v for k, v in info.items() if k != "std_resid"}
        for name, info in results.items()
    }


# =============================================================================
# 4. Seasonal SEIR POMP
# =============================================================================

NSTEP_PER_MONTH = 20
# Month index of March 2020 in a series that starts at Jan 1999 with t=1.
# March 2020 = 1 + 21*12 + 2 = 255.
COVID_T_INDEX = 255.0
ESTIMATED_PARAMS = [
    "b0", "b1", "b2", "mu_EI", "mu_IR", "mu_RS",   # ADD mu_RS
    "rho", "k", "sigma_env", "eta", "iota", "covid_shift",
]


def nbinom_logpmf(x, r, mu):
    x = jnp.maximum(x, 0.0)
    r = jnp.maximum(r, EPS)
    mu = jnp.maximum(mu, EPS)
    return (
        jsp.special.gammaln(x + r)
        - jsp.special.gammaln(r)
        - jsp.special.gammaln(x + 1.0)
        + r * jnp.log(r / (r + mu))
        + x * jnp.log(mu / (r + mu))
    )


def rinit(theta_, key, covars, t0):
    N = jnp.maximum(theta_["N"], 1.0)
    eta = jnp.clip(theta_["eta"], 1e-4, 1.0 - 1e-4)
    iota = jnp.maximum(theta_["iota"], 0.0)

    S0 = eta * N
    I0 = iota
    E0 = iota
    R0 = jnp.maximum(N - S0 - E0 - I0, 0.0)

    return {"S": S0, "E": E0, "I": I0, "R": R0, "C": jnp.asarray(0.0)}


def rproc(X_, theta_, key, covars, t, dt):
    N = jnp.maximum(theta_["N"], 1.0)
    b0 = jnp.maximum(theta_["b0"], EPS)
    b1 = theta_["b1"]
    b2 = theta_["b2"]
    mu_EI = jnp.maximum(theta_["mu_EI"], EPS)
    mu_IR = jnp.maximum(theta_["mu_IR"], EPS)
    mu_RS = jnp.maximum(theta_["mu_RS"], EPS)   # NEW: waning immunity rate
    sigma_env = jnp.maximum(theta_["sigma_env"], EPS)

    S = jnp.maximum(X_["S"], 0.0)
    E = jnp.maximum(X_["E"], 0.0)
    I = jnp.maximum(X_["I"], 0.0)
    R = jnp.maximum(X_["R"], 0.0)
    C = X_["C"]

    omega = 2.0 * jnp.pi * t / 12.0
    seasonal = jnp.maximum(1.0 + b1 * jnp.sin(omega) + b2 * jnp.cos(omega), 0.05)

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)   # NEW: 5 keys, one for R->S
    dW = jax.random.gamma(k1, dt / (sigma_env ** 2)) * (sigma_env ** 2)

    covid_on = jnp.where(t >= COVID_T_INDEX, 1.0, 0.0)
    covid_mult = jnp.exp(theta_["covid_shift"] * covid_on)

    beta_t = b0 * seasonal * covid_mult
    lam = beta_t * I / N

    p_SE = 1.0 - jnp.exp(-lam * dW)
    p_EI = 1.0 - jnp.exp(-mu_EI * dt)
    p_IR = 1.0 - jnp.exp(-mu_IR * dt)
    p_RS = 1.0 - jnp.exp(-mu_RS * dt)   # NEW

    dSE = jnp.clip(
        S * p_SE + jax.random.normal(k2) * jnp.sqrt(S * p_SE * (1 - p_SE) + EPS),
        0.0, S,
    )
    dEI = jnp.clip(
        E * p_EI + jax.random.normal(k3) * jnp.sqrt(E * p_EI * (1 - p_EI) + EPS),
        0.0, E,
    )
    dIR = jnp.clip(
        I * p_IR + jax.random.normal(k4) * jnp.sqrt(I * p_IR * (1 - p_IR) + EPS),
        0.0, I,
    )
    dRS = jnp.clip(   # NEW: recovered individuals losing immunity
        R * p_RS + jax.random.normal(k5) * jnp.sqrt(R * p_RS * (1 - p_RS) + EPS),
        0.0, R,
    )

    return {
        "S": S - dSE + dRS,
        "E": E + dSE - dEI,
        "I": I + dEI - dIR,
        "R": R + dIR - dRS,
        "C": C + dIR,
    }

def dmeas(Y_, X_, theta_, covars, t):
    rho = jnp.clip(theta_["rho"], 1e-6, 1.0)
    k = jnp.maximum(theta_["k"], EPS)
    mu_y = jnp.maximum(rho * X_["C"] + EPS, EPS)
    return nbinom_logpmf(jnp.maximum(Y_["deaths"], 0.0), 1.0 / k, mu_y)


def rmeas(X_, theta_, key, covars, t):
    rho = jnp.clip(theta_["rho"], 1e-6, 1.0)
    k = jnp.maximum(theta_["k"], EPS)
    mu_y = jnp.maximum(rho * X_["C"] + EPS, EPS)
    r = 1.0 / k

    k1, k2 = jax.random.split(key)
    rate = jax.random.gamma(k1, r) * (mu_y / r)
    return jnp.array([jax.random.poisson(k2, rate)])


def init_theta_from_data(race_df_or_ys, mock: bool = False) -> Dict[str, float]:
    if isinstance(race_df_or_ys, pd.DataFrame) and "Deaths" in race_df_or_ys.columns:
        y = race_df_or_ys["Deaths"].to_numpy(dtype=float)
    else:
        y = race_df_or_ys.iloc[:, 0].to_numpy(dtype=float)
    
    y_mean = max(float(y.mean()), 1.0)

    if mock:
        return {
            "N": 1_000_000.0,
            "b0": 8.0, "b1": 0.35, "b2": 0.1,
            "mu_EI": 2.0, "mu_IR": 1.0,
            "mu_RS": 0.1,                      # NEW: ~10-month immunity
            "rho": 0.05,
            "k": 0.08, "sigma_env": 0.3,
            "eta": 0.3,
            "iota": 20_000.0,
            "covid_shift": -0.2,
        }

    return {
        "N": 1_000_000.0,
        "b0": 3.5, "b1": 0.4, "b2": 0.2,
        "mu_EI": 1.8, "mu_IR": 1.0,
        "mu_RS": 0.1,                          # NEW
        "rho": 0.005, "k": 0.1, "sigma_env": 0.15,
        "eta": 0.4,
        "iota": 1000.0,
        "covid_shift": 0.0,
    }

def build_seir_model(ys, theta):
    theta_in = theta if isinstance(theta, list) else [theta]
    return pp.Pomp(
        ys=ys,
        theta=theta_in,
        statenames=["S", "E", "I", "R", "C"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ydim=1,
        nstep=NSTEP_PER_MONTH,
        accumvars=["C"],
    )


def make_rw_sigma(scale_reg: float = 0.02, scale_ivp: float = 0.05):
    regular = ["b0", "b1", "b2", "mu_EI", "mu_IR", "mu_RS",   # ADD mu_RS
           "rho", "k", "sigma_env", "covid_shift"]
    sigmas = {p: scale_reg for p in regular}
    sigmas["N"] = 0.0  # fixed
    sigmas["eta"] = scale_ivp
    sigmas["iota"] = scale_ivp
    return RWSigma(sigmas=sigmas, init_names=["eta", "iota"])


# --- IF2 and pfilter wrappers ---

def _pfilter_batched_logliks(mod, key, J: int, reps: int):
    """Run pfilter once on a model that may carry multiple thetas, and return
    a 1-D numpy array of per-theta log-likelihoods (logmeanexp over reps).

    This avoids the JAX JIT-recompile trap of building a fresh Pomp per theta.
    """
    mod.pfilter(J=J, key=key, reps=reps, ESS=False)
    res = mod.results_history.last()
    ll_arr = np.asarray(res.logLiks.values, dtype=float)

    if ll_arr.ndim == 1:
        return np.array([float(logmeanexp(ll_arr))])
    # Shape is (n_thetas, reps) or (reps, n_thetas); pick the axis matching theta count.
    n_thetas = len(mod.theta.to_list())
    if ll_arr.shape[0] == n_thetas:
        return np.array([float(logmeanexp(ll_arr[i])) for i in range(n_thetas)])
    return np.array([float(logmeanexp(ll_arr[:, i])) for i in range(n_thetas)])


def _pfilter_loglik(mod, key, J: int, reps: int, ESS: bool = False):
    mod.pfilter(J=J, key=key, reps=reps, ESS=ESS)
    res = mod.results_history.last()

    ll_arr = np.asarray(res.logLiks.values, dtype=float)
    row = ll_arr if ll_arr.ndim == 1 else ll_arr[0]
    ll = float(logmeanexp(row))

    ess_df = None
    if ESS:
        ess_attr = getattr(res, "ESS", None)
        if ess_attr is not None and hasattr(ess_attr, "to_dataframe"):
            ess_df = ess_attr.to_dataframe(name="ESS").reset_index()
    return ll, ess_df


def _mif_once(mod, key, J: int, M: int, rw_sd, a: float = 0.5):
    mod.mif(J=J, M=M, rw_sd=rw_sd, a=a, key=key)
    return mod.results_history.last()


def _plot_mif_traces(traces, params, title: str, out_path: str):
    if traces is None or len(traces) == 0:
        return

    df = traces.copy()
    if "iteration" not in df.columns:
        df = df.reset_index()

    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, p in enumerate(params):
        ax = axes[i]
        if p not in df.columns:
            ax.set_title(f"{p} (no data)")
            ax.set_xlabel("mif iter")
            continue

        if "theta_idx" in df.columns:
            for _, grp in df.groupby("theta_idx"):
                grp = grp.sort_values("iteration")
                ax.plot(grp["iteration"], grp[p], alpha=0.5, lw=0.8)
        else:
            sub = df.sort_values("iteration")
            ax.plot(sub["iteration"], sub[p], alpha=0.8)

        ax.set_title(p)
        ax.set_xlabel("mif iter")

    for j in range(len(params), len(axes)):
        axes[j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --- 4a. Simulation study ---

def run_simulation_study(ys, race: str, cfg: RunConfig):
    tag = safe_name(race)
    theta_true = init_theta_from_data(ys, mock=True)


    mod = build_seir_model(ys, theta_true)
    _, y_sims = mod.simulate(key=jax.random.key(20260419), nsim=1)
    y_sim1 = y_sims[y_sims["sim"] == 0].sort_values("time")
    ys_fake = pd.DataFrame(
        {"deaths": y_sim1["obs_0"].to_numpy()},
        index=y_sim1["time"].to_numpy(),
    )

    mod_fake = build_seir_model(ys_fake, theta_true)
    ll_truth, _ = _pfilter_loglik(mod_fake, jax.random.key(1), J=cfg.pf_J, reps=cfg.pf_reps)

    # Perturb + refit to confirm IF2 recovers a loglik close to truth.
    theta_pert = dict(theta_true)
    theta_pert["b0"] *= 0.6
    theta_pert["rho"] *= 2.0
    theta_pert["sigma_env"] *= 1.5

    mod_refit = build_seir_model(ys_fake, theta_pert)
    _mif_once(
        mod_refit, jax.random.key(2),
        J=cfg.mif_J, M=cfg.mif_M,
        rw_sd=make_rw_sigma(), a=0.5,
    )
    theta_hat = mod_refit.theta.to_list()[0]
    ll_refit, _ = _pfilter_loglik(
        build_seir_model(ys_fake, theta_hat),
        jax.random.key(3), J=cfg.pf_J, reps=cfg.pf_reps,
    )

    plt.figure(figsize=(11, 4))
    plt.plot(ys.index, ys.iloc[:, 0].to_numpy(), color="black", lw=1.8, label="Observed")
    plt.plot(
        ys_fake.index, ys_fake.iloc[:, 0].to_numpy(),
        color="#1f77b4", lw=1.2, alpha=0.8, label="Simulation (truth theta)",
    )
    plt.title(
        f"Simulation study ({race})\n"
        f"logLik at truth = {ll_truth:.1f};  logLik after refit = {ll_refit:.1f}"
    )
    plt.xlabel("Month index")
    plt.ylabel("Deaths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"simstudy_{tag}.png", dpi=150)
    plt.close()

    return {
        "loglik_at_truth": float(ll_truth),
        "loglik_after_refit": float(ll_refit),
        "theta_hat_from_sim": theta_hat,
    }


# --- 4b. Local IF2 search ---

def run_local_search(ys, race: str, cfg: RunConfig, theta0: Dict[str, float]):
    tag = safe_name(race)
    rng = np.random.default_rng(123)
    jitter_params = ["b0", "b1", "b2", "mu_EI", "mu_IR", "mu_RS", "rho", "sigma_env", "eta", "iota"]
    thetas_start: List[Dict[str, float]] = []
    for _ in range(cfg.n_local):
        th = dict(theta0)
        for p in jitter_params:
            th[p] = float(th[p] * np.exp(rng.normal() * 0.02))
        th["rho"] = float(np.clip(th["rho"], 1e-4, 1 - 1e-4))
        th["eta"] = float(np.clip(th["eta"], 1e-3, 1 - 1e-3))
        th["k"] = float(max(th["k"], 1e-3))
        thetas_start.append(th)

    mod = build_seir_model(ys, thetas_start)
    _mif_once(
        mod, jax.random.key(42),
        J=cfg.mif_J, M=cfg.mif_M,
        rw_sd=make_rw_sigma(), a=0.5,
    )
    traces = mod.results_history.last().traces()

    # Particle-filter log-likelihood at each final theta, batched in ONE pfilter
    # call on the MIF'd model to avoid JAX JIT recompilation per chain.
    thetas_end = mod.theta.to_list()
    ll_per = _pfilter_batched_logliks(
        mod, jax.random.key(1000), J=cfg.pf_J, reps=cfg.pf_reps,
    )
    theta_best = thetas_end[int(np.argmax(ll_per))]

    # ESS trace for the best chain.
    ll_best, ess_df = _pfilter_loglik(
        build_seir_model(ys, theta_best),
        jax.random.key(777),
        J=cfg.pf_J, reps=cfg.pf_reps, ESS=True,
    )

    _plot_mif_traces(
        traces, ["logLik"] + ESTIMATED_PARAMS,
        title=f"Local IF2 convergence ({race}); best logLik={ll_best:.1f}",
        out_path=f"mif_local_traces_{tag}.png",
    )

    if ess_df is not None and len(ess_df) > 0:
        time_col = next((c for c in ("time", "t", "obs_time") if c in ess_df.columns), None)
        if time_col is not None:
            group_cols = [c for c in ("theta_idx", "rep", "replicate") if c in ess_df.columns]

            plt.figure(figsize=(11, 3.5))
            if group_cols:
                for _, g in ess_df.groupby(group_cols):
                    g = g.sort_values(time_col)
                    plt.plot(g[time_col], g["ESS"], color="#1f77b4", alpha=0.5, lw=1.0)
            else:
                g = ess_df.sort_values(time_col)
                plt.plot(g[time_col], g["ESS"], color="#1f77b4", alpha=0.7, lw=1.2)

            plt.axhline(
                cfg.pf_J * 0.1, ls="--", color="gray",
                label=f"10% of J ({cfg.pf_J})",
            )
            plt.title(f"Effective sample size ({race})")
            plt.xlabel("Month index")
            plt.ylabel("ESS")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"pfilter_ess_{tag}.png", dpi=150)
            plt.close()

    return {
        "theta_best": theta_best,
        "ll_best": float(ll_best),
        "ll_per_chain": ll_per.tolist(),
        "traces": traces,
    }


# --- 4c. Global IF2 search ---

def _sample_start(rng, anchor: Dict[str, float]) -> Dict[str, float]:
    th = dict(anchor)
    th["b0"] = float(np.exp(rng.uniform(np.log(1.0), np.log(15.0))))
    th["b1"] = float(rng.uniform(-0.8, 0.8))
    th["b2"] = float(rng.uniform(-0.5, 0.5))
    th["mu_EI"] = float(np.exp(rng.uniform(np.log(0.5), np.log(4.0))))
    th["mu_IR"] = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    th["mu_RS"] = float(np.exp(rng.uniform(np.log(0.03), np.log(0.5))))  # 2-month to 2.8-year immunity
    th["rho"] = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2))))
    th["covid_shift"] = float(rng.uniform(-0.6, 0.3))
    th["k"] = float(np.exp(rng.uniform(np.log(1e-2), np.log(0.5))))
    th["sigma_env"] = float(np.exp(rng.uniform(np.log(0.02), np.log(0.4))))
    th["eta"] = float(rng.uniform(0.3, 0.98))
    th["iota"] = float(np.exp(rng.uniform(np.log(100.0), np.log(50_000.0))))
    return th


def run_global_search(ys, race: str, cfg: RunConfig, theta_anchor: Dict[str, float]):
    tag = safe_name(race)
    rng = np.random.default_rng(2026)
    starts = [_sample_start(rng, theta_anchor) for _ in range(cfg.n_global)]

    mod = build_seir_model(ys, starts)
    _mif_once(
        mod, jax.random.key(9001),
        J=cfg.mif_J, M=cfg.mif_M,
        rw_sd=make_rw_sigma(scale_reg=0.04, scale_ivp=0.08), a=0.5,
    )
    traces = mod.results_history.last().traces()

    thetas_end = mod.theta.to_list()
    ll_per = _pfilter_batched_logliks(
        mod, jax.random.key(500),
        J=cfg.pf_J, reps=max(2, cfg.pf_reps // 2),
    )
    theta_best = thetas_end[int(np.argmax(ll_per))]

    _plot_mif_traces(
        traces, ["logLik"] + ESTIMATED_PARAMS,
        title=f"Global IF2 convergence ({race}); best logLik = {ll_per.max():.1f}",
        out_path=f"mif_global_traces_{tag}.png",
    )

    # Scatter of final logLik vs each parameter (loglik surface shape).
    scatter_params = ["b0", "b1", "b2", "rho", "sigma_env", "mu_EI", "mu_IR", "eta"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, p in zip(axes.flatten(), scatter_params):
        ax.scatter([t[p] for t in thetas_end], ll_per, s=18)
        ax.set_xlabel(p)
        ax.set_ylabel("logLik")
        ax.set_title(p)
    plt.suptitle(f"logLik vs parameter at final IF2 point ({race})")
    plt.tight_layout()
    plt.savefig(f"mif_global_scatter_{tag}.png", dpi=150)
    plt.close()

    return {
        "theta_best": theta_best,
        "ll_best": float(ll_per.max()),
        "ll_per_chain": ll_per.tolist(),
        "thetas_end": thetas_end,
        "traces": traces,
    }


# --- 4d. Simulations + probes at MLE ---

def plot_fitted_simulations(ys, race_df, race: str, theta_hat, cfg: RunConfig):
    tag = safe_name(race)
    mod = build_seir_model(ys, theta_hat)
    _, y_sims = mod.simulate(key=jax.random.key(424242), nsim=cfg.sim_nsim)

    if not {"sim", "time", "obs_0"}.issubset(y_sims.columns):
        return

    plt.figure(figsize=(11, 5))
    for _, one in y_sims.groupby("sim"):
        one = one.sort_values("time")
        plt.plot(one["time"], one["obs_0"], color="steelblue", alpha=0.08)
    plt.plot(race_df["time"], race_df["Deaths"], color="black", lw=1.8, label="Observed")
    plt.title(f"Simulations from fitted SEIR-POMP ({race})")
    plt.xlabel("Month index")
    plt.ylabel("Deaths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"simulations_{tag}.png", dpi=150)
    plt.close()


def _probe_stats(df):
    """Return (growth_rate, residual_sd, acf1, acf12) for one simulated series."""
    y = np.log(np.maximum(df.iloc[:, 0].to_numpy(dtype=float), 0.5))
    t = np.arange(len(y), dtype=float)

    if len(y) < 2 or np.var(t) <= 0:
        return {"growth_rate": np.nan, "residual_sd": np.nan, "acf1": 0.0, "acf12": 0.0}

    b1 = np.cov(t, y)[0, 1] / np.var(t)
    b0 = y.mean() - b1 * t.mean()
    resid = y - (b0 + b1 * t)

    yc = y - y.mean()
    var_yc = np.var(yc)
    acf1 = float(np.corrcoef(yc[:-1], yc[1:])[0, 1]) if var_yc > 0 else 0.0
    if len(yc) >= 13 and var_yc > 0:
        acf12 = float(np.corrcoef(yc[:-12], yc[12:])[0, 1])
    else:
        acf12 = 0.0

    return {
        "growth_rate": float(b1),
        "residual_sd": float(np.std(resid)),
        "acf1": acf1,
        "acf12": acf12,
    }


def run_probe_diagnostics(ys, race: str, theta_hat, cfg: RunConfig):
    tag = safe_name(race)
    mod = build_seir_model(ys, theta_hat)

    probe_names = ["growth_rate", "residual_sd", "acf1", "acf12"]
    probes = {
        name: (lambda df, name=name: _probe_stats(df)[name])
        for name in probe_names
    }

    out = mod.probe(probes=probes, nsim=cfg.sim_nsim, key=jax.random.key(3131))
    data_vals = out[out["is_real_data"]]
    sim_vals = out[~out["is_real_data"]]

    ncols = 2
    nrows = int(np.ceil(len(probe_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).flatten()

    summary = {}
    for i, name in enumerate(probe_names):
        sv = sim_vals.loc[sim_vals["probe"] == name, "value"].to_numpy(dtype=float)
        dv = float(data_vals.loc[data_vals["probe"] == name, "value"].iloc[0])

        axes[i].hist(sv, bins=25, alpha=0.7)
        axes[i].axvline(dv, ls="--", lw=2, color="red", label=f"data = {dv:.3f}")
        axes[i].set_title(name)
        axes[i].legend(fontsize=8)

        summary[name] = {
            "data": dv,
            "sim_mean": float(np.mean(sv)),
            "tail_p": float(np.mean(sv <= dv)),
        }

    for j in range(len(probe_names), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Probe diagnostics ({race})")
    plt.tight_layout()
    plt.savefig(f"probes_{tag}.png", dpi=150)
    plt.close()

    return summary


# --- 4e. Profile likelihood (true profile, with re-optimization) ---

def run_profile_likelihood(
    ys,
    race: str,
    cfg: RunConfig,
    theta_hat: Dict[str, float],
    param_name: str,
    grid: np.ndarray,
):
    """For each grid value g of `param_name`, re-optimize the remaining
    parameters via IF2 while holding `param_name = g` (zero-SD random walk),
    then evaluate the particle-filter log-likelihood. This is the Ionides-style
    profile: maximization over nuisance parameters at each fixed value.
    """
    lls = []
    for i, g in enumerate(grid):
        th = dict(theta_hat)
        th[param_name] = float(g)

        mod = build_seir_model(ys, th)
        rw = make_rw_sigma(scale_reg=0.02, scale_ivp=0.05)
        rw.sigmas[param_name] = 0.0  # clamp the profile parameter

        _mif_once(
            mod, jax.random.key(7000 + i),
            J=max(200, cfg.mif_J // 2),
            M=max(15, cfg.mif_M // 2),
            rw_sd=rw, a=0.5,
        )
        th_star = mod.theta.to_list()[0]
        th_star[param_name] = float(g)

        ll, _ = _pfilter_loglik(
            build_seir_model(ys, th_star),
            jax.random.key(8000 + i),
            J=max(300, cfg.pf_J // 2),
            reps=max(2, cfg.pf_reps // 2),
        )
        lls.append(ll)

    return np.asarray(lls, dtype=float)


def plot_profiles(race: str, profs: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    tag = safe_name(race)
    plt.figure(figsize=(5 * len(profs), 4))
    for i, (name, (grid, ll)) in enumerate(profs.items(), 1):
        ax = plt.subplot(1, len(profs), i)
        ax.plot(grid, ll, "o-", color="#1f77b4")
        ax.axhline(ll.max() - 1.92, ls="--", color="red", label="95% cutoff (-1.92)")
        ax.set_title(f"Profile logLik: {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("logLik")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"profiles_{tag}.png", dpi=150)
    plt.close()


# =============================================================================
# 5. Model comparison
# =============================================================================

def save_model_comparison(race, arma, garch, pomp_loglik, pomp_nparams, pomp_aic):
    tag = safe_name(race)
    rows = []

    if arma is not None:
        rows.append({
            "model": f"ARMA({arma['p']},{arma['q']})",
            "loglik": arma["loglik"],
            "aic": arma["aic"],
            "n_params": arma["n_params"],
        })

    if garch is not None:
        for name, info in garch.items():
            rows.append({
                "model": name,
                "loglik": info["loglik"],
                "aic": info["aic"],
                "n_params": info["n_params"],
            })

    rows.append({
        "model": "SEIR-POMP (IF2)",
        "loglik": pomp_loglik,
        "aic": pomp_aic,
        "n_params": pomp_nparams,
    })

    df = pd.DataFrame(rows)
    df.to_csv(f"model_comparison_{tag}.csv", index=False)
    return df


# =============================================================================
# 6. Across-race comparison (baseline only)
# =============================================================================

def compare_across_races(df, cfg: RunConfig):
    out = []
    for i, race in enumerate(sorted(df["Race"].dropna().unique().tolist())):
        race_df, ys = prepare_race_data(df, race)
        mod = build_seir_model(ys, init_theta_from_data(race_df))
        ll, _ = _pfilter_loglik(
            mod, jax.random.key(33000 + i),
            J=max(300, cfg.pf_J // 2),
            reps=max(2, cfg.pf_reps // 2),
        )
        out.append({
            "race": race,
            "n_months": len(race_df),
            "mean_deaths": float(race_df["Deaths"].mean()),
            "init_loglik": float(ll),
        })

    comp = pd.DataFrame(out).sort_values("init_loglik", ascending=False)
    comp.to_csv("race_comparison_summary.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.barh(comp["race"], comp["init_loglik"])
    plt.gca().invert_yaxis()
    plt.xlabel("Initial pfilter log-likelihood")
    plt.title("Across-race baseline model fit")
    plt.tight_layout()
    plt.savefig("race_comparison_loglik.png", dpi=150)
    plt.close()

    return comp


# =============================================================================
# Main driver
# =============================================================================

def run(cfg: RunConfig):
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = load_data(DATA_FILE)
    race_df, ys = prepare_race_data(df, cfg.race)

    print("=" * 66)
    print(f"STATS 531 Final Project: race = {cfg.race}, n_months = {len(race_df)}")
    print("=" * 66)

    # 1. EDA
    eda = run_eda(race_df, cfg.race)
    print("\n[1] EDA summary")
    for k, v in eda.items():
        print(f"    {k}: {v}")

    # 2. ARMA
    arma = maybe_load(cache_path("arma", cfg.race), cfg.force_recompute)
    if arma is None:
        arma = run_arma_benchmark(race_df, cfg.race)
        save_cache(cache_path("arma", cfg.race), arma)
    if arma is not None:
        print(
            f"\n[2] ARMA benchmark: best = ARMA({arma['p']},{arma['q']}),"
            f" logLik={arma['loglik']:.2f}, AIC={arma['aic']:.2f}"
        )

    # 3. GARCH
    garch = maybe_load(cache_path("garch", cfg.race), cfg.force_recompute)
    if garch is None:
        garch = run_garch_benchmarks(race_df, cfg.race)
        save_cache(cache_path("garch", cfg.race), garch)
    if garch is not None:
        print("\n[3] GARCH benchmarks")
        for name, info in garch.items():
            print(f"    {name}: logLik={info['loglik']:.2f}, AIC={info['aic']:.2f}")

    # 4a. Simulation study
    simstudy = maybe_load(cache_path("simstudy", cfg.race), cfg.force_recompute)
    if simstudy is None:
        simstudy = run_simulation_study(ys, cfg.race, cfg)
        save_cache(cache_path("simstudy", cfg.race), simstudy)
    print(
        f"\n[4a] Simulation-study logLiks: at_truth={simstudy['loglik_at_truth']:.2f},"
        f" after_refit={simstudy['loglik_after_refit']:.2f}"
    )

    # 4b. Local IF2
    local = maybe_load(cache_path("local", cfg.race), cfg.force_recompute)
    if local is None:
        local = run_local_search(
            ys, cfg.race, cfg,
            theta0=init_theta_from_data(race_df),
        )
        save_cache(
            cache_path("local", cfg.race),
            {k: local[k] for k in ["theta_best", "ll_best", "ll_per_chain"]},
        )
    print(f"\n[4b] Local IF2: best logLik = {local['ll_best']:.2f}")

    # 4c. Global IF2
    glob = maybe_load(cache_path("global", cfg.race), cfg.force_recompute)
    if glob is None:
        glob = run_global_search(
            ys, cfg.race, cfg,
            theta_anchor=local["theta_best"],
        )
        save_cache(
            cache_path("global", cfg.race),
            {k: glob[k] for k in ["theta_best", "ll_best", "ll_per_chain"]},
        )
    print(f"\n[4c] Global IF2: best logLik = {glob['ll_best']:.2f}")

    # Select overall best.
    if glob["ll_best"] >= local["ll_best"]:
        theta_mle = glob["theta_best"]
        ll_mle = glob["ll_best"]
    else:
        theta_mle = local["theta_best"]
        ll_mle = local["ll_best"]

    print("\n[4] Point estimate (MLE) parameters:")
    for k in ESTIMATED_PARAMS:
        print(f"    {k}: {theta_mle[k]:.5g}")

    # 4d. Simulations + probes
    plot_fitted_simulations(ys, race_df, cfg.race, theta_mle, cfg)
    probe_summary = run_probe_diagnostics(ys, cfg.race, theta_mle, cfg)
    print("\n[4d] Probe diagnostics (tail_p near 0.5 means the model matches the data)")
    for name, stats in probe_summary.items():
        print(
            f"    {name}: data={stats['data']:.4f}, "
            f"sim_mean={stats['sim_mean']:.4f}, tail_p={stats['tail_p']:.3f}"
        )

    # 4e. Profiles
    profs = maybe_load(cache_path("profile", cfg.race), cfg.force_recompute)
    if profs is None:
        b0_grid = np.linspace(
            max(0.2, 0.5 * theta_mle["b0"]),
            1.8 * theta_mle["b0"],
            cfg.profile_grid_n,
        )
        rho_grid = np.linspace(
            max(1e-3, 0.3 * min(theta_mle["rho"], 0.5)),
            min(0.5, 2.5 * min(theta_mle["rho"], 0.2)),
            cfg.profile_grid_n,
        )
        profs = {
            "b0": (
                b0_grid,
                run_profile_likelihood(ys, cfg.race, cfg, theta_mle, "b0", b0_grid),
            ),
            "rho": (
                rho_grid,
                run_profile_likelihood(ys, cfg.race, cfg, theta_mle, "rho", rho_grid),
            ),
        }
        save_cache(cache_path("profile", cfg.race), profs)

    plot_profiles(cfg.race, profs)
    for name, (grid, ll) in profs.items():
        print(
            f"\n[4e] Profile on {name}: argmax at {grid[int(np.argmax(ll))]:.4g},"
            f" max logLik = {ll.max():.2f}"
        )

    # 5. Comparison
    pomp_n = len(ESTIMATED_PARAMS)
    pomp_aic = -2.0 * ll_mle + 2.0 * pomp_n
    comparison = save_model_comparison(cfg.race, arma, garch, ll_mle, pomp_n, pomp_aic)
    print("\n[5] Model comparison")
    print(comparison.to_string(index=False))

    # 6. Across-race
    race_comp = compare_across_races(df, cfg)
    print("\n[6] Across-race baseline fit")
    print(race_comp.to_string(index=False))

    # Narrative summary
    lines = [
        f"Race analyzed: {cfg.race} ({len(race_df)} months)",
        f"COVID-era / pre-COVID mean-death ratio: {eda['covid_ratio']:.2f}",
        f"ARMA logLik: {arma['loglik']:.2f}  AIC: {arma['aic']:.2f}" if arma else "ARMA skipped",
        *(f"{k}: logLik={v['loglik']:.2f}  AIC={v['aic']:.2f}" for k, v in (garch or {}).items()),
        f"SEIR-POMP logLik: {ll_mle:.2f}  AIC: {pomp_aic:.2f}",
        f"Best b0 on profile: {profs['b0'][0][int(np.argmax(profs['b0'][1]))]:.3f}",
        f"Best rho on profile: {profs['rho'][0][int(np.argmax(profs['rho'][1]))]:.4f}",
        "Seasonal forcing and environmental noise dominate; the COVID-era mean shift is not",
        "fully explained by the stationary SEIR and deserves a time-varying extension.",
    ]
    with open(f"analysis_summary_{safe_name(cfg.race)}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nDone. Plots, CSV tables, and the summary file saved in the working directory.")


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="STATS 531 SEIR-POMP final project workflow")
    p.add_argument("--race", type=str, default="White")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    a = p.parse_args()
    return RunConfig(race=a.race, quick=a.quick, force_recompute=a.force_recompute)


if __name__ == "__main__":
    run(parse_args())
