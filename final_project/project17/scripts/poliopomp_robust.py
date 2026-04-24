#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def parse_args():
    p = argparse.ArgumentParser(
        description="Robust monthly logged TSIR-POMP search for national polio data."
    )
    p.add_argument(
        "--data",
        type=str,
        default="aggregated_monthly_polio_1930_1964.csv",
        help="Path to aggregated monthly CSV.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="poliopomp_robust_outputs",
        help="Directory to save outputs.",
    )
    p.add_argument("--year-start", type=int, default=1930)
    p.add_argument("--year-end", type=int, default=1964)
    p.add_argument("--global-starts", type=int, default=15)
    p.add_argument("--global-iters", type=int, default=50)
    p.add_argument("--global-particles", type=int, default=1000)
    p.add_argument("--global-cooling", type=float, default=0.5)
    p.add_argument("--top-trace-per-fit", type=int, default=3)
    p.add_argument("--local-top-n", type=int, default=2)
    p.add_argument("--local-iters", type=int, default=50)
    p.add_argument("--local-particles", type=int, default=1000)
    p.add_argument("--local-cooling", type=float, default=0.3)
    p.add_argument("--rep-particles", type=int, default=3000)
    p.add_argument("--rep-reps", type=int, default=10)
    p.add_argument("--filter-particles", type=int, default=1200)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--verbose-if2", action="store_true")
    p.add_argument("--run-profile", action="store_true")
    p.add_argument("--profile-points", type=int, default=7)
    p.add_argument("--profile-restarts", type=int, default=5)
    p.add_argument("--profile-iters", type=int, default=40)
    p.add_argument("--profile-particles", type=int, default=1000)
    return p.parse_args()


def load_monthly_data(path, year_start=1930, year_end=1964):
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    dat = pd.read_csv(data_path)
    dat["date"] = pd.to_datetime(dat["date"])
    dat = dat.sort_values("date").reset_index(drop=True)

    dat["year"] = dat["date"].dt.year
    dat["month"] = dat["date"].dt.month

    dat = (
        dat[(dat["year"] >= year_start) & (dat["year"] <= year_end)]
        .copy()
        .reset_index(drop=True)
    )
    dat["month_angle"] = 2.0 * np.pi * (dat["month"] - 1) / 12.0
    dat["post_1955"] = (dat["date"] >= pd.Timestamp("1955-01-01")).astype(int)
    dat["year_change"] = (dat["year"] != dat["year"].shift(1)).astype(int)
    dat.loc[0, "year_change"] = 0
    dat["log_cases"] = np.log1p(dat["cases"])

    return dat


# Globals initialized after loading data
SEASON_COS1 = SEASON_SIN1 = SEASON_COS2 = SEASON_SIN2 = None
post = year_change = YEAR = MONTH = None
N_EFF = MAX_NU = MAX_LATENT_I = MAX_POISSON_LAM = None
y_raw = y = None
T = None

IDX_LOG_BETA0 = 0
IDX_A1 = 1
IDX_B1 = 2
IDX_A2 = 3
IDX_B2 = 4
IDX_DELTA_POST = 5
IDX_Z_ALPHA = 6
IDX_LOG_NU = 7
IDX_LOG_KPROC = 8
IDX_LOG_SIGMA_A = 9
IDX_LOGIT_ETA = 10
IDX_LOG_I0 = 11
IDX_LOG_SIGMA_OBS = 12

N_PARAMS = 13
PARAM_NAMES = [
    "beta0",
    "a1",
    "b1",
    "a2",
    "b2",
    "delta_post",
    "alpha",
    "nu",
    "k_proc",
    "sigma_A",
    "eta",
    "I0",
    "sigma_obs",
]

THETA_LOWER = None
THETA_UPPER = None


def initialize_covariates(dat):
    global SEASON_COS1, SEASON_SIN1, SEASON_COS2, SEASON_SIN2
    global \
        post, \
        year_change, \
        YEAR, \
        MONTH, \
        N_EFF, \
        MAX_NU, \
        MAX_LATENT_I, \
        MAX_POISSON_LAM
    global y_raw, y, T, THETA_LOWER, THETA_UPPER

    SEASON_COS1 = np.cos(dat["month_angle"].to_numpy())
    SEASON_SIN1 = np.sin(dat["month_angle"].to_numpy())
    SEASON_COS2 = np.cos(2.0 * dat["month_angle"].to_numpy())
    SEASON_SIN2 = np.sin(2.0 * dat["month_angle"].to_numpy())

    post = dat["post_1955"].to_numpy(dtype=float)
    year_change = dat["year_change"].to_numpy(dtype=int)
    YEAR = dat["year"].to_numpy(dtype=int)
    MONTH = dat["month"].to_numpy(dtype=int)

    y_raw = dat["cases"].to_numpy(dtype=float)
    y = dat["log_cases"].to_numpy(dtype=float)
    T = len(y)

    N_EFF = float(max(200_000, 35.0 * np.quantile(y_raw, 0.95)))
    MAX_NU = float(0.05 * N_EFF)
    MAX_LATENT_I = float(max(50_000.0, 6.0 * np.max(y_raw), 0.35 * N_EFF))
    MAX_POISSON_LAM = float(MAX_LATENT_I)

    THETA_LOWER = np.array(
        [
            np.log(1e-5),
            -2.5,
            -2.5,
            -1.5,
            -1.5,
            -6.0,
            -6.0,
            np.log(1.0),
            np.log(1e-3),
            np.log(1e-3),
            -8.0,
            np.log(1e-3),
            np.log(1e-3),
        ],
        dtype=float,
    )

    THETA_UPPER = np.array(
        [
            np.log(100.0),
            2.5,
            2.5,
            1.5,
            1.5,
            2.0,
            6.0,
            np.log(MAX_NU),
            np.log(1e3),
            np.log(2.0),
            8.0,
            np.log(MAX_LATENT_I),
            np.log(5.0),
        ],
        dtype=float,
    )


def clamp_theta(theta):
    theta = np.asarray(theta, dtype=float)
    return np.clip(theta, THETA_LOWER, THETA_UPPER)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1.0 - p))


def untransform(theta):
    theta = clamp_theta(theta)
    return {
        "beta0": np.exp(theta[..., IDX_LOG_BETA0]),
        "a1": theta[..., IDX_A1],
        "b1": theta[..., IDX_B1],
        "a2": theta[..., IDX_A2],
        "b2": theta[..., IDX_B2],
        "delta_post": theta[..., IDX_DELTA_POST],
        "alpha": 0.6 + 0.5 * sigmoid(theta[..., IDX_Z_ALPHA]),
        "nu": np.exp(theta[..., IDX_LOG_NU]),
        "k_proc": np.exp(theta[..., IDX_LOG_KPROC]),
        "sigma_A": np.exp(theta[..., IDX_LOG_SIGMA_A]),
        "eta": sigmoid(theta[..., IDX_LOGIT_ETA]),
        "I0": np.exp(theta[..., IDX_LOG_I0]),
        "sigma_obs": np.exp(theta[..., IDX_LOG_SIGMA_OBS]),
    }


def transform(
    beta0,
    a1,
    b1,
    a2,
    b2,
    delta_post,
    alpha,
    nu,
    k_proc,
    sigma_A,
    eta,
    I0,
    sigma_obs,
):
    alpha_scaled = np.clip((alpha - 0.6) / 0.5, 1e-6, 1 - 1e-6)
    z_alpha = logit(alpha_scaled)
    theta = np.array(
        [
            np.log(beta0),
            a1,
            b1,
            a2,
            b2,
            delta_post,
            z_alpha,
            np.log(nu),
            np.log(k_proc),
            np.log(sigma_A),
            logit(eta),
            np.log(I0),
            np.log(sigma_obs),
        ],
        dtype=float,
    )
    return clamp_theta(theta)


def seasonal_log_effect(theta_particles, t_idx):
    p = untransform(theta_particles)
    return (
        p["a1"] * SEASON_COS1[t_idx]
        + p["b1"] * SEASON_SIN1[t_idx]
        + p["a2"] * SEASON_COS2[t_idx]
        + p["b2"] * SEASON_SIN2[t_idx]
    )


def sample_nb_mean_disp(mean, k, rng):
    mean = np.clip(np.asarray(mean, dtype=float), 1e-10, MAX_POISSON_LAM)
    k = np.clip(np.asarray(k, dtype=float), 1e-10, 1e6)
    lam = rng.gamma(shape=k, scale=mean / k)
    lam = np.clip(lam, 0.0, MAX_POISSON_LAM)
    return rng.poisson(lam)


def systematic_resample(w, rng):
    n = len(w)
    positions = (rng.uniform() + np.arange(n)) / n
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    return np.clip(idx, 0, n - 1)


def init_state(theta_particles, rng):
    theta_particles = clamp_theta(theta_particles)
    n_particles = theta_particles.shape[0]
    p = untransform(theta_particles)

    S0 = p["eta"] * N_EFF
    I0 = p["I0"] * rng.lognormal(mean=0.0, sigma=0.15, size=n_particles)
    A0 = np.zeros(n_particles)

    state = np.zeros((n_particles, 3), dtype=float)
    state[:, 0] = np.clip(S0, 0.0, N_EFF)
    state[:, 1] = np.clip(I0, 0.0, MAX_LATENT_I)
    state[:, 2] = A0
    return state


def step_particles(state, theta_particles, t_next, rng):
    theta_particles = clamp_theta(theta_particles)
    p = untransform(theta_particles)

    S = np.clip(state[:, 0], 0.0, N_EFF)
    I = np.clip(state[:, 1], 0.0, MAX_LATENT_I)
    A = state[:, 2].copy()

    if year_change[t_next] == 1:
        A = rng.normal(loc=0.0, scale=p["sigma_A"], size=A.shape[0])
        A = np.clip(A, -4.0, 4.0)

    season = seasonal_log_effect(theta_particles, t_next)
    post_shift = p["delta_post"] * post[t_next]
    depletion = np.log1p(np.maximum(S, 0.0)) - np.log1p(N_EFF)

    log_lam = (
        np.log(p["beta0"])
        + season
        + post_shift
        + A
        + depletion
        + p["alpha"] * np.log1p(np.maximum(I, 0.0))
    )
    log_lam = np.clip(log_lam, -20.0, np.log(MAX_POISSON_LAM))
    lam = np.exp(log_lam)

    I_next = sample_nb_mean_disp(lam, p["k_proc"], rng).astype(float)
    I_next = np.clip(I_next, 0.0, MAX_LATENT_I)

    replenishment = p["nu"] * (1.0 - S / N_EFF)
    S_next = np.clip(S + replenishment - I_next, 0.0, N_EFF)

    out = np.empty_like(state)
    out[:, 0] = S_next
    out[:, 1] = I_next
    out[:, 2] = A
    return out


def dmeasure_log(state, theta_particles, y_t):
    theta_particles = clamp_theta(theta_particles)
    p = untransform(theta_particles)
    mu_log = np.log1p(np.maximum(state[:, 1], 0.0))
    sigma = np.maximum(p["sigma_obs"], 1e-8)
    return norm.logpdf(y_t, loc=mu_log, scale=sigma)


def rmeasure_log(state, theta_particles, rng):
    theta_particles = clamp_theta(theta_particles)
    p = untransform(theta_particles)
    mu_log = np.log1p(np.maximum(state[:, 1], 0.0))
    y_log = rng.normal(mu_log, np.maximum(p["sigma_obs"], 1e-8))
    y_raw_sim = np.maximum(np.expm1(y_log), 0.0)
    return y_log, y_raw_sim


def pfilter(y_obs, theta, n_particles=600, seed=None, return_filtered=False):
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, dtype=float)

    theta_particles = np.broadcast_to(theta, (n_particles, N_PARAMS)).copy()
    state = init_state(theta_particles, rng)

    log_lik = 0.0
    log_lik_t = np.zeros(T)
    ess = np.zeros(T)
    filt_q_raw = np.zeros((T, 3)) if return_filtered else None
    filt_mean_raw = np.zeros(T) if return_filtered else None
    filt_mean_log = np.zeros(T) if return_filtered else None

    for t in range(T):
        log_w = dmeasure_log(state, theta_particles, y_obs[t])
        max_lw = np.max(log_w)
        if not np.isfinite(max_lw):
            log_lik_t[t] = -np.inf
            log_lik = -np.inf
            break

        w_unnorm = np.exp(log_w - max_lw)
        sum_w = np.sum(w_unnorm)
        inc = max_lw + np.log(sum_w) - np.log(n_particles)

        log_lik_t[t] = inc
        log_lik += inc

        w = w_unnorm / sum_w
        ess[t] = 1.0 / np.sum(w**2)

        if return_filtered:
            latent_I = state[:, 1]
            latent_logI = np.log1p(np.maximum(latent_I, 0.0))
            order = np.argsort(latent_I)
            cw = np.cumsum(w[order])
            for j, q in enumerate((0.025, 0.5, 0.975)):
                pos = np.searchsorted(cw, q, side="right")
                pos = min(pos, n_particles - 1)
                filt_q_raw[t, j] = latent_I[order][pos]
            filt_mean_raw[t] = np.sum(w * latent_I)
            filt_mean_log[t] = np.sum(w * latent_logI)

        idx = systematic_resample(w, rng)
        state = state[idx]
        theta_particles = theta_particles[idx]

        if t < T - 1:
            state = step_particles(state, theta_particles, t + 1, rng)

    return {
        "log_lik": float(log_lik),
        "log_lik_t": log_lik_t,
        "ess": ess,
        "filtered_q_raw": filt_q_raw,
        "filtered_mean_raw": filt_mean_raw,
        "filtered_mean_log": filt_mean_log,
    }


def pfilter_replicated(y_obs, theta, n_particles=3000, n_reps=10, seed=None):
    rng = np.random.default_rng(seed)
    lls = np.zeros(n_reps)
    for r in range(n_reps):
        out = pfilter(
            y_obs,
            theta,
            n_particles=n_particles,
            seed=int(rng.integers(2**31)),
        )
        lls[r] = out["log_lik"]
    return {
        "log_lik_reps": lls,
        "log_lik_mean": float(np.mean(lls)),
        "log_lik_se": float(np.std(lls, ddof=1) / np.sqrt(n_reps))
        if n_reps > 1
        else 0.0,
    }


def simulate_from_model(theta, n_sims=100, seed=None):
    rng = np.random.default_rng(seed)
    theta = clamp_theta(theta)
    sims_raw = np.zeros((n_sims, T), dtype=float)
    sims_log = np.zeros((n_sims, T), dtype=float)

    for s in range(n_sims):
        theta_particles = theta[np.newaxis, :].copy()
        state = init_state(theta_particles, rng)
        y_log0, y_raw0 = rmeasure_log(state, theta_particles, rng)
        sims_log[s, 0] = y_log0[0]
        sims_raw[s, 0] = y_raw0[0]

        for t in range(1, T):
            state = step_particles(state, theta_particles, t, rng)
            y_log_t, y_raw_t = rmeasure_log(state, theta_particles, rng)
            sims_log[s, t] = y_log_t[0]
            sims_raw[s, t] = y_raw_t[0]

    return {"raw": sims_raw, "log": sims_log}


def mif2(
    y_obs,
    theta_start,
    n_iterations=50,
    n_particles=1000,
    rw_sd=None,
    cooling=0.5,
    seed=None,
    verbose=True,
    fixed_indices=None,
    fixed_values=None,
):
    rng = np.random.default_rng(seed)
    theta = clamp_theta(np.asarray(theta_start, dtype=float).copy())

    if rw_sd is None:
        rw_sd = default_rw_sd()
    else:
        rw_sd = np.asarray(rw_sd, dtype=float)

    if fixed_indices is None:
        fixed_indices = np.array([], dtype=int)
        fixed_values = np.array([], dtype=float)
    else:
        fixed_indices = np.asarray(fixed_indices, dtype=int)
        fixed_values = np.asarray(fixed_values, dtype=float)

    def apply_fixed(theta_array):
        arr = np.array(theta_array, copy=True)
        if fixed_indices.size > 0:
            arr[..., fixed_indices] = fixed_values
        return clamp_theta(arr)

    theta = apply_fixed(theta)
    log_liks = np.zeros(n_iterations)
    theta_trace = np.zeros((n_iterations + 1, N_PARAMS))
    theta_trace[0] = theta

    for m in range(n_iterations):
        cool = cooling ** (m / max(n_iterations - 1, 1))
        sd_m = rw_sd * cool
        if fixed_indices.size > 0:
            sd_m = sd_m.copy()
            sd_m[fixed_indices] = 0.0

        theta_particles = theta[np.newaxis, :] + rng.normal(
            0.0, sd_m, size=(n_particles, N_PARAMS)
        )
        theta_particles = apply_fixed(theta_particles)
        state = init_state(theta_particles, rng)

        ll_m = 0.0
        for t in range(T):
            theta_particles = theta_particles + rng.normal(
                0.0, sd_m, size=theta_particles.shape
            )
            theta_particles = apply_fixed(theta_particles)

            log_w = dmeasure_log(state, theta_particles, y_obs[t])
            max_lw = np.max(log_w)
            if not np.isfinite(max_lw):
                ll_m = -np.inf
                break

            w_unnorm = np.exp(log_w - max_lw)
            sum_w = np.sum(w_unnorm)
            inc = max_lw + np.log(sum_w) - np.log(n_particles)
            ll_m += inc

            w = w_unnorm / sum_w
            idx = systematic_resample(w, rng)
            state = state[idx]
            theta_particles = theta_particles[idx]

            if t < T - 1:
                state = step_particles(state, theta_particles, t + 1, rng)

        theta = apply_fixed(np.mean(theta_particles, axis=0))
        log_liks[m] = ll_m
        theta_trace[m + 1] = theta

        if verbose:
            nat = untransform(theta)
            print(
                f"iter {m + 1:3d}/{n_iterations}  "
                f"loglik={ll_m:11.2f}  "
                f"beta0={nat['beta0']:.3f}  "
                f"alpha={nat['alpha']:.3f}  "
                f"nu={nat['nu']:.1f}  "
                f"k_proc={nat['k_proc']:.3f}  "
                f"sigma_A={nat['sigma_A']:.3f}  "
                f"sigma_obs={nat['sigma_obs']:.3f}  "
                f"eta={nat['eta']:.3f}  "
                f"I0={nat['I0']:.1f}"
            )

    best_iter = int(np.nanargmax(log_liks))
    theta_best_if2 = theta_trace[1 + best_iter].copy()
    return {
        "theta": theta,
        "theta_best_if2": theta_best_if2,
        "best_iter": best_iter + 1,
        "log_liks": log_liks,
        "theta_trace": theta_trace,
    }


def default_rw_sd():
    return np.array(
        [
            0.03,
            0.02,
            0.02,
            0.015,
            0.015,
            0.03,
            0.02,
            0.015,
            0.03,
            0.02,
            0.02,
            0.03,
            0.02,
        ],
        dtype=float,
    )


def random_theta0(rng):
    beta0 = np.exp(rng.uniform(np.log(0.2), np.log(10.0)))
    a1 = rng.uniform(-1.0, 1.0)
    b1 = rng.uniform(-1.0, 1.0)
    a2 = rng.uniform(-0.4, 0.4)
    b2 = rng.uniform(-0.4, 0.4)
    delta_post = rng.uniform(-2.0, -0.1)
    alpha = rng.uniform(0.75, 1.02)
    nu = np.exp(rng.uniform(np.log(50.0), np.log(MAX_NU)))
    k_proc = np.exp(rng.uniform(np.log(2.0), np.log(100.0)))
    sigma_A = np.exp(rng.uniform(np.log(0.03), np.log(0.5)))
    eta = rng.uniform(0.10, 0.80)
    I0 = np.exp(
        rng.uniform(
            np.log(max(1.0, np.median(y_raw[:12]) / 4)),
            np.log(max(20.0, 4 * np.median(y_raw[:12]))),
        )
    )
    sigma_obs = np.exp(rng.uniform(np.log(0.10), np.log(1.20)))
    return transform(
        beta0,
        a1,
        b1,
        a2,
        b2,
        delta_post,
        alpha,
        nu,
        k_proc,
        sigma_A,
        eta,
        I0,
        sigma_obs,
    )


def evaluate_trace_candidates(
    y_obs, fit, top_k=5, n_particles=3000, n_reps=10, seed=1
):
    log_liks = fit["log_liks"]
    trace = fit["theta_trace"][1:]
    ord_idx = np.argsort(log_liks)[::-1]
    keep = ord_idx[: min(top_k, len(ord_idx))]

    rows = []
    for rank, idx in enumerate(keep, start=1):
        theta_cand = trace[idx]
        rep = pfilter_replicated(
            y_obs,
            theta_cand,
            n_particles=n_particles,
            n_reps=n_reps,
            seed=seed + rank,
        )
        nat = untransform(theta_cand)
        rows.append(
            {
                "trace_rank": rank,
                "iter": int(idx + 1),
                "if2_loglik": float(log_liks[idx]),
                "rep_loglik": float(rep["log_lik_mean"]),
                "rep_se": float(rep["log_lik_se"]),
                "beta0": float(nat["beta0"]),
                "delta_post": float(nat["delta_post"]),
                "alpha": float(nat["alpha"]),
                "nu": float(nat["nu"]),
                "k_proc": float(nat["k_proc"]),
                "sigma_A": float(nat["sigma_A"]),
                "eta": float(nat["eta"]),
                "I0": float(nat["I0"]),
                "sigma_obs": float(nat["sigma_obs"]),
                "theta": theta_cand.copy(),
            }
        )
    return pd.DataFrame(rows)


def run_global_search(
    y_obs,
    n_starts=15,
    n_iterations=50,
    n_particles=1000,
    cooling=0.5,
    rw_sd=None,
    seed=123,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    fits = []
    summary_rows = []
    for s in range(n_starts):
        theta0 = random_theta0(rng)
        fit = mif2(
            y_obs,
            theta0,
            n_iterations=n_iterations,
            n_particles=n_particles,
            rw_sd=rw_sd,
            cooling=cooling,
            seed=int(rng.integers(2**31)),
            verbose=verbose,
        )
        fit["theta0"] = theta0
        fit["start_id"] = s
        fits.append(fit)
        nat = untransform(fit["theta_best_if2"])
        summary_rows.append(
            {
                "start_id": s,
                "best_if2_loglik": float(np.max(fit["log_liks"])),
                "best_iter": int(fit["best_iter"]),
                "beta0": float(nat["beta0"]),
                "delta_post": float(nat["delta_post"]),
                "alpha": float(nat["alpha"]),
                "nu": float(nat["nu"]),
                "k_proc": float(nat["k_proc"]),
                "sigma_A": float(nat["sigma_A"]),
                "eta": float(nat["eta"]),
                "I0": float(nat["I0"]),
                "sigma_obs": float(nat["sigma_obs"]),
            }
        )
    return fits, pd.DataFrame(summary_rows).sort_values(
        "best_if2_loglik", ascending=False
    ).reset_index(drop=True)


def gather_global_candidates(
    y_obs, fits, top_trace_per_fit=3, rep_particles=3000, rep_reps=10, seed=100
):
    cand_frames = []
    for j, fit in enumerate(fits):
        df = evaluate_trace_candidates(
            y_obs,
            fit,
            top_k=top_trace_per_fit,
            n_particles=rep_particles,
            n_reps=rep_reps,
            seed=seed + 50 * j,
        )
        fit_id = fit.get("start_id", fit.get("local_id", j))
        df["start_id"] = fit_id
        cand_frames.append(df)
    out = pd.concat(cand_frames, ignore_index=True)
    out = out.sort_values(
        ["rep_loglik", "if2_loglik"], ascending=False
    ).reset_index(drop=True)
    return out


def run_local_refinement(
    y_obs,
    candidate_thetas,
    n_iterations=50,
    n_particles=1000,
    cooling=0.3,
    rw_sd=None,
    seed=456,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    if rw_sd is None:
        rw_sd = 0.5 * default_rw_sd()
    fits = []
    for j, theta0 in enumerate(candidate_thetas):
        fit = mif2(
            y_obs,
            theta0,
            n_iterations=n_iterations,
            n_particles=n_particles,
            rw_sd=rw_sd,
            cooling=cooling,
            seed=int(rng.integers(2**31)),
            verbose=verbose,
        )
        fit["local_id"] = j
        fit["start_id"] = j
        fit["theta0"] = theta0
        fits.append(fit)
    return fits


def profile_delta_post(
    y_obs,
    theta_center,
    grid,
    n_restarts=5,
    n_iterations=40,
    n_particles=1000,
    rw_sd=None,
    cooling=0.3,
    rep_particles=3000,
    rep_reps=10,
    seed=999,
):
    rng = np.random.default_rng(seed)
    rows = []
    if rw_sd is None:
        rw_sd = 0.5 * default_rw_sd()

    for val in grid:
        best_rep = -np.inf
        best_se = np.nan
        for _ in range(n_restarts):
            theta_start = theta_center.copy()
            theta_start[IDX_DELTA_POST] = val
            theta_start = clamp_theta(
                theta_start + rng.normal(0.0, 0.1 * rw_sd, size=N_PARAMS)
            )
            theta_start[IDX_DELTA_POST] = val

            fit = mif2(
                y_obs,
                theta_start,
                n_iterations=n_iterations,
                n_particles=n_particles,
                rw_sd=rw_sd,
                cooling=cooling,
                seed=int(rng.integers(2**31)),
                verbose=False,
                fixed_indices=[IDX_DELTA_POST],
                fixed_values=[val],
            )
            cand = evaluate_trace_candidates(
                y_obs,
                fit,
                top_k=3,
                n_particles=rep_particles,
                n_reps=rep_reps,
                seed=int(rng.integers(2**31)),
            )
            row = cand.iloc[0]
            if row["rep_loglik"] > best_rep:
                best_rep = row["rep_loglik"]
                best_se = row["rep_se"]
        rows.append(
            {
                "delta_post": float(val),
                "profile_loglik": float(best_rep),
                "rep_se": float(best_se),
            }
        )
    return pd.DataFrame(rows)


def save_trace_plot(best_fit, outdir):
    trace_nat = pd.DataFrame(
        [untransform(th) for th in best_fit["theta_trace"]]
    )
    trace_nat.to_csv(outdir / "local_parameter_traces.csv", index=False)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    plot_names = [
        "beta0",
        "delta_post",
        "alpha",
        "nu",
        "k_proc",
        "sigma_A",
        "eta",
        "I0",
    ]
    for ax, nm in zip(axes.ravel(), plot_names):
        ax.plot(trace_nat[nm].to_numpy(), lw=1.6)
        ax.set_title(nm)
    axes[-1, -1].axis("off")
    fig.suptitle("Local refinement parameter traces", y=0.995)
    fig.tight_layout()
    fig.savefig(
        outdir / "local_parameter_traces.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def save_global_hist(global_summary, outdir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(global_summary["best_if2_loglik"], bins=10, edgecolor="black")
    ax.set_title(
        "Distribution of best IF2 log-likelihoods across global starts"
    )
    ax.set_xlabel("best IF2-reported log-likelihood")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(
        outdir / "global_search_histogram.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def save_ess_plots(dat, out_hat, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(dat["date"], out_hat["ess"], lw=1.5)
    axes[0].set_title("ESS over time")
    axes[0].set_ylabel("ESS")
    axes[1].plot(dat["date"], out_hat["log_lik_t"], lw=1.5)
    axes[1].set_title("Per-observation log-likelihood contribution")
    axes[1].set_ylabel("loglik contribution")
    fig.tight_layout()
    fig.savefig(
        outdir / "ess_and_loglik_contrib.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def save_fit_plots(dat, out_hat, sims, outdir):
    q_lo = np.quantile(sims["raw"], 0.025, axis=0)
    q_med = np.quantile(sims["raw"], 0.50, axis=0)
    q_hi = np.quantile(sims["raw"], 0.975, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(dat["date"], y_raw, lw=1.0, label="observed monthly cases")
    axes[0].plot(
        dat["date"],
        out_hat["filtered_mean_raw"],
        lw=2.0,
        label="filtered mean latent cases",
    )
    axes[0].fill_between(
        dat["date"],
        out_hat["filtered_q_raw"][:, 0],
        out_hat["filtered_q_raw"][:, 2],
        alpha=0.25,
    )
    axes[0].set_title("Observed vs TSIR-POMP filtered fit (raw scale)")
    axes[0].legend(loc="upper right")

    axes[1].plot(dat["date"], y, lw=1.0, label="observed log(1+cases)")
    axes[1].plot(
        dat["date"],
        out_hat["filtered_mean_log"],
        lw=2.0,
        label="filtered mean log latent",
    )
    axes[1].set_title("Observed vs TSIR-POMP filtered fit (log scale)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "fit_raw_log.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dat["date"], y_raw, color="black", lw=1.2, label="observed")
    ax.plot(dat["date"], q_med, lw=1.5, label="sim median")
    ax.fill_between(dat["date"], q_lo, q_hi, alpha=0.25, label="sim 95% band")
    ax.set_title("Simulation-from-fitted-model check (raw scale)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(
        outdir / "simulation_check_raw.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def save_seasonal_profile(seasonal_profile, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(
        seasonal_profile["month_name"],
        seasonal_profile["cases"],
        marker="o",
        label="observed mean month",
    )
    axes[0].plot(
        seasonal_profile["month_name"],
        seasonal_profile["filtered_mean_raw"],
        marker="o",
        label="fitted mean month",
    )
    axes[0].set_title("Mean monthly profile (raw scale)")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        seasonal_profile["month_name"],
        seasonal_profile["log_cases"],
        marker="o",
        label="observed mean log month",
    )
    axes[1].plot(
        seasonal_profile["month_name"],
        seasonal_profile["filtered_mean_log"],
        marker="o",
        label="fitted mean log month",
    )
    axes[1].set_title("Mean monthly profile (log scale)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "seasonal_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_profile_plot(profile_df, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        profile_df["delta_post"],
        profile_df["profile_loglik"],
        yerr=profile_df["rep_se"],
        marker="o",
    )
    ax.set_title("Profile likelihood for delta_post")
    ax.set_xlabel("delta_post")
    ax.set_ylabel("replicated log-likelihood")
    fig.tight_layout()
    fig.savefig(
        outdir / "profile_delta_post.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    dat = load_monthly_data(args.data, args.year_start, args.year_end)
    initialize_covariates(dat)

    with open(outdir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    theta0 = transform(
        beta0=4.0,
        a1=0.60,
        b1=-0.20,
        a2=0.15,
        b2=-0.05,
        delta_post=-1.20,
        alpha=0.92,
        nu=min(800.0, MAX_NU),
        k_proc=20.0,
        sigma_A=0.20,
        eta=0.45,
        I0=max(20.0, float(np.median(y_raw[:6]))),
        sigma_obs=0.45,
    )

    out0 = pfilter(y, theta0, n_particles=600, seed=123, return_filtered=True)
    baseline_metrics = {
        "initial_log_likelihood_logscale": float(out0["log_lik"]),
        "initial_mean_ess": float(np.mean(out0["ess"])),
    }

    global_rw_sd = default_rw_sd()
    global_fits, global_summary = run_global_search(
        y,
        n_starts=args.global_starts,
        n_iterations=args.global_iters,
        n_particles=args.global_particles,
        cooling=args.global_cooling,
        rw_sd=global_rw_sd,
        seed=args.seed,
        verbose=args.verbose_if2,
    )
    global_summary.to_csv(outdir / "global_summary.csv", index=False)
    save_global_hist(global_summary, outdir)

    global_candidates = gather_global_candidates(
        y,
        global_fits,
        top_trace_per_fit=args.top_trace_per_fit,
        rep_particles=args.rep_particles,
        rep_reps=args.rep_reps,
        seed=args.seed + 505,
    )
    global_candidates_for_csv = global_candidates.copy()
    global_candidates_for_csv["theta"] = global_candidates_for_csv[
        "theta"
    ].apply(lambda x: json.dumps([float(v) for v in x]))
    global_candidates_for_csv.to_csv(
        outdir / "global_candidates.csv", index=False
    )

    local_seeds = list(global_candidates["theta"].iloc[: args.local_top_n])
    local_rw_sd = 0.5 * default_rw_sd()
    local_fits = run_local_refinement(
        y,
        local_seeds,
        n_iterations=args.local_iters,
        n_particles=args.local_particles,
        cooling=args.local_cooling,
        rw_sd=local_rw_sd,
        seed=args.seed + 909,
        verbose=args.verbose_if2,
    )

    local_candidates = gather_global_candidates(
        y,
        local_fits,
        top_trace_per_fit=max(5, args.top_trace_per_fit),
        rep_particles=args.rep_particles,
        rep_reps=args.rep_reps,
        seed=args.seed + 1111,
    )
    local_candidates_for_csv = local_candidates.copy()
    local_candidates_for_csv["theta"] = local_candidates_for_csv[
        "theta"
    ].apply(lambda x: json.dumps([float(v) for v in x]))
    local_candidates_for_csv.to_csv(
        outdir / "local_candidates.csv", index=False
    )

    best_row = local_candidates.iloc[0].copy()
    theta_hat = best_row["theta"].copy()
    theta_hat_nat = {k: float(v) for k, v in untransform(theta_hat).items()}
    pd.DataFrame([theta_hat_nat]).to_csv(
        outdir / "best_parameters_natural_scale.csv", index=False
    )

    best_local_fit = local_fits[int(best_row["start_id"])]
    save_trace_plot(best_local_fit, outdir)

    rep = pfilter_replicated(
        y,
        theta_hat,
        n_particles=args.rep_particles,
        n_reps=args.rep_reps,
        seed=99,
    )
    out_hat = pfilter(
        y,
        theta_hat,
        n_particles=args.filter_particles,
        seed=321,
        return_filtered=True,
    )
    save_ess_plots(dat, out_hat, outdir)

    sims = simulate_from_model(theta_hat, n_sims=args.simulations, seed=222)
    save_fit_plots(dat, out_hat, sims, outdir)

    fit_df = dat.copy()
    fit_df["filtered_mean_raw"] = out_hat["filtered_mean_raw"]
    fit_df["filtered_mean_log"] = out_hat["filtered_mean_log"]
    fit_df["ess"] = out_hat["ess"]
    fit_df["loglik_t"] = out_hat["log_lik_t"]
    fit_df.to_csv(outdir / "fitted_series.csv", index=False)

    obs_idx = fit_df.groupby("year")["cases"].idxmax()
    fit_idx = fit_df.groupby("year")["filtered_mean_raw"].idxmax()

    obs_peaks = (
        fit_df.loc[obs_idx, ["year", "cases", "date"]]
        .rename(columns={"cases": "obs_peak", "date": "obs_peak_date"})
        .sort_values("year")
    )
    fit_peaks = (
        fit_df.loc[fit_idx, ["year", "filtered_mean_raw", "date"]]
        .rename(
            columns={"filtered_mean_raw": "fit_peak", "date": "fit_peak_date"}
        )
        .sort_values("year")
    )
    annual = obs_peaks.merge(fit_peaks, on="year", how="inner")
    annual["peak_abs_error"] = np.abs(annual["obs_peak"] - annual["fit_peak"])
    annual["peak_timing_error_days"] = (
        annual["fit_peak_date"] - annual["obs_peak_date"]
    ).dt.days.abs()
    annual.to_csv(outdir / "annual_peak_diagnostics.csv", index=False)

    seasonal_df = fit_df.copy()
    seasonal_df["month_name"] = seasonal_df["date"].dt.strftime("%b")
    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    seasonal_profile = seasonal_df.groupby("month_name", as_index=False)[
        ["cases", "filtered_mean_raw", "log_cases", "filtered_mean_log"]
    ].mean()
    seasonal_profile["month_name"] = pd.Categorical(
        seasonal_profile["month_name"], categories=month_order, ordered=True
    )
    seasonal_profile = seasonal_profile.sort_values("month_name").reset_index(
        drop=True
    )
    seasonal_profile.to_csv(outdir / "seasonal_profile.csv", index=False)
    save_seasonal_profile(seasonal_profile, outdir)

    profile_df = None
    if args.run_profile:
        delta_grid = np.linspace(
            theta_hat[IDX_DELTA_POST] - 0.6,
            theta_hat[IDX_DELTA_POST] + 0.6,
            args.profile_points,
        )
        profile_df = profile_delta_post(
            y,
            theta_hat,
            delta_grid,
            n_restarts=args.profile_restarts,
            n_iterations=args.profile_iters,
            n_particles=args.profile_particles,
            rw_sd=0.5 * default_rw_sd(),
            cooling=args.local_cooling,
            rep_particles=args.rep_particles,
            rep_reps=args.rep_reps,
            seed=args.seed + 1500,
        )
        profile_df.to_csv(outdir / "profile_delta_post.csv", index=False)
        save_profile_plot(profile_df, outdir)

    metrics = {
        "overall_log_likelihood_logscale": float(rep["log_lik_mean"]),
        "overall_log_likelihood_logscale_se": float(rep["log_lik_se"]),
        "mae_raw": float(
            np.mean(np.abs(y_raw - out_hat["filtered_mean_raw"]))
        ),
        "rmse_raw": float(
            np.sqrt(np.mean((y_raw - out_hat["filtered_mean_raw"]) ** 2))
        ),
        "mae_log": float(np.mean(np.abs(y - out_hat["filtered_mean_log"]))),
        "rmse_log": float(
            np.sqrt(np.mean((y - out_hat["filtered_mean_log"]) ** 2))
        ),
        "peak_mae_raw": float(annual["peak_abs_error"].mean()),
        "peak_timing_mae_days": float(annual["peak_timing_error_days"].mean()),
        "initial_log_likelihood_logscale": baseline_metrics[
            "initial_log_likelihood_logscale"
        ],
        "initial_mean_ess": baseline_metrics["initial_mean_ess"],
        "elapsed_seconds": float(time.time() - t_start),
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_summary = {
        "winner_start_id": int(best_row["start_id"]),
        "winner_trace_rank": int(best_row["trace_rank"]),
        "winner_iter": int(best_row["iter"]),
        "winner_if2_loglik": float(best_row["if2_loglik"]),
        "winner_rep_loglik": float(best_row["rep_loglik"]),
        "winner_rep_se": float(best_row["rep_se"]),
        "theta_hat_natural_scale": theta_hat_nat,
    }
    with open(outdir / "winner_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("Finished robust monthly logged TSIR-POMP run.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
