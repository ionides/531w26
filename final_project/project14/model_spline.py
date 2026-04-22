"""
Spline seasonality measles model for STATS 531 final project.

Identical to model_001b EXCEPT the fixed school-term seasonal forcing
is replaced by a flexible 6-basis Fourier spline, following the
dacca (pp.dacca) approach from Ionides et al.

Instead of:
    seas = step_function(school_terms, amplitude)

We use:
    log(beta(t)) = bs1*s1(t) + bs2*s2(t) + ... + bs6*s6(t)

where s1..s6 are periodic basis functions (sin/cos pairs) covering
the annual cycle. This allows the data to determine the seasonal
shape rather than imposing school-term windows.

New parameters: bs1, bs2, bs3, bs4, bs5, bs6
Removed parameters: amplitude (subsumed by spline)
Kept: cohort (school-entry pulse is still a distinct mechanism)

Scientific question: does a flexible spline fit London measles
significantly better than fixed school-term forcing?
"""

import jax.numpy as jnp
import jax
import jax.scipy.special as jspecial
from pypomp.random.poissoninvf import fast_approx_rpoisson
from pypomp.random.binominvf import fast_approx_rmultinom
from pypomp.random.gammainvf import fast_approx_rgamma
from pypomp.types import (
    StateDict,
    ParamDict,
    CovarDict,
    TimeFloat,
    StepSizeFloat,
    RNGKey,
    InitialTimeFloat,
    ObservationDict,
)

# ── parameter names ───────────────────────────────────────────────────
param_names = (
    "R0",       # basic reproduction number
    "sigma",    # latency rate
    "gamma",    # recovery rate
    "iota",     # imported cases
    "rho",      # reporting rate
    "sigmaSE",  # environmental stochasticity
    "psi",      # obs overdispersion
    "cohort",   # school cohort pulse (kept from 001b)
    # spline coefficients — replace amplitude
    "bs1", "bs2", "bs3", "bs4", "bs5", "bs6",
    # initial compartment fractions
    "S_0", "E_0", "I_0", "R_0",
)

statenames = ["S", "E", "I", "R", "W", "C"]
accumvars  = ["W", "C"]


def _spline_seas(t_mod: jax.Array, bs: jax.Array) -> jax.Array:
    """
    Compute the seasonal multiplier at fractional time t_mod in [0,1).

    Uses 6 Fourier basis functions (3 sin/cos pairs) to represent
    an arbitrary annual periodic function, following the dacca
    spline approach recommended by Ionides.

    The basis is:
        s1 = sin(2*pi*t),  s2 = cos(2*pi*t)
        s3 = sin(4*pi*t),  s4 = cos(4*pi*t)
        s5 = sin(6*pi*t),  s6 = cos(6*pi*t)

    The seasonal log-transmission multiplier is:
        log_seas = dot(bs, [s1,s2,s3,s4,s5,s6])
        seas = exp(log_seas)

    This is strictly positive and captures arbitrary smooth
    annual periodicity without assuming school-term windows.
    """
    two_pi_t = 2.0 * jnp.pi * t_mod
    basis = jnp.array([
        jnp.sin(two_pi_t),
        jnp.cos(two_pi_t),
        jnp.sin(2.0 * two_pi_t),
        jnp.cos(2.0 * two_pi_t),
        jnp.sin(3.0 * two_pi_t),
        jnp.cos(3.0 * two_pi_t),
    ])
    return jnp.exp(jnp.dot(bs, basis))


# ── rinit — identical to 001b ─────────────────────────────────────────
def rinit(theta_: ParamDict, key: RNGKey,
          covars: CovarDict, t0: InitialTimeFloat):
    S_0 = theta_["S_0"]
    E_0 = theta_["E_0"]
    I_0 = theta_["I_0"]
    R_0 = theta_["R_0"]
    m = covars["pop"] / (S_0 + E_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    E = jnp.round(m * E_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    W = 0
    C = 0
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


# ── rproc — spline seasonality replaces school-term step ──────────────
def rproc(X_: StateDict, theta_: ParamDict, key: RNGKey,
          covars: CovarDict, t: TimeFloat, dt: StepSizeFloat):
    S, E, I, R, W, C = (X_["S"], X_["E"], X_["I"],
                         X_["R"], X_["W"], X_["C"])
    R0      = theta_["R0"]
    sigma   = theta_["sigma"]
    gamma   = theta_["gamma"]
    iota    = theta_["iota"]
    sigmaSE = theta_["sigmaSE"]
    cohort  = theta_["cohort"]
    pop        = covars["pop"]
    birthrate  = covars["birthrate"]
    mu = 0.02

    # spline coefficients
    bs = jnp.array([theta_[f"bs{i}"] for i in range(1, 7)])

    # fractional year
    t_mod = t - jnp.floor(t)

    # cohort pulse — identical to 001b (school entry in September)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    br = jnp.where(
        is_cohort_time,
        cohort * birthrate / dt + (1 - cohort) * birthrate,
        (1 - cohort) * birthrate,
    )

    # ── KEY CHANGE: spline seasonality replaces school-term step ──────
    seas = _spline_seas(t_mod, bs)
    # ------------------------------------------------------------------

    # transmission rate (same formula as 001b, different seas)
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # force of infection
    foi = beta * (I + iota) / pop

    # white noise
    keys = jax.random.split(key, 3)
    dw   = fast_approx_rgamma(keys[0], dt / sigmaSE**2) * sigmaSE**2

    rate = jnp.array([foi * dw / dt, mu, sigma, mu, gamma, mu])

    # Poisson births
    births = fast_approx_rpoisson(keys[1], br * dt)

    # transitions — identical to 001b
    rt_final   = jnp.zeros((3, 3))
    rate_pairs = jnp.array([
        [rate[0], rate[1]],
        [rate[2], rate[3]],
        [rate[4], rate[5]],
    ])
    populations = jnp.array([S, E, I])
    rate_sums   = jnp.sum(rate_pairs, axis=1)
    p0_values   = jnp.exp(-rate_sums * dt)
    rt_final = (
        rt_final
        .at[:, 0:2].set(
            jnp.einsum("ij,i,i->ij",
                       rate_pairs, 1 / rate_sums, 1 - p0_values))
        .at[:, 2].set(p0_values)
    )
    transitions = fast_approx_rmultinom(keys[2], populations, rt_final)
    trans_S, trans_E, trans_I = transitions

    S = S + births - trans_S[0] - trans_S[1]
    E = E + trans_S[0] - trans_E[0] - trans_E[1]
    I = I + trans_E[0] - trans_I[0] - trans_I[1]
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + trans_I[0]
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


# ── dmeas / rmeas — identical to 001b ────────────────────────────────
def dmeas(Y_: ObservationDict, X_: StateDict,
          theta_: ParamDict, covars: CovarDict, t: TimeFloat):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C   = X_["C"]
    tol = 1.0e-18
    y   = Y_["cases"]
    m   = rho * C
    v   = m * (1.0 - rho + psi**2 * m)
    sqrt_v_tol = jnp.sqrt(v) + tol
    upper_cdf  = jax.scipy.stats.norm.cdf(y + 0.5, m, sqrt_v_tol)
    lower_cdf  = jax.scipy.stats.norm.cdf(y - 0.5, m, sqrt_v_tol)
    lik = jnp.where(y > tol, upper_cdf - lower_cdf, upper_cdf) + tol
    lik = jnp.where(C < 0,          0.0, lik)
    lik = jnp.where(jnp.isnan(y),   1.0, lik)
    return jnp.log(lik)


def rmeas(X_: StateDict, theta_: ParamDict,
          key: RNGKey, covars: CovarDict, t: TimeFloat):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C   = X_["C"]
    m   = rho * C
    v   = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)


# ── parameter transformations ─────────────────────────────────────────
def to_est(theta: dict) -> dict:
    SEIR_0 = jnp.array([theta["S_0"], theta["E_0"],
                         theta["I_0"], theta["R_0"]])
    S_0, E_0, I_0, R_0 = jnp.log(SEIR_0 / jnp.sum(SEIR_0))
    return {
        "R0":      jnp.log(theta["R0"]),
        "sigma":   jnp.log(theta["sigma"]),
        "gamma":   jnp.log(theta["gamma"]),
        "iota":    jnp.log(theta["iota"]),
        "sigmaSE": jnp.log(theta["sigmaSE"]),
        "psi":     jnp.log(theta["psi"]),
        "cohort":  jspecial.logit(theta["cohort"]),
        "rho":     jspecial.logit(theta["rho"]),
        # spline coefficients are unconstrained
        **{f"bs{i}": theta[f"bs{i}"] for i in range(1, 7)},
        "S_0": S_0, "E_0": E_0, "I_0": I_0, "R_0": R_0,
    }


def from_est(theta: dict) -> dict:
    SEIR_0 = jnp.exp(jnp.array([theta["S_0"], theta["E_0"],
                                  theta["I_0"], theta["R_0"]]))
    S_0, E_0, I_0, R_0 = SEIR_0 / jnp.sum(SEIR_0)
    return {
        "R0":      jnp.exp(theta["R0"]),
        "sigma":   jnp.exp(theta["sigma"]),
        "gamma":   jnp.exp(theta["gamma"]),
        "iota":    jnp.exp(theta["iota"]),
        "sigmaSE": jnp.exp(theta["sigmaSE"]),
        "psi":     jnp.exp(theta["psi"]),
        "cohort":  jspecial.expit(theta["cohort"]),
        "rho":     jspecial.expit(theta["rho"]),
        **{f"bs{i}": theta[f"bs{i}"] for i in range(1, 7)},
        "S_0": S_0, "E_0": E_0, "I_0": I_0, "R_0": R_0,
    }

# accumvars exposed for measlesPomp.py
accumvars  = ["W", "C"]
statenames = ["S", "E", "I", "R", "W", "C"]
