import numpy as np
import pandas as pd
import jax.numpy as jnp
import pypomp as pp
import jax
import time
from pathlib import Path
import pickle
from jax.scipy.special import gammaln
from search_helpers import run_local_search


y = pd.read_csv('google_returns_cleaned.csv')
y.set_index('Date', inplace = True)
y = y['Close']

from pypomp.types import (
    StateDict, ParamDict, CovarDict,
    TimeFloat, StepSizeFloat, RNGKey,
    ObservationDict, InitialTimeFloat,
)

N = len(y)
statenames = ["H", "G", "Y_state"]

def hansen_skewt_constants(nu, lam):
    """
    Constants from Hansen (1994), page 6.

    nu > 2
    -1 < lam < 1
    """
    nu = jnp.asarray(nu)
    lam = jnp.asarray(lam)

    log_c = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
    )
    c = jnp.exp(log_c)

    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = jnp.sqrt(1.0 + 3.0 * lam**2 - a**2)

    return a, b, log_c


def r_hansen_skewt(key, nu, lam, shape=()):
    """
    Direct sampler for Hansen's standardized skewed t.

    This is derived from the piecewise density in Hansen (1994),
    but implemented without inverse-CDF uniforms.

    Returns draws with mean 0 and variance 1 when nu > 2 and |lam| < 1.
    """
    nu = jnp.asarray(nu)
    lam = jnp.asarray(lam)

    key_side, key_t = jax.random.split(key)
    a, b, _ = hansen_skewt_constants(nu, lam)

    # Side probabilities implied by the paper's skew construction
    p_right = (1.0 + lam) / 2.0
    go_right = jax.random.bernoulli(key_side, p_right, shape=shape)

    # Ordinary Student t draw, then fold to half-t
    t_raw = jax.random.t(key_t, df=nu, shape=shape)
    t_abs = jnp.abs(t_raw)

    # Put mass on left/right half with the correct probabilities
    t_half = jnp.where(go_right, t_abs, -t_abs)

    # Standardization factor used throughout Hansen / arch
    s = jnp.sqrt(1.0 - 2.0 / nu)

    # Branch-specific scale: (1-lam) on left, (1+lam) on right
    branch_scale = jnp.where(go_right, 1.0 + lam, 1.0 - lam)

    z = (s * branch_scale * t_half - a) / b
    return z
  
def hansen_skewt_logpdf(x, nu, lam):
    """
    Log density of Hansen's standardized skewed t.
    """
    x = jnp.asarray(x)
    nu = jnp.asarray(nu)
    lam = jnp.asarray(lam)

    a, b, log_c = hansen_skewt_constants(nu, lam)

    sign_term = jnp.sign(x + a / b)
    denom = 1.0 + sign_term * lam
    q = (a + b * x) / denom

    return (
        jnp.log(b)
        + log_c
        - 0.5 * (nu + 1.0) * jnp.log1p(q**2 / (nu - 2.0))
    )
  

def rinit_skewt(theta_: ParamDict, key: RNGKey,
        covars: CovarDict, t0: InitialTimeFloat):
  G_0 = theta_["G_0"]
  H_0 = theta_["H_0"]
  nu = theta_["nu"]
  lam = theta_["lam"]
  Y_state = r_hansen_skewt(key, nu, lam) * jnp.exp(H_0 / 2)
  return {"H": H_0, "G": G_0, "Y_state": Y_state}


def rproc_filt_basic(X_: StateDict, theta_: ParamDict,
        key: RNGKey, covars: CovarDict,
        t: TimeFloat, dt: StepSizeFloat):
  H = X_["H"]
  G = X_["G"]
  Y_state = X_["Y_state"]
  sigma_nu = theta_["sigma_nu"]
  mu_h = theta_["mu_h"]
  phi = theta_["phi"]
  sigma_eta = theta_["sigma_eta"]
  key1, key2 = jax.random.split(key)
  omega = jax.random.normal(key1) * (
      sigma_eta * jnp.sqrt(1 - phi**2)
      * jnp.sqrt(1 - jnp.tanh(G)**2))
  nu_shock = jax.random.normal(key2) * sigma_nu
  G = G + nu_shock
  beta = Y_state * sigma_eta * jnp.sqrt(1 - phi**2)
  H = (mu_h * (1 - phi) + phi * H
      + beta * jnp.tanh(G) * jnp.exp(-H / 2)
      + omega)
  Y_state = covars["covaryt"]
  return {"H": H, "G": G, "Y_state": Y_state}


def dmeas_skewt(Y_, X_, theta_, covars, t):
  y = Y_["y"]
  H = X_["H"]

  nu = theta_["nu"]
  lam = theta_["lam"]

  z = y * jnp.exp(-H / 2.0)
  return hansen_skewt_logpdf(z, nu=nu, lam=lam) - H / 2.0
  

def rmeas(X_: StateDict, theta_: ParamDict,
        key: RNGKey, covars: CovarDict,
        t: TimeFloat):
  H = X_["H"]
  nu = theta_["nu"]
  lam = theta_["lam"]
  return jnp.array(
      [r_hansen_skewt(key, nu, lam) * jnp.exp(H / 2)])



def to_est_skewt(theta):
  eps = 1e-6
  phi = jnp.clip(theta["phi"], -1 + eps, 1 - eps)
  
  return {
      "sigma_nu": jnp.log(theta["sigma_nu"]),
      "mu_h": theta["mu_h"],
      "phi": jnp.arctanh(phi),
      "sigma_eta": jnp.log(theta["sigma_eta"]),
      "G_0": theta["G_0"],
      "H_0": theta["H_0"],
      "nu": jnp.log(theta["nu"] - 2.1),
      "lam": jnp.arctanh(theta["lam"])
  }

def from_est_skewt(theta):
  return {
      "sigma_nu": jnp.exp(theta["sigma_nu"]),
      "mu_h": theta["mu_h"],
      "phi": jnp.tanh(theta["phi"]),
      "sigma_eta": jnp.exp(theta["sigma_eta"]),
      "G_0": theta["G_0"],
      "H_0": theta["H_0"],
      "nu": 2.1 + jnp.exp(theta["nu"]),
      "lam": jnp.tanh(theta["lam"])
  }
  



#par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)

y_vals = y.to_numpy()

ys = pd.DataFrame({"y": y_vals},
                  index=np.arange(1, N + 1).astype(float))

covars = pd.DataFrame({"covaryt": np.concatenate([[0.0], y_vals])},
                      index=np.arange(0, N + 1).astype(float))


rw_sd_rp = 0.02
rw_sd_ivp = 0.1
cooling_fraction_50 = 0.5


if __name__ == "__main__":
  # Make output directory
  outdir = Path("skewt_model_results")
  outdir.mkdir(parents=True, exist_ok=True)

  # RL2 run
  output = run_local_search(ys = ys,
  n_local = 30, J = 2000, M = 500,
  rinit = rinit_skewt,
  rproc = rproc_filt_basic,
  dmeas = dmeas_skewt,
  to_est = to_est_skewt,
  from_est = from_est_skewt,
  covars = covars,
  theta = {
    "sigma_nu": .01,
    "mu_h": -0.1,
    "phi": .95,
    "sigma_eta":.9,
    "G_0": 0.0,
    "H_0": 0.0,
    "nu": 5.0,
    "lam": 0
  },
  sigmas = {
        "sigma_nu":  rw_sd_rp,
        "mu_h":      rw_sd_rp,
        "phi":       rw_sd_rp,
        "sigma_eta": rw_sd_rp,
        "G_0":       rw_sd_ivp,
        "H_0":       rw_sd_ivp,
        "nu": .1,
        "lam": .02
    },
  )

  outfile = outdir / "skewt_local_search.pkl"
  with open(outfile, "wb") as f:
      pickle.dump(output, f)

  print(f"Saved results to: {outfile}")
  print(f"Runtime: {output['runtime']:.2f} seconds")
  print(f"Dataset head:\n{ys.head()}")
  
  
