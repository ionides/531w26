import numpy as np
import pandas as pd
import jax.numpy as jnp
import pypomp as pp
import jax
from pathlib import Path
import pickle
from pypomp.types import (
    StateDict, ParamDict, CovarDict,
    TimeFloat, StepSizeFloat, RNGKey,
    ObservationDict, InitialTimeFloat,
)
from search_helpers import run_global_search

y = pd.read_csv('google_returns_cleaned.csv')
y.set_index('Date', inplace = True)
y = y['Close']

N = len(y)
statenames = ["H", "H_prev", "G", "Y_state"]

y_vals = y.to_numpy()

ys = pd.DataFrame({"y": y_vals},
                  index=np.arange(1, N + 1).astype(float))

covars = pd.DataFrame({"covaryt": np.concatenate([[0.0], y_vals])},
                      index=np.arange(0, N + 1).astype(float))
                      


rw_sd_rp = 0.02
rw_sd_ivp = 0.1


def rinit_ar2(theta_, key, covars, t0):
  H0 = theta_["H_0"]
  G0 = theta_["G_0"]
  return {
      "H": H0,          # current H
      "H_prev": H0,     # previous lag, i.e. H_{-1} or chosen initial lag
      "G": G0,
      "Y_state": 0.0, 
  }

def rproc_filt_ar2(X_, theta_, key, covars, t, dt):
  H = X_["H"]           # H_{n-1}
  H_prev = X_["H_prev"] #  H_{n-2}
  G = X_["G"]
  Y_state = X_["Y_state"]

  mu_h = theta_["mu_h"]
  phi_1 = theta_["phi_1"]
  phi_2 = theta_["phi_2"]
  sigma_eta = theta_["sigma_eta"]
  sigma_nu = theta_["sigma_nu"]

  key1, key2 = jax.random.split(key)

  # update leverage state
  nu_shock = jax.random.normal(key1) * sigma_nu
  G_new = G + nu_shock
  R_new = jnp.tanh(G_new)

  beta = Y_state * sigma_eta
  
  omega = jax.random.normal(key2) * (
      sigma_eta * jnp.sqrt(1 - jnp.tanh(G)**2))

  H_new = (
      mu_h * (1 - phi_1 - phi_2)
      + phi_1 * H
      + phi_2 * H_prev
      + beta * R_new * jnp.exp(-H / 2)
      + omega
  )
  
  Y_state = covars["covaryt"]

  return {
      "H": H_new,
      "H_prev": H,     # shift the lag forward
      "G": G_new,
      "Y_state": Y_state,
  }
  
def dmeas_t_errors(Y_: ObservationDict, X_: StateDict,
        theta_: ParamDict, covars: CovarDict,
        t: TimeFloat):
  H = X_["H"]
  y = Y_["y"]
  nu = theta_["nu"]
  
  sd = jnp.sqrt(nu / (nu - 2.0)) # standard dev of t distribution with df degrees of freedom
  
  # log density of t distribution using b * t has pdf (1 / b) f(x / b)
  # t distribution is normalized to have variance 1
  return jnp.log(sd) + (-H / 2.0) + jax.scipy.stats.t.logpdf(
      y * sd * jnp.exp(-H / 2.0), df = nu) 

def rmeas(X_: StateDict, theta_: ParamDict,
        key: RNGKey, covars: CovarDict,
        t: TimeFloat):
  H = X_["H"]
  return jnp.array(
      [jax.random.normal(key) * jnp.exp(H / 2)])


def to_est_ar2_t(theta):
  est = theta.copy()

  # positive parameters
  est["sigma_nu"] = jnp.log(theta["sigma_nu"])
  est["sigma_eta"] = jnp.log(theta["sigma_eta"])

  phi_1 = theta["phi_1"]
  phi_2 = theta["phi_2"]

  eps = 1e-6

  # protect denominator away from 0
  denom = jnp.maximum(1.0 - phi_2, eps)

  kappa_2 = phi_2
  kappa_1 = phi_1 / denom

  # clip to valid PACF region (strictly inside (-1,1))
  kappa_1 = jnp.clip(kappa_1, -1 + eps, 1 - eps)
  kappa_2 = jnp.clip(kappa_2, -1 + eps, 1 - eps)

  est["phi_1"] = jnp.arctanh(kappa_1)
  est["phi_2"] = jnp.arctanh(kappa_2)

  est["mu_h"] = theta["mu_h"]
  est["G_0"] = theta["G_0"]
  est["H_0"] = theta["H_0"]
  est["nu"] = jnp.log(theta["nu"] - 2.1)
  #est["H_prev_0"] = theta["H_prev_0"]

  return est

def from_est_ar2_t(est):
  theta = est.copy()

  eps = 1e-6

  # positive parameters
  theta["sigma_nu"] = jnp.exp(est["sigma_nu"])
  theta["sigma_eta"] = jnp.exp(est["sigma_eta"])

  # unconstrained -> PACF
  kappa_1 = jnp.tanh(est["phi_1"])
  kappa_2 = jnp.tanh(est["phi_2"])

  # optional: keep slightly away from boundary
  kappa_1 = jnp.clip(kappa_1, -1 + eps, 1 - eps)
  kappa_2 = jnp.clip(kappa_2, -1 + eps, 1 - eps)

  # PACF -> AR(2)
  phi_2 = kappa_2
  phi_1 = kappa_1 * (1.0 - kappa_2)

  theta["phi_2"] = phi_2
  theta["phi_1"] = phi_1

  theta["mu_h"] = est["mu_h"]
  theta["G_0"] = est["G_0"]
  theta["H_0"] = est["H_0"]
  theta['nu'] = 2.1 + jnp.exp(est["nu"])
  #theta["H_prev_0"] = est["H_prev_0"]

  return theta

if __name__ == "__main__":
  # Make output directory
  outdir = Path("ar2_t_model_results")
  outdir.mkdir(parents=True, exist_ok=True)

  # run the global search
  output = run_global_search(ys = ys,
        n_starts=260,
        reps = 10,
        rinit = rinit_ar2,
        rproc = rproc_filt_ar2,
        dmeas = dmeas_t_errors,
        to_est = to_est_ar2_t,
        from_est = from_est_ar2_t,
        covars = covars,
        statenames = statenames,
        J1 = 2000,
        J2 = 2000,
        J3 = 2000,
        M1 = 50,
        M2 = 250,
         theta = {
    "sigma_nu": .01,
    "mu_h": -0.1,
    "phi_1": .95,
    "phi_2": 0.0,
    "sigma_eta":.9,
    "G_0": 0.1,
    "H_0": 0.1,
    "nu": 5.0
  },
        grid = {
    "sigma_nu":  (0.002, 0.01),
    "mu_h":      (-3, 3),
    "phi_1":       (-0.95, 0.95),
    "phi_2": (-0.95, 0.95),
    "sigma_eta": (0.1, 1),
    "G_0":       (-5, 5),
    "H_0":       (-5, 5),
    "nu": (2.5, 8)
    },
    sigmas = {
        "sigma_nu":  rw_sd_rp,
        "mu_h":      rw_sd_rp,
        "phi_1":       rw_sd_rp,
        "phi_2":       rw_sd_rp,
        "sigma_eta": rw_sd_rp,
        "G_0":       rw_sd_ivp,
        "H_0":       rw_sd_ivp,
        "nu": .1
    }
    )

  outfile = outdir / "ar2_t_global_search.pkl"
  with open(outfile, "wb") as f:
      pickle.dump(output, f)

  print(f"Saved results to: {outfile}")
  print(f"Runtime: {output['runtime']:.2f} seconds")
