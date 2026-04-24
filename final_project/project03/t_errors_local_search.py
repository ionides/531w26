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

def rinit_t_errors(theta_: ParamDict, key: RNGKey,
        covars: CovarDict, t0: InitialTimeFloat):
  G_0 = theta_["G_0"]
  H_0 = theta_["H_0"]
  nu = theta_["nu"]
  Y_state = jax.random.t(key, df = nu) * jnp.sqrt((nu - 2.0) / nu) * jnp.exp(H_0 / 2)
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


def to_est_t_errors(theta):
  eps = 1e-6
  phi = jnp.clip(theta["phi"], -1 + eps, 1 - eps)
  
  return {
      "sigma_nu": jnp.log(theta["sigma_nu"]),
      "mu_h": theta["mu_h"],
      "phi": jnp.arctanh(phi),
      "sigma_eta": jnp.log(theta["sigma_eta"]),
      "G_0": theta["G_0"],
      "H_0": theta["H_0"],
      "nu": jnp.log(theta["nu"] - 2.1)
  }

def from_est_t_errors(theta):
  return {
      "sigma_nu": jnp.exp(theta["sigma_nu"]),
      "mu_h": theta["mu_h"],
      "phi": jnp.tanh(theta["phi"]),
      "sigma_eta": jnp.exp(theta["sigma_eta"]),
      "G_0": theta["G_0"],
      "H_0": theta["H_0"],
      "nu": 2.1 + jnp.exp(theta["nu"])
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
  outdir = Path("t_errors_model_results")
  outdir.mkdir(parents=True, exist_ok=True)

  # RL2 run
  output = run_local_search(ys = ys,
  n_local = 30, J = 2000, M = 500,
  rinit = rinit_t_errors,
  rproc = rproc_filt_basic,
  dmeas = dmeas_t_errors,
  to_est = to_est_t_errors,
  from_est = from_est_t_errors,
  covars = covars,
  theta = {
    "sigma_nu": .01,
    "mu_h": -0.1,
    "phi": .95,
    "sigma_eta":.9,
    "G_0": 0.0,
    "H_0": 0.0,
    "nu": 5.0
  },
  sigmas = {
        "sigma_nu":  rw_sd_rp,
        "mu_h":      rw_sd_rp,
        "phi":       rw_sd_rp,
        "sigma_eta": rw_sd_rp,
        "G_0":       rw_sd_ivp,
        "H_0":       rw_sd_ivp,
        "nu": .1
    },
  )

  outfile = outdir / "t_errors_local_search.pkl"
  with open(outfile, "wb") as f:
      pickle.dump(output, f)

  print(f"Saved results to: {outfile}")
  print(f"Runtime: {output['runtime']:.2f} seconds")
  
  
