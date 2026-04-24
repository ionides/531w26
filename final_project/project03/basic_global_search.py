import numpy as np
import pandas as pd
import jax.numpy as jnp
import pypomp as pp
import jax
from pathlib import Path
import pickle
from search_helpers import run_global_search
from pypomp.types import (
    StateDict, ParamDict, CovarDict,
    TimeFloat, StepSizeFloat, RNGKey,
    ObservationDict, InitialTimeFloat,
)


y = pd.read_csv('google_returns_cleaned.csv')
y.set_index('Date', inplace = True)
y = y['Close']

N = len(y)
statenames = ["H", "G", "Y_state"]
def rinit_basic(theta_: ParamDict, key: RNGKey,
        covars: CovarDict, t0: InitialTimeFloat):
  G_0 = theta_["G_0"]
  H_0 = theta_["H_0"]
  Y_state = jax.random.normal(key) * jnp.exp(H_0 / 2)
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
  nu = jax.random.normal(key2) * sigma_nu
  G = G + nu
  beta = Y_state * sigma_eta * jnp.sqrt(1 - phi**2)
  H = (mu_h * (1 - phi) + phi * H
      + beta * jnp.tanh(G) * jnp.exp(-H / 2)
      + omega)
  Y_state = covars["covaryt"]
  return {"H": H, "G": G, "Y_state": Y_state}


def dmeas_basic(Y_: ObservationDict, X_: StateDict,
        theta_: ParamDict, covars: CovarDict,
        t: TimeFloat):
  H = X_["H"]
  y = Y_["y"]
  return jax.scipy.stats.norm.logpdf(
      y, 0, jnp.exp(H / 2))

def rmeas(X_: StateDict, theta_: ParamDict,
        key: RNGKey, covars: CovarDict,
        t: TimeFloat):
  H = X_["H"]
  return jnp.array(
      [jax.random.normal(key) * jnp.exp(H / 2)])


def to_est_basic(theta):
  eps = 1e-6
  phi = jnp.clip(theta["phi"], -1 + eps, 1 - eps)
  
  return {
      "sigma_nu": jnp.log(theta["sigma_nu"]),
      "mu_h": theta["mu_h"],
      "phi": jnp.arctanh(phi),
      "sigma_eta": jnp.log(theta["sigma_eta"]),
      "G_0": theta["G_0"],
      "H_0": theta["H_0"],
  }

def from_est_basic(theta):
  return {
      "sigma_nu": jnp.exp(theta["sigma_nu"]),
      "mu_h": theta["mu_h"],
      "phi": jnp.tanh(theta["phi"]),
      "sigma_eta": jnp.exp(theta["sigma_eta"]),
      "G_0": theta["G_0"],
      "H_0": theta["H_0"],
  }


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
  outdir = Path("basic_model_results")
  outdir.mkdir(parents=True, exist_ok=True)

  # run the global search
  output = run_global_search(ys = ys,
        n_starts=250,
        reps = 10,
        rinit = rinit_basic,
        rproc = rproc_filt_basic,
        dmeas = dmeas_basic,
        to_est = to_est_basic,
        from_est = from_est_basic,
        covars = covars,
        J1 = 2000,
        J2 = 2000,
        J3 = 2000,
        M1 = 50,
        M2 = 250,
        grid = {
    "sigma_nu":  (0.002, 0.05),
    "mu_h":      (-5, 5),
    "phi":       (0.01, 0.99),
    "sigma_eta": (0.1, 5),
    "G_0":       (-5, 5),
    "H_0":       (-5, 5),
    }
    )

  outfile = outdir / "basic_global_search.pkl"
  with open(outfile, "wb") as f:
     pickle.dump(output, f)

  print(f"Saved results to: {outfile}")
  print(f"Runtime: {output['runtime']:.2f} seconds")
  
