import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pypomp as pp
import jax
import time
from pathlib import Path
from scipy.stats import chi2
from pathlib import Path
import pickle
import os

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
      "phi":  jnp.arctanh(phi),
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

#par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)

y_vals = y.to_numpy()

ys = pd.DataFrame({"y": y_vals},
                  index=np.arange(1, N + 1).astype(float))

covars = pd.DataFrame({"covaryt": np.concatenate([[0.0], y_vals])},
                      index=np.arange(0, N + 1).astype(float))



rw_sd_rp = 0.02
rw_sd_ivp = 0.1
cooling_fraction_50 = 0.5



def run_local_search(n_local, J, M,
  rinit,
  rproc, 
  dmeas,
  to_est,
  from_est,
  theta = None, 
  sigmas = None, 
  init_names = ["G_0", "H_0"],
  a = 0.5, 
  key = 42):

  start_time = time.time()
  if theta is None:
    theta = {
    "sigma_nu": .01,
    "mu_h": -0.1,
    "phi": .95,
    "sigma_eta":.9,
    "G_0": 0.0,
    "H_0": 0.0,
  }
  
  if sigmas is None:
    sigmas = {
        "sigma_nu":  rw_sd_rp,
        "mu_h":      rw_sd_rp,
        "phi":       rw_sd_rp,
        "sigma_eta": rw_sd_rp,
        "G_0":       rw_sd_ivp,
        "H_0":       rw_sd_ivp,
    }
  
  
  rw_sd_local = pp.RWSigma(sigmas=sigmas, init_names=init_names)
  theta_list = [theta.copy() for _ in range(n_local)]
  par_trans_local = pp.ParTrans(to_est=to_est, from_est=from_est)
  
  mod_local = pp.Pomp(
      rinit=rinit, rproc=rproc,
      dmeas=dmeas, rmeas=rmeas,
      ys=ys, theta=theta_list,
      statenames=statenames,
      par_trans=par_trans_local,
      t0=0.0, nstep=1,
      ydim=1, covars=covars
  )
  
  key_jax = jax.random.key(key)
  
  mod_local.mif(
      J=J, M=M,
      rw_sd=rw_sd_local, a=a, key=key_jax
  )
  
  mif_result = mod_local.results_history.last()
  #jax.block_until_ready(mif_result.logLiks)
  elapsed_time = time.time() - start_time
  return {
        "mif_result": mif_result,
        "runtime": elapsed_time,
        "settings": {
            "n_local": n_local,
            "J": J,
            "M": M,
            "a": a,
            "key": key,
            "theta": theta,
            "sigmas": sigmas,
            "init_names": init_names
        }
    }
    

if __name__ == "__main__":
  # Make output directory
  outdir = Path("basic_model_results")
  outdir.mkdir(parents=True, exist_ok=True)

  # RL2 run
  output = run_local_search(
      n_local=30,
      J=2000,
      M=500,
      rinit = rinit_basic,
      rproc = rproc_filt_basic,
      dmeas = dmeas_basic,
      to_est = to_est_basic,
      from_est = from_est_basic,
      key=42
  )

  outfile = outdir / "basic_local_search.pkl"
  with open(outfile, "wb") as f:
      pickle.dump(output, f)

  print(f"Saved results to: {outfile}")
  print(f"Runtime: {output['runtime']:.2f} seconds")
  print(f"Dataset head:\n{ys.head()}")
    


