import numpy as np
import pandas as pd
import jax.numpy as jnp
import pypomp as pp
import jax
import time

from pypomp.types import (
    StateDict, ParamDict, CovarDict,
    TimeFloat, StepSizeFloat, RNGKey,
    ObservationDict, InitialTimeFloat,
)

# local search

def run_local_search(ys,
  n_local, 
  J, 
  M,
  rinit,
  rproc, 
  dmeas,
  to_est,
  from_est,
  covars,
  theta = None, 
  sigmas = None, 
  init_names = ["G_0", "H_0"],
  statenames = ["H", "G", "Y_state"],
  a = 0.5, 
  key = 42):
    
  rw_sd_rp = 0.02
  rw_sd_ivp = 0.1
  def rmeas(X_: StateDict, theta_: ParamDict,
        key: RNGKey, covars: CovarDict,
        t: TimeFloat):
    H = X_["H"]
    return jnp.array(
        [jax.random.normal(key) * jnp.exp(H / 2)])

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





# global search

def run_global_search(ys,
  n_starts, 
  reps,
  rinit,
  rproc,
  dmeas,
  to_est,
  from_est,
  covars,
  grid = None, # grid specifies bounds to search through for each param
  J1 = 2000,
  J2 = 2000,
  J3 = 5000,
  M1 = 50,
  M2 = 100,
  init_names = ["G_0", "H_0"],
  statenames = ["H", "G", "Y_state"],
  a = 0.5,
  theta = None,
  sigmas = None,
  seed = 42, 
  key = 42):
    
  rw_sd_rp = 0.02
  rw_sd_ivp = 0.1
  
  def rmeas(X_: StateDict, theta_: ParamDict,
        key: RNGKey, covars: CovarDict,
        t: TimeFloat):
    H = X_["H"]
    return jnp.array(
        [jax.random.normal(key) * jnp.exp(H / 2)])
  
  
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
  
  if grid is None:
    grid = {
            "sigma_nu": (0.005, 0.05),
            "mu_h": (-1.0, 1.0),
            "phi": (0.9, 0.99),
            "sigma_eta": (0.5, 2.0),
            "G_0": (-1.0, 1.0),
            "H_0": (-1.0, 1.0),
        }
  
  rw_sd_local = pp.RWSigma(sigmas=sigmas, init_names=init_names)
  par_trans_local = pp.ParTrans(to_est=to_est, from_est=from_est)

  np.random.seed(seed)
  theta_list = []
  for i in range(n_starts):
    th = theta.copy()
    for pname, bounds in grid.items():
      low, high = bounds
      th[pname] = np.random.uniform(low, high)
    
    theta_list.append(th)

  mod = pp.Pomp(
    rinit=rinit, rproc=rproc,
    dmeas=dmeas, rmeas=rmeas,
    ys=ys, theta=theta_list,
    statenames=statenames,
    par_trans=par_trans_local,
    t0=0.0, nstep=1,
    ydim=1, covars=covars)
    
  base_key = jax.random.key(key)
  key1, key2, key3 = jax.random.split(base_key, 3)
  mod.mif(J=J1, M=M1,
            rw_sd=rw_sd_local, a=a, key=key1)
  mif1 = mod.results_history.last()
  
  mod.mif(J=J2, M=M2,
            rw_sd=rw_sd_local, a=a, key=key2)
  mif2 = mod.results_history.last()

  mod.pfilter(key=key3, J=J3, reps=reps, ESS = True)
  pf = mod.results_history.last()

  rows = []
  for i in range(n_starts):
    lls = pf.logLiks.values[i, :]
    rows.append({
          **mod.theta[i],
          'loglik': pp.logmeanexp(lls),
          'loglik_se': pp.logmeanexp_se(lls)
      })
  results_df = pd.DataFrame(rows)
  results_df = results_df[np.isfinite(results_df['loglik'])]
  elapsed_time = time.time() - start_time
  return {
      "results": results_df,
      "mif1_traces": mif1.traces_da,
      "mif2_traces": mif2.traces_da,
      "pf": pf,
      "runtime": elapsed_time,
      "settings": {
          "n_starts": n_starts,
          "reps": reps,
          "grid": grid,
          "J1": J1,
          "J2": J2,
          "J3": J3,
          "M1": M1,
          "M2": M2,
          "key": key,
          "seed": seed,
          "theta": theta,
          "sigmas": sigmas,
          "init_names": init_names
      }
  }



