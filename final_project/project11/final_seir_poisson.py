import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
import jax.random as random
import jax.numpy as jnp
import jax
from pypomp.random import (fast_approx_rbinom, fast_approx_rgamma, fast_approx_rpoisson)
import pypomp as pp
import numpy as np

data = pd.read_csv("FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv")
time_series = data['TOTAL SPECIMENS']




theta = {
'N': 2300000.0,
'beta': 1.928,
'gamma': 2.728,
'eta': 0.0121,
'omega': 0.15,
'sigma': 0.45,
'amplitude': 20.0,
'rho': 0.05,
'k': 1.0
}


def rinit(theta_, key, covars, t0):
	N = theta_["N"]
	S0 = jnp.round(N)
	E0 = 0.0
	I0 = 1.0
	R0 = 0.0
	return {"S": S0, "E": E0, "I": I0, "R": R0}

def rproc(X_, theta_, key, covars, t, dt):
	S = jnp.asarray(X_["S"])
	E = jnp.asarray(X_["E"])
	I = jnp.asarray(X_["I"])
	R = jnp.asarray(X_["R"])

	beta = theta_["beta"]
	N = theta_["N"]
	gamma = theta_["gamma"]
	omega = theta_["omega"]
	eta = theta_["eta"]
	sigma = theta_["sigma"]
	amp = theta_["amplitude"]

	beta_t = beta * (1 + 0.2 * jnp.sin(amp * jnp.pi * t / 365))

	rate_SI = (beta_t * S * I / N  + eta * S ) * dt
	rate_EI = sigma * E * dt
	rate_IR = gamma * I * dt
	rate_RS = omega * R * dt

	key_SI, key_EI, key_IR, key_RS = jax.random.split(key, 4)

	dN_SI = jax.random.poisson(key_SI, rate_SI, shape=None, dtype=None)
	dN_EI = jax.random.poisson(key_EI, rate_EI, shape=None, dtype=None)
	dN_IR = jax.random.poisson(key_IR, rate_IR, shape=None, dtype=None)
	dN_SR = jax.random.poisson(key_RS, rate_RS, shape=None, dtype=None)

	S_new = S - dN_SI + dN_SR
	E_new = E + dN_SI - dN_EI
	I_new = I + dN_EI - dN_IR
	R_new = R + dN_IR - dN_SR

	return {"S": S_new, "E": E_new, "I": I_new, "R": R_new}

# evaluate f(y_n | x_n, theta)
def dmeas(Y_, X_, theta_, covars, t):
  rho = theta_["rho"]
  k = theta_["k"]
  R = X_["R"]
  mu = rho * R
  return nbinom_logpmf(Y_["reports"], k, mu)

# draw f(y_n |x_n, theta)
def rmeas(X_, theta_, key, covars, t):
  rho = theta_["rho"]
  k = theta_["k"]
  R = X_["R"]
  mu = rho * R
  reports = rnbinom(key, k, mu)
  return jnp.array([reports])

def nbinom_logpmf(x, k, mu):
  x = jnp.asarray(x)
  k = jnp.asarray(k)
  mu = jnp.asarray(mu)
  logp_zero = jnp.where(x == 0, 0.0,-jnp.inf)
  safe_mu = jnp.where(mu == 0.0, 1.0, mu)
  core = (jax.scipy.special.gammaln(k + x)
    - jax.scipy.special.gammaln(k)
    - jax.scipy.special.gammaln(x + 1)
    + k * jnp.log(k / (k + safe_mu))
    + x * jnp.log(safe_mu / (k + safe_mu)))
  return jnp.where(mu == 0.0, logp_zero, core)

def rnbinom(key, k, mu):
  key_g, key_p = jax.random.split(key)
  lam = jax.random.gamma(key_g, k) * (mu / k)
  return jax.random.poisson(key_p, lam)


statenames = ["S", "E", "I", "R"]


raw_data = [i for i in range(0,572)]

ys = pd.DataFrame({'reports': time_series})


sir_obj = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta,statenames=statenames,
t0=0.0, nstep=20, ydim=1, covars=None)

import jax.random as random

key = random.PRNGKey(0)


theta = {
'N': 2300000.0,
'beta': 1.928,
'gamma': 2.728,
'eta': 0.0121,
'omega': 0.15,
'sigma': 0.45,
'amplitude': 20.0,
'rho': 0.05,
'k': 1.0
}


def to_est(theta):
	"""Natural scale -> estimation scale."""
	return {
	"beta": jnp.log(theta["beta"]),
	"gamma": jnp.log(theta["gamma"]),
	"eta": jnp.log(theta["eta"]),
	"omega": jnp.log(theta["omega"]),
	"sigma": jnp.log(theta["sigma"]),
	"amplitude": theta["amplitude"],
	"N": theta["N"],
	"rho": jax.scipy.special.logit(theta["rho"]),
	"eta": jax.scipy.special.logit(theta["eta"]),
	"k": theta["k"],
	}

def from_est(theta):
	"""Estimation scale -> natural scale."""
	return {
	"beta": jnp.log(theta["beta"]),
	"gamma": jnp.log(theta["gamma"]),
	"eta": jnp.log(theta["eta"]),
	"omega": jnp.log(theta["omega"]),
	"sigma": jnp.log(theta["sigma"]),
	"amplitude": theta["amplitude"],
	"N": theta["N"],
	"rho": jax.scipy.special.logit(theta["rho"]),
	"eta": jax.scipy.special.logit(theta["eta"]),
	"k": theta["k"],
	}



def conduct_local_search_ser():
    n_local = 3
    key = jax.random.PRNGKey(482947940)
    results = []
    for _ in range(n_local):
        key, subkey = jax.random.split(key)
        t_i = theta.copy()
        t_i["beta"] *= np.exp(np.random.normal(0, 0.2))
        t_i["gamma"] *= np.exp(np.random.normal(0, 0.2))
        t_i["eta"] *= np.exp(np.random.normal(0, 0.2))
        t_i["omega"] *= np.exp(np.random.normal(0, 0.2))
        t_i["sigma"] *= np.exp(np.random.normal(0, 0.2))
        t_i["amplitude"] *= np.exp(np.random.normal(0, 0.2))
        t_i["rho"] *= np.exp(np.random.normal(0, 0.2))
        t_i["k"] *= np.exp(np.random.normal(0, 0.2))
        mod_local = pp.Pomp(
            rinit=rinit,
            rproc=rproc,
            dmeas=dmeas,
            rmeas=rmeas,
            ys=ys,
            theta=t_i,
            statenames=statenames,
            t0=0.0,
            nstep=20,
            ydim=1,
            covars=None
            )
        mod_local.mif(J=2000, M=50, rw_sd=pp.RWSigma(sigmas={"beta": 0.01, "gamma": 0.01, "eta": 0.01, "omega": 0.01, "sigma": 0.01, "amplitude": 0.01, "N": 0.0, "rho": 0.01, "k": 0.01}, init_names=["beta", "gamma", "eta", "omega", "sigma", "amplitude", "rho", "k"]), a=0.5, key=subkey)
        results.append(mod_local.results_history.last())
    print("\nLocal Search:")
    print(results)
    print("log likelihoods")
    all_traces = []
    for i, res in enumerate(results):
        loglik_trace = res.traces_da.sel(variable="logLik").values.flatten()
        df_trace = pd.DataFrame({
            "iteration": range(len(loglik_trace)),
            "loglik": loglik_trace,
            "run": i
            })
        df_trace.to_csv(f"loglik_run_{i}.csv", index=False)
        all_traces.append(df_trace)
        print(res.traces_da.sel(variable="logLik"))
        print(res.traces_da.sel(variable="logLik").mean(skipna=True).item())
    df_all = pd.concat(all_traces, ignore_index=True)
    df_all.to_csv("loglik_all_runs.csv", index=False)
    param_traces = []
    param_names = ["beta", "gamma", "eta", "omega", "sigma", "amplitude", "rho", "k", "N"]

    for i, res in enumerate(results):

        variables = res.traces_da.coords["variable"].values

        for param in param_names:
            if param not in variables:
                continue

            values = res.traces_da.sel(variable=param).values.flatten()

            df_param = pd.DataFrame({
                "iteration": range(len(values)),
                "value": values,
                "parameter": param,
                "run": i
                })
            param_traces.append(df_param)

    if param_traces:
        df_params = pd.concat(param_traces, ignore_index=True)
        df_params.to_csv("param_traces_all_runs.csv", index=False)
    else:
        print("param_traces is empty — no parameter traces were collected.")
    final_thetas = [res.theta for res in results]
    print(final_thetas)
    print("params: ")
    pd.set_option('display.float_format', '{:.10f}'.format) 
    pd.set_option('display.max_rows', 20)
    df = pd.DataFrame(final_thetas)
    print(df)
    print(df.head())
    print(df.shape)
    for index, row in df.iterrows():
        print(row)
    # Log Likelihoods
    logliks = [
        np.nanmax(res.traces_da.sel(variable="logLik").values)
        for res in results
    ]
    # Best Parameters
    best_idx = int(np.argmax(logliks))
    best_result = results[best_idx]
    best_theta = best_result.theta
    print("\nBest run index:", best_idx)
    print("Best logLik:", logliks[best_idx])
    print("Best parameters:")
    print(best_theta)

def conduct_global_search_ser():
    n_local = 2
    key = jax.random.PRNGKey(482947940)
    results = []
    for _ in range(n_local):
        key, subkey = jax.random.split(key)
        print(f"Starting Local Search ...")
        theta_start = []
        for _ in range(n_local):
            t_i = theta.copy()
            t_i["beta"] *= np.exp(np.random.normal(0, 0.2))
            t_i["gamma"] *= np.exp(np.random.normal(0, 0.2))
            t_i["eta"] *= np.exp(np.random.normal(0, 0.2))
            t_i["omega"] *= np.exp(np.random.normal(0, 0.2))
            t_i["sigma"] *= np.exp(np.random.normal(0, 0.2))
            t_i["amplitude"] *= np.exp(np.random.normal(0, 0.2))
            t_i["rho"] *= np.exp(np.random.normal(0, 0.2))
            t_i["k"] *= np.exp(np.random.normal(0, 0.2))
            theta_start.append(t_i)

        mod_local = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta_start,statenames=statenames, t0=0.0, nstep=20, ydim=1, covars=None)

        stages = [
            {
                "M": 50,
                "rw_sd": pp.RWSigma(
                    sigmas={"beta": 0.08, "gamma": 0.08, "eta": 0.08, "omega": 0.08, "sigma": 0.08, "amplitude": 0.08, "N": 0.0, "rho": 0.08, "k": 0.08},
                    init_names=["beta", "gamma", "eta", "omega", "sigma", "amplitude", "rho", "k"]
                    )
                },
            {
                "M": 50, 
                "rw_sd": pp.RWSigma(
                    sigmas={"beta": 0.04, "gamma": 0.04, "eta": 0.04, "omega": 0.04, "sigma": 0.04, "amplitude": 0.04, "N": 0.0, "rho": 0.04, "k": 0.04},
                    init_names=["beta", "gamma", "eta", "omega", "sigma", "amplitude", "rho", "k"]
                    )
                },
            {
                "M": 50, 
                "rw_sd": pp.RWSigma(
                    sigmas={"beta": 0.01, "gamma": 0.01, "eta": 0.01, "omega": 0.01, "sigma": 0.01, "amplitude": 0.01, "N": 0.0, "rho": 0.01, "k": 0.01},
                    init_names=["beta", "gamma", "eta", "omega", "sigma", "amplitude", "rho", "k"]
                    )
                }
            ]

        for stage in stages:
            key, subkey = jax.random.split(key)
            mod_local.mif(
                J=2000, 
                M=stage['M'], 
                rw_sd=stage['rw_sd'], 
                a=0.5, 
                key=subkey
                )
            mod_local.theta = mod_local.results_history.last().theta
        results.append(mod_local.results_history.last())

    print("\nLocal Search:")
    print(results)
    print("log likelihoods")
    all_traces = []
    for i, res in enumerate(results):
        loglik_trace = res.traces_da.sel(variable="logLik").values.flatten()
        df_trace = pd.DataFrame({
            "iteration": range(len(loglik_trace)),
            "loglik": loglik_trace,
            "run": i
            })
        df_trace.to_csv(f"loglik_run_{i}.csv", index=False)
        all_traces.append(df_trace)
        print(res.traces_da.sel(variable="logLik"))
        print(res.traces_da.sel(variable="logLik").mean(skipna=True).item())
    df_all = pd.concat(all_traces, ignore_index=True)
    df_all.to_csv("loglik_all_runs.csv", index=False)
    param_traces = []
    param_names = ["beta", "gamma", "eta", "omega", "sigma", "amplitude", "rho", "k", "N"]

    for i, res in enumerate(results):

        variables = res.traces_da.coords["variable"].values

        for param in param_names:
            if param not in variables:
                continue

            values = res.traces_da.sel(variable=param).values.flatten()

            df_param = pd.DataFrame({
                "iteration": range(len(values)),
                "value": values,
                "parameter": param,
                "run": i
                })
            param_traces.append(df_param)

    if param_traces:
        df_params = pd.concat(param_traces, ignore_index=True)
        df_params.to_csv("param_traces_all_runs.csv", index=False)
    else:
        print("param_traces is empty — no parameter traces were collected.")
    final_thetas = [res.theta for res in results]
    print(final_thetas)
    print("params: ")
    pd.set_option('display.float_format', '{:.10f}'.format) 
    pd.set_option('display.max_rows', 20)
    df = pd.DataFrame(final_thetas)
    print(df)
    print(df.head())
    print(df.shape)
    for index, row in df.iterrows():
        print(row)
    # Log Likelihoods
    logliks = [
        np.nanmax(res.traces_da.sel(variable="logLik").values)
        for res in results
    ]
    # Best Parameters
    best_idx = int(np.argmax(logliks))
    best_result = results[best_idx]
    best_theta = best_result.theta
    print("\nBest run index:", best_idx)
    print("Best logLik:", logliks[best_idx])
    print("Best parameters:")
    print(best_theta)


conduct_local_search_ser()
conduct_global_search_ser()

