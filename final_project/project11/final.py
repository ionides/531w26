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

# print(time_series)

theta = {'Beta': 1.8903463841445274, 'mu_IR': 0.23804138385096346, 'N': 2300000.0, 'eta': 0.23521023924004322, 'rho': 0.015150643307051232, 'k': 1.0}


key = random.PRNGKey(0)


def rinit(theta_, key, covars, t0):
	N = theta_["N"]
	eta = theta_["eta"]
	S0 = jnp.round(N * eta)
	I0 = 1.0
	R0 = jnp.round(N * (1 - eta)) - 1.0
	H0 = 0.0
	return {"S": S0, "I": I0, "R": R0, "H": H0}

# draw from f(x_n | x_n -1, theta)
def rproc(X_, theta_, key, covars, t, dt):
	S = jnp.asarray(X_["S"])
	I = jnp.asarray(X_["I"])
	R = jnp.asarray(X_["R"])
	H = jnp.asarray(X_["H"])

	Beta = theta_["Beta"]
	mu_IR = theta_["mu_IR"]
	N = theta_["N"]

	p_SI = 1.0 - jnp.exp(-Beta * I / N * dt)
	p_IR = 1.0 - jnp.exp(-mu_IR * dt)

	key_SI, key_IR = jax.random.split(key)
	dN_SI = jax.random.binomial(key_SI, n=jnp.int32(S), p=p_SI)
	dN_IR = jax.random.binomial(key_IR, n=jnp.int32(I), p=p_IR)

	S_new = S - dN_SI
	I_new = I + dN_SI - dN_IR
	R_new = R + dN_IR
	H_new = H + dN_IR

	return {"S": S_new, "I": I_new, "R": R_new, "H": H_new}

# evaluate f(y_n | x_n, theta)
def dmeas(Y_, X_, theta_, covars, t):
	"""Measurement density: log P(reports | H, rho, k)."""
	rho = theta_["rho"]
	k = theta_["k"]
	H = X_["H"]
	mu = rho * H
	return nbinom_logpmf(Y_["reports"], k, mu)

# draw f(y_n |x_n, theta)
def rmeas(X_, theta_, key, covars, t):
	"""Measurement simulator."""
	rho = theta_["rho"]
	k = theta_["k"]
	H = X_["H"]
	mu = rho * H
	reports = pp.random.fast_approx_rbinom(key, k, mu)
	return jnp.array([reports])

def nbinom_logpmf(x, k, mu):
	"""Log PMF of NegBin(k, mu) that is robust when mu == 0."""
	x = jnp.asarray(x)
	k = jnp.asarray(k)
	mu = jnp.asarray(mu)
	logp_zero = jnp.where(x == 0, 0.0,-jnp.inf)
	safe_mu = jnp.where(mu == 0.0, 1.0, mu)
	core = (jax.scipy.special.gammaln(k + x) - jax.scipy.special.gammaln(k) - jax.scipy.special.gammaln(x + 1) + k * jnp.log(k / (k + safe_mu)) + x * jnp.log(safe_mu / (k + safe_mu)))
	return jnp.where(mu == 0.0, logp_zero, core)

def rnbinom(key, k, mu):
	"""Sample from NegBin(k, mu) via Gamma-Poisson mixture."""
	key_g, key_p = jax.random.split(key)
	lam = jax.random.gamma(key_g, k) * (mu / k)
	return jax.random.poisson(key_p, lam)


statenames = ["S", "I", "R", "H"]

raw_data = [i for i in range(0,572)]

ys = pd.DataFrame({'reports': time_series})


sir_obj = pp.Pomp(rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas, ys=ys, theta=theta,
statenames=statenames, t0=0.0, nstep=1, ydim=1, covars=None, accumvars=("H",))



def to_est(theta):
	"""Natural scale -> estimation scale."""
	return {
	"Beta": jnp.log(theta["Beta"]),
	"mu_IR": theta["mu_IR"],
	"N": theta["N"],
	"rho": jax.scipy.special.logit(theta["rho"]),
	"eta": jax.scipy.special.logit(theta["eta"]),
	"k": theta["k"],
	}

def from_est(theta):
	"""Estimation scale -> natural scale."""
	return {
	"Beta": jnp.exp(theta["Beta"]),
	"mu_IR": theta["mu_IR"],
	"N": theta["N"],
	"rho": jax.scipy.special.expit(theta["rho"]),
	"eta": jax.scipy.special.expit(theta["eta"]),
	"k": theta["k"],
	}







def conduct_local_search_sir():
	print("Initiating Local Search ... ")
	rw_sd = pp.RWSigma(
		sigmas={
		"Beta": 0.3,
		"mu_IR": 0.05,
		"N": 0.0,
		"rho": 0.2,
		"eta": 0.2,
		"k": 0.05
		},
		init_names=["Beta", "mu_IR", "N", "rho", "eta", "k"]
		)
	par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)
	n_local = 20
	theta_list = []
	for _ in range(n_local):
		t = theta.copy()
		t["Beta"] *= np.random.uniform(0.9, 1.1)
		t["eta"] += np.random.normal(0, 0.01)
		t["rho"] += np.random.normal(0, 0.01)
		t["mu_IR"] += np.random.normal(0, 0.01)
		theta_list.append(t)
	mod_local = pp.Pomp(rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas, ys=ys, theta=theta_list, statenames=statenames, par_trans=par_trans, t0=0.0, nstep=7, accumvars=("H",), ydim=1, covars=None)
	key = jax.random.key(482947940)
	mod_local.mif(J=2000, M=900, rw_sd=rw_sd, a=0.5, key=key)
	mif_result = mod_local.results_history
	final_thetas = [mif_result.last().theta[r] for r in range(n_local)]
	# Get the best Thetas with the Best Log Likelihood
	loglik_vector = mif_result.last().traces_da.sel(variable='logLik').values
	best_index = loglik_vector.argmax()
	best_theta = final_thetas[best_index]
	print("Best Parameters (Local Search): ")
	print(best_theta)
	print("Best Index:")
	print(best_index)
	print("Log Likelihood: ")
	print(loglik_vector[best_index])


def conduct_global_search_sir():
    print("Initiating Global Search ... ")
    n_local = 20
    key = jax.random.PRNGKey(482947940)
    keys = jax.random.split(key, n_local)
    results = []
    par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)
    for i in range(n_local):
        print(f"Starting Local Search {i+1}/{n_local}...")
        theta_start = []
        for _ in range(1):
            t = theta.copy()
            t["Beta"] *= np.random.uniform(0.9, 1.1)
            t["eta"] += np.random.normal(0, 0.01)
            t["rho"] += np.random.normal(0, 0.01)
            t["mu_IR"] += np.random.normal(0, 0.01)
            theta_start.append(t)
        mod_local = pp.Pomp(rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas, ys=ys, theta=theta_start, statenames=statenames, par_trans=par_trans, t0=0.0, nstep=7, accumvars=("H",), ydim=1, covars=None)

        stages = [
            {
                "M": 50,
                "rw_sd": pp.RWSigma(
                    sigmas={"Beta": 0.02, "mu_IR": 0.0, "N": 0.0, "rho": 0.02, "eta": 0.02, "k": 0.0},
                    init_names=["eta"]
                )
            },
            {
                "M": 30, 
                "rw_sd": pp.RWSigma(
                    sigmas={"Beta": 0.01, "mu_IR": 0.0, "N": 0.0, "rho": 0.01, "eta": 0.01, "k": 0.0},
                    init_names=["eta"]
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


    best_result = results
    print("\nGlobal Search Multi Start:")
    print(results)

    print("log likelihoods")
    for res in results:
        print(res.traces_da.sel(variable="logLik"))
        print(res.traces_da.sel(variable="logLik").mean(skipna=True).item())



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
    logliks = [ res.traces_da.sel(variable="logLik").mean(skipna=True).item() for res in results]
    # Best Parameters
    best_idx = int(np.argmax(logliks))
    best_result = results[best_idx]
    best_theta = best_result.theta
    print("\nBest run index:", best_idx)
    print("Best logLik:", logliks[best_idx])
    print("Best parameters:")
    print(best_theta)

conduct_local_search_sir()
conduct_global_search_sir()



########

theta = {'Beta': 1.8903463841445274, 'mu_IR': 0.23804138385096346, 'N': 2300000.0, 'eta': 0.23521023924004322, 'rho': 0.015150643307051232, 'k': 1.0, 'sigma': 0.5}


def rinit(theta_, key, covars, t0):
  N = theta_["N"]
  eta = theta_["eta"]

  S0 = jnp.round(N * eta)
  E0 = 0.0
  I0 = 1.0
  R0 = jnp.round(N * (1 - eta)) - 1.0
  H0 = 0.0

  return {"S": S0, "E": E0, "I": I0, "R": R0, "H": H0}

def rproc(X_, theta_, key, covars, t, dt):
  S, E, I, R, H = X_["S"], X_["E"], X_["I"], X_["R"], X_["H"]

  Beta = theta_["Beta"]
  mu_IR = theta_["mu_IR"]
  sigma = theta_["sigma"]
  N = theta_["N"]

  p_SE = 1.0 - jnp.exp(-Beta * I / N * dt)
  p_EI = 1.0 - jnp.exp(-sigma * dt)
  p_IR = 1.0 - jnp.exp(-mu_IR * dt)
  key_SE, key_EI, key_IR = jax.random.split(key, 3)


  dN_SE = jax.random.binomial(key_SE, n=jnp.int32(S), p=p_SE)
  dN_EI = jax.random.binomial(key_EI, n=jnp.int32(E), p=p_EI)
  dN_IR = jax.random.binomial(key_IR, n=jnp.int32(I), p=p_IR)

  S_new = S - dN_SE
  E_new = E + dN_SE - dN_EI
  I_new = I + dN_EI - dN_IR
  R_new = R + dN_IR
  H_new = H + dN_SE


  return {"S": S_new, "E": E_new, "I": I_new, "R": R_new, "H": H_new}

def dmeas(Y_, X_, theta_, covars, t):
  rho = theta_["rho"]
  k = theta_["k"]
  H = X_["H"]
  mu = rho * H
  return nbinom_logpmf(Y_["reports"], k, mu)

def rmeas(X_, theta_, key, covars, t):
  rho = theta_["rho"]
  k = theta_["k"]
  H = X_["H"]
  mu = rho * H
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


#######

def to_est(theta):
  return {
  "Beta": jnp.log(theta["Beta"]),
  "mu_IR": theta["mu_IR"],
  "N": theta["N"],
  "rho": jax.scipy.special.logit(theta["rho"]),
  "eta": jax.scipy.special.logit(theta["eta"]),
  "k": theta["k"],
  "sigma": theta["sigma"]
  }

def from_est(theta):
  return {
  "Beta": jnp.exp(theta["Beta"]),
  "mu_IR": theta["mu_IR"],
  "N": theta["N"],
  "rho": jax.scipy.special.expit(theta["rho"]),
  "eta": jax.scipy.special.expit(theta["eta"]),
  "k": theta["k"],
  "sigma": theta["sigma"]
  }

statenames = ["S", "E", "I", "R", "H"]

def conduct_local_search_ser():
    print("Initiating Local Search ... ")
    rw_sd = pp.RWSigma(
        sigmas={
        "Beta": 0.3,
        "mu_IR": 0.05,
        "N": 0.0,
        "rho": 0.2,
        "eta": 0.2,
        "k": 0.05,
        "sigma": 0.5
        },
        init_names=["Beta", "mu_IR", "N", "rho", "eta", "k", "sigma"])

    par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)
    n_local = 20
    # theta_list = [theta.copy() for _ in range(n_local)]
    theta_list = []
    for _ in range(n_local):
        t = theta.copy()
        t["Beta"] *= np.random.uniform(0.9, 1.1)
        t["eta"] += np.random.normal(0, 0.01)
        t["rho"] += np.random.normal(0, 0.01)
        t["mu_IR"] += np.random.normal(0, 0.01)
        theta_list.append(t)

    mod_local = pp.Pomp( rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas, ys=ys, theta=theta_list, statenames=statenames,
        par_trans=par_trans, t0=0.0, nstep=7, accumvars=("H",), ydim=1, covars=None)

    key = jax.random.key(482947940)

    mod_local.mif(J=2000, M=900, rw_sd=rw_sd, a=0.5, key=key)

    mif_result = mod_local.results_history
    print("Local Search Result: ")
    print(mif_result)
    print(mif_result.last().traces_da.sel(variable="logLik"))
    print(mif_result.last().traces_da.sel(variable="logLik").mean(skipna=True).item())

    for i, step in enumerate(mif_result):
        print(f"Step {i}:")
        print(step)
        print("-" * 40)

    # Final Thetas
    final_thetas = [mif_result.last().theta[r] for r in range(n_local)]
    print(final_thetas)
    print("params: ")
    pd.set_option('display.float_format', '{:.10f}'.format) 
    pd.set_option('display.max_rows', 20)
    df = pd.DataFrame(final_thetas)
    print(df)
    print(df.shape)
    # Get the best Thetas with the Best Log Likelihood
    # Get the best Thetas with the Best Log Likelihood
    loglik_vector = mif_result.last().traces_da.sel(variable='logLik').values
    best_index = loglik_vector.argmax()
    best_theta = final_thetas[best_index]
    print("Best Parameters (Local Search): ")
    print(best_theta)
    print("Best Index:")
    print(best_index)
    print("Log Likelihood: ")
    print(loglik_vector[best_index])

def conduct_global_search_ser():
    print("Initiating Global Search ... ")
    n_local = 20
    key = jax.random.PRNGKey(482947940)
    keys = jax.random.split(key, n_local)
    results = []
    for i in range(n_local):
        print(f"Starting Local Search {i+1}/{n_local}...")
        theta_start = []
        for _ in range(1):
            t = theta.copy()
            t["Beta"] *= np.random.uniform(0.9, 1.1)
            t["eta"] += np.random.normal(0, 0.01)
            t["rho"] += np.random.normal(0, 0.01)
            t["mu_IR"] += np.random.normal(0, 0.01)
            theta_start.append(t)

        par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)
        mod_local = pp.Pomp(
            rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas, 
            ys=ys, theta=theta_start, statenames=statenames, 
            par_trans=par_trans, t0=0.0, nstep=7, 
            accumvars=("H",), ydim=1, covars=None)

        stages = [
            {
                "M": 50,
                "rw_sd": pp.RWSigma(
                    sigmas={"Beta": 0.02, "mu_IR": 0.0, "N": 0.0, "rho": 0.02, "eta": 0.02, "k": 0.0, "sigma": 0.5},
                    init_names=["eta"]
                    )
                },
            {
                "M": 30, 
                "rw_sd": pp.RWSigma(
                    sigmas={"Beta": 0.01, "mu_IR": 0.0, "N": 0.0, "rho": 0.01, "eta": 0.01, "k": 0.0, "sigma": 0.5},
                    init_names=["eta"]
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

    best_result = results
    print("\nGlobal Search Multi Start:")
    print(results)
    print("log likelihoods")
    for res in results:
       print(res.traces_da.sel(variable="logLik"))
       print(res.traces_da.sel(variable="logLik").mean(skipna=True).item())
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
    logliks = [ res.traces_da.sel(variable="logLik").mean(skipna=True).item() for res in results]
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






















