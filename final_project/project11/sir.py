import numpy as np
from pypomp.random import (fast_approx_rbinom, fast_approx_rgamma, fast_approx_rpoisson)
import pypomp as pp
import jax.numpy as jnp
import jax
import jax.random as random
import pandas as pd


theta = {
"Beta": 1.75,
"mu_IR": 0.25,
"N": 2300000.0,
"eta": 0.25,
"rho": 0.01,
"k": 1.0
}


# {'Beta': 1.8903463841445274, 'mu_IR': 0.23804138385096346, 'N': 2300000.0, 'eta': 0.23521023924004322, 'rho': 0.015150643307051232, 'k': 1.0}


data = pd.read_csv("/Users/darklizard979/531_final_project/FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv")
time_series = data['TOTAL SPECIMENS']


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

def conduct_local_search():
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
	n_local = 1
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
	print("Log Likelihood: ")
	print(loglik_vector[best_index])


conduct_local_search()







