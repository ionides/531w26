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

