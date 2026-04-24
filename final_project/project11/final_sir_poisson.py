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


plt.plot(time_series, label = "Start")
plt.show()


theta = {
'N': 2300000.0,
'beta': 0.728,
'gamma': 1.928,
'eta': 0.00121,
'omega': 0.15
}


def rinit(theta_, key, covars, t0):
	N = theta_["N"]
	S0 = jnp.round(N)
	I0 = 1.0
	R0 = 0.0
	return {"S": S0, "I": I0, "R": R0}

def rproc(X_, theta_, key, covars, t, dt):
	S = jnp.asarray(X_["S"])
	I = jnp.asarray(X_["I"])
	R = jnp.asarray(X_["R"])

	beta = theta_["beta"]
	N = theta_["N"]
	gamma = theta_["gamma"]
	omega = theta_["omega"]
	eta = theta_["eta"]

	beta_t = beta * (1 + 0.2 * jnp.sin(2 * jnp.pi * t / 365))

	rate_SI = (beta_t * S * I / N  + eta * S ) * dt
	rate_IR = gamma * I * dt
	rate_RS = omega * R * dt

	key_SI, key_IR, key_RS = jax.random.split(key, 3)

	dN_SI = jax.random.poisson(key_SI, rate_SI, shape=None, dtype=None)
	dN_IR = jax.random.poisson(key_IR, rate_IR, shape=None, dtype=None)
	dN_SR = jax.random.poisson(key_RS, rate_RS, shape=None, dtype=None)

	S_new = S - dN_SI + dN_SR
	I_new = I + dN_SI - dN_IR
	R_new = R + dN_IR - dN_SR

	return {"S": S_new, "I": I_new, "R": R_new}

# evaluate f(y_n | x_n, theta)
def dmeas(Y_, X_, theta_, covars, t):
	"""Measurement density: log P(reports | H, rho, k)."""
	# Y - reports
	# X - SIR states
	print("dmeas")
	return 0.5

# draw f(y_n |x_n, theta)
def rmeas(X_, theta_, key, covars, t):
	"""Measurement simulator."""
	# Y - reports
	# X - SIR states
	print("rmeas")
	return jnp.array([1])


statenames = ["S", "I", "R"]


raw_data = [i for i in range(0,572)]

ys = pd.DataFrame({'reports': time_series})


sir_obj = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta,statenames=statenames,
t0=0.0, nstep=20,ydim=1, covars=None)

import jax.random as random

key = random.PRNGKey(0)

results_df = sir_obj.simulate(nsim=1, key=key)[0]

print(results_df)

y = results_df["state_1"]


print(y)

plt.plot(time_series, label = "Raw")
plt.plot(y, label = "Simulated")
plt.legend()
plt.show()






