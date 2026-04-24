import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("/Users/darklizard979/531_final_project/FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv")

print(data)

time_series = data['TOTAL SPECIMENS']


# Time Series Graph - Influenza
print(time_series)
plt.plot(time_series)
plt.title("US Influenza By Week - 2015-2026")
plt.show()


# ARIMA
model = ARIMA(time_series, order=(3,1,2))
result = model.fit()


pred = result.predict(start=0, end=500, typ='levels')


plt.plot(time_series)
plt.plot(pred)
plt.title("US Influenza By Week - 2015-2026")
plt.show()


# POMP
import numpy as np
import pandas as pd
import pypomp as pp
import jax.numpy as jnp
import jax

def rinit(theta_, key, covars, t0):
  N = theta_["N"]
  eta = theta_["eta"]
  S0 = jnp.round(N * eta)
  I0 = 1.0
  R0 = jnp.round(N * (1 - eta)) - 1.0
  H0 = 0.0
  return {"S": S0, "I": I0, "R": R0, "H": H0}

def rproc(X_, theta_, key, covars, t, dt):
  S, I, R, H = X_["S"], X_["I"], X_["R"], X_["H"]
  S_new = S - 1
  I_new = I + 1
  R_new = R
  H_new = H
  return {"S": S_new, "I": I_new, "R": R_new, "H": H_new}

def dmeas(Y_, X_, theta_, covars, t):
  """Measurement density: log P(reports | H, rho, k)."""
  return 1

def rmeas(X_, theta_, key, covars, t):
  """Measurement simulator: sample reports ~ NegBin(k, rho*H)."""
  return jnp.asarray([1])

def nbinom_logpmf(x, k, mu):
  """Log PMF of NegBin(k, mu) that is robust when mu == 0."""
  return jnp.log(1)

def rnbinom(key, k, mu):
  """Sample from NegBin(k, mu) via Gamma-Poisson mixture."""
  return 1

theta = {
"Beta": 9.0,
"mu_IR": 0.5,
"N": 320000000.0,
"eta": 0.00001875,
"rho": 0.5,
"k": 10.0
}

statenames = ["S", "I", "R", "H"]

raw_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

ys = pd.DataFrame({'reports': raw_data})

sir_obj = pp.Pomp(
rinit=rinit,
rproc=rproc,
dmeas=dmeas,
rmeas=rmeas,
ys=ys,
theta=theta,
statenames=statenames,
t0=0.0, nstep=20,
ydim=1, covars=None, accumvars=("H",)
)

import jax.random as random

key = random.PRNGKey(0)

results_df = sir_obj.simulate(nsim=1, key=key)[0]

print()
print()
print("Raw Results: ")

infected = results_df[0:20]['state_1']
print(infected)


plt.plot(infected)
plt.show()





