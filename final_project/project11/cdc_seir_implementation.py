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
import jax.scipy.stats as stats

data = pd.read_csv("FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv")
time_series = data['TOTAL SPECIMENS']


theta = {
'N': 2300000.0,
'beta': 20.0,
'mu_EI': 0.5,
'mu_IR': 0.33,
'rho': 0.5
}


def rinit(theta_, key, covars, t0):
	N = theta_["N"]
	S0 = N - 1.0
	E0 = 0.0
	I0 = 1.0
	R0 = 0.0
	return {"S": S0, "E": E0, "I": I0, "R": R0}

def rproc(X_, theta_, key, covars, t, dt):
	S = X_["S"]
	E = X_["E"]
	I = X_["I"]
	R = X_["R"]
	N = theta_["N"]

	beta = theta_["beta"]
	mu_EI = theta_["mu_EI"]
	mu_IR = theta_["mu_IR"]

	p_SE = 1 - jnp.exp(-beta * I / N * dt)
	p_EI = 1 - jnp.exp(-mu_EI * dt)
	p_IR = 1 - jnp.exp(-mu_IR * dt)

	k1, k2, k3 = random.split(key, 3)

	S_int = jnp.floor(S).astype(jnp.int32)
	E_int = jnp.floor(E).astype(jnp.int32)
	I_int = jnp.floor(I).astype(jnp.int32)

	new_E = random.binomial(k1, S_int, p_SE)
	new_I = random.binomial(k2, E_int, p_EI)
	new_R = random.binomial(k3, I_int, p_IR)

	S_new = S - new_E
	E_new = E + new_E - new_I
	I_new = I + new_I - new_R
	R_new = R + new_R

	return {"S": S_new, "E": E_new, "I": I_new, "R": R_new}

# evaluate f(y_n | x_n, theta)
def dmeas(Y_, X_, theta_, covars, t):
	I = X_["I"]
	rho = theta_["rho"]
	lam = jnp.maximum(rho * I, 1e-8)
	y = Y_["reports"]
	return stats.poisson.logpmf(y, lam)

# draw f(y_n |x_n, theta)
def rmeas(X_, theta_, key, covars, t):
	I = X_["I"]
	rho = theta_["rho"]
	lam = rho * I
	y = random.poisson(key, lam)
	return jnp.asarray(y)


statenames = ["S", "E", "I", "R"]


raw_data = [i for i in range(0,572)]

ys = pd.DataFrame({'reports': time_series})


sir_obj = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta,statenames=statenames, t0=0.0, nstep=572, ydim=1, covars=None)



#####################
#####################
### Local Search ####
#####################
#####################


theta = {
'N': 2300000.0,
'beta': 20.0,
'mu_EI': 0.5,
'mu_IR': 0.33,
'rho': 0.5
}


def conduct_local_search_ser():
	n_local = 5
	key = jax.random.PRNGKey(482947940)
	results = []
	for _ in range(n_local):
		key, subkey = jax.random.split(key)
		t_i = theta.copy()
		t_i["beta"] *= np.exp(np.random.normal(0, 0.2))
		t_i["mu_EI"] *= np.exp(np.random.normal(0, 0.2))
		t_i["mu_IR"] *= np.exp(np.random.normal(0, 0.2))
		t_i["rho"] *= np.exp(np.random.normal(0, 0.2))
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
		mod_local.mif(J=2000, M=50, rw_sd=pp.RWSigma(sigmas={"beta":0.2,"mu_EI":0.2,"mu_IR":0.1,"N":0.0,"rho":0.1}, init_names=["beta", "mu_EI", "mu_IR", "rho"]), a=0.5, key=subkey)
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
	param_names = ["beta", "mu_EI", "mu_IR", "N", "rho"]

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
			t_i["mu_EI"] *= np.exp(np.random.normal(0, 0.2))
			t_i["mu_IR"] *= np.exp(np.random.normal(0, 0.2))
			t_i["rho"] *= np.exp(np.random.normal(0, 0.2))
			theta_start.append(t_i)

		mod_local = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta_start,statenames=statenames, t0=0.0, nstep=20, ydim=1, covars=None)

		stages = [
			{
				"M": 50,
				"rw_sd": pp.RWSigma(
					sigmas={"beta": 0.08, "mu_EI": 0.08, "mu_IR": 0.08, "N": 0.0, "rho": 0.08},
					init_names=["beta", "mu_EI", "mu_IR", "rho"]
					)
				},
			{
				"M": 50, 
				"rw_sd": pp.RWSigma(
					sigmas={"beta": 0.04, "mu_EI": 0.04, "mu_IR": 0.04, "N": 0.0, "rho": 0.04},
					init_names=["beta", "mu_EI", "mu_IR", "rho"]
					)
				},
			{
				"M": 50, 
				"rw_sd": pp.RWSigma(
					sigmas={"beta": 0.01, "mu_EI": 0.01, "mu_IR": 0.01, "N": 0.0, "rho": 0.01},
					init_names=["beta", "mu_EI", "mu_IR", "rho"]
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
	param_names = ["beta", "mu_EI", "mu_IR", "N", "rho"]

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



def get_reporting_rate_ci():
    # We use the theta from the Local Search, as we have a log likelihood closer to 0.
    theta = {
        "beta": 22.70865335380483,
        "mu_IR": 0.3961116621569491,
        "mu_EI": 0.39714361142110044,
        "N": 2300000.0,
        "rho": 0.5058308187492441,
    }

    seir = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta,statenames=statenames, t0=0.0, nstep=20, ydim=1, covars=None)


    rho_values = np.linspace(0.1, 1, num=9)
    profile_results = []


    for rho_fixed in rho_values:
        # 1) Update theta with fixed rho
        current_theta = theta.copy()
        current_theta['rho'] = rho_fixed

        # 2) Run the particle filter
        seir.theta = current_theta
        seir.pfilter(key=jax.random.key(42), J=500, reps=1)

        # 3) Store result
        loglik = float(seir.results_history.last().logLiks.values[0, 0])
        profile_results.append((rho_fixed, loglik))

    profile_results = np.array(profile_results)
    x = profile_results[:, 0]
    y = profile_results[:, 1]
    rhos = profile_results[:, 0]
    logliks = profile_results[:, 1]
    max_loglik = np.max(logliks)
    threshold = max_loglik - 1.92
    valid = logliks >= threshold
    rho_lower = rhos[valid].min()
    rho_upper = rhos[valid].max()
    print("Approx 95% CI:", (rho_lower, rho_upper))
    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker='o', label = "Log Likelihood")
    plt.axhline(0, color='black', linestyle='-', linewidth=1, label='Perfect Likelihood')
    plt.axvline(rho_lower, color='red', linestyle='-', linewidth=1, label='95% CI Lower Bound')
    plt.axvline(rho_upper, color='red', linestyle='-', linewidth=1, label='95% CI Upper Bound')
    plt.xlabel("Reporting Rate")
    plt.ylabel("Log Likelihood")
    plt.legend()
    plt.title(r'Profile Likelihood - $\rho$')
    plt.show()

def compute_2d_surface():
    theta = {
        "beta": 22.70865335380483,
        "mu_IR": 0.3961116621569491,
        "mu_EI": 0.39714361142110044,
        "N": 2300000.0,
        "rho": 0.5058308187492441,
    }

    seir = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta,statenames=statenames, t0=0.0, nstep=20, ydim=1, covars=None)

    beta_range = np.linspace(15, 30, num=15)
    mu_ir_range = np.linspace(0.1, 0.8, num=15)

    loglik_surface = np.zeros((len(beta_range), len(mu_ir_range)))
    
    for i, b_val in enumerate(beta_range):
        for j, m_val in enumerate(mu_ir_range):
        	# 1) Update theta
            current_theta = theta.copy()
            current_theta['beta'] = b_val
            current_theta['mu_IR'] = m_val

             # 2) Run the particle filter
            seir.theta = current_theta
            seir.pfilter(key=jax.random.key(42), J=500, reps=1)

            # 3) Store result
            loglik = float(seir.results_history.last().logLiks.values[0, 0])
            loglik_surface[i, j] = loglik
            print(f"beta: {b_val:.2f}, mu_IR: {m_val:.2f}, LogLik: {loglik:.2f}")

    B, M = np.meshgrid(beta_range, mu_ir_range)
    plt.figure(figsize=(8,6))
    contour = plt.contourf(B, M, loglik_surface.T, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Log Likelihood')
    plt.contour(B, M, loglik_surface.T, levels=10, colors='white', alpha=0.5)
    plt.scatter(theta['beta'], theta['mu_IR'], color='red', marker='*', s=200, label='Best Fit')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\mu_{IR}$')
    plt.title('2D Likelihood Surface (beta vs mu_IR)')
    plt.legend()
    plt.show()


get_reporting_rate_ci()
conduct_local_search_ser()
conduct_global_search_ser()
compute_2d_surface()





