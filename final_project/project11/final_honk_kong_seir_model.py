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
'beta': 0.8,
'sigma': 0.667,
'gamma': 0.33,
'kappa': 0.5,
'phi': 0.25,
'theta': 0.10,
'rho': 0.1
}


def rinit(theta_, key, covars, t0):
    N = theta_["N"]
    return {
        "S": N - 4500.0,
        "E": 0.0,
        "I": 4500.0,
        "H": 0.0,
        "D": 0.0,
        "R": 0.0,
        "incD": 0.0
    }

def rproc(X_, theta_, key, covars, t, dt):
    S, E, I, H, D, R = X_["S"], X_["E"], X_["I"], X_["H"], X_["D"], X_["R"]
    N = theta_["N"]

    beta = theta_["beta"]
    sigma = theta_["sigma"]
    gamma = theta_["gamma"]
    phi = theta_["phi"]
    theta_h = theta_["theta"]
    kappa = theta_["kappa"]

    key1, key2, key3, key4 = random.split(key, 4)


    rate_SE = beta * S * I / N
    new_E = random.poisson(key1, rate_SE * dt)

    new_I = random.poisson(key2, sigma * E * dt)

    leave_I = random.poisson(key3, gamma * I * dt)


    new_H = random.poisson(key4, phi * leave_I)
    new_R_from_I = leave_I - new_H

    key5, key6 = random.split(key4)

    leave_H = random.poisson(key5, kappa * H * dt)

    new_D = random.poisson(key6, theta_h * leave_H)
    new_R_from_H = leave_H - new_D

    S_new = S - new_E
    E_new = E + new_E - new_I
    I_new = I + new_I - leave_I
    H_new = H + new_H - leave_H
    D_new = D + new_D
    R_new = R + new_R_from_I + new_R_from_H

    incD = new_D

    return {
        "S": S_new,
        "E": E_new,
        "I": I_new,
        "H": H_new,
        "D": D_new,
        "R": R_new,
        "incD": incD
    }

# evaluate f(y_n | x_n, theta)
def dmeas(Y_, X_, theta_, covars, t):
    incD = X_["incD"]
    rho = theta_["rho"]

    lam = jnp.maximum(rho * incD, 1e-8)
    y = Y_["reports"]

    return stats.poisson.logpmf(y, lam)

# draw f(y_n |x_n, theta)
def rmeas(X_, theta_, key, covars, t):
    incD = X_["incD"]
    rho = theta_["rho"]

    lam = rho * incD
    y = random.poisson(key, lam)

    return jnp.asarray(y)


statenames = ["S", "E", "I", "H", "D", "R", "incD"]


raw_data = [i for i in range(0,572)]

ys = pd.DataFrame({'reports': time_series})


sir_obj = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta,statenames=statenames, t0=0.0, nstep=572, ydim=1, covars=None)

import jax.random as random

key = random.PRNGKey(0)


def conduct_local_search_ser():
    n_local = 3
    key = jax.random.PRNGKey(482947940)
    results = []
    for _ in range(n_local):
        key, subkey = jax.random.split(key)
        t_i = theta.copy()
        t_i["beta"] *= np.exp(np.random.normal(0, 0.2))
        t_i["gamma"] *= np.exp(np.random.normal(0, 0.2))
        t_i["sigma"] *= np.exp(np.random.normal(0, 0.2))
        t_i["kappa"] *= np.exp(np.random.normal(0, 0.2))
        t_i["phi"] *= np.exp(np.random.normal(0, 0.2))
        t_i["rho"] *= np.exp(np.random.normal(0, 0.2))
        t_i["theta"] *= np.exp(np.random.normal(0, 0.2))
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
        mod_local.mif(J=2000, M=50, rw_sd=pp.RWSigma(sigmas={"beta": 0.01, "gamma": 0.01, "kappa": 0.01, "sigma": 0.01, "phi": 0.01, "N": 0.0, "rho": 0.01, "theta": 0.01}, init_names=["beta", "gamma", "kappa", "sigma", "phi", "N", "rho", "theta"]), a=0.5, key=subkey)
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
    param_names = ["beta", "gamma", "kappa", "sigma", "phi", "N", "rho", "theta"]

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
            t_i["sigma"] *= np.exp(np.random.normal(0, 0.2))
            t_i["kappa"] *= np.exp(np.random.normal(0, 0.2))
            t_i["phi"] *= np.exp(np.random.normal(0, 0.2))
            t_i["rho"] *= np.exp(np.random.normal(0, 0.2))
            t_i["theta"] *= np.exp(np.random.normal(0, 0.2))
            theta_start.append(t_i)

        mod_local = pp.Pomp(rinit=rinit,rproc=rproc, dmeas=dmeas,rmeas=rmeas,ys=ys,theta=theta_start,statenames=statenames, t0=0.0, nstep=20, ydim=1, covars=None)

        stages = [
            {
                "M": 50,
                "rw_sd": pp.RWSigma(
                    sigmas={"beta": 0.08, "gamma": 0.08, "kappa": 0.08, "sigma": 0.08, "phi": 0.08, "N": 0.0, "rho": 0.08, "theta": 0.08},
                    init_names= ["beta", "gamma", "kappa", "sigma", "phi", "N", "rho", "theta"]
                    )
                },
            {
                "M": 50, 
                "rw_sd": pp.RWSigma(
                    sigmas={"beta": 0.01, "gamma": 0.01, "kappa": 0.01, "sigma": 0.01, "phi": 0.01, "N": 0.0, "rho": 0.01, "theta": 0.01},
                    init_names= ["beta", "gamma", "kappa", "sigma", "phi", "N", "rho", "theta"]
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
    param_names =  ["beta", "gamma", "kappa", "sigma", "phi", "N", "rho", "theta"]

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


