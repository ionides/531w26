# fish_pomp_m1.py — M1: Linear Gaussian Gompertz + state-space SOI
# Simulated data, TMB/KF exact, IF2 should match.
#
# Usage:
#   python fish_pomp_m1.py              # simulate + PF check + IF2 + save results
#   python fish_pomp_m1.py --pf-only    # just PF sanity check

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import json, time, sys, os

jax.config.update("jax_enable_x64", True)
from pypomp import Pomp, RWSigma
from pypomp.types import (StateDict, ParamDict, CovarDict, TimeFloat,
                           StepSizeFloat, RNGKey, ObservationDict, InitialTimeFloat)

# ── True parameters ──
TRUE_THETA = {
    "a": 0.3, "b": 0.9, "c_env": -0.05, "phi_E": 0.7,
    "log_sigma_p": np.log(0.25), "log_sigma_o": np.log(0.15),
    "log_sigma_E": np.log(0.25), "log_sigma_S": np.log(0.15),
    "X_0": 3.0, "E_0": 0.0,
}
T_SIM = 200  # time steps

# ── Model components ──

def rinit(theta_: ParamDict, key: RNGKey, covars: CovarDict, t0: InitialTimeFloat):
    return {"X": theta_["X_0"], "E": theta_["E_0"]}

def rproc(X_: StateDict, theta_: ParamDict, key: RNGKey, covars: CovarDict,
          t: TimeFloat, dt: StepSizeFloat):
    a = theta_["a"]; b = theta_["b"]; c = theta_["c_env"]; phi = theta_["phi_E"]
    sp = jnp.exp(theta_["log_sigma_p"]); sE = jnp.exp(theta_["log_sigma_E"])
    k1, k2 = jax.random.split(key)
    E_new = phi * X_["E"] + sE * jax.random.normal(k1)
    X_new = a + b * X_["X"] + c * X_["E"] + sp * jax.random.normal(k2)
    return {"X": X_new, "E": E_new}

def dmeas(Y_: ObservationDict, X_: StateDict, theta_: ParamDict,
          covars: CovarDict, t: TimeFloat):
    so = jnp.exp(theta_["log_sigma_o"]); sS = jnp.exp(theta_["log_sigma_S"])
    return (jax.scipy.stats.norm.logpdf(Y_["Y"], loc=X_["X"], scale=so) +
            jax.scipy.stats.norm.logpdf(Y_["S"], loc=X_["E"], scale=sS))

def rmeas(X_: StateDict, theta_: ParamDict, key: RNGKey,
          covars: CovarDict, t: TimeFloat):
    so = jnp.exp(theta_["log_sigma_o"]); sS = jnp.exp(theta_["log_sigma_S"])
    k1, k2 = jax.random.split(key)
    return jnp.array([X_["X"] + so * jax.random.normal(k1),
                       X_["E"] + sS * jax.random.normal(k2)])

# ── Build + simulate ──

def make_m1_simulated(seed=42):
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T_SIM+1, dtype=float),
                            columns=["Y", "S"])
    m1 = Pomp(rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas,
              ys=ys_dummy, t0=0.0, nstep=1, ydim=2,
              theta=TRUE_THETA, statenames=["X", "E"])
    m1 = m1.simulate(key=jax.random.key(seed), nsim=1, as_pomp=True)
    return m1

# ── Main ──

if __name__ == "__main__":
    pf_only = "--pf-only" in sys.argv

    m1 = make_m1_simulated()
    print(f"M1: T={len(m1.ys)}, simulated from true params")

    # Save simulated data for QMD
    m1.ys.to_csv("m1_simulated_data.csv")
    with open("m1_true_theta.json", "w") as f:
        json.dump({k: float(v) for k, v in TRUE_THETA.items()}, f, indent=2)

    # PF at true params
    print("\n=== PF at TRUE params ===")
    for J in [500, 2000, 10000]:
        m1.pfilter(J=J, theta=TRUE_THETA, reps=5, key=jax.random.key(99))
        lls = m1.results_history[-1].logLiks.values.flatten()
        print(f"  J={J:>6d}: {np.mean(lls):.1f} ± {np.std(lls):.1f}")

    if pf_only:
        sys.exit(0)

    # IF2 local search
    print("\n=== IF2: 10 chains, J=5000, M=100 ===")
    rw_sd = RWSigma(sigmas={
        "a": 0.005, "b": 0.003, "c_env": 0.005, "phi_E": 0.005,
        "log_sigma_p": 0.005, "log_sigma_o": 0.005,
        "log_sigma_E": 0.005, "log_sigma_S": 0.005,
        "X_0": 0.0, "E_0": 0.0,
    })

    np.random.seed(100)
    starts = [dict(TRUE_THETA)]  # start 0 = truth
    for _ in range(9):
        s = dict(TRUE_THETA)
        for k in s:
            if k not in ("X_0", "E_0"):
                s[k] += np.random.normal(0, 0.1)
        starts.append(s)

    t0 = time.time()
    m1.mif(J=5000, M=100, rw_sd=rw_sd, a=0.5, theta=starts, key=jax.random.key(2026))
    print(f"IF2 done in {time.time()-t0:.1f}s")

    # Evaluate endpoints
    print("\n=== Endpoint evaluation (J=10000, 10 reps) ===")
    tr = m1.traces()
    pnames = [k for k in rw_sd.sigmas.keys() if k not in ("X_0", "E_0")]
    results = []

    for cid in sorted(tr["replicate"].unique()):
        grp = tr[tr["replicate"] == cid].sort_values("iteration")
        ep = {p: float(grp.iloc[-1][p]) for p in pnames}
        ep["X_0"] = TRUE_THETA["X_0"]
        ep["E_0"] = TRUE_THETA["E_0"]
        fe = make_m1_simulated()
        fe.pfilter(J=10000, theta=ep, reps=10, key=jax.random.key(500 + int(cid)))
        lls = fe.results_history[-1].logLiks.values.flatten()
        ll_mean = float(np.mean(lls))
        ll_sd = float(np.std(lls))
        results.append({"chain": int(cid), "mean_ll": ll_mean, "sd_ll": ll_sd, **ep})
        print(f"  Chain {int(cid)}: ll={ll_mean:.1f} ± {ll_sd:.1f}")

    # Save best
    best = max(results, key=lambda x: x["mean_ll"])
    print(f"\nBest: chain {best['chain']}, ll={best['mean_ll']:.2f}")

    with open("m1_if2_results.json", "w") as f:
        json.dump({"best": best, "all_chains": results}, f, indent=2)

    pd.DataFrame(results).to_csv("m1_if2_endpoints.csv", index=False)
    print("Saved: m1_simulated_data.csv, m1_true_theta.json, m1_if2_results.json, m1_if2_endpoints.csv")
