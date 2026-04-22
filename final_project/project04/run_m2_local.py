# run_m2_local.py — M2 local IF2 search

from fish_pomp_m2 import *
from pypomp.RWSigma_class import RWSigma
import jax, numpy as np, json, time
jax.config.update("jax_enable_x64", True)

fish, theta0 = make_fish_m2()
y0 = float(fish.ys.iloc[0]["Y"])
s0 = float(fish.ys.iloc[0]["S"])

# 合理起点
start = {
    "a": 0.1, "b": 0.95, "c_env": 0.0, "phi_E": 0.65,
    "log_sigma_p": np.log(0.25), "log_sigma_o": np.log(0.07),
    "log_sigma_E": np.log(0.25), "log_sigma_S": np.log(0.18),
    "X_0": y0, "E_0": s0,
}

rw_sd = RWSigma(sigmas={
    "a": 0.002, "b": 0.001, "c_env": 0.002, "phi_E": 0.002,
    "log_sigma_p": 0.002, "log_sigma_o": 0.001,
    "log_sigma_E": 0.002, "log_sigma_S": 0.001,
    "X_0": 0.0, "E_0": 0.0,
})

# 5 chains, jittered
np.random.seed(42)
starts = [start]
for _ in range(4):
    s = dict(start)
    for k in s:
        s[k] += np.random.normal(0, 0.02)
    starts.append(s)

t0 = time.time()
fish.mif(J=5000, M=150, rw_sd=rw_sd, a=0.5, theta=starts, key=jax.random.key(2026))
print(f"IF2 done in {time.time()-t0:.1f}s")

# Evaluate
tr = fish.traces()
pnames = list(rw_sd.sigmas.keys())

print("\n=== Endpoint evaluation (J=10000, 10 reps) ===")
for cid in sorted(tr["replicate"].unique()):
    grp = tr[tr["replicate"] == cid].sort_values("iteration")
    ep = {p: float(grp.iloc[-1][p]) for p in pnames}
    ep["X_0"] = y0
    ep["E_0"] = s0
    fe, _ = make_fish_m2()
    fe.pfilter(J=10000, theta=ep, reps=10, key=jax.random.key(500 + int(cid)))
    lls = fe.results_history[-1].logLiks.values.flatten()
    print(f"  Chain {int(cid)}: ll={np.mean(lls):.1f} ± {np.std(lls):.1f}  "
          f"sigma_o={np.exp(ep['log_sigma_o']):.4f}")

# Print best params
print("\nDone. Check which chain has highest loglik above.")
