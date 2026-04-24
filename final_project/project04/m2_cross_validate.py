# m2_cross_validate.py — Cross-validate TMB vs IF2 for M2
# 1. PF at TMB params
# 2. Print TMB params for R script to evaluate TMB at IF2 params

from fish_pomp_m2 import *
import jax, numpy as np
jax.config.update("jax_enable_x64", True)

fish, _ = make_fish_m2()
y0 = float(fish.ys.iloc[0]["Y"])
s0 = float(fish.ys.iloc[0]["S"])

# ── TMB MLE (constrained: sigma_b=0.01, sigma_o>=0.05) ──
tmb_params = {
    "a": 0.2510, "b": 0.9363, "c_env": -0.0056, "phi_E": 0.6420,
    "log_sigma_p": -1.3865, "log_sigma_o": -3.0000,
    "log_sigma_E": -1.2237, "log_sigma_S": -2.6324,
    "X_0": y0, "E_0": s0,
}

# ── IF2 MLE (Deepan Run 1, chain 9) ──
if2_params = {
    "a": 0.2699, "b": 0.9215, "c_env": -0.0035, "phi_E": 0.6751,
    "log_sigma_p": -1.3188, "log_sigma_o": -2.8838,
    "log_sigma_E": -1.3374, "log_sigma_S": -2.1465,
    "X_0": y0, "E_0": s0,
}

print("=== M2 Cross-Validation ===\n")
print("TMB params:")
for k, v in tmb_params.items():
    if k not in ("X_0", "E_0"):
        print(f"  {k} = {v:.4f}", end="")
        if "log_sigma" in k:
            print(f"  (sigma = {np.exp(v):.4f})", end="")
        print()

print("\nIF2 params:")
for k, v in if2_params.items():
    if k not in ("X_0", "E_0"):
        print(f"  {k} = {v:.4f}", end="")
        if "log_sigma" in k:
            print(f"  (sigma = {np.exp(v):.4f})", end="")
        print()

# ── PF at TMB params ──
print("\n=== PF at TMB params ===")
for J in [5000, 20000, 50000]:
    fish.pfilter(J=J, theta=tmb_params, reps=10, key=jax.random.key(77))
    lls = fish.results_history[-1].logLiks.values.flatten()
    print(f"  J={J:>6d}: {np.mean(lls):.1f} ± {np.std(lls):.1f}")

# ── PF at IF2 params ──
print("\n=== PF at IF2 params ===")
for J in [5000, 20000, 50000]:
    fish.pfilter(J=J, theta=if2_params, reps=10, key=jax.random.key(88))
    lls = fish.results_history[-1].logLiks.values.flatten()
    print(f"  J={J:>6d}: {np.mean(lls):.1f} ± {np.std(lls):.1f}")

print("\n=== Summary ===")
print("TMB reports:  loglik = -131.56 at TMB params")
print("TMB reports:  loglik = ???     at IF2 params  (run R script below)")
print("PF reports:   loglik = ???     at TMB params  (see above)")
print("PF reports:   loglik = -160.3  at IF2 params  (Deepan)")
print()
print("To get TMB loglik at IF2 params, run in R:")
print('  source("m2_cross_tmb_at_if2.R")')
