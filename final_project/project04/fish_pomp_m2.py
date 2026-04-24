# fish_pomp_m2.py — Complete M2 pypomp model + PF test
# 一个文件搞定，直接跑: python fish_pomp_m2.py
#
# M2: Gompertz + state-space SOI (WHAM Ecov style)
#   E_t = phi * E_{t-1} + sigma_E * N(0,1)
#   S_t = E_t + sigma_S * N(0,1)           [observation]
#   X_t = a + b * X_{t-1} + c * E_{t-1} + sigma_p * N(0,1)
#   Y_t = X_t + sigma_o * N(0,1)           [observation]

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

from pypomp import Pomp

# ── Load data from astsa (hardcoded first/last few + full CSV approach) ──
def load_rec_soi():
    """Load rec/soi from R's astsa package via CSV, or download if needed."""
    try:
        # Try loading from CSV if you exported from R
        rec = pd.read_csv("rec.csv")["x"].values
        soi = pd.read_csv("soi.csv")["x"].values
    except FileNotFoundError:
        # If no CSV, try rpy2
        try:
            import rpy2.robjects as ro
            ro.r('library(astsa); data(rec); data(soi)')
            rec = np.array(ro.r('as.numeric(rec)'))
            soi = np.array(ro.r('as.numeric(soi)'))
        except ImportError:
            print("ERROR: Need rec/soi data.")
            print("In R, run:")
            print('  library(astsa); data(rec); data(soi)')
            print('  write.csv(data.frame(x=as.numeric(rec)), "rec.csv", row.names=FALSE)')
            print('  write.csv(data.frame(x=as.numeric(soi)), "soi.csv", row.names=FALSE)')
            raise SystemExit(1)

    y = np.log(rec)
    T = len(y)
    times = np.arange(1, T + 1, dtype=float)

    # Observations: Y_t (log rec) and S_t (SOI)
    ys = pd.DataFrame({"Y": y, "S": soi}, index=times)

    return ys


# ── Model components ──

def rinit(theta_, key, covars, t0):
    # X0 and E0 are estimated parameters — IF2 perturbation gives particle diversity
    return {"X": theta_["X_0"], "E": theta_["E_0"]}


def rproc(X_, theta_, key, covars, t, dt):
    a = theta_["a"]
    b = theta_["b"]
    c = theta_["c_env"]
    phi = theta_["phi_E"]
    sigma_p = jnp.exp(theta_["log_sigma_p"])
    sigma_E = jnp.exp(theta_["log_sigma_E"])

    k1, k2 = jax.random.split(key)
    E_new = phi * X_["E"] + sigma_E * jax.random.normal(k1)
    X_new = a + b * X_["X"] + c * X_["E"] + sigma_p * jax.random.normal(k2)
    return {"X": X_new, "E": E_new}


def dmeas(Y_, X_, theta_, covars, t):
    sigma_o = jnp.exp(theta_["log_sigma_o"])
    sigma_S = jnp.exp(theta_["log_sigma_S"])

    ll_Y = jax.scipy.stats.norm.logpdf(Y_["Y"], loc=X_["X"], scale=sigma_o)
    ll_S = jax.scipy.stats.norm.logpdf(Y_["S"], loc=X_["E"], scale=sigma_S)
    return ll_Y + ll_S


def rmeas(X_, theta_, key, covars, t):
    sigma_o = jnp.exp(theta_["log_sigma_o"])
    sigma_S = jnp.exp(theta_["log_sigma_S"])

    k1, k2 = jax.random.split(key)
    Y_sim = X_["X"] + sigma_o * jax.random.normal(k1)
    S_sim = X_["E"] + sigma_S * jax.random.normal(k2)
    return jnp.array([Y_sim, S_sim])


# ── Build Pomp object ──

def make_fish_m2(theta0=None):
    ys = load_rec_soi()

    if theta0 is None:
        theta0 = {
            "a":           0.3,
            "b":           0.9,
            "c_env":       0.0,
            "phi_E":       0.6,
            "log_sigma_p": np.log(0.25),
            "log_sigma_o": np.log(0.05),
            "log_sigma_E": np.log(0.3),
            "log_sigma_S": np.log(0.1),
            "X_0":         float(ys.iloc[0]["Y"]),  # start near first obs
            "E_0":         float(ys.iloc[0]["S"]),
        }

    pomp_obj = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys,
        t0=0.0,
        nstep=1,
        ydim=2,
        theta=theta0,
        statenames=["X", "E"],
    )
    return pomp_obj, theta0


# ============================================================
# Main: PF test at KF MLE, then IF2
# ============================================================
if __name__ == "__main__":
    from pypomp import RWSigma
    import time

    # ── Step 0: Build model with simulated data (like LG model) ──
    # Use known true parameters, simulate data, then PF should give correct loglik
    true_theta = {
        "a": 0.3, "b": 0.9, "c_env": -0.05, "phi_E": 0.7,
        "log_sigma_p": np.log(0.25), "log_sigma_o": np.log(0.15),
        "log_sigma_E": np.log(0.25), "log_sigma_S": np.log(0.15),
        "X_0": 3.0, "E_0": 0.0,
    }

    # Create a dummy ys with T=100 (shorter for quick test)
    T_sim = 100
    ys_dummy = pd.DataFrame(
        0.0, index=np.arange(1, T_sim + 1, dtype=float), columns=["Y", "S"]
    )

    fish_sim = Pomp(
        rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas,
        ys=ys_dummy, t0=0.0, nstep=1, ydim=2,
        theta=true_theta, statenames=["X", "E"],
    )

    # Simulate data from the model
    fish_sim = fish_sim.simulate(key=jax.random.key(42), nsim=1, as_pomp=True)
    print(f"Simulated T={len(fish_sim.ys)} observations from M2")

    # PF at true parameters — should give reasonable loglik with low SD
    print("\n=== PF at TRUE params on SIMULATED data ===")
    for J in [500, 2000, 10000]:
        fish_sim.pfilter(J=J, theta=true_theta, reps=5, key=jax.random.key(99))
        lls = fish_sim.results_history[-1].logLiks.values.flatten()
        print(f"  J={J:>6d}: {np.mean(lls):.1f} ± {np.std(lls):.1f}")

    # ── Step 1: Now test on real data ──
    print("\n=== Loading real data ===")
    fish_real, theta0 = make_fish_m2()
    print(f"T = {len(fish_real.ys)}, states = {fish_real.statenames}")

    # Get first obs for initial state
    y0 = float(fish_real.ys.iloc[0]["Y"])
    s0 = float(fish_real.ys.iloc[0]["S"])

    # PF at balanced params on real data
    balanced = {
        "a": 0.15, "b": 0.95, "c_env": 0.0, "phi_E": 0.65,
        "log_sigma_p": np.log(0.25), "log_sigma_o": np.log(0.15),
        "log_sigma_E": np.log(0.25), "log_sigma_S": np.log(0.15),
        "X_0": y0, "E_0": s0,
    }
    print("\n=== PF at balanced params on REAL data (sigma_o=sigma_S=0.15) ===")
    for J in [1000, 5000, 20000]:
        fish_real.pfilter(J=J, theta=balanced, reps=5, key=jax.random.key(42))
        lls = fish_real.results_history[-1].logLiks.values.flatten()
        print(f"  J={J:>6d}: {np.mean(lls):.1f} ± {np.std(lls):.1f}")

    # PF at IF2 HW8 best on real data
    if2_hw8 = {
        "a": 0.0527, "b": 0.9743, "c_env": 0.0422, "phi_E": 0.6778,
        "log_sigma_p": np.log(0.2516), "log_sigma_o": np.log(0.0729),
        "log_sigma_E": np.log(0.2247), "log_sigma_S": np.log(0.1840),
        "X_0": y0, "E_0": s0,
    }
    print("\n=== PF at IF2 HW8 params on REAL data (sigma_o=0.073) ===")
    for J in [1000, 5000, 20000]:
        fish_real.pfilter(J=J, theta=if2_hw8, reps=5, key=jax.random.key(99))
        lls = fish_real.results_history[-1].logLiks.values.flatten()
        print(f"  J={J:>6d}: {np.mean(lls):.1f} ± {np.std(lls):.1f}")

    print("\nDone. If simulated data PF looks good but real data doesn't,")
    print("the model equations are correct and the issue is parameter regime.")
