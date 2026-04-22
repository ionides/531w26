import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO


df = pd.read_csv("loglik_all_runs.csv")

# ensure numeric (handles blank first row per run)
df["loglik"] = pd.to_numeric(df["loglik"], errors="coerce")

plt.figure(figsize=(8,5))

for run_id, g in df.groupby("run"):
    g = g.sort_values("iteration")
    plt.plot(g["iteration"], g["loglik"], label=f"run {run_id}")

plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.title("Likelihood convergence across runs")
plt.legend()
plt.tight_layout()
plt.show()