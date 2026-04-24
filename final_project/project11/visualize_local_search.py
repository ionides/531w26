import matplotlib.pyplot as plt
import pandas as pd

dfp = pd.read_csv("cdc_local_search/param_traces_all_runs.csv")

for p in dfp["parameter"].unique():
    plt.figure()

    for run_id, g in dfp[dfp["parameter"] == p].groupby("run"):
        plt.plot(g["iteration"], g["value"], alpha=0.7)

    plt.title(p)
    plt.xlabel("iteration")
    plt.ylabel(p)
    plt.show()