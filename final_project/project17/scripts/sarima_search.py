import pandas as pd
import numpy as np
import warnings
import itertools
import statsmodels.api as sm
from pathlib import Path
from numpy.linalg import LinAlgError

# ignore convergence warnings during grid search
warnings.filterwarnings("ignore")


# added d_range to the function signature
def grid_search_sarima(
    series, exog, p_range, d_range, q_range, P_range, Q_range, D=1, m=12
):
    """
    Loops through parameters, finds the best BIC, and logs all results.
    """
    best_bic = float("inf")
    best_order = None
    best_seasonal = None

    search_history = []

    # added d_range to itertools.product
    parameters = list(
        itertools.product(p_range, d_range, q_range, P_range, Q_range)
    )
    print(f"Starting Grid Search: {len(parameters)} combinations to test.")

    # unpack 'd' from the parameters loop
    for p, d, q, P, Q in parameters:
        order = (p, d, q)
        seasonal_order = (P, D, Q, m)
        try:
            mod = sm.tsa.statespace.SARIMAX(
                endog=series,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = mod.fit(disp=False)

            current_bic = results.bic
            search_history.append(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "P": P,
                    "D": D,
                    "Q": Q,
                    "m": m,
                    "bic": current_bic,
                }
            )

            if current_bic < best_bic:
                best_bic = current_bic
                best_order = order
                best_seasonal = seasonal_order
                print(
                    f"New Best: {order}x{seasonal_order} | BIC: {best_bic:.2f}"
                )

        except (LinAlgError, ValueError):
            continue

    return best_order, best_seasonal, pd.DataFrame(search_history)


if __name__ == "__main__":
    # setup paths and load monthly data
    parent_dir = Path(__file__).parent
    input_file = parent_dir / "data" / "aggregated_monthly_polio_1930_1964.csv"
    out_dir = parent_dir / "sarima_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Run aggregate.py first to create {input_file}"
        )

    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"

    # pre-processing
    target_col = "log_cases"
    df[target_col] = np.log(df["cases"] + 1)

    # 1955 vaccine introduction as exogenous variable
    df["post_1955"] = (df.index >= "1955-01-01").astype(int)
    exog_series = df["post_1955"]

    # define grid ranges
    p_vals = range(0, 4)
    q_vals = range(0, 4)
    d_vals = range(0, 2)
    P_vals = range(0, 3)
    Q_vals = range(0, 3)

    # perform search and capture history
    # UPDATE 4: Passed d_range to the function correctly
    best_order, best_seasonal, history_df = grid_search_sarima(
        series=df[target_col],
        exog=exog_series,
        p_range=p_vals,
        d_range=d_vals,
        q_range=q_vals,
        P_range=P_vals,
        Q_range=Q_vals,
        D=1,
        m=12,
    )

    # save the full BIC history
    history_path = out_dir / "sarima_grid_search_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved full search history to {history_path}")

    # retrain final model with the exogenous variable
    final_model = sm.tsa.statespace.SARIMAX(
        endog=df[target_col],
        exog=exog_series,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_results = final_model.fit(disp=False)

    print(
        f"\nAll processes complete. Final Best: {best_order} x {best_seasonal}"
    )

    print("\nFINAL MODEL SUMMARY:")
    print(final_results.summary())
