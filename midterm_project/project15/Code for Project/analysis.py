"""
Time series project: USD/INR and WTI crude oil.
Data are daily, Monday–Friday only. We do EDA, decomposition, stationarity
checks, then fit ARMA / ARIMA / SARIMA and compare with AIC. Finally we
look at whether one series helps predict the other (Granger, cross-correlation).
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from data_utils import load_usd_inr, load_wti, get_aligned_series, data_report

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# Seasonal period: 5 = one week (Mon–Fri). Set to 4 if you prefer a 4-step season
# (e.g. when some days are imputed and you want a shorter seasonal cycle).
SEASONAL_PERIOD = 5
DECOMPOSE_PERIOD = 21
ACF_LAGS = 40
FORECAST_STEPS = 21
# EDA plots: extend to this last date (e.g. last day of WTI data: 2026-02-09)
EDA_END_DATE = "2026-02-09"


def run_preprocessing():
    print("1. DATA & PREPROCESSING")
    data_report()
    usd = load_usd_inr()
    wti = load_wti()
    u, w = get_aligned_series(usd, wti)
    return u, w


def eda_and_decompose(series, name, end_date=None):
    """
    Plot the series and its decomposition into trend, seasonal, and residual.
    If end_date is given (e.g. 2026-02-09), trim the series so plots run to that last day.
    """
    if end_date is not None:
        end = pd.Timestamp(end_date)
        series = series.loc[series.index <= end]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    series.plot(ax=axes[0], title="{} – daily series (weekdays)".format(name))
    axes[0].set_ylabel("Level")

    decomp = seasonal_decompose(
        series.dropna(), model="additive", period=DECOMPOSE_PERIOD, extrapolate_trend="freq"
    )
    decomp.trend.plot(ax=axes[1], title="Trend")
    axes[1].set_ylabel("Trend")
    decomp.seasonal.plot(ax=axes[2], title="Seasonal")
    axes[2].set_ylabel("Seasonal")
    decomp.resid.dropna().plot(ax=axes[3], title="Residuals")
    axes[3].set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_eda_decompose_{}.png".format(name), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved 01_eda_decompose_{}.png".format(name))
    return decomp


# ---------- 3. Stationarity & Differencing ----------
def choose_d(series, alpha=0.05):
    """
    Use ADF tests to decide differencing order: d=0 if level is stationary,
    else d=1 (or d=2 if needed). Returns (d, dict of ADF results).
    """
    y = series.dropna()
    adf_level = adfuller(y, autolag="AIC")
    if adf_level[1] < alpha:
        return 0, {"adf_level": adf_level, "adf_diff1": None, "adf_diff2": None}
    z1 = y.diff().dropna()
    adf_diff1 = adfuller(z1, autolag="AIC")
    if adf_diff1[1] < alpha:
        return 1, {"adf_level": adf_level, "adf_diff1": adf_diff1, "adf_diff2": None}
    z2 = z1.diff().dropna()
    adf_diff2 = adfuller(z2, autolag="AIC")
    return 2, {"adf_level": adf_level, "adf_diff1": adf_diff1, "adf_diff2": adf_diff2}


def stationarity_and_differencing(series, name):
    """ACF and PACF for level and first difference; ADF to choose d."""
    z = series.diff().dropna()
    d_choice, adf_info = choose_d(series, alpha=0.05)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(series.dropna(), lags=ACF_LAGS, ax=axes[0, 0], title="{} – ACF (level)".format(name))
    plot_pacf(series.dropna(), lags=ACF_LAGS, ax=axes[0, 1], title="{} – PACF (level)".format(name))
    plot_acf(z, lags=ACF_LAGS, ax=axes[1, 0], title="{} – ACF (first difference)".format(name))
    plot_pacf(z, lags=ACF_LAGS, ax=axes[1, 1], title="{} – PACF (first difference)".format(name))
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"02_acf_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    a0 = adf_info["adf_level"]
    print("   {} ADF level:   stat={:.4f}, p={:.4f}  -> {}".format(
        name, a0[0], a0[1], "stationary" if a0[1] < 0.05 else "non-stationary"))
    if adf_info["adf_diff1"] is not None:
        a1 = adf_info["adf_diff1"]
        print("   {} ADF diff(1): stat={:.4f}, p={:.4f}  -> {}".format(
            name, a1[0], a1[1], "stationary" if a1[1] < 0.05 else "non-stationary"))
    if adf_info["adf_diff2"] is not None:
        a2 = adf_info["adf_diff2"]
        print("   {} ADF diff(2): stat={:.4f}, p={:.4f}  -> {}".format(
            name, a2[0], a2[1], "stationary" if a2[1] < 0.05 else "non-stationary"))
    print("   Suggested d = {} (from ADF)".format(d_choice))
    return z, adf_info


# ---------- 4. ARMA / ARIMA / SARIMA model selection ----------
# Seasonal period: 5 for daily weekdays (one “week” of trading days)
S = SEASONAL_PERIOD

# Seasonal orders to try: (P, D, Q). (0,0,0) = no seasonal = plain ARMA/ARIMA.
# Use fewer when series is short to avoid long runtimes.
SEASONAL_ORDERS_FULL = [
    (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
    (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
]
SEASONAL_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]  # default: no D=1 to save time


def _fit_candidate(series, order, seasonal_order):
    """Fit one SARIMAX. If seasonal_order is (0,0,0) we get plain ARMA/ARIMA."""
    try:
        so = (seasonal_order[0], seasonal_order[1], seasonal_order[2], S)
        model = SARIMAX(series, order=order, seasonal_order=so, trend="c")
        return model.fit(disp=False)
    except Exception:
        return None


def full_model_search(series, name, max_p=4, max_q=4, use_seasonal_D=False):
    """
    Try ARMA (d=0), ARIMA (d=1), and SARIMA with seasonal period S (5 for daily weekdays).
    We include both d=0 and d=1 so the table has ARMA, ARIMA, and SARIMA in the same format.
    Pick best by AIC; report BIC and LRT vs second-best when nested.
    """
    n = len(series.dropna())
    d_suggested, _ = choose_d(series, alpha=0.05)
    # Search over d=0 (ARMA / SARIMA with no diff) and d=1 (ARIMA / SARIMA) so ARMA appears in the table
    d_values = [0, 1]
    seasonal_orders = list(SEASONAL_ORDERS)
    if use_seasonal_D and n > 100:
        seasonal_orders = [s for s in SEASONAL_ORDERS_FULL if s[1] == 0 or (s[1] == 1 and n > 2 * S)]
    if not seasonal_orders:
        seasonal_orders = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]

    # Grid: (p,q) from (0,0) to (max_p, max_q); d in {0, 1} so we get ARMA + ARIMA + SARIMA
    results = []
    for d in d_values:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                order = (p, d, q)
                for (P, D, Q) in seasonal_orders:
                    fit = _fit_candidate(series, order, (P, D, Q))
                    if fit is None:
                        continue
                    model_type = "ARMA" if (d == 0 and (P, D, Q) == (0, 0, 0)) else (
                        "ARIMA" if (P, D, Q) == (0, 0, 0) else "SARIMA"
                    )
                    results.append({
                        "order": order,
                        "seasonal": (P, D, Q),
                        "AIC": fit.aic,
                        "BIC": fit.bic,
                        "llf": fit.llf,
                        "fit": fit,
                        "model_type": model_type,
                    })
    if not results:
        raise RuntimeError("No model converged for {}".format(name))
    results.sort(key=lambda x: x["AIC"])
    best = results[0]

    # Build table rows for export (no fit object): p, d, q, P, D, Q, model_type, model_label, AIC, BIC, rank
    table_rows = []
    for rank, r in enumerate(results, start=1):
        ord_ = r["order"]
        sea_ = r["seasonal"]
        label = "{} ({},{},{})".format(r["model_type"], ord_[0], ord_[1], ord_[2])
        if sea_ != (0, 0, 0):
            label += " ({},{},{})[{}]".format(sea_[0], sea_[1], sea_[2], S)
        table_rows.append({
            "p": ord_[0], "d": ord_[1], "q": ord_[2],
            "P": sea_[0], "D": sea_[1], "Q": sea_[2],
            "model_type": r["model_type"], "model_label": label,
            "AIC": r["AIC"], "BIC": r["BIC"],
            "rank_AIC": rank,
        })
    results_table = table_rows

    # Report top 5 and best model (table includes ARMA d=0 and ARIMA/SARIMA d=1)
    print("   {} model search (d=0 and d=1): top 5 by AIC".format(name))
    for i, r in enumerate(results[:5]):
        label = "({},{},{})".format(r["order"][0], r["order"][1], r["order"][2])
        if r["seasonal"] != (0, 0, 0):
            label += "({},{},{})[{}]".format(r["seasonal"][0], r["seasonal"][1], r["seasonal"][2], S)
        print("      {}. {} {}  AIC={:.2f}  BIC={:.2f}".format(i + 1, r["model_type"], label, r["AIC"], r["BIC"]))
    print("   -> Best (AIC): {} {}".format(best["model_type"], best["order"]), end="")
    if best["seasonal"] != (0, 0, 0):
        print(" {}[{}]".format(best["seasonal"], S), end="")
    print("  AIC={:.2f}  BIC={:.2f}".format(best["AIC"], best["BIC"]))

    best_bic = min(results, key=lambda x: x["BIC"])
    if best_bic is not best and (best_bic["AIC"] != best["AIC"] or best_bic["BIC"] != best["BIC"]):
        print("   Best (BIC):  {} {}".format(best_bic["model_type"], best_bic["order"]), end="")
        if best_bic["seasonal"] != (0, 0, 0):
            print(" {}[{}]".format(best_bic["seasonal"], S), end="")
        print("  BIC={:.2f}".format(best_bic["BIC"]))

    # LRT: best vs second if nested
    if len(results) >= 2:
        second = results[1]
        df_diff = (best["order"][0] - second["order"][0]) + (best["order"][2] - second["order"][2])
        df_diff += (best["seasonal"][0] - second["seasonal"][0]) + (best["seasonal"][2] - second["seasonal"][2])
        if df_diff > 0:
            lr, pval = lrt(second["llf"], best["llf"], df_diff)
            print("   LRT (best vs 2nd): LR={:.4f}, p={:.4f}".format(lr, pval))

    return best["fit"], best["order"], best["seasonal"], best["model_type"], results_table


def lrt(l0, l1, df_diff):
    """Likelihood ratio test: 2*(l1 - l0) ~ chi^2(df_diff)."""
    lr = 2 * (l1 - l0)
    p = 1 - stats.chi2.cdf(lr, df_diff)
    return lr, p


def results_to_dataframe(results_table):
    """Turn list of result dicts (p, d, q, P, D, Q, model_type, AIC, BIC, rank_AIC) into a DataFrame."""
    return pd.DataFrame(results_table)


def export_model_results_to_excel(results_by_series, filepath):
    """
    Export full model results (ARMA/ARIMA/SARIMA) to one Excel file, one sheet per series.
    results_by_series: dict like {"USD_INR": results_table, "WTI": results_table}.
    """
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for series_name, table in results_by_series.items():
            df = results_to_dataframe(table)
            # Sheet name must be <= 31 chars and no invalid chars
            sheet_name = series_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("   Full model table exported to: {}".format(filepath))


# ---------- 6. Diagnostics ----------
def diagnostics(model, series, name):
    """Plot residuals over time, ACF of residuals, and a Q-Q plot."""
    res = model.resid
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    res.plot(ax=axes[0], title="{} – residuals over time".format(name))
    plot_acf(res.dropna(), lags=min(ACF_LAGS, len(res) // 2 - 1), ax=axes[1], title="ACF of residuals")
    stats.probplot(res.dropna(), dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot (normality)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_diagnostics_{}.png".format(name), dpi=150, bbox_inches="tight")
    plt.close()
    lb_lag = min(21, len(res) // 3)
    if lb_lag >= 1:
        lb = acorr_ljungbox(res.dropna(), lags=[lb_lag], return_df=True)
        print("   {} Ljung-Box (lag {}): p={:.4f}".format(name, lb_lag, lb["lb_pvalue"].iloc[-1]))


# ---------- 7. Forecasting ----------
def forecast_plot(model, series, name, steps=None):
    """Forecast the next `steps` trading days with 95% CI."""
    if steps is None:
        steps = FORECAST_STEPS
    f = model.get_forecast(steps=steps)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_show = min(252, len(series))
    hist = series.iloc[-n_show:]
    hist.plot(ax=ax, label="History")
    f.predicted_mean.plot(ax=ax, label="Forecast")
    ax.fill_between(f.predicted_mean.index, f.conf_int().iloc[:, 0], f.conf_int().iloc[:, 1], alpha=0.3)
    ax.set_title("{} – {} trading-day forecast (95% CI)".format(name, steps))
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_forecast_{}.png".format(name), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved 04_forecast_{}.png".format(name))


# ---------- 8. Cross-series: does one affect the other? ----------
def cross_analysis(usd, wti):
    """Cross-correlation and Granger causality in both directions."""
    u, w = get_aligned_series(usd, wti)
    u_diff = u.diff().dropna()
    w_diff = w.diff().dropna()
    common = u_diff.index.intersection(w_diff.index)
    u_diff = u_diff.reindex(common).dropna()
    w_diff = w_diff.reindex(common).dropna()
    n = min(len(u_diff), len(w_diff))
    u_diff = u_diff.iloc[-n:]
    w_diff = w_diff.iloc[-n:]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    max_lag_ccf = 20
    ccf_vals = [u_diff.corr(w_diff.shift(lag)) for lag in range(-max_lag_ccf, max_lag_ccf + 1)]
    axes[0].bar(range(-max_lag_ccf, max_lag_ccf + 1), ccf_vals)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_title("Cross-correlation: USD/INR vs WTI (lag in trading days)")
    axes[0].set_xlabel("Lag (negative = USD/INR leads)")

    df = pd.DataFrame({"usd": u_diff, "wti": w_diff}).dropna()
    max_lag_granger = 10
    print("   Granger: WTI -> USD/INR")
    try:
        g1 = grangercausalitytests(df[["usd", "wti"]], maxlag=max_lag_granger, verbose=False)
        for lag in range(1, max_lag_granger + 1):
            p = g1[lag][0]["ssr_ftest"][1]
            print("      lag {}: p = {:.4f}".format(lag, p))
    except Exception as e:
        print("      Error: {}".format(e))
    print("   Granger: USD/INR -> WTI")
    try:
        g2 = grangercausalitytests(df[["wti", "usd"]], maxlag=max_lag_granger, verbose=False)
        for lag in range(1, max_lag_granger + 1):
            p = g2[lag][0]["ssr_ftest"][1]
            print("      lag {}: p = {:.4f}".format(lag, p))
    except Exception as e:
        print("      Error: {}".format(e))

    u_level, w_level = get_aligned_series(usd, wti)
    axes[1].plot(u_level.index, u_level.values, label="USD/INR", alpha=0.8)
    ax2 = axes[1].twinx()
    ax2.plot(w_level.index, w_level.values, color="orange", label="WTI", alpha=0.8)
    axes[1].set_title("USD/INR vs WTI (aligned weekdays)")
    axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_cross_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved 05_cross_series.png")


def main():
    print("=" * 60)
    print("STATS 531: USD/INR & WTI Time Series Analysis")
    print("=" * 60)

    usd, wti = run_preprocessing()
    u, w = get_aligned_series(usd, wti)

    print("\n2. EDA & DECOMPOSITION (to end date {})".format(EDA_END_DATE))
    eda_and_decompose(u, "USD_INR", end_date=EDA_END_DATE)
    eda_and_decompose(w, "WTI", end_date=EDA_END_DATE)

    print("\n3. STATIONARITY & DIFFERENCING")
    z_u, _ = stationarity_and_differencing(u, "USD_INR")
    z_w, _ = stationarity_and_differencing(w, "WTI")

    print("\n4. MODEL SELECTION: ARMA / ARIMA / SARIMA (AIC, LRT)")
    best_u, order_u, seasonal_u, type_u, table_u = full_model_search(u, "USD_INR", max_p=4, max_q=4)
    best_w, order_w, seasonal_w, type_w, table_w = full_model_search(w, "WTI", max_p=4, max_q=4)

    # Full table: all (p,d,q) and (P,D,Q) from (0,0,0) to (4,4,4) including ARMA, ARIMA, SARIMA
    df_u = results_to_dataframe(table_u)
    df_w = results_to_dataframe(table_w)
    print("\n   Full model table (USD_INR) — all ARMA/ARIMA/SARIMA from (0,0,0) to (4,4,4):")
    print(df_u.to_string(index=False))
    print("\n   Full model table (WTI) — all ARMA/ARIMA/SARIMA from (0,0,0) to (4,4,4):")
    print(df_w.to_string(index=False))
    export_model_results_to_excel(
        {"USD_INR": table_u, "WTI": table_w},
        OUT_DIR / "model_results_arma_arima_sarima.xlsx",
    )

    print("\n5. MODEL DIAGNOSTICS (best model per series)")
    diagnostics(best_u, u, "USD_INR")
    diagnostics(best_w, w, "WTI")

    print("\n6. FORECASTING ({} trading days ahead)".format(FORECAST_STEPS))
    forecast_plot(best_u, u, "USD_INR")
    forecast_plot(best_w, w, "WTI")

    print("\n7. CROSS-SERIES (Granger and cross-correlation)")
    cross_analysis(usd, wti)

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
