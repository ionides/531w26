"""
Load and clean USD/INR and WTI data. We keep daily data only, and only
trading days (Monday through Friday). Weekends are dropped so the series
are aligned to the same weekday calendar.
"""
import pandas as pd
from pathlib import Path

# Project folder (where the CSV files live)
PROJECT_DIR = Path(__file__).resolve().parent


def parse_usd_inr_date(date_string):
    """Our USD/INR file has two date formats; handle both."""
    date_string = str(date_string).strip()
    if "/" in date_string:
        return pd.to_datetime(date_string, format="%m/%d/%Y")
    return pd.to_datetime(date_string, format="%d-%m-%Y")


def keep_weekdays_only(series):
    """Keep only Monday (0) through Friday (4). Drop weekends."""
    weekdays = series.index.dayofweek
    return series.loc[weekdays < 5].sort_index()


def load_usd_inr(path=None):
    """
    Load USD/INR from CSV, clean it, and return a daily series.
    Only weekdays (Mon–Fri) are kept. No monthly aggregation.
    """
    if path is None:
        path = PROJECT_DIR / "USD_INR Historical Data.csv"

    df = pd.read_csv(path)
    df["Date"] = df["Date"].apply(parse_usd_inr_date)
    df = df.dropna(subset=["Date", "Price"])

    # Make sure Price is numeric (in case of commas etc.)
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["Price"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])

    series = df.set_index("Date")["Price"]
    series.name = "USD_INR"
    series = keep_weekdays_only(series)
    return series


def load_wti(path=None, missing="ffill"):
    """
    Load WTI crude oil from CSV, clean it, and return a daily series.
    Only weekdays (Mon–Fri) are kept.
    Blank/missing oil prices: we do NOT drop. We impute:
      - "ffill": forward-fill (carry last observed price forward).
      - "average": linear interpolation (average of previous and next observed price).
      - "mean": fill all blanks with the mean of observed prices (computed over the series).
    Any leading NaNs (before first observed price) use backward-fill, except for "mean".
    """
    if path is None:
        path = PROJECT_DIR / "DCOILWTICO.csv"

    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["DCOILWTICO"] = pd.to_numeric(df["DCOILWTICO"], errors="coerce")
    df = df.sort_values("observation_date").drop_duplicates(subset=["observation_date"])
    series = df.set_index("observation_date")["DCOILWTICO"]

    n_missing = series.isna().sum()
    if n_missing > 0:
        if missing == "mean":
            series = series.fillna(series.mean())
        elif missing == "average":
            series = series.interpolate(method="linear")
            series = series.bfill()
        else:
            series = series.ffill()
            series = series.bfill()

    series.name = "WTI"
    series = keep_weekdays_only(series)
    return series


def count_wti_missing(path=None):
    """Return how many WTI rows have blank/missing price (for reporting)."""
    if path is None:
        path = PROJECT_DIR / "DCOILWTICO.csv"
    df = pd.read_csv(path)
    df["DCOILWTICO"] = pd.to_numeric(df["DCOILWTICO"], errors="coerce")
    return df["DCOILWTICO"].isna().sum()


def wti_missing_by_weekday(path=None):
    """
    Return counts of missing WTI price by weekday (0=Mon, ..., 4=Fri).
    Helps answer whether missing values are e.g. on Fridays (US holidays often fall on Mon/Fri).
    """
    if path is None:
        path = PROJECT_DIR / "DCOILWTICO.csv"
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["DCOILWTICO"] = pd.to_numeric(df["DCOILWTICO"], errors="coerce")
    missing = df["DCOILWTICO"].isna()
    dow = df.loc[missing, "observation_date"].dt.dayofweek
    return dow.value_counts().sort_index()


def _week_key(d):
    """(year, iso_week) for a date so we can group by week (Mon–Fri)."""
    return (d.year, d.isocalendar()[1])


def fill_oil_with_weekly_average(wti, target_dates):
    """
    For each date in target_dates where WTI is missing, fill with the mean of
    observed WTI in that same week (Mon–Fri). Keeps all 5 weekdays when forex
    has 5 and oil has 4 — the missing oil day gets the average of the other 4.
    """
    w = wti.reindex(target_dates)
    missing = w.isna()
    if not missing.any():
        return w
    # Build week -> list of dates in that week (from target_dates)
    week_to_dates = {}
    for d in target_dates:
        wk = _week_key(d)
        week_to_dates.setdefault(wk, []).append(d)
    # For each missing date, set value = mean of observed WTI in that week
    for d in w.index[missing]:
        wk = _week_key(d)
        same_week_dates = week_to_dates.get(wk, [d])
        observed_in_week = wti.reindex(same_week_dates).dropna()
        if len(observed_in_week) > 0:
            w.loc[d] = observed_in_week.mean()
        else:
            w.loc[d] = pd.NA
    # Any still missing (e.g. week with no oil at all): forward/back fill or series mean
    if w.isna().any():
        w = w.ffill().bfill()
    if w.isna().any():
        w = w.fillna(wti.mean())
    return w


def get_aligned_series(usd_inr=None, wti=None, keep_all_forex_days=True):
    """
    Return both series on the same set of dates.

    - If keep_all_forex_days=True (default): use all USD/INR (forex) weekdays as
      the index. Oil has only 4 days per week in many weeks; we fill the missing
      5th day with the **average of observed oil in that same week**, so we don't
      drop any forex dates.
    - If keep_all_forex_days=False: use intersection of dates only (drop days
      where either series is missing).
    """
    if usd_inr is None:
        usd_inr = load_usd_inr()
    if wti is None:
        wti = load_wti()

    if keep_all_forex_days:
        # Keep every weekday we have for forex; fill missing oil with weekly average
        target_dates = usd_inr.index.sort_values()
        u = usd_inr.loc[target_dates]
        w = fill_oil_with_weekly_average(wti, target_dates)
        return u, w

    common_dates = usd_inr.index.intersection(wti.index).sort_values()
    u = usd_inr.reindex(common_dates).dropna()
    w = wti.reindex(common_dates).dropna()
    common = u.index.intersection(w.index)
    u = u.loc[common]
    w = w.loc[common]
    return u, w


def data_report():
    """
    Print how many points we have: raw daily, after keeping weekdays only,
    and how many common (aligned) weekdays we use for the analysis.
    Also reports how many WTI rows were dropped due to blank/missing prices.
    """
    usd = load_usd_inr()
    wti = load_wti()
    u, w = get_aligned_series(usd, wti)

    n_missing_wti = count_wti_missing()
    overlap_dates = usd.index.intersection(wti.index).sort_values()
    n_overlap = len(overlap_dates)

    print("  Data are daily; only Monday–Friday are kept (weekends dropped).")
    if n_missing_wti > 0:
        print("  WTI: {} missing prices in CSV were imputed (forward-fill) when loading.".format(n_missing_wti))
        missing_by_dow = wti_missing_by_weekday()
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        parts = ["{}: {}".format(dow_names[i], missing_by_dow.get(i, 0)) for i in range(5)]
        print("  Missing WTI by weekday (raw CSV): " + ",  ".join(parts))
    print("  Overlapping days (dates where BOTH have data before any fill): {:>6}  from {}  to {}.".format(
        n_overlap,
        overlap_dates.min().strftime("%Y-%m-%d") if n_overlap else "N/A",
        overlap_dates.max().strftime("%Y-%m-%d") if n_overlap else "N/A",
    ))
    print("  Alignment: we keep ALL forex (USD/INR) weekdays; oil filled by weekly avg where missing.")
    print("")
    print("  USD/INR (weekdays):  {:>6} days   from {}  to {}".format(
        len(usd),
        usd.index.min().strftime("%Y-%m-%d"),
        usd.index.max().strftime("%Y-%m-%d"),
    ))
    print("  WTI (weekdays):      {:>6} days   from {}  to {}".format(
        len(wti),
        wti.index.min().strftime("%Y-%m-%d"),
        wti.index.max().strftime("%Y-%m-%d"),
    ))
    print("")
    print("  Aligned series used in analysis: {:>6} days  from {}  to {}".format(
        len(u),
        u.index.min().strftime("%Y-%m-%d"),
        u.index.max().strftime("%Y-%m-%d"),
    ))
    print("  (Same as USD/INR count; WTI filled for {} days where oil was missing.)".format(max(0, len(u) - n_overlap)))


if __name__ == "__main__":
    print("Data report (daily, Mon–Fri only):\n")
    data_report()
