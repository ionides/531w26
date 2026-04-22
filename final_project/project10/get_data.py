#!/usr/bin/env python3
"""
get_data.py — Download and cache S&P 500 weekly log-returns.

Usage:
    python get_data.py

Produces: spx_weekly.csv (weekly percentage log-returns, index = 0,1,...,N-1)
"""

import yfinance as yf
import numpy as np
import pandas as pd

START = "2010-01-01"
END   = "2024-01-01"
SKIP  = 261           # first 261 weeks = 2010-01-04 to 2014-12-28

raw   = yf.download("^GSPC", start=START, end=END,
                    interval="1wk", progress=True, auto_adjust=True)
close = raw["Close"].squeeze().dropna()
log_ret = np.log(close / close.shift(1)).dropna() * 100

# Trim to 2015-onward
y = log_ret.values.flatten()[SKIP:]
df = pd.DataFrame({"y": y}, index=np.arange(len(y), dtype=float))
df.to_csv("spx_weekly.csv")

print(f"Saved {len(df)} weekly observations to spx_weekly.csv")
print(f"  Date range approx: 2015-01-02 to 2023-12-29")
print(f"  Mean   : {y.mean():.4f}%")
print(f"  Std    : {y.std():.4f}%")
print(f"  Skew   : {pd.Series(y).skew():.4f}")
print(f"  Kurtosis: {pd.Series(y).kurt():.4f}")
