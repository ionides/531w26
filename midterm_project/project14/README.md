# Review comments on Project 14, STATS 531 W26

The authors find and investigate an interesting dataset, suitable for time series analysis using the methods covered in class. 
Reviewers found quite a few errors, not all small, but that is the point of the exercise!

1. Sec 4. ADF is not a good choice here, because a unit-root linear process is not a good explanation of the changing situation for road accidents. Differencing also does not help to address that. This is an example of a situation where the only justification for differencing is a desire to stay in the "Box-Jenkins framework". But Box and Jenkins would have wanted to use their own methods more wisely.

1. The ADF test is applied to raw counts, not the log-transformed version `(adf_result = adfuller(y))`. The code runs the function on the untransformed crash counts, but it might be
better for the test to be run on the log-transformed `y`, since the model being built operates on
the log scale (The log-transform is done after the ADF test).

1. Sec 5. There is clear pattern in this plot. Viewing it as "approximately stationary" is wishful thinking.

1. Sec 7. The code shows that the periodograms were computed on the series after both first and seasonal differencing are applied. The
correct workflow would have been to have done the spectral analysis on the log transformed or
first-differenced series to identify the seasonal frequency and to use the results showing that
there is seasonal frequency, as justification to apply seasonal differencing. This may be why
the actual periodogram seems to show the highest peak at around 0.05, closer to a 20-month
cycle, than the reported 12 months. The report seems to conclude that the periodogram
corresponds to a 12 month cycle without clear derivation or steps and it might be better to
note the computation of how it was derived. Then, the discrepancy might have been clearer.

1. Sec 9. Ljung-Box is not designed to pick up the non-stationarity that is evident from the time plot.

1. The choice of SARIMA(1,1,1)x(1,1,1) is not well motivated. It is easy enough to compare against alternatives, for example by an AIC table

1. Sec 10. The Q-Q plot is far from a straight line. It may be unclear what to do about this, since not all non-normality can be controlled by a transformation, but at least it needs to be recognized.

1. The authors claim to have shown that their analysis shows how SARIMA analysis can work despite the pandemic, but they don't show whether better fit and better prediction would be available if some allowance was made for the pandemic (e.g., regression with ARMA errors and a suitable covariate).

1. Coding Decisions and Algorithmic Constraints: In the source code, the
SARIMAX function is called with `enforce_stationarity=False` and `enforce_invertibility=False`. While this prevents the optimizer from failing, it requires extra care for interpreting the output. Combining this with `warnings.filterwarnings("ignore")` adds to the danger of invisible numerical problems.

1. A notable related issue arises in section 8, with the reported MA1 coefficient of 2.1844. This
violates the invertibility of the model that requiring $|\theta_1| < 1$ for an MA1 model component. With the reported MA coefficient of 2.1844, residual calculations (from inverting the model) could be unstable.

1. The project is written to a high standard of scholarship and reproducibility.
